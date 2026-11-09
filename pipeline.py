
import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import APIError

import planner
import simulator
import repair
from feedback import format_feedback
from safeguards import SafeguardConfig

from schemas import Schedule, ValidationReport


MAX_REPAIRS = 3

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def delivery_regressions(instance, before: ValidationReport, after: ValidationReport):
    """Find shipments that lost delivered units in a proposed repair."""
    return [
        {
            "shipment_id": shipment["id"],
            "required_units": shipment["quantity"],
            "previously_delivered": before.delivered.get(shipment["id"], 0),
            "candidate_delivered": after.delivered.get(shipment["id"], 0),
            "lost_units": before.delivered.get(shipment["id"], 0)
                          - after.delivered.get(shipment["id"], 0),
        }
        for shipment in instance["shipments"]
        if after.delivered.get(shipment["id"], 0) < before.delivered.get(shipment["id"], 0)
    ]


def run_pipeline(
    instance_path, max_repairs=MAX_REPAIRS, *,
    provider="openai", model=None, reasoning=None,
    safeguard=False, grounded_feedback=False, output_recovery=False,
    delivery_preservation=False, planning_support=False,
    guided_repair=False, repair_guards=False, all_safeguards=False,
    tiered_safeguards=False,
    initial_schedule=None,
    resume_final=False, request_timeout=None,
    outline="off", trace=False, config=None,
):
    """Run plan -> simulate -> repair.

    Safeguards come from `config` if given; otherwise from the flags.
    Tiered mode starts with a guided repair and only adds the outline to
    later repairs if the simulator has not yet accepted a schedule.
    """
    if tiered_safeguards:
        if config is not None or any((
            safeguard, grounded_feedback, output_recovery,
            delivery_preservation, planning_support, guided_repair,
            repair_guards, all_safeguards, outline != "off", trace,
        )):
            raise ValueError("tiered_safeguards cannot be combined with other safeguard controls")
        config = SafeguardConfig(
            grounded_feedback=True,
            output_recovery=True,
            delivery_preservation=True,
            outline="rolling",
            trace=True,
        )
    elif config is None:
        config = SafeguardConfig.from_flags(
            safeguard=safeguard, grounded_feedback=grounded_feedback,
            output_recovery=output_recovery,
            delivery_preservation=delivery_preservation,
            planning_support=planning_support, guided_repair=guided_repair,
            repair_guards=repair_guards, all_safeguards=all_safeguards,
            outline=outline, trace=trace,
        )

    # ======================================
    # 1. LOAD PROBLEM INSTANCE
    # ======================================

    instance = planner.load_instance(instance_path)
    model = model or planner.get_model(provider)
    if reasoning is None:
        reasoning_key = {
            "soclaas": "SOCLAAS_REASONING_EFFORT",
            "openrouter": "OPENROUTER_REASONING_EFFORT",
            "openai": "REASONING_EFFORT",
        }[provider]
        reasoning = os.getenv(reasoning_key) or None

    history = []
    regression_warning = None

    # ======================================
    # 2. PLAN
    # ======================================

    if initial_schedule is None:
        client = planner.get_client(provider)
        if request_timeout is not None:
            client = client.with_options(timeout=request_timeout)
        candidate = planner.generate_candidate(
            client=client,
            instance=instance,
            model=model,
            reasoning=reasoning,
            outline=config.uses_outline,
        )
    else:
        saved = json.loads(Path(initial_schedule).read_text(encoding="utf-8"))
        saved_outline = None
        saved_raw = None
        if "voyages" in saved:
            saved_schedule = saved
        elif saved.get("status") == "plan_parse_failed" and saved.get("attempts"):
            saved_schedule = None
            saved_raw = saved["attempts"][0].get("raw_response")
        elif resume_final and saved.get("schedule"):
            saved_schedule = saved["schedule"]
            saved_outline = next(
                (attempt.get("outline") for attempt in reversed(saved.get("attempts", []))
                 if attempt.get("schedule") == saved_schedule and attempt.get("outline")),
                None,
            )
        elif saved.get("attempts") and saved["attempts"][0].get("schedule"):
            saved_schedule = saved["attempts"][0]["schedule"]
            saved_outline = saved["attempts"][0].get("outline") or saved.get("planner_metadata", {}).get("outline")
        else:
            saved_schedule = saved.get("schedule")
        candidate = {
            "schedule": Schedule.model_validate(saved_schedule).model_dump() if saved_schedule is not None else None,
            "model": model,
            "usage": None,
            "finish_reason": "reused_initial_schedule",
            "outline": saved_outline,
            "raw_response": saved_raw,
            "parse_error": saved.get("attempts", [{}])[0].get("parse_error") if saved_raw else None,
        }

    # Apply the same bounded JSON recovery to an initial reply when requested.
    # This also permits replaying a saved malformed reply without another API call.
    if candidate["schedule"] is None and config.output_recovery and candidate.get("raw_response"):
        try:
            recovered, recovery_metadata = repair.parse_repair_response(
                candidate["raw_response"], instance, recover=True,
            )
            candidate["schedule"] = recovered.model_dump()
            candidate["initial_recovery"] = recovery_metadata
        except ValueError:
            pass

    def planner_metadata():
        return {
            "model": candidate["model"],
            "provider": provider,
            "reasoning_effort": reasoning,
            "usage": candidate["usage"],
            "finish_reason": candidate["finish_reason"],
            "initial_schedule": str(initial_schedule) if initial_schedule else None,
            "safeguard": config.grounded_feedback and config.output_recovery and config.delivery_preservation,
            "safeguards": config.to_dict(),
            "outline": candidate.get("outline"),
            "resume_final": resume_final,
            "request_timeout": request_timeout,
            "initial_recovery": candidate.get("initial_recovery"),
            "tiered_safeguards": tiered_safeguards,
        }

    # Check if planner returned a parseable schedule
    if candidate["schedule"] is None:

        print("Planner failed to generate valid JSON.")
        print(candidate["parse_error"])

        return {
            "success": False,
            "status": "plan_parse_failed",
            "schedule": None,
            "validation": None,
            "attempts": [
                {
                    "attempt": 0,
                    "stage": "plan",
                    "raw_response": candidate["raw_response"],
                    "parse_error": candidate["parse_error"],
                    "outline": candidate.get("outline")
                }
            ],
            "planner_metadata": planner_metadata(),
        }

    schedule = Schedule.model_validate(
        candidate["schedule"]
    )

    # The outline that matches the current accepted schedule.
    current_outline = candidate.get("outline")

    # ======================================
    # 3. SIMULATE - REPAIR LOOP
    # ======================================

    for attempt in range(max_repairs + 1):

        # SIMULATE
        report = ValidationReport.model_validate(
            simulator.simulate(instance, schedule, with_trace=config.trace)
        )

        # Record results
        history.append({
            "attempt": attempt,
            "stage": "plan" if attempt == 0 else "repair",
            "schedule": schedule.model_dump(),
            "validation": report.model_dump()
        })
        if config.uses_outline:
            history[-1]["outline"] = current_outline

        print(f"\nAttempt {attempt}")
        print(f"Valid: {report.valid}")
        print(f"Violations: {len(report.violations)}")

        # Success: stop immediately
        if report.valid:
            print("Feasible schedule found!")
            break

        # Stop if maximum repairs reached
        if attempt == max_repairs:
            print("Maximum repair attempts reached.")
            break

        feedback = (
            format_feedback(report.violations, instance=instance, schedule=schedule)
            if config.grounded_feedback else [v.model_dump() for v in report.violations]
        )
        history[-1]["repair_feedback"] = feedback

        # The first tier uses the guided-repair prompt unchanged. Recovery
        # and delivery checks act after the reply; an outline is added only
        # if another repair is needed.
        repair_outline = config.uses_outline and not (tiered_safeguards and attempt == 0)
        if tiered_safeguards:
            history[-1]["repair_safeguard_stage"] = (
                "guided_repair" if attempt == 0 else "outline_escalation"
            )

        # ==================================
        # 4. REPAIR
        # ==================================

        try:
            revised_schedule, repair_metadata = repair.repair(
                instance,
                schedule,
                report.violations,
                model=model,
                provider=provider,
                reasoning=reasoning,
                feedback=feedback,
                additional_feedback=regression_warning,
                format_recovery=config.output_recovery,
                include_metadata=True,
                request_timeout=request_timeout,
                outline=repair_outline,
                previous_outline=current_outline if repair_outline and config.outline == "rolling" else None,
                trace=report.trace if config.trace else None,
            )
            history[-1]["repair_metadata"] = repair_metadata

            revised_schedule = Schedule.model_validate(revised_schedule)
            if config.delivery_preservation:
                revised_report = ValidationReport.model_validate(
                    simulator.simulate(instance, revised_schedule)
                )
                losses = delivery_regressions(instance, report, revised_report)
                if losses and not revised_report.valid:
                    regression_warning = {
                        "instruction": (
                            "The previous candidate was rejected because it lost delivered "
                            "cargo. Revise the accepted previous_schedule while preserving "
                            "these shipment deliveries."
                        ),
                        "lost_shipments": losses,
                        "rejected_schedule": revised_schedule.model_dump(),
                    }
                    history[-1]["delivery_preservation"] = {
                        "accepted": False,
                        "lost_shipments": losses,
                        "candidate_violation_count": len(revised_report.violations),
                        "candidate_delivered": revised_report.delivered,
                        "rejected_schedule": revised_schedule.model_dump(),
                    }
                    print(f"Delivery preservation check rejected repair: {len(losses)} shipment(s) lost units.")
                    continue
                history[-1]["delivery_preservation"] = {"accepted": True, "lost_shipments": []}
            regression_warning = None
            schedule = revised_schedule
            if config.uses_outline:
                current_outline = repair_metadata.get("outline") if repair_outline else None

        except (ValueError, TypeError, APIError) as exc:

            print(f"Repair failed: {exc}")

            if tiered_safeguards and isinstance(exc, ValueError) and attempt + 1 < max_repairs:
                history[-1]["repair_error"] = str(exc)
                print("Escalating to an outline-assisted repair.")
                continue

            history.append({
                "attempt": attempt + 1,
                "stage": "repair",
                "schedule": None,
                "validation": None,
                "error": str(exc)
            })

            return {
                "success": False,
                "status": "api_failed" if isinstance(exc, APIError) else "repair_failed",
                "schedule": schedule.model_dump(),
                "validation": report.model_dump(),
                "attempts": history,
                "planner_metadata": planner_metadata(),
            }

    # ======================================
    # 5. RETURN RESULTS
    # ======================================

    return {
        "success": report.valid,
        "status": "success" if report.valid else "max_repairs",
        "schedule": schedule.model_dump(),
        "validation": report.model_dump(),
        "attempts": history,
        "planner_metadata": planner_metadata(),
    }


# ======================================
# 6. RUN PIPELINE
# ======================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the plan-simulate-repair pipeline")
    parser.add_argument("--provider", choices=("openai", "soclaas", "openrouter"), default="openai")
    parser.add_argument("--model", help="Override the provider's configured model")
    parser.add_argument("--instance", type=Path, default=BASE_DIR / "data" / "hard.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results" / "pipeline_result.json")
    parser.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    parser.add_argument("--reasoning-effort", help="Use only with a model that supports this setting")
    groups = parser.add_argument_group("safeguard groups")
    groups.add_argument("--planning-support", action="store_true", help="Use a rolling plan outline")
    groups.add_argument("--guided-repair", action="store_true", help="Use grounded feedback and the simulation trace")
    groups.add_argument("--repair-guards", action="store_true", help="Use output recovery and delivery preservation")
    groups.add_argument("--all-safeguards", action="store_true", help="Enable all three safeguard groups together")
    groups.add_argument("--tiered-safeguards", action="store_true",
                        help="Try guided repair first; add an outline only if another repair is needed")
    parser.add_argument("--initial-schedule", type=Path, help="Reuse a saved pipeline result or schedule for a controlled comparison")
    parser.add_argument("--resume-final", action="store_true", help="Use the final accepted schedule from --initial-schedule")
    parser.add_argument("--request-timeout", type=float, help="Seconds to wait for each model response")
    args = parser.parse_args()
    if args.max_repairs < 0:
        parser.error("--max-repairs cannot be negative")
    if args.resume_final and not args.initial_schedule:
        parser.error("--resume-final requires --initial-schedule")
    if args.request_timeout is not None and args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.tiered_safeguards and any((
        args.planning_support, args.guided_repair,
        args.repair_guards, args.all_safeguards,
    )):
        parser.error("--tiered-safeguards cannot be combined with other safeguard groups")
    started_at = time.perf_counter()
    result = run_pipeline(
        args.instance, max_repairs=args.max_repairs,
        provider=args.provider, model=args.model, reasoning=args.reasoning_effort,
        planning_support=args.planning_support,
        guided_repair=args.guided_repair,
        repair_guards=args.repair_guards,
        all_safeguards=args.all_safeguards,
        tiered_safeguards=args.tiered_safeguards,
        initial_schedule=args.initial_schedule,
        resume_final=args.resume_final,
        request_timeout=args.request_timeout,
    )
    result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nPipeline completed.")
    print(f"Success: {result['success']}")
    print(f"Total attempts: {len(result['attempts'])}")
