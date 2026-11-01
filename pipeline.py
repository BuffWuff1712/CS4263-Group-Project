
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import APIError

import planner
import simulator
import repair
from feedback import format_feedback

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
    safeguard=False, initial_schedule=None,
    resume_final=False, request_timeout=None,
):

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
        )
    else:
        saved = json.loads(Path(initial_schedule).read_text(encoding="utf-8"))
        if "voyages" in saved:
            saved_schedule = saved
        elif resume_final and saved.get("schedule"):
            saved_schedule = saved["schedule"]
        elif saved.get("attempts") and saved["attempts"][0].get("schedule"):
            saved_schedule = saved["attempts"][0]["schedule"]
        else:
            saved_schedule = saved.get("schedule")
        candidate = {
            "schedule": Schedule.model_validate(saved_schedule).model_dump(),
            "model": model,
            "usage": None,
            "finish_reason": "reused_initial_schedule",
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
                    "parse_error": candidate["parse_error"]
                }
            ]
        }

    schedule = Schedule.model_validate(
        candidate["schedule"]
    )

    # ======================================
    # 3. SIMULATE - REPAIR LOOP
    # ======================================

    for attempt in range(max_repairs + 1):

        # SIMULATE
        report = ValidationReport.model_validate(
            simulator.simulate(instance, schedule)
        )

        # Record results
        history.append({
            "attempt": attempt,
            "stage": "plan" if attempt == 0 else "repair",
            "schedule": schedule.model_dump(),
            "validation": report.model_dump()
        })

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
            if safeguard else [v.model_dump() for v in report.violations]
        )
        history[-1]["repair_feedback"] = feedback

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
                format_recovery=safeguard,
                include_metadata=True,
                request_timeout=request_timeout,
            )
            history[-1]["repair_metadata"] = repair_metadata

            revised_schedule = Schedule.model_validate(revised_schedule)
            if safeguard:
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

        except (ValueError, TypeError, APIError) as exc:

            print(f"Repair failed: {exc}")

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
                "attempts": history
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
        "planner_metadata": {
            "model": candidate["model"],
            "usage": candidate["usage"],
            "finish_reason": candidate["finish_reason"],
            "initial_schedule": str(initial_schedule) if initial_schedule else None,
            "safeguard": safeguard,
            "resume_final": resume_final,
            "request_timeout": request_timeout,
        }
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
    parser.add_argument("--safeguard", action="store_true", help="Use grounded feedback, output recovery, and delivery preservation")
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

    result = run_pipeline(
        args.instance, max_repairs=args.max_repairs,
        provider=args.provider, model=args.model, reasoning=args.reasoning_effort,
        safeguard=args.safeguard, initial_schedule=args.initial_schedule,
        resume_final=args.resume_final,
        request_timeout=args.request_timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nPipeline completed.")
    print(f"Success: {result['success']}")
    print(f"Total attempts: {len(result['attempts'])}")
