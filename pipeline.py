
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import planner
import simulator
import repair

from schemas import Schedule, ValidationReport


MAX_REPAIRS = 3

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


def run_pipeline(instance_path, max_repairs=MAX_REPAIRS):

    # ======================================
    # 1. LOAD PROBLEM INSTANCE
    # ======================================

    instance = planner.load_instance(instance_path)

    history = []

    # ======================================
    # 2. PLAN
    # ======================================

    client = planner.get_client()

    candidate = planner.generate_candidate(
        client=client,
        instance=instance,
        model=MODEL,
        reasoning=os.getenv("REASONING_EFFORT", "")
    )

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

        # ==================================
        # 4. REPAIR
        # ==================================

        try:
            revised_schedule = repair.repair(
                instance,
                schedule,
                report.violations,
                model=MODEL
            )

            schedule = Schedule.model_validate(
                revised_schedule
            )

        except (ValueError, TypeError) as exc:

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
                "status": "repair_failed",
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
            "finish_reason": candidate["finish_reason"]
        }
    }


# ======================================
# 6. RUN PIPELINE
# ======================================

if __name__ == "__main__":

    instance_path = BASE_DIR / "data" / "hard.json"

    result = run_pipeline(instance_path)

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    with open(
        output_dir / "pipeline_result.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)

    print("\nPipeline completed.")
    print(f"Success: {result['success']}")
    print(f"Total attempts: {len(result['attempts'])}")
