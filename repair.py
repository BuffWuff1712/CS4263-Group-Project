
import json
import os
import re
from pathlib import Path
from time import time_ns

from pydantic import ValidationError

from planner import get_client, load_prompt, parse_schedule
from schemas import Schedule, Violation


ROOT = Path(__file__).resolve().parent

def repair(
    instance: dict,
    previous_schedule: Schedule,
    violations: list[Violation],
    model: str = "qwen3.5:9b"
) -> Schedule:

    # ======================================
    # 1. LOAD REPAIR PROMPT
    # ======================================

    repair_prompt = load_prompt(
        ROOT / "prompts" / "repair.txt"
    )

    # ======================================
    # 2. PREPARE REPAIR INPUT
    # ======================================

    repair_input = {
        "problem_instance": instance,

        "previous_schedule": previous_schedule.model_dump(),

        "violations": [
            violation.model_dump()
            for violation in violations
        ]
    }

    # ======================================
    # 3. CALL LLM
    # ======================================

    client = get_client()

    model = model

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": repair_prompt
            },
            {
                "role": "user",
                "content": json.dumps(
                    repair_input,
                    indent=2,
                    ensure_ascii=False
                )
            }
        ],
        reasoning_effort="high"
    )

    
    # ======================================
    # 4. PARSE REVISED SCHEDULE
    # ======================================

    content = response.choices[0].message.content or ""

    try:
        if not content.strip():
            raise ValueError("LLM returned an empty response")

        # Remove optional Markdown JSON fences
        cleaned = content.strip()

        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            cleaned = match.group(1).strip()

        # Parse JSON
        data = json.loads(cleaned)

        # Accept Qwen's extra revised_schedule wrapper
        if (
            isinstance(data, dict)
            and set(data.keys()) == {"revised_schedule"}
        ):
            data = data["revised_schedule"]

        # Validate using the original strict schema
        revised_schedule = Schedule.model_validate(data)

    except (ValidationError, ValueError, TypeError) as exc:

        output_dir = ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"failed_repair_{time_ns()}.txt"
        )

        output_path.write_text(
            content,
            encoding="utf-8"
        )

        raise ValueError(
            f"Invalid revised schedule: {exc}. "
            f"Raw response saved to {output_path}"
        ) from exc

    return revised_schedule

