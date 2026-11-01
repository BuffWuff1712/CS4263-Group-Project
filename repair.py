
import json
import re
import time
import random
from pathlib import Path
from time import time_ns

from pydantic import ValidationError

from planner import get_client, load_prompt
from schemas import Schedule, Violation
from openai import RateLimitError
import simulator


ROOT = Path(__file__).resolve().parent


def _save_failed_response(content: str) -> Path:
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"failed_repair_{time_ns()}.txt"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _strict_schedule(content: str) -> Schedule:
    if not content.strip():
        raise ValueError("LLM returned an empty response")
    cleaned = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    data = json.loads(cleaned)
    if isinstance(data, dict) and set(data) == {"revised_schedule"}:
        data = data["revised_schedule"]
    return Schedule.model_validate(data)


def parse_repair_response(content: str, instance: dict, *, recover: bool = False):
    """Strict parse first; optionally score complete embedded schedules."""
    try:
        return _strict_schedule(content), {"method": "strict", "candidates": 1}
    except (ValidationError, ValueError, TypeError) as strict_error:
        if not recover:
            raise ValueError(str(strict_error)) from strict_error
        strict_message = str(strict_error)

    decoder = json.JSONDecoder()
    candidates = []
    seen = set()
    # Bound scanning and simulator work even for a very long model reply.
    for match in re.finditer(r"\{", content[:200_000]):
        try:
            data, _ = decoder.raw_decode(content[match.start():])
            if isinstance(data, dict) and set(data) == {"revised_schedule"}:
                data = data["revised_schedule"]
            schedule = Schedule.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            continue
        key = schedule.model_dump_json()
        if key not in seen:
            seen.add(key)
            report = simulator.simulate(instance, schedule)
            candidates.append((schedule, len(report.violations)))
            if len(candidates) == 10:
                break

    if not candidates:
        raise ValueError(f"No complete schedule JSON found: {strict_message}")
    best_index = min(range(len(candidates)), key=lambda i: candidates[i][1])
    return candidates[best_index][0], {
        "method": "embedded_json",
        "candidates": len(candidates),
        "candidate_violation_counts": [count for _, count in candidates],
        "selected_candidate": best_index,
    }

def call_with_retry(client, **kwargs):
    max_attempts = 6

    for attempt in range(max_attempts):
        try:
            return client.with_options(
                max_retries=0
            ).chat.completions.create(**kwargs)

        except RateLimitError as e:
            if attempt == max_attempts - 1:
                raise

            headers = e.response.headers
            retry_after = headers.get("retry-after")

            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                match = re.search(
                    r"try again in ([\d.]+)s",
                    str(e),
                    re.IGNORECASE
                )
                delay = (
                    float(match.group(1))
                    if match
                    else min(20 * (2 ** attempt), 120)
                )

            delay = max(delay, 1) + random.uniform(1, 3)

            print(f"Rate limited. Retrying in {delay:.1f}s")
            time.sleep(delay)


def repair(
    instance: dict,
    previous_schedule: Schedule,
    violations: list[Violation],
    model: str = "qwen3.5:9b",
    provider: str = "openai",
    reasoning: str | None = None,
    feedback=None,
    additional_feedback: dict | None = None,
    format_recovery: bool = False,
    include_metadata: bool = False,
    request_timeout: float | None = None,
) -> Schedule | tuple[Schedule, dict]:

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

        "violations": feedback if feedback is not None else [
            violation.model_dump() for violation in violations
        ]
    }
    if additional_feedback:
        repair_input["delivery_preservation_warning"] = additional_feedback

    # ======================================
    # 3. CALL LLM
    # ======================================

    client = get_client(provider)
    if request_timeout is not None:
        client = client.with_options(timeout=request_timeout)

    request = {
        "model": model,
        "messages": [
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
    }
    if reasoning:
        request["reasoning_effort"] = reasoning
    response = call_with_retry(client=client, **request)

    # ======================================
    # 4. PARSE REVISED SCHEDULE
    # ======================================

    content = response.choices[0].message.content or ""
    metadata = {"format_retries": 0, "finish_reason": response.choices[0].finish_reason}

    try:
        revised_schedule, parse_metadata = parse_repair_response(
            content, instance, recover=format_recovery,
        )
    except ValueError as first_error:
        first_path = _save_failed_response(content)
        if not format_recovery:
            raise ValueError(
                f"Invalid revised schedule: {first_error}. Raw response saved to {first_path}"
            ) from first_error

        # One bounded retry. The verifier still decides whether the result is feasible.
        retry_request = dict(request)
        retry_request["messages"] = [
            {"role": "system", "content": (
                "Revise previous_schedule to satisfy the constraints in problem_instance. "
                "Return only one complete JSON object with exactly the key voyages. "
                "No explanation, Markdown, or additional schedules."
            )},
            request["messages"][1],
        ]
        retry_response = call_with_retry(client=client, **retry_request)
        content = retry_response.choices[0].message.content or ""
        metadata.update(format_retries=1, first_failed_response=str(first_path),
                        finish_reason=retry_response.choices[0].finish_reason)
        try:
            revised_schedule, parse_metadata = parse_repair_response(
                content, instance, recover=True,
            )
        except ValueError as retry_error:
            retry_path = _save_failed_response(content)
            raise ValueError(
                f"Invalid revised schedule after one format retry: {retry_error}. "
                f"Raw responses saved to {first_path} and {retry_path}"
            ) from retry_error

    metadata.update(parse_metadata)
    return (revised_schedule, metadata) if include_metadata else revised_schedule

