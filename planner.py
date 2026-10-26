import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from schemas import Schedule

ROOT = Path(__file__).resolve().parent


def get_client() -> Any:
    """Keep the user's tested OPENAI / SOCLAAS OpenAI-compatible connection."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "Set API_KEY and BASE_URL in the environment "
            "or in the project's .env file."
        )
    from openai import OpenAI  # Installed by requirements.txt; only needed for real API calls.
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))


def load_instance(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_prompt(path: str | Path = ROOT / "prompts" / "plan.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_schedule(text: str) -> Schedule:
    """Parse well-formed JSON, optionally enclosed in one Markdown code fence."""
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return Schedule.model_validate_json(cleaned)


def generate_candidate(
    client: Any,
    instance: dict[str, Any],
    *,
    model: str = "qwen3.5:9b",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """One-shot LLM call. No semantic constraint checking or automated schedule repair."""
    if system_prompt is None:
        system_prompt = load_prompt(ROOT / "prompts" / "plan.txt")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Find a complete feasible voyage schedule for this "
                    "Return only JSON.\n\n"
                    + json.dumps(instance, indent=2, ensure_ascii=False)
                ),
            },
        ],
        reasoning_effort="high",
    )

    message = response.choices[0].message
    raw = message.content or ""
    usage = response.usage.model_dump() if response.usage else None
    result: dict[str, Any] = {
        "model": model,
        "instance_id": instance.get("instance_id"),
        "raw_response": raw,
        "schedule": None,
        "parse_error": None,
        "finish_reason": response.choices[0].finish_reason,
        "usage": usage,
    }

    try:
        result["schedule"] = parse_schedule(raw).model_dump()
    except (ValidationError, ValueError) as exc:
        result["parse_error"] = str(exc)

    return result
