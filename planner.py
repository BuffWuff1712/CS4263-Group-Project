import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from schemas import Schedule

ROOT = Path(__file__).resolve().parent


def get_client(provider: str = "openai") -> Any:
    """Connect to OpenAI, SOCLAAS, or OpenRouter with separate credentials."""
    load_dotenv(ROOT / ".env")
    if provider not in ("openai", "soclaas", "openrouter"):
        raise ValueError(f"Unknown provider: {provider}")
    prefix = provider.upper()
    api_key = os.environ.get(f"{prefix}_API_KEY")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            f"Set {prefix}_API_KEY and {prefix}_BASE_URL in the environment "
            "or in the project's .env file."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))


def get_model(provider: str) -> str:
    """Choose a model from the selected provider without mixing credentials."""
    load_dotenv(ROOT / ".env")
    if provider == "soclaas":
        return os.getenv("SOCLAAS_MODEL", "qwen3.6:35b")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-max-prime")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-5")
    raise ValueError(f"Unknown provider: {provider}")


def load_instance(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_prompt(path: str | Path = ROOT / "prompts" / "plan.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


# A line starting with "SCHEDULE:" (Markdown "#" or "**" around it is tolerated).
_SCHEDULE_MARKER = re.compile(r"^[ \t#*]*SCHEDULE:[ \t*]*", re.MULTILINE)
_OUTLINE_HEADER = re.compile(r"^\s*[#*]*\s*OUTLINE:[ \t*]*\n?")


def split_outline(text: str) -> tuple[str | None, str]:
    """Split an "OUTLINE: ... SCHEDULE: <json>" reply into (outline, json_text).

    Uses the last line that starts with "SCHEDULE:". Without that line the
    whole text is returned unchanged, so replies with no outline parse as before.
    """
    markers = list(_SCHEDULE_MARKER.finditer(text))
    if not markers:
        return None, text
    last = markers[-1]
    outline = _OUTLINE_HEADER.sub("", text[:last.start()], count=1).strip()
    return outline or None, text[last.end():].strip()


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
    reasoning: str | None = None,
    system_prompt: str | None = None,
    outline: bool = False,
) -> dict[str, Any]:
    """One-shot LLM call. No semantic constraint checking or automated schedule repair.

    With outline=True the model writes a plan outline before the JSON; the
    outline is returned under "outline" and only the JSON part is parsed.
    """
    if system_prompt is None:
        system_prompt = load_prompt(ROOT / "prompts" / "plan.txt")
    if outline:
        system_prompt = system_prompt.rstrip() + "\n\n" + load_prompt(ROOT / "prompts" / "plan_outline.txt")
    output_rule = (
        "Write the OUTLINE, then SCHEDULE: and the JSON."
        if outline else "Return only JSON."
    )

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Find a complete feasible voyage schedule for this instance. "
                    + output_rule + "\n\n"
                    + json.dumps(instance, indent=2, ensure_ascii=False)
                ),
            },
        ],
    }
    if reasoning:
        request["reasoning_effort"] = reasoning
    response = client.chat.completions.create(**request)

    message = response.choices[0].message
    raw = message.content or ""
    usage = response.usage.model_dump() if response.usage else None
    result: dict[str, Any] = {
        "model": model,
        "instance_id": instance.get("instance_id"),
        "raw_response": raw,
        "schedule": None,
        "parse_error": None,
        "outline": None,
        "finish_reason": response.choices[0].finish_reason,
        "usage": usage,
    }

    json_text = raw
    if outline:
        result["outline"], json_text = split_outline(raw)
        result["schedule_marker_found"] = bool(_SCHEDULE_MARKER.search(raw))
    try:
        result["schedule"] = parse_schedule(json_text).model_dump()
    except (ValidationError, ValueError) as exc:
        result["parse_error"] = str(exc)

    return result
