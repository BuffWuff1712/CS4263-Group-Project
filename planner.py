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
) -> dict[str, Any]:
    """One-shot LLM call. No semantic constraint checking or automated schedule repair."""
    if system_prompt is None:
        system_prompt = load_prompt(ROOT / "prompts" / "plan.txt")

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Find a complete feasible voyage schedule for this instance. "
                    "Return only JSON.\n\n"
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
        "finish_reason": response.choices[0].finish_reason,
        "usage": usage,
    }

    try:
        result["schedule"] = parse_schedule(raw).model_dump()
    except (ValidationError, ValueError) as exc:
        result["parse_error"] = str(exc)

    return result
