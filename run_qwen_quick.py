"""One-shot Qwen baseline on the existing MTRSP instances.

The model generates each schedule independently. The simulator scores its output
afterward and never sends validation feedback to the model.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from planner import ROOT, load_instance, load_prompt, parse_schedule
from simulator import simulate


DEFAULT_MODELS = ("qwen3.5:9b", "qwen3.6:35b")
OPENROUTER_MODELS = ("qwen/qwen3.8-max-prime",)
DEFAULT_INSTANCES = ("easy", "medium", "hard")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("soclaas", "openrouter"), default="soclaas")
    parser.add_argument("--models", nargs="+", help="Model IDs; defaults depend on provider")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "qwen_quick")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds to wait for each API request")
    parser.add_argument(
        "--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="Optional reasoning effort; supported levels depend on the model",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    models = args.models or (OPENROUTER_MODELS if args.provider == "openrouter" else DEFAULT_MODELS)

    load_dotenv(ROOT / ".env")
    prefix = args.provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY")
    if not api_key:
        parser.error(f"Set {prefix}_API_KEY in the project's .env file first")

    client = OpenAI(
        base_url=os.getenv(f"{prefix}_BASE_URL") or (
            "https://openrouter.ai/api/v1" if args.provider == "openrouter"
            else "https://soclaas-api.comp.nus.edu.sg/v1"
        ),
        api_key=api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    prompt = load_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for name in args.instances:
            instance_path = ROOT / "data" / f"{name}.json"
            instance = load_instance(instance_path)
            result = {
                "provider": args.provider,
                "model": model,
                "instance": name,
                "instance_id": instance.get("instance_id"),
                "reasoning_effort": args.reasoning_effort,
            }
            print(f"Running {model} on {name}...", flush=True)
            started = time.perf_counter()

            try:
                request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": "Find a complete feasible voyage schedule. Return only JSON.\n\n"
                            + json.dumps(instance, ensure_ascii=False),
                        },
                    ],
                }
                if args.reasoning_effort:
                    request["reasoning_effort"] = args.reasoning_effort
                response = client.chat.completions.create(**request)
                raw = response.choices[0].message.content or ""
                result["raw_response"] = raw
                result["finish_reason"] = response.choices[0].finish_reason
                result["usage"] = response.usage.model_dump() if response.usage else None

                schedule = parse_schedule(raw)
                result["schedule"] = schedule.model_dump()
                report = simulate(instance, schedule)
                result["validation"] = report.model_dump()
                print(f"  valid={report.valid}, violations={len(report.violations)}", flush=True)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  error={result['error']}", flush=True)

            result["seconds"] = round(time.perf_counter() - started, 2)
            run_name = f"{args.provider}_{model}_{name}"
            if args.reasoning_effort:
                run_name += f"_{args.reasoning_effort}"
            filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name) + ".json"
            (args.output_dir / filename).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    summary = []
    for result_path in sorted(args.output_dir.glob("*.json")):
        if result_path.name == "summary.json":
            continue
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        validation = saved.get("validation") or {}
        summary.append({
            "provider": saved.get("provider", "soclaas"),
            "model": saved.get("model"),
            "instance": saved.get("instance"),
            "reasoning_effort": saved.get("reasoning_effort"),
            "valid": validation.get("valid"),
            "violations": len(validation.get("violations", [])) if validation else None,
            "seconds": saved.get("seconds"),
            "error": saved.get("error"),
            "file": result_path.name,
        })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Summary saved to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
