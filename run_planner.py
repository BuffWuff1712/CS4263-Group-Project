import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from planner import ROOT, generate_candidate, get_client, load_instance, load_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a candidate MTRSP voyage schedule")
    parser.add_argument("--instance", type=Path, default=ROOT / "data" / "easy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "easy_candidate.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts" / "plan.txt")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = os.getenv("SOCLAAS_MODEL", "qwen3.5:9b")
    instance = load_instance(args.instance)
    result = generate_candidate(get_client(), instance, model=model, system_prompt=load_prompt(args.prompt))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save raw output and metadata even when the JSON cannot be parsed.
    args.output.with_name(args.output.stem + "_attempt.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if result["schedule"] is None:
        print("Model returned a non-parseable schedule. Raw attempt was saved.")
        print(result["parse_error"])
        raise SystemExit(1)

    args.output.write_text(
        json.dumps(result["schedule"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Candidate saved to: {args.output}")
    print(f"Voyages generated: {len(result['schedule']['voyages'])}")
    print("WARNING: JSON structure parsed successfully, but feasibility is NOT verified.")


if __name__ == "__main__":
    main()
