import argparse
import json
from pathlib import Path

from planner import ROOT, generate_candidate, get_client, get_model, load_instance, load_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a candidate MTRSP voyage schedule")
    parser.add_argument("--instance", type=Path, default=ROOT / "data" / "easy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "easy_candidate.json")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts" / "plan.txt")
    parser.add_argument("--provider", choices=("soclaas", "openai", "openrouter"), default="soclaas")
    parser.add_argument("--model", help="Override the provider's configured model")
    parser.add_argument("--reasoning-effort", help="Use only with a model that supports this setting")
    args = parser.parse_args()

    model = args.model or get_model(args.provider)
    instance = load_instance(args.instance)
    result = generate_candidate(
        get_client(args.provider), instance, model=model,
        reasoning=args.reasoning_effort,
        system_prompt=load_prompt(args.prompt),
    )

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
