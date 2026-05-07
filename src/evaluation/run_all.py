"""
Batch evaluator: runs all 60 test cases across multiple models.

Usage:
    python -m src.evaluation.run_all
    python -m src.evaluation.run_all --models gemini groq-llama --limit 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.evaluation.evaluator import run_eval

DEFAULT_MODELS = ["gemini", "groq-llama", "ollama-llama3b"]


def run_all(
    models: list[str] | None = None,
    cases_path: Path = Path("data/eval/test_cases.json"),
    limit: int | None = None,
) -> dict[str, dict]:
    models = models or DEFAULT_MODELS
    all_results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running evaluation: {model}")
        print(f"{'='*60}")

        try:
            result = run_eval(model_name=model, cases_path=cases_path, limit=limit)
            all_results[model] = result
        except Exception as exc:
            print(f"FAILED for {model}: {exc}")
            all_results[model] = {"model": model, "error": str(exc)}

        time.sleep(2)

    print(f"\n{'='*60}")
    print("All evaluations complete.")
    for model, result in all_results.items():
        if "error" in result:
            print(f"  {model}: FAILED - {result['error']}")
        else:
            agg = result.get("aggregate", {})
            print(f"  {model}: {agg.get('num_cases', 0)} cases, "
                  f"pass={agg.get('constraint_pass_rate', 'N/A')}, "
                  f"latency={agg.get('avg_latency_ms', 'N/A')}ms")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to evaluate")
    parser.add_argument("--cases", default="data/eval/test_cases.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only first N cases per model")
    args = parser.parse_args()

    run_all(
        models=args.models,
        cases_path=Path(args.cases),
        limit=args.limit,
    )
