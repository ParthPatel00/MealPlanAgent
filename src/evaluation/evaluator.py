"""
Evaluator: run a subset of the 60 test cases against one or more models.

Usage:
    python -m src.evaluation.evaluator --model gemini --cases data/eval/test_cases.json --limit 10

Output is written to data/eval/results_<model>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.agent.pipeline import run_pipeline
from src.evaluation.metrics import aggregate_scores, score_case

RESULTS_DIR = Path("data/eval")


def run_eval(
    model_name: str,
    cases_path: Path = Path("data/eval/test_cases.json"),
    limit: int | None = None,
    delay_between: float = 0.0,
) -> dict:
    """
    Run evaluation cases against a model and return aggregated results.

    Args:
        model_name: "gemini", "groq-llama", "ollama-llama3b", or "ollama-granite2b".
        cases_path: Path to test_cases.json.
        limit: If set, only run the first `limit` cases (for quick testing).
        delay_between: Seconds to wait between cases (for rate-limited APIs).

    Returns:
        Dict with per-case scores and aggregate metrics.
    """
    with open(cases_path) as f:
        cases = json.load(f)

    if limit:
        cases = cases[:limit]

    per_case_scores = []
    latencies = []
    errors = []

    for i, case in enumerate(cases):
        if i > 0 and delay_between > 0:
            time.sleep(delay_between)
        print(f"  Case {case['id']}/{len(cases)} — {case['constraints'].get('tags', [])}")
        t0 = time.time()
        try:
            result = run_pipeline(case["constraints"], model_name=model_name)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)
            scored = score_case(result, case)
            scored["latency_ms"] = round(latency, 1)
            per_case_scores.append(scored)
        except Exception as exc:
            latency = (time.time() - t0) * 1000
            exc_str = str(exc).lower()
            is_rate_limit = any(kw in exc_str for kw in ["429", "rate", "quota", "exhausted", "limit"])
            if is_rate_limit:
                print(f"    RATE LIMITED: {str(exc)[:80]}")
            else:
                print(f"    ERROR: {exc}")
            errors.append({"id": case["id"], "error": str(exc)})
            per_case_scores.append(
                {"id": case["id"], "error": str(exc), "latency_ms": round(latency, 1)}
            )

    agg = aggregate_scores([s for s in per_case_scores if "error" not in s])
    agg["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
    agg["error_count"] = len(errors)

    output = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg,
        "per_case": per_case_scores,
        "errors": errors,
    }

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{model_name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Aggregate: {json.dumps(agg, indent=2)}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini",
                        choices=["gemini", "groq-llama", "ollama-llama3b", "ollama-granite2b"])
    parser.add_argument("--cases", default="data/eval/test_cases.json")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between cases (for rate-limited APIs)")
    args = parser.parse_args()

    run_eval(
        model_name=args.model,
        cases_path=Path(args.cases),
        delay_between=args.delay,
        limit=args.limit,
    )
