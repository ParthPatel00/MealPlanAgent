"""
Ablation study over retrieval components.

Systematically disables retrieval components to measure their individual
and combined contributions to retrieval quality. Includes an LLM-only
baseline for comparison against the full RAG pipeline.

Configurations tested:
  1. vector_only      - Dense retrieval alone
  2. bm25_only        - Sparse retrieval alone
  3. vector_bm25      - RRF fusion without KG
  4. vector_kg        - Dense + KG re-ranking
  5. bm25_kg          - Sparse + KG re-ranking
  6. full_hybrid      - All three components (default)
  7. llm_baseline     - No RAG, LLM generates from pretraining knowledge
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.retrieval_eval import (
    RetrieverConfig,
    run_llm_baseline,
    run_retrieval_eval,
    save_eval_result,
)

ABLATION_CONFIGS = [
    RetrieverConfig(name="vector_only", use_vector=True, use_bm25=False, use_kg=False),
    RetrieverConfig(name="bm25_only", use_vector=False, use_bm25=True, use_kg=False),
    RetrieverConfig(name="vector_bm25", use_vector=True, use_bm25=True, use_kg=False),
    RetrieverConfig(name="vector_kg", use_vector=True, use_bm25=False, use_kg=True),
    RetrieverConfig(name="bm25_kg", use_vector=False, use_bm25=True, use_kg=True),
    RetrieverConfig(name="full_hybrid", use_vector=True, use_bm25=True, use_kg=True),
]

RESULTS_DIR = Path("data/eval/ablation_results")


def run_ablation_study(
    configs: list[RetrieverConfig] | None = None,
    include_llm_baseline: bool = True,
    llm_model: str = "gemini",
    max_queries: int | None = None,
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full ablation study across all retriever configurations.

    Returns a summary dict with per-config aggregated metrics for comparison.
    """
    if configs is None:
        configs = ABLATION_CONFIGS
    if k_values is None:
        k_values = [5, 10, 20]

    results = {}

    for config in configs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"ABLATION: {config.name}")
            print(f"{'='*60}")

        result = run_retrieval_eval(
            config=config,
            max_queries=max_queries,
            k_values=k_values,
            verbose=verbose,
        )
        results[config.name] = result
        save_eval_result(result, RESULTS_DIR)

    if include_llm_baseline:
        if verbose:
            print(f"\n{'='*60}")
            print(f"BASELINE: LLM-only ({llm_model})")
            print(f"{'='*60}")

        llm_result = run_llm_baseline(
            model_name=llm_model,
            max_queries=max_queries,
            k_values=k_values,
            verbose=verbose,
        )
        results[f"llm_baseline_{llm_model}"] = llm_result
        save_eval_result(llm_result, RESULTS_DIR)

    summary = build_ablation_summary(results)

    summary_path = RESULTS_DIR / f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("ABLATION SUMMARY")
        print(f"{'='*60}")
        print_ablation_table(summary)

    return summary


def build_ablation_summary(results: dict) -> dict:
    """Build a comparison summary from individual ablation results."""
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configs": {},
    }

    key_metrics = [
        "mean_precision@5", "mean_precision@10", "mean_recall@10",
        "mean_recall@20", "mean_mrr", "mean_ap",
        "mean_f1@10", "mean_ndcg@10",
    ]

    for name, result in results.items():
        agg = result.get("aggregate", {})
        entry = {
            "config": result.get("config", {}),
            "num_queries": result.get("num_queries", 0),
            "avg_latency_ms": result.get("avg_latency_ms", 0),
            "key_metrics": {k: round(agg.get(k, 0), 4) for k in key_metrics if k in agg},
            "full_aggregate": agg,
        }
        if "by_query_type" in result:
            entry["by_query_type"] = result["by_query_type"]
        summary["configs"][name] = entry

    return summary


def print_ablation_table(summary: dict) -> None:
    """Print a formatted comparison table."""
    configs = summary["configs"]
    if not configs:
        print("No results to display.")
        return

    metrics_to_show = ["mean_precision@5", "mean_precision@10", "mean_recall@10", "mean_mrr", "mean_ndcg@10"]

    header = f"{'Config':<20}"
    for m in metrics_to_show:
        short = m.replace("mean_", "")
        header += f" {short:>12}"
    header += f" {'latency_ms':>12}"
    print(header)
    print("-" * len(header))

    for name, entry in configs.items():
        row = f"{name:<20}"
        for m in metrics_to_show:
            val = entry["key_metrics"].get(m, 0)
            row += f" {val:>12.4f}"
        row += f" {entry['avg_latency_ms']:>12.1f}"
        print(row)


def load_ablation_results(results_dir: Path = RESULTS_DIR) -> dict:
    """Load the most recent ablation summary."""
    summaries = sorted(results_dir.glob("ablation_summary_*.json"), reverse=True)
    if not summaries:
        return {}
    with open(summaries[0]) as f:
        return json.load(f)


if __name__ == "__main__":
    run_ablation_study(verbose=True)
