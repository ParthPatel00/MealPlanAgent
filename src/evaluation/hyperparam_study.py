"""
Hyperparameter sensitivity study for the retrieval pipeline.

Systematically varies one hyperparameter at a time while holding others at
their defaults, measuring the impact on retrieval quality metrics.

Parameters studied:
  1. top_k (retrieval depth): [5, 10, 20, 50]
  2. rrf_k (RRF fusion constant): [10, 30, 60, 100, 200]
  3. kg_boost_weight (KG re-ranking strength): [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
  4. embed_model (embedding model choice): 3 sentence-transformer variants
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.retrieval_eval import (
    RetrieverConfig,
    run_retrieval_eval,
    save_eval_result,
)

RESULTS_DIR = Path("data/eval/hyperparam_results")

DEFAULTS = {
    "use_vector": True,
    "use_bm25": True,
    "use_kg": True,
    "top_k": 10,
    "rrf_k": 60,
    "kg_boost_weight": 0.1,
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
}

TOP_K_VALUES = [5, 10, 20, 50]
RRF_K_VALUES = [10, 30, 60, 100, 200]
KG_BOOST_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",      # 384-dim, 22M params
    "sentence-transformers/all-MiniLM-L12-v2",      # 384-dim, 33M params
    "sentence-transformers/all-mpnet-base-v2",       # 768-dim, 110M params
]

EMBED_MODEL_INFO = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "short_name": "MiniLM-L6",
        "dimensions": 384,
        "params_millions": 22,
        "layers": 6,
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "short_name": "MiniLM-L12",
        "dimensions": 384,
        "params_millions": 33,
        "layers": 12,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "short_name": "MPNet-base",
        "dimensions": 768,
        "params_millions": 110,
        "layers": 12,
    },
}


def _make_config(name: str, **overrides) -> RetrieverConfig:
    """Create a RetrieverConfig with defaults and specific overrides."""
    params = {**DEFAULTS, **overrides}
    return RetrieverConfig(name=name, **params)


def sweep_top_k(
    max_queries: int | None = None,
    verbose: bool = True,
) -> dict:
    """Sweep retrieval depth (top_k) values."""
    results = {}
    for k in TOP_K_VALUES:
        name = f"top_k_{k}"
        config = _make_config(name, top_k=k)
        result = run_retrieval_eval(config=config, max_queries=max_queries, verbose=verbose)
        results[name] = result
        save_eval_result(result, RESULTS_DIR)
    return results


def sweep_rrf_k(
    max_queries: int | None = None,
    verbose: bool = True,
) -> dict:
    """Sweep RRF fusion constant (rrf_k) values."""
    results = {}
    for k in RRF_K_VALUES:
        name = f"rrf_k_{k}"
        config = _make_config(name, rrf_k=k)
        result = run_retrieval_eval(config=config, max_queries=max_queries, verbose=verbose)
        results[name] = result
        save_eval_result(result, RESULTS_DIR)
    return results


def sweep_kg_boost(
    max_queries: int | None = None,
    verbose: bool = True,
) -> dict:
    """Sweep knowledge graph boost weight values."""
    results = {}
    for w in KG_BOOST_VALUES:
        name = f"kg_boost_{w:.2f}"
        config = _make_config(name, kg_boost_weight=w)
        result = run_retrieval_eval(config=config, max_queries=max_queries, verbose=verbose)
        results[name] = result
        save_eval_result(result, RESULTS_DIR)
    return results


def sweep_embed_models(
    max_queries: int | None = None,
    verbose: bool = True,
    rebuild_index: bool = True,
) -> dict:
    """
    Sweep embedding models.

    WARNING: Each model requires a separate ChromaDB index. If rebuild_index
    is True, builds a new index per model (slow but necessary). Set to False
    if indices already exist at data/chroma_db_{model_short_name}/.
    """
    from src.rag.indexer import build_index

    results = {}
    for model in EMBED_MODELS:
        info = EMBED_MODEL_INFO[model]
        short = info["short_name"]
        name = f"embed_{short}"

        chroma_path = Path(f"data/chroma_db_{short.lower().replace('-', '_')}")

        if rebuild_index and not chroma_path.exists():
            if verbose:
                print(f"\nBuilding index for {short} ({info['dimensions']}-dim, {info['params_millions']}M params)...")
            build_index(
                chroma_path=chroma_path,
                embed_model=model,
                collection_name=f"recipes_{short.lower()}",
            )

        config = RetrieverConfig(
            name=name,
            use_vector=True,
            use_bm25=True,
            use_kg=True,
            top_k=DEFAULTS["top_k"],
            rrf_k=DEFAULTS["rrf_k"],
            kg_boost_weight=DEFAULTS["kg_boost_weight"],
            embed_model=model,
        )

        result = run_retrieval_eval(config=config, max_queries=max_queries, verbose=verbose)
        result["config"]["embed_model_info"] = info
        results[name] = result
        save_eval_result(result, RESULTS_DIR)

    return results


def run_full_hyperparam_study(
    max_queries: int | None = None,
    skip_embed_models: bool = False,
    verbose: bool = True,
) -> dict:
    """Run all hyperparameter sweeps and compile results."""
    all_results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER STUDY: top_k sweep")
        print("=" * 60)
    all_results["top_k"] = sweep_top_k(max_queries=max_queries, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER STUDY: rrf_k sweep")
        print("=" * 60)
    all_results["rrf_k"] = sweep_rrf_k(max_queries=max_queries, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER STUDY: kg_boost_weight sweep")
        print("=" * 60)
    all_results["kg_boost"] = sweep_kg_boost(max_queries=max_queries, verbose=verbose)

    if not skip_embed_models:
        if verbose:
            print("\n" + "=" * 60)
            print("HYPERPARAMETER STUDY: embedding model sweep")
            print("=" * 60)
        all_results["embed_model"] = sweep_embed_models(max_queries=max_queries, verbose=verbose)

    summary = _build_hyperparam_summary(all_results)

    summary_path = RESULTS_DIR / f"hyperparam_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSummary saved to {summary_path}")
        _print_sensitivity_summary(summary)

    return summary


def _build_hyperparam_summary(all_results: dict) -> dict:
    """Compile sweep results into a structured summary for visualization."""
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sweeps": {},
    }

    key_metrics = [
        "mean_precision@5", "mean_precision@10", "mean_recall@10",
        "mean_recall@20", "mean_mrr", "mean_ap", "mean_ndcg@10",
    ]

    for param_name, sweep_results in all_results.items():
        sweep_data = []
        for config_name, result in sweep_results.items():
            agg = result.get("aggregate", {})
            entry = {
                "config_name": config_name,
                "config": result.get("config", {}),
                "metrics": {k: round(agg.get(k, 0), 4) for k in key_metrics if k in agg},
                "avg_latency_ms": result.get("avg_latency_ms", 0),
                "num_queries": result.get("num_queries", 0),
            }
            sweep_data.append(entry)
        summary["sweeps"][param_name] = sweep_data

    return summary


def _print_sensitivity_summary(summary: dict) -> None:
    """Print which hyperparameters had the most impact."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    for param, data in summary["sweeps"].items():
        if not data:
            continue
        mrr_vals = [d["metrics"].get("mean_mrr", 0) for d in data]
        if mrr_vals:
            spread = max(mrr_vals) - min(mrr_vals)
            best_idx = mrr_vals.index(max(mrr_vals))
            print(f"\n  {param}:")
            print(f"    MRR range: {min(mrr_vals):.4f} - {max(mrr_vals):.4f} (spread: {spread:.4f})")
            print(f"    Best config: {data[best_idx]['config_name']} (MRR={max(mrr_vals):.4f})")


if __name__ == "__main__":
    run_full_hyperparam_study(skip_embed_models=True, verbose=True)
