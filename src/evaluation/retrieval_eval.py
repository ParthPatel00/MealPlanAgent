"""
Retrieval evaluation harness.

Runs a configured retriever against ground truth queries and computes
standard IR metrics. Supports ablation by accepting retriever config dicts.

Also includes an LLM-only baseline that bypasses RAG entirely, asking the
LLM to generate recipe names from memory, then matching against the dataset.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.retrieval_metrics import (
    aggregate_query_metrics,
    compute_all_metrics,
)
from src.rag.retriever import HybridRetriever, RecipeHit

GROUND_TRUTH_PATH = Path("data/eval/retrieval_ground_truth.json")
RESULTS_DIR = Path("data/eval/retrieval_results")


@dataclass
class RetrieverConfig:
    """Configuration for a single retrieval experiment."""
    name: str
    use_vector: bool = True
    use_bm25: bool = True
    use_kg: bool = True
    top_k: int = 10
    rrf_k: int = 60
    kg_boost_weight: float = 0.1
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "use_vector": self.use_vector,
            "use_bm25": self.use_bm25,
            "use_kg": self.use_kg,
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
            "kg_boost_weight": self.kg_boost_weight,
            "embed_model": self.embed_model,
        }


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query: str
    query_type: str
    retrieved_ids: list[int] = field(default_factory=list)
    relevant_ids: list[int] = field(default_factory=list)
    graded_relevance: dict[int, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    source_breakdown: dict[str, int] = field(default_factory=dict)


def _load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _stratified_sample(queries: list[dict], max_queries: int, seed: int = 42) -> list[dict]:
    """Sample queries proportionally across query types instead of taking the first N."""
    by_type: dict[str, list] = {}
    for q in queries:
        by_type.setdefault(q["query_type"], []).append(q)
    rng = random.Random(seed)
    sampled = []
    types = sorted(by_type.keys())
    per_type = max(1, max_queries // len(types))
    for qt in types:
        pool = by_type[qt][:]
        rng.shuffle(pool)
        sampled.extend(pool[:per_type])
    remaining = max_queries - len(sampled)
    if remaining > 0:
        used = {id(q) for q in sampled}
        leftover = [q for q in queries if id(q) not in used]
        rng.shuffle(leftover)
        sampled.extend(leftover[:remaining])
    return sampled[:max_queries]


def run_retrieval_eval(
    config: RetrieverConfig,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    query_types: list[str] | None = None,
    max_queries: int | None = None,
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a retriever configuration against ground truth.

    Args:
        config: RetrieverConfig specifying which components to use.
        ground_truth_path: Path to ground truth JSON.
        query_types: Filter to specific query types (e.g., ["tag_single", "ingredient_multi"]).
        max_queries: Limit number of queries (for quick testing).
        k_values: k values for P@k, R@k, NDCG@k (default [5, 10, 20]).
        verbose: Print progress.

    Returns:
        Dict with config, per_query results, aggregate metrics, and timing.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    gt = _load_ground_truth(ground_truth_path)
    queries = gt["queries"]

    if query_types:
        queries = [q for q in queries if q["query_type"] in query_types]
    if max_queries and max_queries < len(queries):
        queries = _stratified_sample(queries, max_queries)

    if verbose:
        type_counts = {}
        for q in queries:
            type_counts[q["query_type"]] = type_counts.get(q["query_type"], 0) + 1
        print(f"Running eval: {config.name} ({len(queries)} queries: {type_counts})")
        print(f"  Config: vector={config.use_vector}, bm25={config.use_bm25}, "
              f"kg={config.use_kg}, top_k={config.top_k}, rrf_k={config.rrf_k}")

    retriever = HybridRetriever(
        top_k=config.top_k,
        use_vector=config.use_vector,
        use_bm25=config.use_bm25,
        use_kg=config.use_kg,
        rrf_k=config.rrf_k,
        kg_boost_weight=config.kg_boost_weight,
    )

    per_query: list[dict] = []
    all_metrics: list[dict[str, float]] = []
    total_latency = 0.0

    for i, q in enumerate(queries):
        t0 = time.time()
        try:
            preferred_ingredients = q.get("ingredients_used", [])
            preferred_tags = q.get("tags_used", [])
            results, trace = retriever.retrieve(
                query=q["query"],
                preferred_ingredients=preferred_ingredients or None,
                preferred_tags=preferred_tags or None,
                return_trace=True,
            )
        except Exception as e:
            if verbose:
                print(f"  Query {i+1} failed: {e}")
            continue

        latency_ms = (time.time() - t0) * 1000
        total_latency += latency_ms

        retrieved_ids = [h.recipe_id for h in results]
        relevant_ids = set(q["relevant_ids"])

        graded = {}
        if "graded_relevance" in q:
            graded = {int(k): v for k, v in q["graded_relevance"].items()}

        metrics = compute_all_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            graded_relevance=graded if graded else None,
            k_values=k_values,
        )

        source_counts = {}
        for h in results:
            src = getattr(h, "source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        qr = {
            "query": q["query"],
            "query_type": q["query_type"],
            "retrieved_ids": retrieved_ids,
            "num_relevant": len(relevant_ids),
            "num_retrieved_relevant": len(set(retrieved_ids) & relevant_ids),
            "metrics": metrics,
            "latency_ms": round(latency_ms, 2),
            "source_breakdown": source_counts,
        }
        per_query.append(qr)
        all_metrics.append(metrics)

        if verbose and (i + 1) % 25 == 0:
            print(f"  Completed {i+1}/{len(queries)} queries")

    aggregate = aggregate_query_metrics(all_metrics)

    by_type: dict[str, list[dict]] = {}
    for qr in per_query:
        qtype = qr["query_type"]
        by_type.setdefault(qtype, []).append(qr["metrics"])

    type_aggregates = {}
    for qtype, type_metrics in by_type.items():
        type_aggregates[qtype] = aggregate_query_metrics(type_metrics)

    result = {
        "config": config.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(per_query),
        "aggregate": aggregate,
        "by_query_type": type_aggregates,
        "avg_latency_ms": round(total_latency / max(len(per_query), 1), 2),
        "total_latency_s": round(total_latency / 1000, 2),
        "per_query": per_query,
    }

    if verbose:
        print(f"\n  Results for {config.name}:")
        for key, val in aggregate.items():
            print(f"    {key}: {val:.4f}")
        print(f"    avg_latency_ms: {result['avg_latency_ms']:.1f}")

    return result


def run_llm_baseline(
    model_name: str = "gemini",
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    query_types: list[str] | None = None,
    max_queries: int | None = None,
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    LLM-only baseline: ask the LLM to suggest recipes without RAG.

    For each ground truth query, prompts the LLM to generate recipe names,
    then fuzzy-matches them against the dataset to get recipe IDs. This
    measures what the LLM knows from pretraining alone vs. the RAG pipeline.
    """
    from src.models.client import LLMClient

    if k_values is None:
        k_values = [5, 10, 20]

    gt = _load_ground_truth(ground_truth_path)
    recipes = json.load(open(Path("data/processed/recipes_clean.json")))
    name_to_id = {r["name"].strip().lower(): r["id"] for r in recipes}

    queries = gt["queries"]
    if query_types:
        queries = [q for q in queries if q["query_type"] in query_types]
    if max_queries and max_queries < len(queries):
        queries = _stratified_sample(queries, max_queries)

    if verbose:
        print(f"Running LLM baseline: {model_name} ({len(queries)} queries)")

    client = LLMClient(model_name)
    system_prompt = (
        "You are a recipe recommendation system. Given a query, suggest exactly 10 "
        "recipe names that match. Return ONLY a JSON list of recipe name strings, "
        "no other text. Example: [\"Chicken Parmesan\", \"Pasta Primavera\", ...]"
    )

    per_query: list[dict] = []
    all_metrics: list[dict[str, float]] = []
    total_latency = 0.0

    for i, q in enumerate(queries):
        t0 = time.time()
        try:
            response = client.chat(
                prompt=f"Query: {q['query']}\n\nSuggest 10 matching recipes.",
                system=system_prompt,
                temperature=0.0,
                max_tokens=1024,
            )
            latency_ms = (time.time() - t0) * 1000
            total_latency += latency_ms

            try:
                from src.agent.json_utils import extract_first_json
                suggested = extract_first_json(response.text)
                if isinstance(suggested, dict):
                    suggested = suggested.get("recipes", [])
            except Exception:
                suggested = []

            retrieved_ids = []
            for name in (suggested or []):
                if not isinstance(name, str):
                    continue
                name_lower = name.strip().lower()
                if name_lower in name_to_id:
                    retrieved_ids.append(name_to_id[name_lower])
                else:
                    best_match = None
                    best_score = 0
                    for db_name, db_id in name_to_id.items():
                        common = len(set(name_lower.split()) & set(db_name.split()))
                        total = max(len(set(name_lower.split()) | set(db_name.split())), 1)
                        score = common / total
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = db_id
                    if best_match is not None:
                        retrieved_ids.append(best_match)

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            total_latency += latency_ms
            if verbose:
                print(f"  Query {i+1} failed: {e}")
            retrieved_ids = []

        relevant_ids = set(q["relevant_ids"])
        graded = {}
        if "graded_relevance" in q:
            graded = {int(k): v for k, v in q["graded_relevance"].items()}

        metrics = compute_all_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            graded_relevance=graded if graded else None,
            k_values=k_values,
        )

        qr = {
            "query": q["query"],
            "query_type": q["query_type"],
            "retrieved_ids": retrieved_ids,
            "num_relevant": len(relevant_ids),
            "num_retrieved_relevant": len(set(retrieved_ids) & relevant_ids),
            "metrics": metrics,
            "latency_ms": round(latency_ms, 2),
            "llm_suggestions_count": len(suggested) if suggested else 0,
            "matched_to_dataset": len(retrieved_ids),
        }
        per_query.append(qr)
        all_metrics.append(metrics)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(queries)} queries")

    aggregate = aggregate_query_metrics(all_metrics)

    result = {
        "config": {"name": f"llm_baseline_{model_name}", "model": model_name, "type": "llm_only"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(per_query),
        "aggregate": aggregate,
        "avg_latency_ms": round(total_latency / max(len(per_query), 1), 2),
        "total_latency_s": round(total_latency / 1000, 2),
        "per_query": per_query,
    }

    if verbose:
        print(f"\n  LLM Baseline Results ({model_name}):")
        for key, val in aggregate.items():
            print(f"    {key}: {val:.4f}")

    return result


def save_eval_result(result: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Save evaluation result to a timestamped JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    name = result["config"]["name"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"retrieval_{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {path}")
    return path
