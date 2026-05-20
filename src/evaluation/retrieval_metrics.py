"""
Standard Information Retrieval metrics for evaluating retrieval quality.

Implements precision@k, recall@k, MRR, NDCG@k, AP, and MAP with support
for both binary and graded relevance judgments.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of all relevant documents found in the top-k retrieved."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Reciprocal of the rank of the first relevant document (0 if none found)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved_ids: list[int], graded_relevance: dict[int, float], k: int) -> float:
    """Discounted Cumulative Gain at rank k using graded relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = graded_relevance.get(doc_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank is 1-indexed
    return dcg


def ndcg_at_k(
    retrieved_ids: list[int], graded_relevance: dict[int, float], k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain at rank k.

    Uses graded relevance scores (e.g., 0=irrelevant, 1=marginal, 2=relevant,
    3=highly relevant). Returns 0 if no relevant documents exist.
    """
    if not graded_relevance or k <= 0:
        return 0.0
    dcg = _dcg_at_k(retrieved_ids, graded_relevance, k)
    ideal_ranking = sorted(graded_relevance.values(), reverse=True)[:k]
    ideal_ids = list(range(len(ideal_ranking)))
    ideal_relevance = dict(zip(ideal_ids, ideal_ranking))
    idcg = _dcg_at_k(ideal_ids, ideal_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Average Precision for a single query.

    Computes precision at each relevant document's rank, then averages.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant_ids)


def mean_average_precision(
    all_query_results: list[tuple[list[int], set[int]]]
) -> float:
    """
    Mean Average Precision across multiple queries.

    Args:
        all_query_results: list of (retrieved_ids, relevant_ids) per query.
    """
    if not all_query_results:
        return 0.0
    aps = [average_precision(ret, rel) for ret, rel in all_query_results]
    return sum(aps) / len(aps)


def f1_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_all_metrics(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    graded_relevance: dict[int, float] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Returns a flat dict with keys like 'precision@5', 'recall@10', 'mrr', 'ndcg@10', etc.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved_ids, relevant_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    metrics["ap"] = average_precision(retrieved_ids, relevant_ids)

    if graded_relevance:
        for k in k_values:
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, graded_relevance, k)

    return metrics


def aggregate_query_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across all queries."""
    if not per_query_metrics:
        return {}
    all_keys: set[str] = set()
    for m in per_query_metrics:
        all_keys.update(m.keys())
    aggregated = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in per_query_metrics if key in m]
        aggregated[f"mean_{key}"] = sum(vals) / len(vals) if vals else 0.0
    return aggregated
