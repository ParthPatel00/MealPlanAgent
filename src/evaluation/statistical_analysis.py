"""
Statistical analysis for retrieval evaluation results.

Provides rigorous statistical testing between retrieval configurations:
- Paired t-test for comparing two configs on per-query metrics
- Wilcoxon signed-rank test (non-parametric alternative)
- Bootstrap confidence intervals for aggregate metrics
- Cohen's d effect size for practical significance
- Multiple comparison correction (Bonferroni)
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """
    Paired t-test between two sets of per-query scores.

    Tests H0: mean(scores_a) = mean(scores_b).
    Returns t-statistic, p-value, and whether to reject H0 at alpha=0.05.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    if len(a) != len(b):
        raise ValueError(f"Score lists must be same length: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False, "n": len(a)}

    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
        "n": len(a),
        "mean_diff": round(float(np.mean(a - b)), 4),
        "std_diff": round(float(np.std(a - b, ddof=1)), 4),
    }


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    More robust when score differences are not normally distributed.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    non_zero = diff[diff != 0]
    if len(non_zero) < 10:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n": len(a),
                "note": "Too few non-zero differences for reliable test"}
    try:
        stat, p_val = stats.wilcoxon(a, b, alternative="two-sided")
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "n": len(a),
        }
    except ValueError as e:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "error": str(e)}


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """
    Cohen's d effect size for paired samples.

    Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,
    0.5-0.8 = medium, > 0.8 = large.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return round(float(np.mean(diff) / np.std(diff, ddof=1)), 4)


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for the mean of a metric.

    Uses percentile method with 1000 resamples by default.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_level": ci_level}

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - ci_level
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "ci_level": ci_level,
        "std": round(float(np.std(arr, ddof=1)), 4),
        "n": len(arr),
    }


def compare_configs(
    results: dict,
    metric_key: str = "precision@10",
) -> dict:
    """
    Run pairwise statistical comparisons between all retrieval configurations.

    Args:
        results: Dict mapping config_name -> eval result (from retrieval_eval).
        metric_key: Which per-query metric to compare.

    Returns:
        Dict with pairwise comparisons, per-config CIs, and Bonferroni correction.
    """
    per_query_scores = {}
    for name, result in results.items():
        scores = [q["metrics"].get(metric_key, 0) for q in result.get("per_query", [])]
        per_query_scores[name] = scores

    config_names = list(per_query_scores.keys())
    n_comparisons = len(list(combinations(config_names, 2)))

    pairwise = []
    for name_a, name_b in combinations(config_names, 2):
        scores_a = per_query_scores[name_a]
        scores_b = per_query_scores[name_b]
        min_len = min(len(scores_a), len(scores_b))
        sa = scores_a[:min_len]
        sb = scores_b[:min_len]

        comparison = {
            "config_a": name_a,
            "config_b": name_b,
            "mean_a": round(float(np.mean(sa)), 4),
            "mean_b": round(float(np.mean(sb)), 4),
            "t_test": paired_t_test(sa, sb),
            "wilcoxon": wilcoxon_test(sa, sb),
            "cohens_d": cohens_d(sa, sb),
        }

        raw_p = comparison["t_test"]["p_value"]
        bonferroni_p = min(raw_p * n_comparisons, 1.0)
        comparison["bonferroni_p"] = round(bonferroni_p, 6)
        comparison["bonferroni_significant"] = bonferroni_p < 0.05

        pairwise.append(comparison)

    per_config_ci = {}
    for name, scores in per_query_scores.items():
        per_config_ci[name] = bootstrap_ci(scores)

    return {
        "metric": metric_key,
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": round(0.05 / max(n_comparisons, 1), 6),
        "pairwise": pairwise,
        "confidence_intervals": per_config_ci,
    }


def full_statistical_analysis(
    results: dict,
    metrics: list[str] | None = None,
) -> dict:
    """
    Run statistical analysis across multiple metrics.

    Args:
        results: Dict mapping config_name -> eval result.
        metrics: List of metric keys to analyze.
    """
    if metrics is None:
        metrics = ["precision@5", "precision@10", "recall@10", "mrr", "ndcg@10"]

    analysis = {}
    for metric in metrics:
        analysis[metric] = compare_configs(results, metric_key=metric)

    ranking = {}
    for metric in metrics:
        cis = analysis[metric]["confidence_intervals"]
        sorted_configs = sorted(cis.items(), key=lambda x: x[1]["mean"], reverse=True)
        ranking[metric] = [
            {"config": name, "mean": ci["mean"], "ci": f"[{ci['ci_lower']}, {ci['ci_upper']}]"}
            for name, ci in sorted_configs
        ]

    return {
        "per_metric": analysis,
        "rankings": ranking,
    }


def print_statistical_summary(analysis: dict) -> None:
    """Print a human-readable summary of the statistical analysis."""
    for metric, data in analysis.get("per_metric", {}).items():
        print(f"\n{'='*60}")
        print(f"Metric: {metric}")
        print(f"{'='*60}")

        print("\nConfidence Intervals (95%):")
        cis = data["confidence_intervals"]
        for name, ci in sorted(cis.items(), key=lambda x: x[1]["mean"], reverse=True):
            print(f"  {name:<20} {ci['mean']:.4f}  [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

        print(f"\nPairwise Comparisons (Bonferroni-corrected alpha = {data['bonferroni_alpha']:.4f}):")
        for pw in data["pairwise"]:
            sig = "***" if pw["bonferroni_significant"] else "   "
            d = pw["cohens_d"]
            effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
            print(f"  {pw['config_a']:<16} vs {pw['config_b']:<16}  "
                  f"p={pw['bonferroni_p']:.4f} {sig}  d={d:+.3f} ({effect})")

    print("\n" + "=" * 60)
    print("Rankings by metric:")
    print("=" * 60)
    for metric, ranked in analysis.get("rankings", {}).items():
        names = [r["config"] for r in ranked]
        print(f"  {metric:<16}: {' > '.join(names[:3])}")
