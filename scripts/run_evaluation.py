#!/usr/bin/env python3
"""
Master evaluation runner.

Orchestrates the complete academic evaluation pipeline:
  1. Generate ground truth relevance judgments
  2. Generate dynamic test scenarios
  3. Run ablation study (6 retriever configs + LLM baseline)
  4. Run hyperparameter sensitivity study
  5. Run statistical analysis
  6. Generate all visualizations

Usage:
    python run_evaluation.py                    # Full pipeline
    python run_evaluation.py --quick            # Quick test (10 queries)
    python run_evaluation.py --step ablation    # Single step
    python run_evaluation.py --skip-embed       # Skip slow embedding model swap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def step_ground_truth():
    """Step 1: Generate ground truth relevance judgments."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating ground truth relevance judgments")
    print("=" * 70)
    from src.evaluation.ground_truth import generate_all_ground_truth
    return generate_all_ground_truth()


def step_scenarios():
    """Step 2: Generate dynamic test scenarios."""
    print("\n" + "=" * 70)
    print("STEP 2: Generating dynamic test scenarios")
    print("=" * 70)
    from src.evaluation.scenario_generator import generate_all_scenarios
    return generate_all_scenarios()


def step_ablation(max_queries: int | None = None, include_llm: bool = True):
    """Step 3: Run ablation study."""
    print("\n" + "=" * 70)
    print("STEP 3: Running ablation study")
    print("=" * 70)
    from src.evaluation.ablation_study import run_ablation_study
    return run_ablation_study(
        max_queries=max_queries,
        include_llm_baseline=include_llm,
    )


def step_hyperparams(max_queries: int | None = None, skip_embed: bool = False):
    """Step 4: Run hyperparameter sensitivity study."""
    print("\n" + "=" * 70)
    print("STEP 4: Running hyperparameter sensitivity study")
    print("=" * 70)
    from src.evaluation.hyperparam_study import run_full_hyperparam_study
    return run_full_hyperparam_study(
        max_queries=max_queries,
        skip_embed_models=skip_embed,
    )


def step_statistics(ablation_results: dict | None = None):
    """Step 5: Run statistical analysis."""
    print("\n" + "=" * 70)
    print("STEP 5: Running statistical analysis")
    print("=" * 70)
    from src.evaluation.statistical_analysis import full_statistical_analysis, print_statistical_summary

    if ablation_results is None:
        results_dir = Path("data/eval/ablation_results")
        ablation_results = {}
        for p in sorted(results_dir.glob("retrieval_*.json")):
            if "summary" in p.name:
                continue
            with open(p) as f:
                data = json.load(f)
            name = data.get("config", {}).get("name", p.stem)
            ablation_results[name] = data

    if not ablation_results:
        print("  No ablation results found. Run ablation first.")
        return None

    analysis = full_statistical_analysis(ablation_results)
    print_statistical_summary(analysis)

    out_path = Path("data/eval/statistical_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return analysis


def step_visualize(stat_analysis: dict | None = None):
    """Step 6: Generate all visualizations."""
    print("\n" + "=" * 70)
    print("STEP 6: Generating visualizations")
    print("=" * 70)
    from src.evaluation.visualization import generate_all_visualizations

    ablation_summaries = sorted(Path("data/eval/ablation_results").glob("ablation_summary_*.json"), reverse=True)
    hyperparam_summaries = sorted(Path("data/eval/hyperparam_results").glob("hyperparam_summary_*.json"), reverse=True)

    paths = generate_all_visualizations(
        ablation_summary_path=ablation_summaries[0] if ablation_summaries else None,
        hyperparam_summary_path=hyperparam_summaries[0] if hyperparam_summaries else None,
        stat_analysis=stat_analysis,
    )
    print(f"\nGenerated {len(paths)} charts:")
    for p in paths:
        print(f"  {p}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run the full academic evaluation pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 queries per config")
    parser.add_argument("--step", type=str, choices=[
        "ground_truth", "scenarios", "ablation", "hyperparams", "statistics", "visualize"
    ], help="Run a single step")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding model sweep (slow)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM baseline (needs API key)")
    parser.add_argument("--max-queries", type=int, default=None, help="Max queries per eval run")
    args = parser.parse_args()

    max_queries = args.max_queries or (10 if args.quick else None)

    t_start = time.time()

    if args.step:
        if args.step == "ground_truth":
            step_ground_truth()
        elif args.step == "scenarios":
            step_scenarios()
        elif args.step == "ablation":
            step_ablation(max_queries=max_queries, include_llm=not args.skip_llm)
        elif args.step == "hyperparams":
            step_hyperparams(max_queries=max_queries, skip_embed=args.skip_embed)
        elif args.step == "statistics":
            step_statistics()
        elif args.step == "visualize":
            step_visualize()
    else:
        step_ground_truth()
        step_scenarios()
        ablation_summary = step_ablation(max_queries=max_queries, include_llm=not args.skip_llm)
        step_hyperparams(max_queries=max_queries, skip_embed=args.skip_embed)

        ablation_results = {}
        for p in sorted(Path("data/eval/ablation_results").glob("retrieval_*.json")):
            if "summary" in p.name:
                continue
            with open(p) as f:
                data = json.load(f)
            name = data.get("config", {}).get("name", p.stem)
            ablation_results[name] = data

        stat_analysis = step_statistics(ablation_results)
        step_visualize(stat_analysis)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETE. Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
