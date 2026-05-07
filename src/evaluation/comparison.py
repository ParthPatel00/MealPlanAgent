"""
Cross-model comparison: loads evaluation results and builds comparison tables + charts.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("data/eval")

COST_PER_1M_TOKENS = {
    "gemini": {"input": 0.0, "output": 0.0},
    "groq-llama": {"input": 0.59, "output": 0.79},
    "ollama-llama3b": {"input": 0.0, "output": 0.0},
    "ollama-granite2b": {"input": 0.0, "output": 0.0},
}


def load_all_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Load the latest result file per model from the results directory."""
    results: dict[str, dict] = {}
    if not results_dir.exists():
        return results

    result_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    seen_models: set[str] = set()

    for f in result_files:
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model = data.get("model", "")
        if model and model not in seen_models:
            seen_models.add(model)
            results[model] = data

    return results


def build_comparison_table(results: dict[str, dict] | None = None) -> list[dict]:
    """Build a comparison table from all model results."""
    if results is None:
        results = load_all_results()

    rows = []
    for model, data in sorted(results.items()):
        agg = data.get("aggregate", {})
        rows.append({
            "Model": model,
            "Cases": agg.get("num_cases", 0),
            "Constraint Pass %": round((agg.get("constraint_pass_rate", 0) or 0) * 100, 1),
            "Allergy Violation %": round((agg.get("avg_allergy_violation_rate", 0) or 0) * 100, 1),
            "Citation Pass %": round((agg.get("avg_citation_pass_rate", 0) or 0) * 100, 1),
            "Tool Success %": round((agg.get("avg_tool_success_rate", 0) or 0) * 100, 1),
            "Critic Valid %": round((agg.get("critic_valid_rate", 0) or 0) * 100, 1),
            "Avg Latency (ms)": round(agg.get("avg_latency_ms", 0) or 0, 0),
            "Avg Retries": round(agg.get("avg_retries", 0) or 0, 2),
            "Errors": agg.get("error_count", 0),
        })
    return rows


def build_comparison_charts(results: dict[str, dict] | None = None) -> dict:
    """Build Plotly figures for model comparison. Returns dict of name->Figure."""
    import plotly.graph_objects as go

    if results is None:
        results = load_all_results()

    table = build_comparison_table(results)
    if not table:
        return {}

    models = [r["Model"] for r in table]
    charts = {}

    metrics = [
        ("Constraint Pass %", "Constraint Pass Rate", "seagreen"),
        ("Citation Pass %", "Citation Pass Rate", "steelblue"),
        ("Tool Success %", "Tool Success Rate", "mediumpurple"),
        ("Critic Valid %", "Critic Validity Rate", "coral"),
    ]
    for key, title, color in metrics:
        values = [r[key] for r in table]
        fig = go.Figure(go.Bar(x=models, y=values, marker_color=color))
        fig.update_layout(title=title, yaxis_title="%", yaxis_range=[0, 105], height=350)
        charts[title] = fig

    allergy_vals = [r["Allergy Violation %"] for r in table]
    fig = go.Figure(go.Bar(x=models, y=allergy_vals, marker_color="crimson"))
    fig.update_layout(title="Allergy Violation Rate (lower is better)", yaxis_title="%", height=350)
    charts["Allergy Violations"] = fig

    latencies = [r["Avg Latency (ms)"] for r in table]
    fig = go.Figure(go.Bar(x=models, y=latencies, marker_color="darkorange"))
    fig.update_layout(title="Average Latency", yaxis_title="ms", height=350)
    charts["Latency"] = fig

    return charts


def build_cost_latency_scatter(results: dict[str, dict] | None = None):
    """Build a cost-vs-latency scatter plot."""
    import plotly.graph_objects as go

    if results is None:
        results = load_all_results()

    table = build_comparison_table(results)
    if not table:
        return None

    models = [r["Model"] for r in table]
    latencies = [r["Avg Latency (ms)"] for r in table]
    pass_rates = [r["Constraint Pass %"] for r in table]

    costs = []
    for r in table:
        model = r["Model"]
        pricing = COST_PER_1M_TOKENS.get(model, {"input": 0, "output": 0})
        avg_cost = (pricing["input"] + pricing["output"]) / 2 * 1000 / 1_000_000
        costs.append(avg_cost)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=latencies,
        y=costs,
        mode="markers+text",
        text=models,
        textposition="top center",
        marker=dict(
            size=[max(p / 3, 10) for p in pass_rates],
            color=pass_rates,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Pass %"),
        ),
    ))
    fig.update_layout(
        title="Cost vs Latency Tradeoff (bubble size = pass rate)",
        xaxis_title="Avg Latency (ms)",
        yaxis_title="Estimated Cost per Request ($)",
        height=450,
    )
    return fig


def build_tradeoff_summary(results: dict[str, dict] | None = None) -> str:
    """Generate a text summary of cost/latency tradeoffs."""
    if results is None:
        results = load_all_results()

    table = build_comparison_table(results)
    if not table:
        return "No evaluation results found."

    best_quality = max(table, key=lambda r: r["Constraint Pass %"])
    fastest = min(table, key=lambda r: r["Avg Latency (ms)"])

    lines = [
        f"Best quality: {best_quality['Model']} with {best_quality['Constraint Pass %']}% constraint pass rate.",
        f"Fastest: {fastest['Model']} at {fastest['Avg Latency (ms)']:.0f}ms average latency.",
    ]

    free_models = [r for r in table if COST_PER_1M_TOKENS.get(r["Model"], {}).get("input", 0) == 0]
    if free_models:
        best_free = max(free_models, key=lambda r: r["Constraint Pass %"])
        lines.append(f"Best free option: {best_free['Model']} ({best_free['Constraint Pass %']}% pass, {best_free['Avg Latency (ms)']:.0f}ms).")

    return " ".join(lines)
