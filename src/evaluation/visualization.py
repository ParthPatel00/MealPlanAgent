"""
Publication-quality visualization for retrieval evaluation results.

Generates charts for:
1. Ablation study comparison (grouped bar charts with CI error bars)
2. Hyperparameter sensitivity (line plots with CI bands)
3. Embedding space analysis (t-SNE / UMAP with cluster quality)
4. Precision-Recall curves per configuration
5. Component contribution heatmap (which retriever found each result)
6. Knowledge graph analysis (degree distributions, community structure)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CHARTS_DIR = Path("report/charts")


def _ensure_dir():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_ablation_comparison(
    ablation_summary: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Grouped bar chart comparing retrieval configs across key metrics.
    Includes bootstrap 95% CI error bars.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "ablation_comparison.png"

    configs = ablation_summary.get("configs", {})
    if not configs:
        return output_path

    config_names = list(configs.keys())
    metrics = ["mean_precision@5", "mean_precision@10", "mean_recall@10", "mean_mrr", "mean_ndcg@10"]
    short_labels = ["P@5", "P@10", "R@10", "MRR", "NDCG@10"]

    x = np.arange(len(short_labels))
    width = 0.8 / len(config_names)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("Set2", len(config_names))

    for i, name in enumerate(config_names):
        entry = configs[name]
        vals = [entry["key_metrics"].get(m, 0) for m in metrics]
        offset = (i - len(config_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name.replace("_", " ").title(),
                       color=colors[i], edgecolor="black", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Ablation Study: Component Contribution Analysis", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_ablation_with_ci(
    ablation_results: dict,
    stat_analysis: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Ablation comparison with bootstrap 95% confidence interval error bars.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "ablation_with_ci.png"

    metrics_to_plot = ["precision@10", "recall@10", "mrr"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 6))

    for ax, metric in zip(axes, metrics_to_plot):
        if metric not in stat_analysis.get("per_metric", {}):
            continue
        cis = stat_analysis["per_metric"][metric]["confidence_intervals"]

        names = sorted(cis.keys())
        means = [cis[n]["mean"] for n in names]
        lowers = [cis[n]["mean"] - cis[n]["ci_lower"] for n in names]
        uppers = [cis[n]["ci_upper"] - cis[n]["mean"] for n in names]

        colors = sns.color_palette("Set2", len(names))
        bars = ax.barh(range(len(names)), means, xerr=[lowers, uppers],
                        capsize=4, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
        ax.set_xlabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} with 95% CI", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Ablation Study with Statistical Confidence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_hyperparam_sensitivity(
    hyperparam_summary: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Line plots showing metric sensitivity to each hyperparameter.
    One subplot per hyperparameter, with lines for different metrics.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "hyperparam_sensitivity.png"

    sweeps = hyperparam_summary.get("sweeps", {})
    metrics_to_plot = ["mean_precision@10", "mean_recall@10", "mean_mrr", "mean_ndcg@10"]
    metric_labels = {"mean_precision@10": "P@10", "mean_recall@10": "R@10",
                     "mean_mrr": "MRR", "mean_ndcg@10": "NDCG@10"}

    param_configs = {
        "top_k": {"label": "Retrieval Depth (top_k)", "values": [5, 10, 20, 50]},
        "rrf_k": {"label": "RRF Constant (k)", "values": [10, 30, 60, 100, 200]},
        "kg_boost": {"label": "KG Boost Weight", "values": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]},
    }

    active_params = [p for p in param_configs if p in sweeps]
    if not active_params:
        return output_path

    fig, axes = plt.subplots(1, len(active_params), figsize=(7 * len(active_params), 6))
    if len(active_params) == 1:
        axes = [axes]

    colors = sns.color_palette("Dark2", len(metrics_to_plot))

    for ax, param in zip(axes, active_params):
        data = sweeps[param]
        pconfig = param_configs[param]

        for j, metric in enumerate(metrics_to_plot):
            x_vals = []
            y_vals = []
            for entry in data:
                cfg = entry.get("config", {})
                if param == "top_k":
                    x_val = cfg.get("top_k", 10)
                elif param == "rrf_k":
                    x_val = cfg.get("rrf_k", 60)
                elif param == "kg_boost":
                    x_val = cfg.get("kg_boost_weight", 0.1)
                else:
                    continue
                x_vals.append(x_val)
                y_vals.append(entry["metrics"].get(metric, 0))

            label = metric_labels.get(metric, metric)
            ax.plot(x_vals, y_vals, "o-", color=colors[j], label=label,
                    linewidth=2, markersize=6)

        ax.set_xlabel(pconfig["label"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Sensitivity: {pconfig['label']}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: list[str],
    recipe_names: list[str] | None = None,
    method: str = "tsne",
    output_path: Path | None = None,
) -> Path:
    """
    t-SNE or UMAP visualization of the embedding space, colored by cuisine tag.
    Reports silhouette score as a measure of cluster quality.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / f"embedding_space_{method}.png"

    from sklearn.metrics import silhouette_score

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    else:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            method = "tsne"

    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    if len(unique_labels) > 15:
        top_labels = sorted(set(labels), key=lambda x: labels.count(x), reverse=True)[:12]
        labels_mapped = [l if l in top_labels else "other" for l in labels]
        unique_labels = sorted(set(labels_mapped))
    else:
        labels_mapped = labels

    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    label_ints = np.array([label_to_int[l] for l in labels_mapped])

    try:
        sil_score = silhouette_score(coords, label_ints, sample_size=min(len(coords), 5000))
    except Exception:
        sil_score = 0.0

    fig, ax = plt.subplots(figsize=(12, 10))
    palette = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels_mapped) == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[i]], label=label,
                   s=15, alpha=0.6, edgecolors="none")

    ax.set_title(f"Recipe Embedding Space ({method.upper()})\n"
                 f"Silhouette Score: {sil_score:.3f}  |  {len(embeddings)} recipes, "
                 f"{len(unique_labels)} cuisine categories",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, markerscale=2, ncol=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_precision_recall_curves(
    ablation_results: dict,
    ground_truth_path: Path = Path("data/eval/retrieval_ground_truth.json"),
    output_path: Path | None = None,
) -> Path:
    """
    Precision-Recall curves for each ablation config as k varies.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "precision_recall_curves.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Set2", len(ablation_results))
    k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50]

    for idx, (name, result) in enumerate(ablation_results.items()):
        per_query = result.get("per_query", [])
        if not per_query:
            continue

        precisions = []
        recalls = []
        for k in k_values:
            p_key = f"precision@{k}"
            r_key = f"recall@{k}"
            p_vals = [q["metrics"].get(p_key) for q in per_query if p_key in q["metrics"]]
            r_vals = [q["metrics"].get(r_key) for q in per_query if r_key in q["metrics"]]
            if p_vals and r_vals:
                precisions.append(np.mean(p_vals))
                recalls.append(np.mean(r_vals))

        if precisions and recalls:
            label = name.replace("_", " ").title()
            ax.plot(recalls, precisions, "o-", color=colors[idx], label=label,
                    linewidth=2, markersize=6)

            for k_val, r, p in zip(k_values[:len(recalls)], recalls, precisions):
                if k_val in [5, 10, 20]:
                    ax.annotate(f"k={k_val}", (r, p), textcoords="offset points",
                                xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves by Retrieval Configuration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_component_contribution_heatmap(
    ablation_results: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Heatmap showing which retrieval components found each result.
    Shows the fraction of top-10 results sourced from vector, BM25, or both.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "component_contribution.png"

    configs = []
    source_data = []

    for name, result in ablation_results.items():
        per_query = result.get("per_query", [])
        if not per_query:
            continue

        source_counts = {"vector": 0, "bm25": 0, "vector+bm25": 0, "unknown": 0}
        total = 0
        for q in per_query:
            for src, count in q.get("source_breakdown", {}).items():
                source_counts[src] = source_counts.get(src, 0) + count
                total += count

        if total > 0:
            configs.append(name.replace("_", " "))
            source_data.append({k: v / total for k, v in source_counts.items()})

    if not configs:
        return output_path

    sources = ["vector", "bm25", "vector+bm25"]
    data_matrix = np.array([[d.get(s, 0) for s in sources] for d in source_data])

    fig, ax = plt.subplots(figsize=(10, max(6, len(configs) * 0.6)))
    sns.heatmap(data_matrix, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=["Vector Only", "BM25 Only", "Both (RRF)"],
                yticklabels=configs, ax=ax, linewidths=0.5, vmin=0, vmax=1)

    ax.set_title("Source Component Contribution to Retrieved Results", fontsize=13, fontweight="bold")
    ax.set_xlabel("Result Source", fontsize=11)
    ax.set_ylabel("Configuration", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_kg_analysis(
    graph_path: Path = Path("data/recipe_graph.pkl"),
    output_path: Path | None = None,
) -> Path:
    """
    Knowledge graph structural analysis: degree distribution, node type breakdown,
    and connectivity statistics.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "kg_analysis.png"

    import pickle
    import networkx as nx

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "unknown")
        node_types.setdefault(ntype, []).append(node)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Node type distribution
    ax = axes[0, 0]
    types = list(node_types.keys())
    counts = [len(node_types[t]) for t in types]
    colors = sns.color_palette("Set2", len(types))
    bars = ax.bar(types, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{c:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Node Type Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2. Recipe degree distribution (how many ingredients/tags per recipe)
    ax = axes[0, 1]
    recipe_degrees = [G.degree(n) for n in node_types.get("recipe", [])]
    if recipe_degrees:
        ax.hist(recipe_degrees, bins=50, color=sns.color_palette("Set2")[0],
                edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.axvline(np.mean(recipe_degrees), color="red", linestyle="--",
                   label=f"Mean: {np.mean(recipe_degrees):.1f}")
        ax.axvline(np.median(recipe_degrees), color="blue", linestyle="--",
                   label=f"Median: {np.median(recipe_degrees):.1f}")
        ax.legend(fontsize=9)
    ax.set_title("Recipe Node Degree Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Degree (# ingredients + tags)")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 3. Ingredient popularity (top 20 most connected ingredients)
    ax = axes[1, 0]
    ing_degrees = [(n.split(":", 1)[1], G.degree(n))
                   for n in node_types.get("ingredient", []) if ":" in n]
    ing_degrees.sort(key=lambda x: x[1], reverse=True)
    top_ings = ing_degrees[:20]
    if top_ings:
        names, degs = zip(*top_ings)
        y_pos = range(len(names))
        ax.barh(y_pos, degs, color=sns.color_palette("Set2")[1],
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
    ax.set_title("Top 20 Most Connected Ingredients", fontsize=12, fontweight="bold")
    ax.set_xlabel("# Recipes Using This Ingredient")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = (
        f"Knowledge Graph Statistics\n"
        f"{'='*35}\n\n"
        f"Total Nodes: {G.number_of_nodes():,}\n"
        f"Total Edges: {G.number_of_edges():,}\n"
        f"Graph Density: {nx.density(G):.6f}\n\n"
        f"Node Breakdown:\n"
    )
    for t, nodes in node_types.items():
        stats_text += f"  {t}: {len(nodes):,}\n"
    stats_text += f"\nAvg Recipe Degree: {np.mean(recipe_degrees):.1f}\n" if recipe_degrees else ""
    stats_text += f"Avg Ingredient Degree: {np.mean([G.degree(n) for n in node_types.get('ingredient', [])]):.1f}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Knowledge Graph Structural Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_query_type_breakdown(
    ablation_results: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Performance breakdown by query type (tag, ingredient, KG-graded, NL).
    Shows how different retriever configs perform on different query types.
    """
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "query_type_breakdown.png"

    config_names = []
    all_types = set()
    by_type_data = {}

    for name, result in ablation_results.items():
        config_names.append(name)
        bt = result.get("by_query_type", {})
        for qt in bt:
            all_types.add(qt)
        by_type_data[name] = bt

    if not config_names or not all_types:
        return output_path

    query_types = sorted(all_types)
    metric = "mean_mrr"

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(query_types))
    width = 0.8 / len(config_names)
    colors = sns.color_palette("Set2", len(config_names))

    for i, name in enumerate(config_names):
        vals = []
        for qt in query_types:
            bt = by_type_data.get(name, {}).get(qt, {})
            vals.append(bt.get(metric, 0))
        offset = (i - len(config_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name.replace("_", " ").title(),
               color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Query Type", fontsize=12)
    ax.set_ylabel("Mean MRR", fontsize=12)
    ax.set_title("Retrieval Performance by Query Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace("_", " ").title() for qt in query_types], fontsize=10)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_latency_comparison(
    ablation_summary: dict,
    output_path: Path | None = None,
) -> Path:
    """Bar chart comparing latency across configurations."""
    _ensure_dir()
    if output_path is None:
        output_path = CHARTS_DIR / "retrieval_latency.png"

    configs = ablation_summary.get("configs", {})
    if not configs:
        return output_path

    names = list(configs.keys())
    latencies = [configs[n]["avg_latency_ms"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if l < 100 else "orange" if l < 500 else "red" for l in latencies]

    bars = ax.barh(range(len(names)), latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=10)
    ax.set_xlabel("Avg Latency (ms)", fontsize=11)
    ax.set_title("Retrieval Latency by Configuration", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for bar, l in zip(bars, latencies):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{l:.0f}ms", va="center", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def generate_embedding_visualization(
    chroma_path: str = "data/chroma_db",
    sample_size: int = 2000,
    seed: int = 42,
) -> list[Path]:
    """
    Extract embeddings from ChromaDB and generate t-SNE and UMAP visualizations.
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("recipes")

    total = collection.count()
    if total == 0:
        return []

    result = collection.get(
        limit=min(sample_size, total),
        include=["embeddings", "metadatas"],
    )

    embeddings = np.array(result["embeddings"])
    metadatas = result["metadatas"]

    cuisine_tags = set()
    for t in list(CUISINE_TAGS_FOR_VIZ):
        cuisine_tags.add(t)

    labels = []
    for meta in metadatas:
        tags_raw = meta.get("tags_json", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except Exception:
            tags = []

        found = "other"
        for t in tags:
            if t.lower() in cuisine_tags:
                found = t.lower()
                break
        labels.append(found)

    paths = []
    for method in ["tsne", "umap"]:
        try:
            path = plot_embedding_space(embeddings, labels, method=method)
            paths.append(path)
        except Exception as e:
            print(f"Skipping {method}: {e}")

    return paths


CUISINE_TAGS_FOR_VIZ = {
    "mexican", "italian", "chinese", "japanese", "thai", "indian", "greek",
    "french", "korean", "mediterranean", "caribbean", "german",
}


def generate_all_visualizations(
    ablation_summary_path: Path | None = None,
    hyperparam_summary_path: Path | None = None,
    ablation_results_dir: Path = Path("data/eval/ablation_results"),
    stat_analysis: dict | None = None,
) -> list[Path]:
    """Generate all visualization charts and return list of output paths."""
    _ensure_dir()
    generated = []

    if ablation_summary_path and ablation_summary_path.exists():
        with open(ablation_summary_path) as f:
            ablation_summary = json.load(f)
        generated.append(plot_ablation_comparison(ablation_summary))
        generated.append(plot_latency_comparison(ablation_summary))

    if hyperparam_summary_path and hyperparam_summary_path.exists():
        with open(hyperparam_summary_path) as f:
            hyperparam_summary = json.load(f)
        generated.append(plot_hyperparam_sensitivity(hyperparam_summary))

    ablation_results = {}
    for p in sorted(ablation_results_dir.glob("retrieval_*.json")):
        if "summary" in p.name:
            continue
        with open(p) as f:
            data = json.load(f)
        name = data.get("config", {}).get("name", p.stem)
        ablation_results[name] = data

    if ablation_results:
        generated.append(plot_precision_recall_curves(ablation_results))
        generated.append(plot_component_contribution_heatmap(ablation_results))
        generated.append(plot_query_type_breakdown(ablation_results))

    if ablation_results and stat_analysis:
        generated.append(plot_ablation_with_ci(ablation_results, stat_analysis))

    graph_path = Path("data/recipe_graph.pkl")
    if graph_path.exists():
        generated.append(plot_kg_analysis(graph_path))

    try:
        paths = generate_embedding_visualization()
        generated.extend(paths)
    except Exception as e:
        print(f"Skipping embedding visualization: {e}")

    print(f"Generated {len(generated)} charts in {CHARTS_DIR}/")
    return generated
