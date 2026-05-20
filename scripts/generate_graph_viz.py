"""
Generate knowledge graph visualizations for the presentation.

Usage:
    python generate_graph_viz.py

Output: report_charts/knowledge_graph_*.png
"""

import os
import pickle
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
GRAPH_PATH = Path("data/recipe_graph.pkl")
CHART_DIR = Path("report/charts")
CHART_DIR.mkdir(exist_ok=True)


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def viz_single_recipe(G, recipe_node, output_name="knowledge_graph_single_recipe.png"):
    """
    Visualize a single recipe and all its ingredient + tag neighbors.
    This is the clearest way to show the graph structure on a slide.
    """
    neighbors = list(G.neighbors(recipe_node))
    subgraph_nodes = [recipe_node] + neighbors
    S = G.subgraph(subgraph_nodes).copy()

    color_map = []
    labels = {}
    sizes = []
    for node in S.nodes():
        ntype = S.nodes[node].get("type", "unknown")
        if ntype == "recipe":
            color_map.append("#3498db")
            sizes.append(1800)
            labels[node] = S.nodes[node].get("name", node)
        elif ntype == "ingredient":
            color_map.append("#2ecc71")
            sizes.append(600)
            labels[node] = node.replace("ingredient:", "")
        elif ntype == "tag":
            color_map.append("#e67e22")
            sizes.append(600)
            labels[node] = node.replace("tag:", "")
        else:
            color_map.append("#95a5a6")
            sizes.append(400)
            labels[node] = node

    edge_colors = []
    for u, v in S.edges():
        rel = S.edges[u, v].get("relation", "")
        if rel == "HAS_INGREDIENT":
            edge_colors.append("#2ecc71")
        elif rel == "HAS_TAG":
            edge_colors.append("#e67e22")
        else:
            edge_colors.append("#cccccc")

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(S, k=2.5, seed=42, iterations=80)

    nx.draw_networkx_edges(S, pos, ax=ax, edge_color=edge_colors, alpha=0.5, width=1.5)
    nx.draw_networkx_nodes(S, pos, ax=ax, node_color=color_map, node_size=sizes,
                           edgecolors="white", linewidths=1.5, alpha=0.9)
    nx.draw_networkx_labels(S, pos, labels, ax=ax, font_size=8, font_weight="bold")

    legend = [
        mpatches.Patch(color="#3498db", label=f"Recipe"),
        mpatches.Patch(color="#2ecc71", label=f"Ingredient ({sum(1 for n in S if S.nodes[n].get('type') == 'ingredient')})"),
        mpatches.Patch(color="#e67e22", label=f"Tag ({sum(1 for n in S if S.nodes[n].get('type') == 'tag')})"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=11, framealpha=0.9)

    recipe_name = G.nodes[recipe_node].get("name", recipe_node)
    ax.set_title(f"Knowledge Graph: \"{recipe_name}\"", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(CHART_DIR / output_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {CHART_DIR / output_name}")


def viz_shared_ingredients(G, recipe_node_a, recipe_node_b, output_name="knowledge_graph_shared.png"):
    """
    Visualize two recipes and highlight their shared ingredients/tags.
    Shows how the graph connects recipes through common nodes.
    """
    neighbors_a = set(G.neighbors(recipe_node_a))
    neighbors_b = set(G.neighbors(recipe_node_b))
    shared = neighbors_a & neighbors_b

    subgraph_nodes = {recipe_node_a, recipe_node_b} | neighbors_a | neighbors_b
    S = G.subgraph(subgraph_nodes).copy()

    color_map = []
    labels = {}
    sizes = []
    for node in S.nodes():
        ntype = S.nodes[node].get("type", "unknown")
        is_shared = node in shared
        if ntype == "recipe":
            color_map.append("#3498db")
            sizes.append(2000)
            labels[node] = S.nodes[node].get("name", node)
        elif ntype == "ingredient":
            color_map.append("#e74c3c" if is_shared else "#2ecc71")
            sizes.append(800 if is_shared else 400)
            labels[node] = node.replace("ingredient:", "") if is_shared else ""
        elif ntype == "tag":
            color_map.append("#e74c3c" if is_shared else "#e67e22")
            sizes.append(800 if is_shared else 400)
            labels[node] = node.replace("tag:", "") if is_shared else ""
        else:
            color_map.append("#95a5a6")
            sizes.append(300)
            labels[node] = ""

    edge_colors = []
    edge_widths = []
    for u, v in S.edges():
        if u in shared or v in shared:
            edge_colors.append("#e74c3c")
            edge_widths.append(2.0)
        else:
            edge_colors.append("#dddddd")
            edge_widths.append(0.5)

    fig, ax = plt.subplots(figsize=(16, 10))
    pos = nx.spring_layout(S, k=1.8, seed=42, iterations=100)

    nx.draw_networkx_edges(S, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(S, pos, ax=ax, node_color=color_map, node_size=sizes,
                           edgecolors="white", linewidths=1.5, alpha=0.9)
    nx.draw_networkx_labels(S, pos, labels, ax=ax, font_size=8, font_weight="bold")

    name_a = G.nodes[recipe_node_a].get("name", "?")
    name_b = G.nodes[recipe_node_b].get("name", "?")
    legend = [
        mpatches.Patch(color="#3498db", label="Recipe"),
        mpatches.Patch(color="#2ecc71", label="Unique ingredient"),
        mpatches.Patch(color="#e67e22", label="Unique tag"),
        mpatches.Patch(color="#e74c3c", label=f"Shared ({len(shared)} nodes)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_title(f"Shared Connections: \"{name_a}\" and \"{name_b}\"",
                 fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(CHART_DIR / output_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {CHART_DIR / output_name}")


def viz_neighborhood_overview(G, output_name="knowledge_graph_overview.png"):
    """
    Visualize a small cluster of 5 recipes and their shared ingredient/tag neighborhood.
    Gives a zoomed-out sense of the graph's density and structure.
    """
    recipe_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "recipe"]
    random.seed(42)

    # Pick a seed recipe with moderate degree (not too sparse, not too dense)
    candidates = [n for n in recipe_nodes if 10 <= G.degree(n) <= 25]
    if not candidates:
        candidates = recipe_nodes
    seed = random.choice(candidates)

    # Find 4 recipes most connected to the seed through shared neighbors
    seed_neighbors = set(G.neighbors(seed))
    related_scores = {}
    for neighbor in seed_neighbors:
        for connected in G.neighbors(neighbor):
            if connected.startswith("recipe:") and connected != seed:
                related_scores[connected] = related_scores.get(connected, 0) + 1

    top_related = sorted(related_scores, key=related_scores.get, reverse=True)[:4]
    focus_recipes = [seed] + top_related

    # Build subgraph: these recipes + their shared neighbors only
    all_neighbors = set()
    for r in focus_recipes:
        all_neighbors.update(G.neighbors(r))

    # Only keep neighbors connected to 2+ focus recipes (to reduce clutter)
    shared_neighbors = set()
    for n in all_neighbors:
        count = sum(1 for r in focus_recipes if G.has_edge(r, n))
        if count >= 2:
            shared_neighbors.add(n)

    subgraph_nodes = set(focus_recipes) | shared_neighbors
    S = G.subgraph(subgraph_nodes).copy()

    color_map = []
    labels = {}
    sizes = []
    for node in S.nodes():
        ntype = S.nodes[node].get("type", "unknown")
        if ntype == "recipe":
            color_map.append("#3498db")
            sizes.append(1500)
            name = S.nodes[node].get("name", node)
            labels[node] = name[:25] + "..." if len(name) > 25 else name
        elif ntype == "ingredient":
            color_map.append("#2ecc71")
            sizes.append(500)
            labels[node] = node.replace("ingredient:", "")
        elif ntype == "tag":
            color_map.append("#e67e22")
            sizes.append(500)
            labels[node] = node.replace("tag:", "")
        else:
            color_map.append("#95a5a6")
            sizes.append(300)
            labels[node] = ""

    fig, ax = plt.subplots(figsize=(16, 11))
    pos = nx.spring_layout(S, k=2.0, seed=42, iterations=100)

    nx.draw_networkx_edges(S, pos, ax=ax, edge_color="#cccccc", alpha=0.4, width=1.0)
    nx.draw_networkx_nodes(S, pos, ax=ax, node_color=color_map, node_size=sizes,
                           edgecolors="white", linewidths=1.5, alpha=0.85)
    nx.draw_networkx_labels(S, pos, labels, ax=ax, font_size=7, font_weight="bold")

    n_ing = sum(1 for n in S if S.nodes[n].get("type") == "ingredient")
    n_tag = sum(1 for n in S if S.nodes[n].get("type") == "tag")
    legend = [
        mpatches.Patch(color="#3498db", label=f"Recipes ({len(focus_recipes)})"),
        mpatches.Patch(color="#2ecc71", label=f"Shared ingredients ({n_ing})"),
        mpatches.Patch(color="#e67e22", label=f"Shared tags ({n_tag})"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_title(f"Knowledge Graph Cluster: {len(S.nodes)} nodes, {len(S.edges)} edges "
                 f"(from full graph: 15,807 nodes, 270,466 edges)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(CHART_DIR / output_name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {CHART_DIR / output_name}")


def main():
    print("Loading knowledge graph...")
    G = load_graph()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    recipe_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "recipe"]
    print(f"  {len(recipe_nodes)} recipe nodes")

    # Pick a recognizable dinner recipe with a moderate number of connections
    best_recipe = "recipe:392598"  # "15 minute chicken and rice dinner" (degree 17)
    if best_recipe not in G:
        scored = [(n, G.degree(n)) for n in recipe_nodes]
        scored.sort(key=lambda x: abs(x[1] - 18))
        best_recipe = scored[0][0]
    print(f"\n1. Single recipe visualization: {G.nodes[best_recipe].get('name')} (degree {G.degree(best_recipe)})")
    viz_single_recipe(G, best_recipe)

    # Find a second recipe that shares ingredients with the first
    best_neighbors = set(G.neighbors(best_recipe))
    related_counts = {}
    for neighbor in best_neighbors:
        for connected in G.neighbors(neighbor):
            if connected.startswith("recipe:") and connected != best_recipe:
                related_counts[connected] = related_counts.get(connected, 0) + 1

    # Pick a recognizable second recipe that shares ingredients with the first
    # "pick me up party chicken kabobs" shares 7 ingredients/tags with the chicken rice dinner
    preferred_second = None
    for r, c in sorted(related_counts.items(), key=lambda x: -x[1]):
        name = G.nodes[r].get("name", "").lower()
        if any(kw in name for kw in ["chicken", "rice", "stir", "kabob", "curry", "pasta"]) and 4 <= c <= 8:
            preferred_second = r
            break
    if preferred_second:
        second_recipe = preferred_second
    else:
        candidates = [(r, c) for r, c in related_counts.items() if 3 <= c <= 6]
        if not candidates:
            candidates = sorted(related_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        second_recipe = candidates[0][0] if candidates else recipe_nodes[1]
    shared_count = related_counts.get(second_recipe, 0)
    print(f"\n2. Shared ingredients visualization: {G.nodes[second_recipe].get('name')} ({shared_count} shared)")
    viz_shared_ingredients(G, best_recipe, second_recipe)

    print(f"\n3. Cluster overview visualization")
    viz_neighborhood_overview(G)

    print(f"\nDone. All images in {CHART_DIR}/")


if __name__ == "__main__":
    main()
