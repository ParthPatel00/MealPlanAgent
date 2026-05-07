"""
CLI script to build the recipe knowledge graph.

Usage:
    python -m src.rag.graph_builder
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.knowledge_graph import build_graph, save_graph

RECIPES_PATH = Path("data/processed/recipes_clean.json")


def main() -> None:
    print(f"Loading recipes from {RECIPES_PATH}...")
    with open(RECIPES_PATH) as f:
        recipes = json.load(f)

    print(f"Building knowledge graph from {len(recipes)} recipes...")
    G = build_graph(recipes)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for t, count in sorted(node_types.items()):
        print(f"  {t}: {count}")

    save_graph(G)
    print(f"Graph saved to {G.graph.get('path', 'data/recipe_graph.pkl')}")


if __name__ == "__main__":
    main()
