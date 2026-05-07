"""
Knowledge graph over recipes, ingredients, and tags.

Builds a NetworkX graph that enables graph-based recipe recommendations:
- Find recipes sharing ingredients with a target recipe
- Find recipes with similar tags/cuisines
- Re-rank retrieval results based on graph proximity to user preferences
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx

GRAPH_PATH = Path("data/recipe_graph.pkl")


def build_graph(recipes: list[dict]) -> nx.Graph:
    """Build a recipe-ingredient-tag knowledge graph from recipe dicts."""
    G = nx.Graph()

    for recipe in recipes:
        rid = recipe.get("id", recipe.get("recipe_id", -1))
        name = recipe.get("name", "unknown")
        r_node = f"recipe:{rid}"

        G.add_node(r_node, type="recipe", name=name, minutes=recipe.get("minutes", 0))

        ingredients = recipe.get("ingredients", [])
        if isinstance(ingredients, str):
            try:
                ingredients = json.loads(ingredients)
            except (json.JSONDecodeError, TypeError):
                ingredients = [ingredients]

        for ing in ingredients:
            ing_norm = ing.strip().lower()
            i_node = f"ingredient:{ing_norm}"
            G.add_node(i_node, type="ingredient")
            G.add_edge(r_node, i_node, relation="HAS_INGREDIENT")

        tags = recipe.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = [tags]

        for tag in tags:
            tag_norm = tag.strip().lower()
            t_node = f"tag:{tag_norm}"
            G.add_node(t_node, type="tag")
            G.add_edge(r_node, t_node, relation="HAS_TAG")

    return G


def save_graph(G: nx.Graph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path: Path = GRAPH_PATH) -> nx.Graph | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def find_related_recipes(
    G: nx.Graph, recipe_id: int, relation: str = "HAS_INGREDIENT", top_k: int = 5
) -> list[tuple[int, float]]:
    """
    Find recipes related to a given recipe through shared ingredients or tags.

    Returns list of (recipe_id, similarity_score) tuples sorted by score desc.
    """
    r_node = f"recipe:{recipe_id}"
    if r_node not in G:
        return []

    neighbors = set()
    for neighbor in G.neighbors(r_node):
        edge_data = G.edges[r_node, neighbor]
        if edge_data.get("relation") == relation:
            neighbors.add(neighbor)

    recipe_scores: Counter = Counter()
    for shared_node in neighbors:
        for connected in G.neighbors(shared_node):
            if connected.startswith("recipe:") and connected != r_node:
                recipe_scores[connected] += 1

    results = []
    for node, count in recipe_scores.most_common(top_k):
        rid = int(node.split(":")[1])
        max_possible = len(neighbors)
        score = count / max_possible if max_possible > 0 else 0
        results.append((rid, round(score, 4)))

    return results


def graph_rerank(
    G: nx.Graph,
    hits: list,
    preferred_ingredients: list[str] | None = None,
    preferred_tags: list[str] | None = None,
    boost_weight: float = 0.1,
) -> list:
    """
    Re-rank retrieval hits using knowledge graph proximity to user preferences.

    Adds a small boost to the relevance score of recipes that share ingredients
    or tags with the user's known preferences.
    """
    if G is None or (not preferred_ingredients and not preferred_tags):
        return hits

    pref_nodes = set()
    for ing in (preferred_ingredients or []):
        pref_nodes.add(f"ingredient:{ing.strip().lower()}")
    for tag in (preferred_tags or []):
        pref_nodes.add(f"tag:{tag.strip().lower()}")

    existing_pref_nodes = pref_nodes & set(G.nodes())

    for hit in hits:
        rid = getattr(hit, "recipe_id", None)
        if rid is None:
            continue
        r_node = f"recipe:{rid}"
        if r_node not in G:
            continue

        recipe_neighbors = set(G.neighbors(r_node))
        overlap = len(recipe_neighbors & existing_pref_nodes)
        max_overlap = len(existing_pref_nodes) if existing_pref_nodes else 1
        boost = (overlap / max_overlap) * boost_weight
        hit.score = getattr(hit, "score", 0) + boost

    hits.sort(key=lambda h: getattr(h, "score", 0), reverse=True)
    return hits
