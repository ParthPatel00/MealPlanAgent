"""
Tool: recipe_search

Uses the HybridRetriever (BM25 + vector) to find recipes matching user constraints.
Returns structured results with citation metadata (recipe_id, name).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.retriever import RecipeHit, get_retriever

_RECIPE_DETAILS: dict[int, dict] | None = None


def _load_recipe_details() -> dict[int, dict]:
    global _RECIPE_DETAILS
    if _RECIPE_DETAILS is not None:
        return _RECIPE_DETAILS
    path = Path("data/processed/recipes_clean.json")
    with open(path) as f:
        recipes = json.load(f)
    _RECIPE_DETAILS = {r["id"]: r for r in recipes}
    return _RECIPE_DETAILS


def recipe_search(
    query: str,
    max_minutes: int | None = None,
    required_tags: list[str] | None = None,
    forbidden_ingredients: list[str] | None = None,
    preferred_ingredients: list[str] | None = None,
    preferred_tags: list[str] | None = None,
    calorie_target: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search for recipes using hybrid RAG retrieval.

    Args:
        query: Free-text query, e.g. "high protein chicken dinner".
        max_minutes: Filter out recipes that take longer than this.
        required_tags: All of these tags must appear in recipe tags.
        forbidden_ingredients: Recipes containing any of these are excluded.
        preferred_ingredients: Ingredients to boost via KG re-ranking.
        preferred_tags: Tags to boost via KG re-ranking.
        calorie_target: Target calories per meal (soft sort, not hard filter).
        top_k: Maximum number of results to return.

    Returns:
        List of recipe dicts with citation fields.
    """
    retriever = get_retriever()
    hits: list[RecipeHit] = retriever.retrieve(
        query,
        preferred_ingredients=preferred_ingredients,
        preferred_tags=preferred_tags,
    )
    details_db = _load_recipe_details()

    results = []
    for hit in hits:
        if max_minutes is not None and hit.minutes > max_minutes:
            continue

        if required_tags:
            hit_tags_lower = {t.lower() for t in hit.tags}
            if not all(t.lower() in hit_tags_lower for t in required_tags):
                continue

        if forbidden_ingredients:
            hit_ingredients_lower = " ".join(hit.ingredients).lower()
            if any(fi.lower() in hit_ingredients_lower for fi in forbidden_ingredients):
                continue

        detail = details_db.get(hit.recipe_id, {})

        import re
        slug = re.sub(r'[^a-z0-9]+', '-', hit.name.lower()).strip('-')

        results.append(
            {
                "citation": {
                    "recipe_id": hit.recipe_id,
                    "name": hit.name,
                    "source": "Food.com",
                    "url": f"https://www.food.com/recipe/{slug}-{hit.recipe_id}",
                },
                "name": hit.name,
                "minutes": hit.minutes,
                "tags": hit.tags,
                "ingredients": hit.ingredients,
                "nutrition": hit.nutrition,
                "steps": detail.get("steps", []),
                "description": detail.get("description", ""),
                "relevance_score": round(hit.score, 4),
            }
        )

        if len(results) >= top_k * 2:
            break

    if calorie_target is not None and len(results) > 1:
        def _calorie_distance(r):
            cal = r.get("nutrition", {}).get("calories_pdv", 0)
            return abs(cal - calorie_target)
        results.sort(key=_calorie_distance)

    return results[:top_k]
