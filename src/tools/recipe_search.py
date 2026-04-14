"""
Tool: recipe_search

Uses the HybridRetriever (BM25 + vector) to find recipes matching user constraints.
Returns structured results with citation metadata (recipe_id, name).
"""

from __future__ import annotations

from src.rag.retriever import RecipeHit, get_retriever


def recipe_search(
    query: str,
    max_minutes: int | None = None,
    required_tags: list[str] | None = None,
    forbidden_ingredients: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search for recipes using hybrid RAG retrieval.

    Args:
        query: Free-text query, e.g. "high protein chicken dinner".
        max_minutes: Filter out recipes that take longer than this.
        required_tags: All of these tags must appear in recipe tags.
        forbidden_ingredients: Recipes containing any of these are excluded.
        top_k: Maximum number of results to return.

    Returns:
        List of recipe dicts with citation fields.
    """
    retriever = get_retriever()
    hits: list[RecipeHit] = retriever.retrieve(query)

    results = []
    for hit in hits:
        # Time filter
        if max_minutes is not None and hit.minutes > max_minutes:
            continue

        # Tag filter
        if required_tags:
            hit_tags_lower = {t.lower() for t in hit.tags}
            if not all(t.lower() in hit_tags_lower for t in required_tags):
                continue

        # Allergy / forbidden ingredient filter
        if forbidden_ingredients:
            hit_ingredients_lower = " ".join(hit.ingredients).lower()
            if any(fi.lower() in hit_ingredients_lower for fi in forbidden_ingredients):
                continue

        results.append(
            {
                "citation": {
                    "recipe_id": hit.recipe_id,
                    "name": hit.name,
                    "source": "Food.com",
                },
                "name": hit.name,
                "minutes": hit.minutes,
                "tags": hit.tags,
                "ingredients": hit.ingredients,
                "nutrition": hit.nutrition,
                "relevance_score": round(hit.score, 4),
            }
        )

        if len(results) >= top_k:
            break

    return results
