"""
Tool: allergy_checker

Two-layer allergy check:
  1. Fast local string match against ingredient list (no network)
  2. Open Food Facts API lookup for product-level allergen labels (network)

Returns a verdict dict: {safe: bool, violations: [...], checked_via: [...]}
"""

from __future__ import annotations

import httpx

OFF_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"


def _local_check(
    ingredients: list[str], allergens: list[str]
) -> tuple[bool, list[str]]:
    """String-match allergens against recipe ingredient names."""
    violations = []
    ingredients_str = " ".join(ingredients).lower()
    for allergen in allergens:
        if allergen.lower() in ingredients_str:
            violations.append(allergen)
    return len(violations) == 0, violations


def _off_search(ingredient_name: str, allergens: list[str]) -> list[str]:
    """
    Query Open Food Facts for a product matching `ingredient_name`
    and return any allergen labels that match the user's list.
    """
    try:
        params = {
            "search_terms": ingredient_name,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 3,
            "fields": "allergens,traces",
        }
        resp = httpx.get(OFF_SEARCH, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        products = data.get("products", [])
        found = []
        for product in products:
            allergen_str = (
                product.get("allergens", "") + " " + product.get("traces", "")
            ).lower()
            for allergen in allergens:
                if allergen.lower() in allergen_str and allergen not in found:
                    found.append(allergen)
        return found
    except Exception:
        return []


def allergy_checker(
    ingredients: list[str],
    allergens: list[str],
    use_api: bool = True,
) -> dict:
    """
    Check a recipe's ingredient list against the user's allergen list.

    Args:
        ingredients: Recipe ingredients, e.g. ["peanut butter", "flour", "eggs"].
        allergens: User's allergen list, e.g. ["peanuts", "gluten"].
        use_api: If True, also query Open Food Facts for label data.

    Returns:
        {
            "safe": bool,
            "violations": [list of flagged allergens],
            "checked_via": ["local_match", "open_food_facts"]
        }
    """
    if not allergens:
        return {"safe": True, "violations": [], "checked_via": []}

    safe_local, local_violations = _local_check(ingredients, allergens)
    checked_via = ["local_match"]
    api_violations: list[str] = []

    if use_api and safe_local:
        # Only hit the API if local check passed (avoid unnecessary calls)
        for ingredient in ingredients[:10]:  # check first 10 ingredients
            found = _off_search(ingredient, allergens)
            for v in found:
                if v not in api_violations:
                    api_violations.append(v)
        if api_violations:
            checked_via.append("open_food_facts")

    all_violations = list(set(local_violations + api_violations))
    return {
        "safe": len(all_violations) == 0,
        "violations": all_violations,
        "checked_via": checked_via,
    }
