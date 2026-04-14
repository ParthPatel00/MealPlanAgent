"""
Tool: nutrition

Parses the Food.com nutrition dict (PDV = Percent Daily Value based on a 2000-cal diet)
into human-readable labels and computes weekly totals across a meal plan.
"""

from __future__ import annotations

# Approximate absolute values per 100% PDV (FDA standard, 2000 cal diet)
PDV_REFERENCE = {
    "calories_pdv": 2000,      # kcal  (stored differently — see note)
    "total_fat_pdv": 78,       # g
    "sugar_pdv": 50,           # g
    "sodium_pdv": 2300,        # mg
    "protein_pdv": 50,         # g
    "saturated_fat_pdv": 20,   # g
    "carbohydrates_pdv": 275,  # g
}

# Display labels
DISPLAY_LABELS = {
    "calories_pdv": "Calories (kcal)",
    "total_fat_pdv": "Total Fat (g)",
    "sugar_pdv": "Sugar (g)",
    "sodium_pdv": "Sodium (mg)",
    "protein_pdv": "Protein (g)",
    "saturated_fat_pdv": "Saturated Fat (g)",
    "carbohydrates_pdv": "Carbohydrates (g)",
}


def pdv_to_absolute(nutrition: dict) -> dict:
    """
    Convert PDV percentages to approximate absolute values.
    Food.com stores calories directly as kcal (not PDV), so calories_pdv
    is already the raw calorie count — no conversion needed.
    """
    result = {}
    for key, ref in PDV_REFERENCE.items():
        pdv = nutrition.get(key, 0) or 0
        if key == "calories_pdv":
            result[DISPLAY_LABELS[key]] = round(pdv, 1)
        else:
            result[DISPLAY_LABELS[key]] = round((pdv / 100) * ref, 1)
    return result


def summarize_recipe_nutrition(recipe: dict) -> dict:
    """Return an absolute-value nutrition dict for a single recipe."""
    raw_nutrition = recipe.get("nutrition", {})
    return pdv_to_absolute(raw_nutrition)


def summarize_plan_nutrition(recipes: list[dict]) -> dict:
    """Sum nutrition across all recipes in the meal plan."""
    totals: dict[str, float] = {}
    for recipe in recipes:
        per_recipe = pdv_to_absolute(recipe.get("nutrition", {}))
        for label, value in per_recipe.items():
            totals[label] = round(totals.get(label, 0.0) + value, 1)
    return totals
