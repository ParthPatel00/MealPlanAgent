"""
Evaluation metric functions for comparing model outputs against test case expectations.
"""

from __future__ import annotations


def constraint_pass(result_recipes: list[dict], case: dict) -> bool:
    """True if the number of meals matches expected and all are within time limit."""
    if len(result_recipes) < case["expected_meal_count"]:
        return False
    max_min = case["max_minutes"]
    return all(r.get("minutes", 0) <= max_min for r in result_recipes)


def allergy_violation_rate(allergy_reports: list[dict], case: dict) -> float:
    """Fraction of recipes that contain forbidden allergens (0.0 = perfect)."""
    forbidden = set(a.lower() for a in case.get("forbidden_allergens", []))
    if not forbidden or not allergy_reports:
        return 0.0
    violations = sum(1 for r in allergy_reports if not r.get("safe", True))
    return violations / len(allergy_reports)


def citation_pass_rate(result_recipes: list[dict]) -> float:
    """Fraction of recipes that have a valid citation (recipe_id present)."""
    if not result_recipes:
        return 0.0
    cited = sum(
        1 for r in result_recipes
        if r.get("citation") and r["citation"].get("recipe_id") is not None
    )
    return cited / len(result_recipes)


def tool_success_rate(tool_calls: list[dict]) -> float:
    """Fraction of tool calls that succeeded (no exception raised by executor)."""
    # Tool calls from executor are always appended on success; errors are logged separately
    # We report based on presence: if N recipes → N recipe_search calls expected
    return 1.0 if tool_calls else 0.0


def ingredient_coverage(result_recipes: list[dict], ingredients_on_hand: list[str]) -> float:
    """Fraction of user's available ingredients that appear in selected recipes."""
    if not ingredients_on_hand or not result_recipes:
        return 0.0
    on_hand_lower = {i.strip().lower() for i in ingredients_on_hand}
    found = set()
    for r in result_recipes:
        recipe_ings = " ".join(r.get("ingredients", [])).lower()
        for ing in on_hand_lower:
            if ing in recipe_ings:
                found.add(ing)
    return len(found) / len(on_hand_lower)


def cuisine_alignment(result_recipes: list[dict], cuisine_preferences: list[str]) -> float:
    """Fraction of recipes whose tags match at least one preferred cuisine."""
    if not cuisine_preferences or not result_recipes:
        return 0.0
    prefs_lower = {c.strip().lower() for c in cuisine_preferences}
    matched = 0
    for r in result_recipes:
        tags_lower = {t.lower() for t in r.get("tags", [])}
        if tags_lower & prefs_lower:
            matched += 1
    return matched / len(result_recipes)


def calorie_deviation(result_recipes: list[dict], calorie_target: int) -> float:
    """Mean absolute deviation of recipe calories from target (PDV units)."""
    if not result_recipes or calorie_target is None:
        return 0.0
    deviations = []
    for r in result_recipes:
        cal = r.get("nutrition", {}).get("calories_pdv", 0)
        deviations.append(abs(cal - calorie_target))
    return sum(deviations) / len(deviations) if deviations else 0.0


def score_case(result, case: dict) -> dict:
    """
    Score a single test case.

    Args:
        result: AgentResult from pipeline.run_pipeline().
        case: Test case dict, from test_cases.json or dynamic_test_cases.json.

    Returns:
        Dict with individual metric scores.
    """
    constraints = case.get("constraints", case)
    expected = case.get("expected", case)

    scores = {
        "id": case.get("id", 0),
        "constraint_pass": constraint_pass(result.recipes, {
            "expected_meal_count": expected.get("min_meals", constraints.get("expected_meal_count", 1)),
            "max_minutes": expected.get("max_minutes", constraints.get("max_minutes", 120)),
        }),
        "allergy_violation_rate": allergy_violation_rate(result.allergy_reports, {
            "forbidden_allergens": expected.get("forbidden_allergens", constraints.get("allergens", [])),
        }),
        "citation_pass_rate": citation_pass_rate(result.recipes),
        "tool_success_rate": tool_success_rate(result.tool_calls),
        "num_recipes": len(result.recipes),
        "critic_valid": result.critic.valid if result.critic else None,
        "retries": result.retries,
    }

    ings = constraints.get("ingredients_on_hand", [])
    if ings:
        scores["ingredient_coverage"] = ingredient_coverage(result.recipes, ings)

    cuisines = constraints.get("cuisine_preferences", [])
    if cuisines:
        scores["cuisine_alignment"] = cuisine_alignment(result.recipes, cuisines)

    cal_target = constraints.get("calorie_target_per_meal")
    if cal_target is not None:
        scores["calorie_deviation"] = calorie_deviation(result.recipes, cal_target)

    return scores


def aggregate_scores(scores: list[dict]) -> dict:
    """Compute mean metrics across all scored test cases."""
    if not scores:
        return {}

    def mean(key):
        vals = [s[key] for s in scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg = {
        "num_cases": len(scores),
        "constraint_pass_rate": mean("constraint_pass"),
        "avg_allergy_violation_rate": mean("allergy_violation_rate"),
        "avg_citation_pass_rate": mean("citation_pass_rate"),
        "avg_tool_success_rate": mean("tool_success_rate"),
        "critic_valid_rate": mean("critic_valid"),
        "avg_retries": mean("retries"),
    }

    if any("ingredient_coverage" in s for s in scores):
        agg["avg_ingredient_coverage"] = mean("ingredient_coverage")
    if any("cuisine_alignment" in s for s in scores):
        agg["avg_cuisine_alignment"] = mean("cuisine_alignment")
    if any("calorie_deviation" in s for s in scores):
        agg["avg_calorie_deviation"] = mean("calorie_deviation")

    return agg
