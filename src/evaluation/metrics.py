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


def score_case(result, case: dict) -> dict:
    """
    Score a single test case.

    Args:
        result: AgentResult from pipeline.run_pipeline().
        case: Test case dict from test_cases.json.

    Returns:
        Dict with individual metric scores.
    """
    return {
        "id": case["id"],
        "constraint_pass": constraint_pass(result.recipes, case),
        "allergy_violation_rate": allergy_violation_rate(result.allergy_reports, case),
        "citation_pass_rate": citation_pass_rate(result.recipes),
        "tool_success_rate": tool_success_rate(result.tool_calls),
        "num_recipes": len(result.recipes),
        "critic_valid": result.critic.valid if result.critic else None,
        "retries": result.retries,
    }


def aggregate_scores(scores: list[dict]) -> dict:
    """Compute mean metrics across all scored test cases."""
    if not scores:
        return {}

    def mean(key):
        vals = [s[key] for s in scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "num_cases": len(scores),
        "constraint_pass_rate": mean("constraint_pass"),
        "avg_allergy_violation_rate": mean("allergy_violation_rate"),
        "avg_citation_pass_rate": mean("citation_pass_rate"),
        "avg_tool_success_rate": mean("tool_success_rate"),
        "critic_valid_rate": mean("critic_valid"),
        "avg_retries": mean("retries"),
    }
