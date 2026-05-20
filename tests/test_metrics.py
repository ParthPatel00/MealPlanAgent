import pytest

from src.evaluation.metrics import (
    aggregate_scores,
    allergy_violation_rate,
    calorie_deviation,
    citation_pass_rate,
    constraint_pass,
    cuisine_alignment,
    ingredient_coverage,
    tool_success_rate,
)


class TestConstraintPass:
    def test_pass_when_count_and_time_match(self):
        recipes = [{"minutes": 20}, {"minutes": 25}, {"minutes": 30}]
        case = {"expected_meal_count": 3, "max_minutes": 30}
        assert constraint_pass(recipes, case) is True

    def test_fail_when_too_few_meals(self):
        recipes = [{"minutes": 20}]
        case = {"expected_meal_count": 3, "max_minutes": 30}
        assert constraint_pass(recipes, case) is False

    def test_fail_when_time_exceeded(self):
        recipes = [{"minutes": 20}, {"minutes": 45}, {"minutes": 10}]
        case = {"expected_meal_count": 3, "max_minutes": 30}
        assert constraint_pass(recipes, case) is False

    def test_pass_with_extra_meals(self):
        recipes = [{"minutes": 10}, {"minutes": 15}, {"minutes": 20}, {"minutes": 25}]
        case = {"expected_meal_count": 3, "max_minutes": 30}
        assert constraint_pass(recipes, case) is True

    def test_edge_case_zero_minutes(self):
        recipes = [{"minutes": 0}]
        case = {"expected_meal_count": 1, "max_minutes": 30}
        assert constraint_pass(recipes, case) is True


class TestAllergyViolationRate:
    def test_zero_violations(self):
        reports = [{"safe": True}, {"safe": True}]
        case = {"forbidden_allergens": ["peanuts"]}
        assert allergy_violation_rate(reports, case) == 0.0

    def test_one_violation(self):
        reports = [{"safe": True}, {"safe": False}]
        case = {"forbidden_allergens": ["peanuts"]}
        assert allergy_violation_rate(reports, case) == 0.5

    def test_all_violations(self):
        reports = [{"safe": False}, {"safe": False}]
        case = {"forbidden_allergens": ["peanuts"]}
        assert allergy_violation_rate(reports, case) == 1.0

    def test_no_allergens_returns_zero(self):
        reports = [{"safe": False}]
        case = {"forbidden_allergens": []}
        assert allergy_violation_rate(reports, case) == 0.0

    def test_empty_reports_returns_zero(self):
        case = {"forbidden_allergens": ["peanuts"]}
        assert allergy_violation_rate([], case) == 0.0


class TestCitationPassRate:
    def test_all_cited(self):
        recipes = [
            {"citation": {"recipe_id": 1}},
            {"citation": {"recipe_id": 2}},
        ]
        assert citation_pass_rate(recipes) == 1.0

    def test_none_cited(self):
        recipes = [{"citation": {}}, {"citation": {"recipe_id": None}}]
        assert citation_pass_rate(recipes) == 0.0

    def test_mixed(self):
        recipes = [{"citation": {"recipe_id": 1}}, {"citation": {}}]
        assert citation_pass_rate(recipes) == 0.5

    def test_empty(self):
        assert citation_pass_rate([]) == 0.0


class TestToolSuccessRate:
    def test_non_empty(self):
        assert tool_success_rate([{"tool": "search"}]) == 1.0

    def test_empty(self):
        assert tool_success_rate([]) == 0.0


class TestIngredientCoverage:
    def test_full_coverage(self):
        recipes = [{"ingredients": ["chicken", "rice"]}]
        assert ingredient_coverage(recipes, ["chicken", "rice"]) == 1.0

    def test_partial_coverage(self):
        recipes = [{"ingredients": ["chicken"]}]
        assert ingredient_coverage(recipes, ["chicken", "rice"]) == 0.5

    def test_no_coverage(self):
        recipes = [{"ingredients": ["tofu"]}]
        assert ingredient_coverage(recipes, ["chicken"]) == 0.0

    def test_empty_inputs(self):
        assert ingredient_coverage([], ["chicken"]) == 0.0
        assert ingredient_coverage([{"ingredients": ["a"]}], []) == 0.0


class TestCuisineAlignment:
    def test_full_alignment(self):
        recipes = [{"tags": ["italian"]}, {"tags": ["italian"]}]
        assert cuisine_alignment(recipes, ["italian"]) == 1.0

    def test_no_alignment(self):
        recipes = [{"tags": ["mexican"]}, {"tags": ["thai"]}]
        assert cuisine_alignment(recipes, ["italian"]) == 0.0

    def test_partial(self):
        recipes = [{"tags": ["italian"]}, {"tags": ["mexican"]}]
        assert cuisine_alignment(recipes, ["italian"]) == 0.5


class TestCalorieDeviation:
    def test_exact_match(self):
        recipes = [{"nutrition": {"calories_pdv": 500}}]
        assert calorie_deviation(recipes, 500) == 0.0

    def test_deviation(self):
        recipes = [{"nutrition": {"calories_pdv": 600}}]
        assert calorie_deviation(recipes, 500) == 100.0


class TestAggregateScores:
    def test_aggregation(self):
        scores = [
            {"constraint_pass": True, "allergy_violation_rate": 0.0,
             "citation_pass_rate": 1.0, "tool_success_rate": 1.0,
             "critic_valid": True, "retries": 0},
            {"constraint_pass": True, "allergy_violation_rate": 0.5,
             "citation_pass_rate": 0.5, "tool_success_rate": 1.0,
             "critic_valid": False, "retries": 1},
        ]
        agg = aggregate_scores(scores)
        assert agg["num_cases"] == 2
        assert agg["constraint_pass_rate"] == 1.0
        assert agg["avg_allergy_violation_rate"] == 0.25
        assert agg["avg_citation_pass_rate"] == 0.75

    def test_empty(self):
        assert aggregate_scores([]) == {}
