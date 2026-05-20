import pytest

from src.agent.critic import CriticResult, run_critic
from src.agent.executor import ExecutorResult


def _make_result(**overrides) -> ExecutorResult:
    defaults = {
        "recipes": [
            {"name": "Chicken Stir Fry", "citation": {"recipe_id": 101}},
            {"name": "Pasta Primavera", "citation": {"recipe_id": 202}},
        ],
        "allergy_reports": [
            {"recipe_name": "Chicken Stir Fry", "safe": True},
            {"recipe_name": "Pasta Primavera", "safe": True},
        ],
        "cooking_blocks": [
            {"meal_name": "Chicken Stir Fry", "day": "Monday", "cook_hour": 18},
            {"meal_name": "Pasta Primavera", "day": "Tuesday", "cook_hour": 19},
        ],
    }
    defaults.update(overrides)
    return ExecutorResult(**defaults)


class TestCritic:
    def test_valid_result_passes(self):
        result = run_critic(_make_result())
        assert result.valid is True
        assert result.issues == []

    def test_empty_recipes_flagged(self):
        result = run_critic(_make_result(recipes=[], cooking_blocks=[]))
        assert result.valid is False
        assert any("No recipes" in i for i in result.issues)

    def test_allergy_violation_flagged(self):
        reports = [
            {"recipe_name": "Bad Dish", "safe": False, "violations": ["peanuts"]},
        ]
        er = _make_result(allergy_reports=reports)
        result = run_critic(er)
        assert result.valid is False
        assert any("Allergy" in i for i in result.issues)

    def test_missing_citation_flagged(self):
        recipes = [{"name": "No Citation", "citation": {}}]
        blocks = [{"meal_name": "No Citation", "day": "Monday", "cook_hour": 18}]
        er = _make_result(
            recipes=recipes,
            allergy_reports=[{"recipe_name": "No Citation", "safe": True}],
            cooking_blocks=blocks,
        )
        result = run_critic(er)
        assert result.valid is False
        assert any("citation" in i.lower() for i in result.issues)

    def test_duplicate_recipe_flagged(self):
        recipes = [
            {"name": "Same Dish", "citation": {"recipe_id": 101}},
            {"name": "Same Dish Copy", "citation": {"recipe_id": 101}},
        ]
        blocks = [
            {"meal_name": "Same Dish", "day": "Monday", "cook_hour": 18},
            {"meal_name": "Same Dish Copy", "day": "Tuesday", "cook_hour": 19},
        ]
        er = _make_result(
            recipes=recipes,
            allergy_reports=[
                {"recipe_name": "Same Dish", "safe": True},
                {"recipe_name": "Same Dish Copy", "safe": True},
            ],
            cooking_blocks=blocks,
        )
        result = run_critic(er)
        assert result.valid is False
        assert any("Duplicate" in i for i in result.issues)

    def test_invalid_day_flagged(self):
        blocks = [{"meal_name": "Test", "day": "Funday", "cook_hour": 18}]
        er = _make_result(cooking_blocks=blocks)
        result = run_critic(er)
        assert result.valid is False
        assert any("Invalid day" in i for i in result.issues)

    def test_invalid_cook_hour_flagged(self):
        blocks = [{"meal_name": "Test", "day": "Monday", "cook_hour": 25}]
        er = _make_result(cooking_blocks=blocks)
        result = run_critic(er)
        assert result.valid is False
        assert any("cook_hour" in i for i in result.issues)

    def test_fix_instructions_present_on_failure(self):
        result = run_critic(_make_result(recipes=[], cooking_blocks=[]))
        assert result.fix_instructions != ""
        assert "Re-plan" in result.fix_instructions
