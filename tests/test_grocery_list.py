import pytest

from src.tools.grocery_list import _categorize, build_grocery_list


class TestCategorize:
    def test_produce(self):
        assert _categorize("spinach") == "Produce"
        assert _categorize("fresh tomato") == "Produce"
        assert _categorize("garlic cloves") == "Produce"

    def test_meat(self):
        assert _categorize("chicken breast") == "Meat & Seafood"
        assert _categorize("ground beef") == "Meat & Seafood"
        assert _categorize("salmon fillet") == "Meat & Seafood"

    def test_dairy(self):
        assert _categorize("cheddar cheese") == "Dairy & Eggs"
        assert _categorize("eggs") == "Dairy & Eggs"
        assert _categorize("butter") == "Dairy & Eggs"

    def test_grains(self):
        assert _categorize("white rice") == "Pasta, Rice & Grains"
        assert _categorize("spaghetti pasta") == "Pasta, Rice & Grains"

    def test_spices(self):
        assert _categorize("ground cumin") == "Spices & Seasonings"
        assert _categorize("paprika") == "Spices & Seasonings"

    def test_oils(self):
        assert _categorize("olive oil") == "Oils & Fats"

    def test_unknown_goes_to_other(self):
        assert _categorize("xylitol supplement") == "Other"


class TestBuildGroceryList:
    def test_basic_grouping(self, sample_recipes):
        result = build_grocery_list(sample_recipes)
        assert isinstance(result, dict)
        assert len(result) > 0
        all_items = [item for items in result.values() for item in items]
        assert len(all_items) > 0

    def test_deduplication(self):
        recipes = [
            {"ingredients": ["garlic", "rice"]},
            {"ingredients": ["garlic", "chicken"]},
        ]
        result = build_grocery_list(recipes)
        all_items = [item.lower() for items in result.values() for item in items]
        assert all_items.count("garlic") == 1

    def test_empty_recipes(self):
        result = build_grocery_list([])
        assert result == {}

    def test_recipe_without_ingredients(self):
        result = build_grocery_list([{"name": "Mystery Meal"}])
        assert result == {}

    def test_categories_sorted(self):
        recipes = [{"ingredients": ["chicken", "rice", "olive oil", "salt"]}]
        result = build_grocery_list(recipes)
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_items_sorted_within_category(self):
        recipes = [{"ingredients": ["zucchini", "avocado", "carrot"]}]
        result = build_grocery_list(recipes)
        produce = result.get("Produce", [])
        assert produce == sorted(produce)
