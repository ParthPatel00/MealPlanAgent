import pytest

from src.tools.budget_estimator import _match_price, estimate_grocery_cost


class TestMatchPrice:
    def test_exact_match(self):
        name, price = _match_price("chicken")
        assert name == "chicken"
        assert price == 4.50

    def test_substring_match(self):
        name, price = _match_price("boneless chicken breast")
        assert "chicken breast" in name
        assert price == 5.50

    def test_unknown_item_default(self):
        _, price = _match_price("dragon fruit extract")
        assert price == 1.50

    def test_case_insensitive(self):
        name, price = _match_price("SALMON")
        assert price == 8.00


class TestEstimateGroceryCost:
    def test_basic_estimate(self):
        grocery_list = {
            "Produce": ["garlic", "onion"],
            "Meat & Seafood": ["chicken"],
        }
        result = estimate_grocery_cost(grocery_list)
        assert "total_estimated_cost" in result
        assert "per_category" in result
        assert "per_item" in result
        assert result["total_estimated_cost"] > 0

    def test_per_category_sums(self):
        grocery_list = {"Produce": ["garlic", "onion"]}
        result = estimate_grocery_cost(grocery_list)
        cat_total = result["per_category"]["Produce"]
        item_sum = sum(i["estimated_price"] for i in result["per_item"])
        assert cat_total == pytest.approx(item_sum)

    def test_empty_list(self):
        result = estimate_grocery_cost({})
        assert result["total_estimated_cost"] == 0
        assert result["per_item"] == []
