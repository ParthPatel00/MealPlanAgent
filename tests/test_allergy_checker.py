import pytest

from src.tools.allergy_checker import _local_check, allergy_checker


class TestLocalCheck:
    def test_safe_no_allergens(self):
        safe, violations = _local_check(["chicken", "rice"], [])
        assert safe is True
        assert violations == []

    def test_safe_no_match(self):
        safe, violations = _local_check(["chicken", "rice"], ["peanuts"])
        assert safe is True
        assert violations == []

    def test_detects_direct_match(self):
        safe, violations = _local_check(["peanut butter", "bread"], ["peanuts"])
        assert safe is False
        assert "peanuts" in violations

    def test_detects_dairy_alias(self):
        safe, violations = _local_check(["cheese", "bread"], ["dairy"])
        assert safe is False
        assert "dairy" in violations

    def test_detects_gluten_alias(self):
        safe, violations = _local_check(["wheat flour", "sugar"], ["gluten"])
        assert safe is False
        assert "gluten" in violations

    def test_multiple_allergens(self):
        safe, violations = _local_check(
            ["peanut butter", "milk", "bread"],
            ["peanuts", "dairy"],
        )
        assert safe is False
        assert len(violations) == 2

    def test_soy_detection(self):
        safe, violations = _local_check(["tofu", "rice"], ["soy"])
        assert safe is False
        assert "soy" in violations

    def test_shellfish_detection(self):
        safe, violations = _local_check(["shrimp", "garlic"], ["shellfish"])
        assert safe is False
        assert "shellfish" in violations


class TestAllergyChecker:
    def test_no_allergens_returns_safe(self):
        result = allergy_checker(["chicken", "rice"], [], use_api=False)
        assert result["safe"] is True
        assert result["violations"] == []
        assert result["checked_via"] == []

    def test_local_only_safe(self):
        result = allergy_checker(["chicken", "rice"], ["peanuts"], use_api=False)
        assert result["safe"] is True
        assert "local_match" in result["checked_via"]

    def test_local_only_unsafe(self):
        result = allergy_checker(["peanut butter", "jam"], ["peanuts"], use_api=False)
        assert result["safe"] is False
        assert "peanuts" in result["violations"]

    def test_egg_detection(self):
        result = allergy_checker(["eggs", "flour", "sugar"], ["eggs"], use_api=False)
        assert result["safe"] is False

    def test_tree_nuts(self):
        result = allergy_checker(["almond milk", "oats"], ["tree nuts"], use_api=False)
        assert result["safe"] is False

    def test_case_insensitive(self):
        result = allergy_checker(["PEANUT BUTTER"], ["Peanuts"], use_api=False)
        assert result["safe"] is False
