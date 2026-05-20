import pytest

from src.tools.nutrition import pdv_to_absolute, summarize_plan_nutrition, summarize_recipe_nutrition


class TestPdvToAbsolute:
    def test_calories_passed_through(self):
        result = pdv_to_absolute({"calories_pdv": 500})
        assert result["Calories (kcal)"] == 500

    def test_fat_conversion(self):
        result = pdv_to_absolute({"total_fat_pdv": 50})
        assert result["Total Fat (g)"] == pytest.approx(39.0)

    def test_protein_conversion(self):
        result = pdv_to_absolute({"protein_pdv": 100})
        assert result["Protein (g)"] == pytest.approx(50.0)

    def test_sodium_conversion(self):
        result = pdv_to_absolute({"sodium_pdv": 50})
        assert result["Sodium (mg)"] == pytest.approx(1150.0)

    def test_zero_values(self):
        result = pdv_to_absolute({})
        assert all(v == 0 for v in result.values())

    def test_all_fields(self):
        nutrition = {
            "calories_pdv": 400,
            "total_fat_pdv": 30,
            "sugar_pdv": 20,
            "sodium_pdv": 10,
            "protein_pdv": 40,
            "saturated_fat_pdv": 25,
            "carbohydrates_pdv": 50,
        }
        result = pdv_to_absolute(nutrition)
        assert len(result) == 7
        assert result["Calories (kcal)"] == 400
        assert result["Total Fat (g)"] == pytest.approx(23.4)
        assert result["Sugar (g)"] == pytest.approx(10.0)
        assert result["Protein (g)"] == pytest.approx(20.0)
        assert result["Carbohydrates (g)"] == pytest.approx(137.5)


class TestSummarizeRecipeNutrition:
    def test_single_recipe(self, sample_recipes):
        result = summarize_recipe_nutrition(sample_recipes[0])
        assert "Calories (kcal)" in result
        assert result["Calories (kcal)"] == 450


class TestSummarizePlanNutrition:
    def test_sums_across_recipes(self, sample_recipes):
        result = summarize_plan_nutrition(sample_recipes)
        expected_cal = 450 + 380 + 520
        assert result["Calories (kcal)"] == expected_cal

    def test_empty_plan(self):
        result = summarize_plan_nutrition([])
        assert result == {}
