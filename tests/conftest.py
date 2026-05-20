import pytest


@pytest.fixture
def sample_recipes():
    return [
        {
            "name": "Chicken Stir Fry",
            "minutes": 25,
            "tags": ["high-protein", "asian"],
            "ingredients": ["chicken breast", "soy sauce", "bell pepper", "rice", "garlic"],
            "nutrition": {
                "calories_pdv": 450,
                "total_fat_pdv": 30,
                "sugar_pdv": 10,
                "sodium_pdv": 40,
                "protein_pdv": 60,
                "saturated_fat_pdv": 15,
                "carbohydrates_pdv": 35,
            },
            "citation": {"recipe_id": 101, "source": "Food.com"},
            "_day": "Monday",
            "_cook_hour": 18,
        },
        {
            "name": "Vegetable Pasta",
            "minutes": 20,
            "tags": ["vegetarian", "italian"],
            "ingredients": ["pasta", "tomato sauce", "spinach", "garlic", "olive oil"],
            "nutrition": {
                "calories_pdv": 380,
                "total_fat_pdv": 20,
                "sugar_pdv": 15,
                "sodium_pdv": 25,
                "protein_pdv": 20,
                "saturated_fat_pdv": 8,
                "carbohydrates_pdv": 55,
            },
            "citation": {"recipe_id": 202, "source": "Food.com"},
            "_day": "Tuesday",
            "_cook_hour": 19,
        },
        {
            "name": "Salmon Bowl",
            "minutes": 30,
            "tags": ["high-protein", "healthy"],
            "ingredients": ["salmon", "rice", "avocado", "cucumber", "soy sauce"],
            "nutrition": {
                "calories_pdv": 520,
                "total_fat_pdv": 35,
                "sugar_pdv": 5,
                "sodium_pdv": 30,
                "protein_pdv": 55,
                "saturated_fat_pdv": 10,
                "carbohydrates_pdv": 40,
            },
            "citation": {"recipe_id": 303, "source": "Food.com"},
            "_day": "Wednesday",
            "_cook_hour": 18,
        },
    ]


@pytest.fixture
def sample_constraints():
    return {
        "num_meals": 3,
        "max_minutes": 30,
        "tags": ["high-protein"],
        "allergens": [],
        "cook_after_hour": 18,
    }


@pytest.fixture
def sample_test_case():
    return {
        "id": 1,
        "constraints": {
            "num_meals": 3,
            "max_minutes": 30,
            "tags": ["high-protein"],
            "allergens": ["peanuts"],
            "cook_after_hour": 18,
        },
        "expected_meal_count": 3,
        "max_minutes": 30,
        "forbidden_allergens": ["peanuts"],
        "required_tags": ["high-protein"],
    }


@pytest.fixture
def sample_cooking_blocks():
    return [
        {"meal_name": "Chicken Stir Fry", "day": "Monday", "cook_hour": 18, "duration_minutes": 25},
        {"meal_name": "Vegetable Pasta", "day": "Tuesday", "cook_hour": 19, "duration_minutes": 20},
        {"meal_name": "Salmon Bowl", "day": "Wednesday", "cook_hour": 18, "duration_minutes": 30},
    ]


@pytest.fixture
def sample_graph_recipes():
    return [
        {"id": 1, "name": "Chicken Rice", "minutes": 30,
         "ingredients": ["chicken", "rice", "garlic", "soy sauce"],
         "tags": ["asian", "high-protein"]},
        {"id": 2, "name": "Chicken Pasta", "minutes": 25,
         "ingredients": ["chicken", "pasta", "garlic", "tomato sauce"],
         "tags": ["italian", "high-protein"]},
        {"id": 3, "name": "Veggie Stir Fry", "minutes": 20,
         "ingredients": ["tofu", "rice", "garlic", "soy sauce", "bell pepper"],
         "tags": ["asian", "vegetarian"]},
        {"id": 4, "name": "Salmon Salad", "minutes": 15,
         "ingredients": ["salmon", "lettuce", "avocado", "lemon"],
         "tags": ["healthy", "low-carb"]},
    ]
