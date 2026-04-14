"""
Tool: grocery_list

Aggregates ingredients across all selected recipes and groups them
into shopping categories (produce, dairy, meat/seafood, pantry, etc.)
using keyword-based rules.  Future improvement: replace with Instacart
dataset aisle/department IDs for precise grouping.
"""

from __future__ import annotations

from collections import defaultdict

# Keyword-to-category mapping (checked in order; first match wins)
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Produce", ["lettuce", "spinach", "kale", "arugula", "cabbage", "broccoli",
                 "cauliflower", "carrot", "celery", "cucumber", "zucchini", "squash",
                 "bell pepper", "pepper", "tomato", "onion", "garlic", "ginger",
                 "potato", "sweet potato", "yam", "mushroom", "corn", "pea",
                 "green bean", "asparagus", "avocado", "lemon", "lime", "orange",
                 "apple", "banana", "berry", "strawberry", "blueberry", "grape",
                 "mango", "pineapple", "peach", "pear", "fruit", "herb",
                 "cilantro", "parsley", "basil", "dill", "mint", "chive",
                 "scallion", "shallot", "leek"]),
    ("Meat & Seafood", ["chicken", "beef", "pork", "lamb", "turkey", "duck",
                        "salmon", "tuna", "shrimp", "prawn", "fish", "tilapia",
                        "cod", "halibut", "crab", "lobster", "clam", "mussel",
                        "sausage", "bacon", "ham", "steak", "ground beef",
                        "ground turkey", "ground pork"]),
    ("Dairy & Eggs", ["milk", "cream", "butter", "cheese", "cheddar", "mozzarella",
                      "parmesan", "ricotta", "feta", "yogurt", "sour cream",
                      "heavy cream", "half and half", "egg", "eggs"]),
    ("Bakery & Bread", ["bread", "baguette", "roll", "bun", "tortilla", "pita",
                        "naan", "bagel", "muffin", "croissant"]),
    ("Pasta, Rice & Grains", ["pasta", "spaghetti", "penne", "fusilli", "rice",
                               "quinoa", "couscous", "barley", "oat", "oats",
                               "noodle", "ramen", "udon", "farro", "bulgur"]),
    ("Canned & Jarred", ["canned", "diced tomatoes", "tomato sauce", "tomato paste",
                         "crushed tomatoes", "coconut milk", "broth", "stock",
                         "beans", "chickpeas", "lentils", "kidney beans",
                         "black beans", "pinto beans"]),
    ("Condiments & Sauces", ["soy sauce", "fish sauce", "oyster sauce", "hot sauce",
                              "ketchup", "mustard", "mayonnaise", "vinegar",
                              "worcestershire", "teriyaki", "hoisin", "sriracha"]),
    ("Oils & Fats", ["olive oil", "vegetable oil", "canola oil", "coconut oil",
                     "sesame oil", "cooking spray", "lard", "shortening"]),
    ("Spices & Seasonings", ["salt", "pepper", "cumin", "paprika", "turmeric",
                              "chili", "cayenne", "cinnamon", "nutmeg", "clove",
                              "oregano", "thyme", "rosemary", "bay leaf", "curry",
                              "garlic powder", "onion powder", "garam masala",
                              "seasoning", "spice"]),
    ("Baking", ["flour", "sugar", "baking soda", "baking powder", "yeast",
                "vanilla", "cocoa", "chocolate", "honey", "syrup", "cornstarch",
                "powdered sugar", "brown sugar"]),
    ("Frozen", ["frozen", "ice cream", "edamame"]),
    ("Beverages", ["wine", "beer", "broth", "juice", "water", "stock"]),
]

OTHER_CATEGORY = "Other"


def _categorize(ingredient: str) -> str:
    ingredient_lower = ingredient.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in ingredient_lower:
                return category
    return OTHER_CATEGORY


def build_grocery_list(recipes: list[dict]) -> dict[str, list[str]]:
    """
    Aggregate and deduplicate ingredients across recipes, then group by category.

    Args:
        recipes: List of recipe dicts (each must have an "ingredients" list).

    Returns:
        Dict mapping category name -> sorted list of unique ingredients.
    """
    seen: set[str] = set()
    grouped: dict[str, list[str]] = defaultdict(list)

    for recipe in recipes:
        for ingredient in recipe.get("ingredients", []):
            ingredient_clean = ingredient.strip().lower()
            if ingredient_clean and ingredient_clean not in seen:
                seen.add(ingredient_clean)
                category = _categorize(ingredient_clean)
                grouped[category].append(ingredient.strip())

    # Sort within each category and sort categories
    return {cat: sorted(items) for cat, items in sorted(grouped.items())}
