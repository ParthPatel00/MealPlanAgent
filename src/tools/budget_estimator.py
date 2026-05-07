"""
Tool: budget_estimator

Estimates the total grocery cost for a meal plan based on average
US retail ingredient prices (sourced from USDA/BLS averages).
"""

from __future__ import annotations

PRICE_DB: dict[str, float] = {
    # Produce
    "onion": 1.20, "garlic": 0.50, "tomato": 1.50, "potato": 1.00,
    "carrot": 1.00, "celery": 1.50, "bell pepper": 1.25, "broccoli": 2.00,
    "spinach": 2.50, "lettuce": 1.75, "cucumber": 1.00, "zucchini": 1.50,
    "mushroom": 2.50, "corn": 0.75, "green bean": 2.00, "pea": 1.50,
    "avocado": 1.50, "lemon": 0.50, "lime": 0.35, "apple": 1.50,
    "banana": 0.60, "orange": 1.00, "berry": 3.50, "strawberry": 3.50,
    "blueberry": 4.00, "grape": 2.50, "mango": 1.50, "pineapple": 3.00,
    "ginger": 0.75, "cilantro": 0.75, "parsley": 0.75, "basil": 1.50,
    "mint": 1.50, "rosemary": 1.50, "thyme": 1.50, "dill": 1.50,
    "scallion": 1.00, "green onion": 1.00, "jalapeño": 0.50, "cabbage": 1.50,
    "kale": 2.50, "sweet potato": 1.50, "asparagus": 3.00, "cauliflower": 2.50,
    "eggplant": 2.00, "squash": 1.75,

    # Meat & Seafood
    "chicken": 4.50, "chicken breast": 5.50, "chicken thigh": 4.00,
    "ground beef": 5.50, "beef": 7.00, "steak": 10.00, "pork": 4.50,
    "pork chop": 5.00, "bacon": 5.50, "sausage": 4.00, "ham": 4.50,
    "turkey": 5.00, "ground turkey": 5.50, "lamb": 8.00,
    "salmon": 8.00, "shrimp": 8.00, "tuna": 3.00, "cod": 6.00,
    "tilapia": 5.00, "fish": 6.00, "crab": 10.00,

    # Dairy & Eggs
    "egg": 3.50, "milk": 3.50, "butter": 4.00, "cheese": 4.50,
    "cream cheese": 3.00, "sour cream": 2.50, "yogurt": 3.00,
    "heavy cream": 3.50, "cream": 3.50, "parmesan": 5.00,
    "mozzarella": 4.00, "cheddar": 4.00, "feta": 4.50,

    # Grains & Pasta
    "rice": 2.50, "pasta": 1.50, "bread": 3.00, "flour": 2.50,
    "tortilla": 3.00, "noodle": 2.00, "oat": 3.50, "quinoa": 5.00,
    "couscous": 3.00, "cornmeal": 2.50, "breadcrumb": 2.00,

    # Canned & Jarred
    "tomato sauce": 1.50, "tomato paste": 1.00, "diced tomato": 1.25,
    "coconut milk": 2.00, "broth": 2.50, "stock": 2.50,
    "black bean": 1.25, "kidney bean": 1.25, "chickpea": 1.50,
    "lentil": 2.00, "bean": 1.25, "corn": 1.00, "olive": 3.00,

    # Oils & Condiments
    "olive oil": 5.00, "vegetable oil": 3.00, "oil": 3.50,
    "soy sauce": 2.50, "vinegar": 2.50, "ketchup": 2.50,
    "mustard": 2.00, "mayonnaise": 3.50, "hot sauce": 2.50,
    "honey": 5.00, "maple syrup": 6.00, "worcestershire": 3.00,

    # Spices & Seasonings
    "salt": 1.00, "pepper": 2.50, "cumin": 3.00, "paprika": 3.00,
    "cinnamon": 3.00, "oregano": 2.50, "chili powder": 2.50,
    "garlic powder": 2.50, "onion powder": 2.50, "cayenne": 3.00,
    "turmeric": 3.50, "nutmeg": 3.50, "bay leaf": 3.00,

    # Baking
    "sugar": 2.50, "brown sugar": 3.00, "baking powder": 2.50,
    "baking soda": 1.50, "vanilla": 4.00, "cocoa": 4.00,
    "chocolate": 3.50, "chocolate chip": 3.50, "cornstarch": 2.00,

    # Nuts & Seeds
    "almond": 6.00, "walnut": 7.00, "pecan": 8.00, "peanut": 4.00,
    "cashew": 7.00, "sesame": 3.50, "sunflower seed": 4.00,
    "pine nut": 10.00, "coconut": 2.50,

    # Misc
    "tofu": 2.50, "tempeh": 3.50, "water": 0.00,
}


def _match_price(ingredient: str) -> tuple[str, float]:
    ingredient_lower = ingredient.lower().strip()

    if ingredient_lower in PRICE_DB:
        return ingredient_lower, PRICE_DB[ingredient_lower]

    # Prefer longest matching key to avoid "chicken" shadowing "chicken breast"
    best_key = ""
    best_price = 1.50
    for key, price in PRICE_DB.items():
        if key in ingredient_lower or ingredient_lower in key:
            if len(key) > len(best_key):
                best_key = key
                best_price = price

    return best_key or ingredient_lower, best_price


def estimate_grocery_cost(grocery_list: dict[str, list[str]]) -> dict:
    """
    Estimate grocery costs from a categorized grocery list.

    Returns:
        Dict with total_estimated_cost, per_category breakdown, and per_item prices.
    """
    per_item = []
    per_category: dict[str, float] = {}
    total = 0.0

    for category, items in grocery_list.items():
        cat_total = 0.0
        for item in items:
            matched_name, price = _match_price(item)
            per_item.append({"item": item, "matched": matched_name, "estimated_price": price})
            cat_total += price
        per_category[category] = round(cat_total, 2)
        total += cat_total

    return {
        "total_estimated_cost": round(total, 2),
        "per_category": per_category,
        "per_item": per_item,
    }
