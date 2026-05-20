"""
Few-shot examples for the Planner LLM prompt.

Each example is a (constraints, plan) pair demonstrating the expected
input/output format. These are injected into the planner prompt to improve
structured JSON output quality, especially for weaker models.
"""

from __future__ import annotations

FEW_SHOT_EXAMPLES = [
    {
        "constraints": {
            "num_meals": 3,
            "max_minutes": 30,
            "tags": ["vegetarian"],
            "allergens": ["peanuts", "dairy"],
            "cook_after_hour": 18,
            "dietary_notes": "Light dinners, Mediterranean-style preferred.",
            "cuisine_preferences": ["mediterranean"],
        },
        "plan": {
            "meal_queries": [
                {"query": "vegetarian mediterranean light dinner", "day": "Monday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": [], "preferred_tags": ["vegetarian", "mediterranean"]},
                {"query": "quick vegetarian pasta no dairy", "day": "Wednesday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": [], "preferred_tags": ["vegetarian", "mediterranean"]},
                {"query": "easy vegetarian stir fry mediterranean", "day": "Friday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": [], "preferred_tags": ["vegetarian", "mediterranean"]}
            ],
            "allergens": ["peanuts", "dairy"],
            "steps": [
                "Search for vegetarian recipes under 30 minutes",
                "Check each recipe for peanut and dairy allergens",
                "Calculate nutrition summary for the weekly plan",
                "Build categorized grocery list from ingredients",
                "Estimate grocery budget"
            ],
            "notes": "Focus on Mediterranean flavors. Avoid all nuts and dairy."
        }
    },
    {
        "constraints": {
            "num_meals": 5,
            "max_minutes": 45,
            "tags": ["high-protein"],
            "allergens": ["gluten"],
            "cook_after_hour": 17,
            "dietary_notes": "",
        },
        "plan": {
            "meal_queries": [
                {"query": "high protein chicken dinner gluten free", "day": "Monday", "cook_hour": 17, "max_minutes": 45, "preferred_ingredients": [], "preferred_tags": ["high-protein"]},
                {"query": "protein rich beef meal no gluten", "day": "Tuesday", "cook_hour": 17, "max_minutes": 45, "preferred_ingredients": [], "preferred_tags": ["high-protein"]},
                {"query": "high protein fish recipe quick", "day": "Wednesday", "cook_hour": 18, "max_minutes": 45, "preferred_ingredients": [], "preferred_tags": ["high-protein"]},
                {"query": "easy high protein turkey dinner", "day": "Thursday", "cook_hour": 17, "max_minutes": 45, "preferred_ingredients": [], "preferred_tags": ["high-protein"]},
                {"query": "protein packed egg and rice bowl", "day": "Friday", "cook_hour": 18, "max_minutes": 45, "preferred_ingredients": [], "preferred_tags": ["high-protein"]}
            ],
            "allergens": ["gluten"],
            "steps": [
                "Search for high-protein recipes under 45 minutes, excluding gluten",
                "Verify allergen safety for each recipe",
                "Compute per-recipe and total nutrition",
                "Aggregate grocery list grouped by category",
                "Build cooking schedule with blocks at 5-6 PM"
            ],
            "notes": "Variety of protein sources across the week."
        }
    },
    {
        "constraints": {
            "num_meals": 7,
            "max_minutes": 60,
            "tags": ["low-calorie", "healthy"],
            "allergens": ["shellfish", "soy"],
            "cook_after_hour": 12,
            "dietary_notes": "Meal prep for the whole week, budget-friendly.",
        },
        "plan": {
            "meal_queries": [
                {"query": "low calorie healthy chicken breast", "day": "Monday", "cook_hour": 12, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "healthy vegetable soup low calorie", "day": "Tuesday", "cook_hour": 12, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "budget friendly healthy turkey meal", "day": "Wednesday", "cook_hour": 13, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "low calorie salmon dinner healthy", "day": "Thursday", "cook_hour": 12, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "easy healthy bean and rice bowl", "day": "Friday", "cook_hour": 12, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "low calorie grilled chicken salad", "day": "Saturday", "cook_hour": 13, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]},
                {"query": "healthy whole grain pasta low calorie", "day": "Sunday", "cook_hour": 12, "max_minutes": 60, "preferred_ingredients": [], "preferred_tags": ["low-calorie", "healthy"]}
            ],
            "allergens": ["shellfish", "soy"],
            "steps": [
                "Search for low-calorie, healthy recipes under 60 minutes",
                "Exclude all shellfish and soy-containing recipes",
                "Compute weekly nutrition totals",
                "Build a comprehensive grocery list for meal prep",
                "Schedule cooking blocks starting at noon"
            ],
            "notes": "Full week meal prep. Prioritize budget-friendly, simple ingredients."
        }
    },
    {
        "constraints": {
            "num_meals": 4,
            "max_minutes": 30,
            "tags": [],
            "allergens": ["dairy"],
            "cook_after_hour": 18,
            "dietary_notes": "",
            "ingredients_on_hand": ["chicken breast", "rice", "bell peppers", "onions"],
            "cuisine_preferences": ["asian", "mexican"],
        },
        "plan": {
            "meal_queries": [
                {"query": "chicken rice stir fry asian style", "day": "Monday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": ["chicken breast", "rice", "bell peppers"], "preferred_tags": ["asian"]},
                {"query": "mexican chicken fajitas bell peppers", "day": "Wednesday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": ["chicken breast", "bell peppers", "onions"], "preferred_tags": ["mexican"]},
                {"query": "asian chicken fried rice quick", "day": "Thursday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": ["chicken breast", "rice", "onions"], "preferred_tags": ["asian"]},
                {"query": "chicken burrito bowl rice peppers", "day": "Saturday", "cook_hour": 18, "max_minutes": 30, "preferred_ingredients": ["chicken breast", "rice", "bell peppers"], "preferred_tags": ["mexican"]}
            ],
            "allergens": ["dairy"],
            "steps": [
                "Search for recipes using chicken, rice, and peppers with Asian and Mexican flavors",
                "Verify each recipe is dairy-free",
                "Calculate per-recipe and weekly nutrition",
                "Build grocery list (user already has chicken, rice, peppers, onions)",
                "Estimate grocery budget"
            ],
            "notes": "Prioritize recipes using ingredients the user already has. Alternate between Asian and Mexican cuisines."
        }
    },
    {
        "constraints": {
            "num_meals": 3,
            "max_minutes": 45,
            "tags": ["keto"],
            "allergens": [],
            "cook_after_hour": 19,
            "dietary_notes": "",
            "ingredients_on_hand": ["salmon", "avocado", "spinach", "eggs"],
            "calorie_target_per_meal": 600,
        },
        "plan": {
            "meal_queries": [
                {"query": "keto salmon avocado dinner low carb", "day": "Monday", "cook_hour": 19, "max_minutes": 45, "preferred_ingredients": ["salmon", "avocado", "spinach"], "preferred_tags": ["keto"]},
                {"query": "keto egg spinach frittata", "day": "Wednesday", "cook_hour": 19, "max_minutes": 45, "preferred_ingredients": ["eggs", "spinach", "avocado"], "preferred_tags": ["keto"]},
                {"query": "keto salmon bowl spinach eggs", "day": "Friday", "cook_hour": 19, "max_minutes": 45, "preferred_ingredients": ["salmon", "spinach", "eggs"], "preferred_tags": ["keto"]}
            ],
            "allergens": [],
            "steps": [
                "Search for keto recipes under 45 minutes using salmon, eggs, spinach, avocado",
                "Target approximately 600 calories per meal",
                "Calculate per-recipe nutrition and macro breakdown",
                "Build grocery list noting user already has key ingredients",
                "Estimate grocery budget"
            ],
            "notes": "Use the ingredients on hand (salmon, avocado, spinach, eggs). Target ~600 cal per meal for keto macros."
        }
    },
]


def format_few_shot_prompt() -> str:
    """Format few-shot examples as a string to prepend to the planner prompt."""
    import json
    parts = []
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input constraints: {json.dumps(example['constraints'], indent=2)}")
        parts.append(f"Output plan: {json.dumps(example['plan'], indent=2)}")
        parts.append("")
    return "\n".join(parts)
