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
            "dietary_notes": "Light dinners, Mediterranean-style preferred."
        },
        "plan": {
            "meal_queries": [
                {"query": "vegetarian mediterranean light dinner", "day": "Monday", "cook_hour": 18, "max_minutes": 30},
                {"query": "quick vegetarian pasta no dairy", "day": "Wednesday", "cook_hour": 18, "max_minutes": 30},
                {"query": "easy vegetarian stir fry", "day": "Friday", "cook_hour": 18, "max_minutes": 30}
            ],
            "allergens": ["peanuts", "dairy"],
            "steps": [
                "Search for vegetarian recipes under 30 minutes",
                "Check each recipe for peanut and dairy allergens",
                "Calculate nutrition summary for the weekly plan",
                "Build categorized grocery list from ingredients",
                "Generate calendar events starting at 6 PM"
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
            "dietary_notes": ""
        },
        "plan": {
            "meal_queries": [
                {"query": "high protein chicken dinner gluten free", "day": "Monday", "cook_hour": 17, "max_minutes": 45},
                {"query": "protein rich beef meal no gluten", "day": "Tuesday", "cook_hour": 17, "max_minutes": 45},
                {"query": "high protein fish recipe quick", "day": "Wednesday", "cook_hour": 18, "max_minutes": 45},
                {"query": "easy high protein turkey dinner", "day": "Thursday", "cook_hour": 17, "max_minutes": 45},
                {"query": "protein packed egg and rice bowl", "day": "Friday", "cook_hour": 18, "max_minutes": 45}
            ],
            "allergens": ["gluten"],
            "steps": [
                "Search for high-protein recipes under 45 minutes, excluding gluten",
                "Verify allergen safety for each recipe",
                "Compute per-recipe and total nutrition",
                "Aggregate grocery list grouped by category",
                "Create .ics calendar with cooking blocks at 5-6 PM"
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
            "dietary_notes": "Meal prep for the whole week, budget-friendly."
        },
        "plan": {
            "meal_queries": [
                {"query": "low calorie healthy chicken breast", "day": "Monday", "cook_hour": 12, "max_minutes": 60},
                {"query": "healthy vegetable soup low calorie", "day": "Tuesday", "cook_hour": 12, "max_minutes": 60},
                {"query": "budget friendly healthy turkey meal", "day": "Wednesday", "cook_hour": 13, "max_minutes": 60},
                {"query": "low calorie salmon dinner healthy", "day": "Thursday", "cook_hour": 12, "max_minutes": 60},
                {"query": "easy healthy bean and rice bowl", "day": "Friday", "cook_hour": 12, "max_minutes": 60},
                {"query": "low calorie grilled chicken salad", "day": "Saturday", "cook_hour": 13, "max_minutes": 60},
                {"query": "healthy whole grain pasta low calorie", "day": "Sunday", "cook_hour": 12, "max_minutes": 60}
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
