"""
Dynamic test scenario generator for end-to-end pipeline evaluation.

Replaces hardcoded test cases with programmatically generated scenarios
drawn from the dataset's actual vocabulary. Generates 200+ diverse scenarios
across three complexity tiers with expected behaviors for validation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROCESSED_PATH = Path("data/processed/recipes_clean.json")
OUTPUT_PATH = Path("data/eval/dynamic_test_cases.json")


def _load_recipes(path: Path = PROCESSED_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _extract_vocabulary(recipes: list[dict]) -> dict:
    """Extract the actual tag/ingredient vocabulary from the dataset."""
    tags = {}
    ingredients = {}
    for r in recipes:
        for t in r.get("tags", []):
            tn = t.strip().lower()
            tags[tn] = tags.get(tn, 0) + 1
        for ing in r.get("ingredients", []):
            inn = ing.strip().lower()
            ingredients[inn] = ingredients.get(inn, 0) + 1

    common_tags = {t for t, c in tags.items() if 10 <= c <= 2000}
    common_ingredients = {i for i, c in ingredients.items() if 10 <= c <= 2000}
    return {"tags": common_tags, "ingredients": common_ingredients}


CUISINE_TAGS = {
    "mexican", "italian", "chinese", "japanese", "thai", "indian", "greek",
    "french", "korean", "mediterranean",
}

DIETARY_TAGS = {
    "vegetarian", "vegan", "low-carb", "low-fat", "low-sodium", "low-calorie",
    "high-protein", "gluten-free", "dairy-free", "keto",
}

ALLERGENS = [
    "peanuts", "dairy", "gluten", "shellfish", "soy", "eggs", "tree nuts", "wheat",
]

PROTEINS = ["chicken", "beef", "salmon", "tofu", "shrimp", "pork", "turkey", "tuna"]


def generate_simple_scenarios(vocab: dict, count: int = 70, seed: int = 42) -> list[dict]:
    """Tier 1: Simple scenarios with 1-2 constraints."""
    rng = random.Random(seed)
    scenarios = []

    available_cuisines = list(CUISINE_TAGS & vocab["tags"])
    available_dietary = list(DIETARY_TAGS & vocab["tags"])

    for _ in range(count):
        scenario_type = rng.choice(["diet_only", "cuisine_only", "time_only", "meals_only"])
        constraints = {
            "num_meals": rng.choice([3, 4, 5]),
            "max_minutes": 30,
            "tags": [],
            "allergens": [],
            "cook_after_hour": 18,
            "dietary_notes": "",
            "ingredients_on_hand": [],
            "cuisine_preferences": [],
            "calorie_target_per_meal": None,
        }
        expected = {"min_meals": constraints["num_meals"]}

        if scenario_type == "diet_only" and available_dietary:
            diet = rng.choice(available_dietary)
            constraints["tags"] = [diet]
            expected["required_tags"] = [diet]
        elif scenario_type == "cuisine_only" and available_cuisines:
            cuisine = rng.choice(available_cuisines)
            constraints["cuisine_preferences"] = [cuisine]
            expected["preferred_cuisine"] = cuisine
        elif scenario_type == "time_only":
            constraints["max_minutes"] = rng.choice([15, 20, 30])
            expected["max_minutes"] = constraints["max_minutes"]
        else:
            constraints["num_meals"] = rng.choice([3, 5, 7])
            expected["min_meals"] = constraints["num_meals"]

        nl_input = _constraints_to_nl(constraints, rng)
        scenarios.append({
            "complexity": "simple",
            "constraints": constraints,
            "nl_input": nl_input,
            "expected": expected,
        })

    return scenarios


def generate_medium_scenarios(vocab: dict, count: int = 70, seed: int = 42) -> list[dict]:
    """Tier 2: Medium scenarios with 3-4 constraints including ingredients."""
    rng = random.Random(seed + 100)
    scenarios = []

    available_cuisines = list(CUISINE_TAGS & vocab["tags"])
    available_dietary = list(DIETARY_TAGS & vocab["tags"])
    available_proteins = [p for p in PROTEINS if p in vocab["ingredients"]]

    for _ in range(count):
        constraints = {
            "num_meals": rng.choice([3, 4, 5]),
            "max_minutes": rng.choice([30, 45, 60]),
            "tags": [],
            "allergens": [],
            "cook_after_hour": rng.choice([12, 17, 18, 19]),
            "dietary_notes": "",
            "ingredients_on_hand": [],
            "cuisine_preferences": [],
            "calorie_target_per_meal": None,
        }
        expected = {"min_meals": constraints["num_meals"], "max_minutes": constraints["max_minutes"]}

        if available_dietary and rng.random() < 0.6:
            diet = rng.choice(available_dietary)
            constraints["tags"] = [diet]
            expected["required_tags"] = [diet]

        if rng.random() < 0.5:
            n_allergens = rng.choice([1, 2])
            constraints["allergens"] = rng.sample(ALLERGENS, min(n_allergens, len(ALLERGENS)))
            expected["forbidden_allergens"] = constraints["allergens"]

        if available_proteins and rng.random() < 0.7:
            n_ings = rng.choice([1, 2, 3])
            constraints["ingredients_on_hand"] = rng.sample(available_proteins, min(n_ings, len(available_proteins)))
            expected["preferred_ingredients"] = constraints["ingredients_on_hand"]

        if available_cuisines and rng.random() < 0.4:
            constraints["cuisine_preferences"] = [rng.choice(available_cuisines)]
            expected["preferred_cuisine"] = constraints["cuisine_preferences"][0]

        nl_input = _constraints_to_nl(constraints, rng)
        scenarios.append({
            "complexity": "medium",
            "constraints": constraints,
            "nl_input": nl_input,
            "expected": expected,
        })

    return scenarios


def generate_complex_scenarios(vocab: dict, count: int = 60, seed: int = 42) -> list[dict]:
    """Tier 3: Complex scenarios with 5+ constraints, calorie targets, edge cases."""
    rng = random.Random(seed + 200)
    scenarios = []

    available_cuisines = list(CUISINE_TAGS & vocab["tags"])
    available_dietary = list(DIETARY_TAGS & vocab["tags"])
    available_proteins = [p for p in PROTEINS if p in vocab["ingredients"]]

    for _ in range(count):
        constraints = {
            "num_meals": rng.choice([3, 5, 7]),
            "max_minutes": rng.choice([20, 30, 45, 60, 90]),
            "tags": [],
            "allergens": [],
            "cook_after_hour": rng.choice([7, 12, 17, 18, 19]),
            "dietary_notes": "",
            "ingredients_on_hand": [],
            "cuisine_preferences": [],
            "calorie_target_per_meal": rng.choice([None, 400, 500, 600, 800]),
        }
        expected = {
            "min_meals": constraints["num_meals"],
            "max_minutes": constraints["max_minutes"],
        }

        if available_dietary:
            n_tags = rng.choice([1, 2])
            constraints["tags"] = rng.sample(available_dietary, min(n_tags, len(available_dietary)))
            expected["required_tags"] = constraints["tags"]

        n_allergens = rng.choice([1, 2, 3])
        constraints["allergens"] = rng.sample(ALLERGENS, min(n_allergens, len(ALLERGENS)))
        expected["forbidden_allergens"] = constraints["allergens"]

        if available_proteins:
            n_ings = rng.choice([2, 3, 4])
            constraints["ingredients_on_hand"] = rng.sample(available_proteins, min(n_ings, len(available_proteins)))
            expected["preferred_ingredients"] = constraints["ingredients_on_hand"]

        if available_cuisines:
            n_cuisines = rng.choice([1, 2])
            constraints["cuisine_preferences"] = rng.sample(available_cuisines, min(n_cuisines, len(available_cuisines)))

        if constraints["calorie_target_per_meal"]:
            expected["calorie_target"] = constraints["calorie_target_per_meal"]

        nl_input = _constraints_to_nl(constraints, rng)
        scenarios.append({
            "complexity": "complex",
            "constraints": constraints,
            "nl_input": nl_input,
            "expected": expected,
        })

    return scenarios


def generate_edge_case_scenarios(count: int = 20, seed: int = 42) -> list[dict]:
    """Edge cases: conflicting constraints, extreme values."""
    rng = random.Random(seed + 300)
    scenarios = []

    edge_cases = [
        {
            "description": "Very tight time limit",
            "constraints": {"num_meals": 3, "max_minutes": 10, "tags": [], "allergens": [],
                            "cook_after_hour": 18, "dietary_notes": "Extremely quick meals only",
                            "ingredients_on_hand": [], "cuisine_preferences": [], "calorie_target_per_meal": None},
            "expected": {"min_meals": 1, "max_minutes": 10},
        },
        {
            "description": "Many allergens",
            "constraints": {"num_meals": 5, "max_minutes": 45, "tags": [],
                            "allergens": ["dairy", "gluten", "peanuts", "soy", "eggs", "shellfish"],
                            "cook_after_hour": 18, "dietary_notes": "",
                            "ingredients_on_hand": [], "cuisine_preferences": [], "calorie_target_per_meal": None},
            "expected": {"min_meals": 3, "forbidden_allergens": ["dairy", "gluten", "peanuts", "soy", "eggs", "shellfish"]},
        },
        {
            "description": "Conflicting diet + cuisine",
            "constraints": {"num_meals": 3, "max_minutes": 30, "tags": ["vegan"],
                            "allergens": [], "cook_after_hour": 18, "dietary_notes": "",
                            "ingredients_on_hand": [], "cuisine_preferences": ["japanese"],
                            "calorie_target_per_meal": None},
            "expected": {"min_meals": 1, "required_tags": ["vegan"]},
        },
        {
            "description": "Maximum meals",
            "constraints": {"num_meals": 7, "max_minutes": 90, "tags": [], "allergens": [],
                            "cook_after_hour": 12, "dietary_notes": "Full week planning",
                            "ingredients_on_hand": [], "cuisine_preferences": [], "calorie_target_per_meal": None},
            "expected": {"min_meals": 7, "max_minutes": 90},
        },
        {
            "description": "Ingredient-only query",
            "constraints": {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": [],
                            "cook_after_hour": 18, "dietary_notes": "",
                            "ingredients_on_hand": ["chicken", "rice", "broccoli", "garlic", "soy sauce"],
                            "cuisine_preferences": [], "calorie_target_per_meal": None},
            "expected": {"min_meals": 3, "preferred_ingredients": ["chicken", "rice", "broccoli"]},
        },
    ]

    for ec in edge_cases:
        ec["complexity"] = "edge_case"
        ec["nl_input"] = ec.get("description", "")
        scenarios.append(ec)

    return scenarios


def _constraints_to_nl(constraints: dict, rng: random.Random) -> str:
    """Convert structured constraints to a natural language request."""
    parts = []

    if constraints.get("ingredients_on_hand"):
        ings = ", ".join(constraints["ingredients_on_hand"])
        parts.append(rng.choice([
            f"I have {ings} at home",
            f"Using {ings}",
            f"I need to use up my {ings}",
        ]))

    n = constraints["num_meals"]
    parts.append(rng.choice([
        f"plan {n} meals",
        f"I need {n} meals",
        f"make me {n} dinners",
    ]))

    if constraints.get("tags"):
        tags = " and ".join(constraints["tags"])
        parts.append(rng.choice([
            f"that are {tags}",
            f"{tags} only",
        ]))

    if constraints.get("cuisine_preferences"):
        cuisines = " or ".join(constraints["cuisine_preferences"])
        parts.append(rng.choice([
            f"preferably {cuisines} style",
            f"I'm in the mood for {cuisines}",
        ]))

    t = constraints["max_minutes"]
    if t <= 30:
        parts.append(rng.choice([f"under {t} minutes", f"quick, {t} min max"]))
    elif t <= 60:
        parts.append(f"under {t} minutes each")

    if constraints.get("allergens"):
        allergens = ", ".join(constraints["allergens"])
        parts.append(rng.choice([
            f"no {allergens}",
            f"avoid {allergens}",
            f"I'm allergic to {allergens}",
        ]))

    if constraints.get("calorie_target_per_meal"):
        cal = constraints["calorie_target_per_meal"]
        parts.append(f"about {cal} calories each")

    return ". ".join(parts) + "."


def generate_all_scenarios(
    recipes_path: Path = PROCESSED_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = 42,
) -> list[dict]:
    """Generate all test scenarios and save to JSON."""
    recipes = _load_recipes(recipes_path)
    vocab = _extract_vocabulary(recipes)

    print(f"Vocabulary: {len(vocab['tags'])} tags, {len(vocab['ingredients'])} ingredients")

    simple = generate_simple_scenarios(vocab, seed=seed)
    medium = generate_medium_scenarios(vocab, seed=seed)
    complex_s = generate_complex_scenarios(vocab, seed=seed)
    edge = generate_edge_case_scenarios(seed=seed)

    all_scenarios = simple + medium + complex_s + edge

    for i, s in enumerate(all_scenarios, 1):
        s["id"] = i

    output = {
        "metadata": {
            "total": len(all_scenarios),
            "by_complexity": {
                "simple": len(simple),
                "medium": len(medium),
                "complex": len(complex_s),
                "edge_case": len(edge),
            },
            "seed": seed,
        },
        "scenarios": all_scenarios,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Generated {len(all_scenarios)} scenarios -> {output_path}")

    return all_scenarios


if __name__ == "__main__":
    generate_all_scenarios()
