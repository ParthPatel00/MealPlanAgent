"""
Generate ground truth relevance judgments for retrieval evaluation.

Three strategies for creating query-relevance pairs from the dataset:
1. Tag-based: queries derived from tag combinations, relevant = recipes with those tags
2. Ingredient-based: queries from ingredient sets, relevant = recipes containing those ingredients
3. Knowledge-graph-based: queries using KG neighborhood structure for graded relevance

All judgments are deterministic and derived from the dataset itself (no LLM calls),
making them reproducible and free to generate at scale.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

PROCESSED_PATH = Path("data/processed/recipes_clean.json")
OUTPUT_PATH = Path("data/eval/retrieval_ground_truth.json")


def _load_recipes(path: Path = PROCESSED_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _build_tag_index(recipes: list[dict]) -> dict[str, set[int]]:
    """Map each tag to the set of recipe IDs that have it."""
    tag_to_ids: dict[str, set[int]] = {}
    for r in recipes:
        for tag in r.get("tags", []):
            tag_norm = tag.strip().lower()
            tag_to_ids.setdefault(tag_norm, set()).add(r["id"])
    return tag_to_ids


def _build_ingredient_index(recipes: list[dict]) -> dict[str, set[int]]:
    """Map each ingredient to the set of recipe IDs that contain it."""
    ing_to_ids: dict[str, set[int]] = {}
    for r in recipes:
        for ing in r.get("ingredients", []):
            ing_norm = ing.strip().lower()
            ing_to_ids.setdefault(ing_norm, set()).add(r["id"])
    return ing_to_ids


NOISY_TAGS = {
    "time-to-make", "course", "main-ingredient", "preparation", "occasion",
    "north-american", "cuisine", "for-1-or-2", "main-dish", "dietary",
    "number-of-servings", "3-steps-or-less", "4-hours-or-less",
    "1-day-or-more", "for-large-groups", "equipment", "technique",
}

CUISINE_TAGS = {
    "mexican", "italian", "chinese", "japanese", "thai", "indian", "greek",
    "french", "korean", "vietnamese", "mediterranean", "middle-eastern",
    "african", "caribbean", "spanish", "german", "british", "brazilian",
    "turkish", "moroccan", "ethiopian", "peruvian", "cuban", "cajun-creole",
    "southern-united-states", "southwestern-united-states",
}

DIETARY_TAGS = {
    "vegetarian", "vegan", "low-carb", "low-fat", "low-sodium", "low-calorie",
    "high-protein", "gluten-free", "dairy-free", "sugar-free", "kosher",
    "paleo", "keto", "whole30",
}

COOKING_TAGS = {
    "grilling", "baking", "slow-cooker", "one-pot", "sheet-pan", "stir-fry",
    "deep-fried", "roasting", "pressure-cooker", "no-cook", "raw",
    "broiling", "steaming", "poaching", "braising", "smoking",
}

MEAL_TYPE_TAGS = {
    "breakfast", "lunch", "dinner", "dessert", "snack", "appetizer",
    "side-dish", "soup", "salad", "bread", "beverage",
}


def generate_tag_queries(
    recipes: list[dict],
    num_single: int = 80,
    num_pair: int = 70,
    min_relevant: int = 5,
    max_relevant: int = 500,
    seed: int = 42,
) -> list[dict]:
    """
    Generate queries from tag combinations with known relevant recipe sets.

    Uses meaningful tags (cuisine, dietary, cooking method, meal type) and
    filters out noisy structural tags.
    """
    rng = random.Random(seed)
    tag_index = _build_tag_index(recipes)

    meaningful_tags = {}
    for tag, ids in tag_index.items():
        if tag in NOISY_TAGS:
            continue
        if min_relevant <= len(ids) <= max_relevant:
            meaningful_tags[tag] = ids

    queries = []

    # Single-tag queries
    single_candidates = list(meaningful_tags.items())
    rng.shuffle(single_candidates)
    for tag, rel_ids in single_candidates[:num_single]:
        category = _categorize_tag(tag)
        query_text = _tag_to_query(tag, category)
        queries.append({
            "query": query_text,
            "query_type": "tag_single",
            "tags_used": [tag],
            "relevant_ids": sorted(rel_ids),
            "num_relevant": len(rel_ids),
        })

    # Paired-tag queries (e.g., "vegetarian mexican")
    tag_list = list(meaningful_tags.keys())
    pair_candidates = []
    for t1, t2 in combinations(tag_list, 2):
        intersection = meaningful_tags[t1] & meaningful_tags[t2]
        if min_relevant <= len(intersection) <= max_relevant:
            pair_candidates.append((t1, t2, intersection))

    rng.shuffle(pair_candidates)
    for t1, t2, rel_ids in pair_candidates[:num_pair]:
        query_text = f"{t1} {t2} recipes"
        queries.append({
            "query": query_text,
            "query_type": "tag_pair",
            "tags_used": [t1, t2],
            "relevant_ids": sorted(rel_ids),
            "num_relevant": len(rel_ids),
        })

    return queries


def generate_ingredient_queries(
    recipes: list[dict],
    num_single: int = 50,
    num_multi: int = 50,
    min_relevant: int = 3,
    max_relevant: int = 300,
    seed: int = 42,
) -> list[dict]:
    """
    Generate queries from ingredient combinations with known relevant recipe sets.

    Focuses on common cooking ingredients and their combinations.
    """
    rng = random.Random(seed)
    ing_index = _build_ingredient_index(recipes)

    viable_ings = {
        ing: ids for ing, ids in ing_index.items()
        if min_relevant <= len(ids) <= max_relevant and len(ing) > 2
    }

    queries = []

    # Single-ingredient queries
    single_candidates = list(viable_ings.items())
    rng.shuffle(single_candidates)
    for ing, rel_ids in single_candidates[:num_single]:
        queries.append({
            "query": f"recipes with {ing}",
            "query_type": "ingredient_single",
            "ingredients_used": [ing],
            "relevant_ids": sorted(rel_ids),
            "num_relevant": len(rel_ids),
        })

    # Multi-ingredient queries (2-3 ingredients)
    ing_list = list(viable_ings.keys())
    multi_candidates = []
    for _ in range(num_multi * 20):
        n = rng.choice([2, 3])
        chosen = rng.sample(ing_list, min(n, len(ing_list)))
        intersection = viable_ings[chosen[0]]
        for ing in chosen[1:]:
            intersection = intersection & viable_ings[ing]
        if min_relevant <= len(intersection):
            multi_candidates.append((chosen, intersection))
        if len(multi_candidates) >= num_multi * 3:
            break

    rng.shuffle(multi_candidates)
    for ings, rel_ids in multi_candidates[:num_multi]:
        query_text = f"recipes with {', '.join(ings)}"
        queries.append({
            "query": query_text,
            "query_type": "ingredient_multi",
            "ingredients_used": ings,
            "relevant_ids": sorted(rel_ids),
            "num_relevant": len(rel_ids),
        })

    return queries


def generate_kg_graded_queries(
    recipes: list[dict],
    num_queries: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Generate queries with graded relevance using knowledge graph structure.

    For a focal recipe, assigns graded relevance based on ingredient overlap:
    - 3 (highly relevant): shares >= 60% of ingredients
    - 2 (relevant): shares 30-59% of ingredients
    - 1 (marginally relevant): shares 10-29% of ingredients
    - 0 (irrelevant): shares < 10% of ingredients

    This provides graded judgments needed for NDCG computation.
    """
    rng = random.Random(seed)

    recipe_map = {r["id"]: r for r in recipes}
    ing_index = _build_ingredient_index(recipes)

    viable = [r for r in recipes if 3 <= len(r.get("ingredients", [])) <= 20]
    rng.shuffle(viable)

    queries = []
    for focal in viable[:num_queries]:
        focal_ings = {ing.strip().lower() for ing in focal.get("ingredients", [])}
        if not focal_ings:
            continue

        neighbor_scores: Counter = Counter()
        for ing in focal_ings:
            for rid in ing_index.get(ing, set()):
                if rid != focal["id"]:
                    neighbor_scores[rid] += 1

        graded: dict[int, int] = {}
        for rid, shared_count in neighbor_scores.most_common(200):
            other = recipe_map.get(rid)
            if not other:
                continue
            other_ings = {i.strip().lower() for i in other.get("ingredients", [])}
            overlap = shared_count / max(len(focal_ings), 1)
            if overlap >= 0.6:
                graded[rid] = 3
            elif overlap >= 0.3:
                graded[rid] = 2
            elif overlap >= 0.1:
                graded[rid] = 1

        if len([v for v in graded.values() if v >= 2]) < 3:
            continue

        top_ings = sorted(focal_ings)[:4]
        query_text = f"{focal['name']} with {', '.join(top_ings)}"

        queries.append({
            "query": query_text,
            "query_type": "kg_graded",
            "focal_recipe_id": focal["id"],
            "focal_recipe_name": focal["name"],
            "ingredients_used": list(focal_ings),
            "relevant_ids": sorted(graded.keys()),
            "graded_relevance": {str(k): v for k, v in graded.items()},
            "num_relevant": len(graded),
            "grade_distribution": {
                "highly_relevant_3": sum(1 for v in graded.values() if v == 3),
                "relevant_2": sum(1 for v in graded.values() if v == 2),
                "marginal_1": sum(1 for v in graded.values() if v == 1),
            },
        })

    return queries


def generate_natural_language_queries(
    recipes: list[dict],
    num_queries: int = 30,
    seed: int = 42,
) -> list[dict]:
    """
    Generate natural language queries that mimic real user requests.

    Combines ingredients, dietary preferences, and time constraints into
    realistic query strings with known relevant recipe sets.
    """
    rng = random.Random(seed)
    tag_index = _build_tag_index(recipes)
    ing_index = _build_ingredient_index(recipes)

    templates = [
        "I have {ingredients} and want to make a {cuisine} dish",
        "Quick {diet} {meal_type} under {time} minutes",
        "{diet} recipes using {ingredients}",
        "Easy {cuisine} {meal_type} with {ingredients}",
        "Healthy {diet} dinner ideas with {ingredients}",
        "{cooking_method} {protein} recipes",
        "Simple {meal_type} for weeknight cooking with {ingredients}",
        "{cuisine} style {diet} {meal_type}",
    ]

    common_proteins = ["chicken", "beef", "salmon", "tofu", "shrimp", "pork", "turkey"]
    time_limits = [15, 20, 30, 45, 60]

    queries = []
    for _ in range(num_queries * 5):
        if len(queries) >= num_queries:
            break

        template = rng.choice(templates)
        replacements = {}

        cuisine = rng.choice(list(CUISINE_TAGS & set(tag_index.keys())) or ["italian"])
        diet = rng.choice(list(DIETARY_TAGS & set(tag_index.keys())) or ["vegetarian"])
        meal_type = rng.choice(list(MEAL_TYPE_TAGS & set(tag_index.keys())) or ["dinner"])
        cooking = rng.choice(list(COOKING_TAGS & set(tag_index.keys())) or ["baking"])
        protein = rng.choice([p for p in common_proteins if p in ing_index] or ["chicken"])
        time_limit = rng.choice(time_limits)

        usable_ings = [p for p in common_proteins if p in ing_index]
        chosen_ings = rng.sample(usable_ings, min(rng.choice([1, 2]), len(usable_ings)))

        replacements = {
            "cuisine": cuisine,
            "diet": diet,
            "meal_type": meal_type,
            "cooking_method": cooking,
            "protein": protein,
            "time": str(time_limit),
            "ingredients": ", ".join(chosen_ings),
        }

        try:
            query_text = template.format(**replacements)
        except KeyError:
            continue

        all_recipe_ids = {r["id"] for r in recipes}
        constraint_sets = []
        used_constraints = {}

        if "cuisine" in template and cuisine in tag_index:
            constraint_sets.append(tag_index[cuisine])
            used_constraints["cuisine"] = cuisine
        if "diet" in template and diet in tag_index:
            constraint_sets.append(tag_index[diet])
            used_constraints["diet"] = diet
        if "meal_type" in template and meal_type in tag_index:
            constraint_sets.append(tag_index[meal_type])
            used_constraints["meal_type"] = meal_type
        if "cooking_method" in template and cooking in tag_index:
            constraint_sets.append(tag_index[cooking])
            used_constraints["cooking_method"] = cooking

        ing_relevant = set()
        for ing in chosen_ings:
            if ing in ing_index:
                ing_relevant |= ing_index[ing]
        if "ingredients" in template and ing_relevant:
            constraint_sets.append(ing_relevant)
            used_constraints["ingredients"] = chosen_ings
        if "protein" in template and protein in ing_index:
            constraint_sets.append(ing_index[protein])
            used_constraints["protein"] = protein

        if not constraint_sets:
            continue
        relevant = constraint_sets[0]
        for cs in constraint_sets[1:]:
            relevant = relevant & cs

        if 3 <= len(relevant) <= 500:
            queries.append({
                "query": query_text,
                "query_type": "natural_language",
                "constraints_used": used_constraints,
                "relevant_ids": sorted(relevant),
                "num_relevant": len(relevant),
            })

    return queries


def _categorize_tag(tag: str) -> str:
    if tag in CUISINE_TAGS:
        return "cuisine"
    if tag in DIETARY_TAGS:
        return "dietary"
    if tag in COOKING_TAGS:
        return "cooking"
    if tag in MEAL_TYPE_TAGS:
        return "meal_type"
    return "general"


def _tag_to_query(tag: str, category: str) -> str:
    if category == "cuisine":
        return f"{tag} recipes"
    if category == "dietary":
        return f"{tag} recipes"
    if category == "cooking":
        return f"{tag} recipes"
    if category == "meal_type":
        return f"{tag} recipes"
    return f"{tag} recipes"


def generate_all_ground_truth(
    recipes_path: Path = PROCESSED_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = 42,
) -> dict:
    """Generate all ground truth queries and save to JSON."""
    recipes = _load_recipes(recipes_path)
    print(f"Loaded {len(recipes)} recipes")

    print("Generating tag-based queries...")
    tag_queries = generate_tag_queries(recipes, seed=seed)
    print(f"  {len(tag_queries)} tag queries")

    print("Generating ingredient-based queries...")
    ingredient_queries = generate_ingredient_queries(recipes, seed=seed)
    print(f"  {len(ingredient_queries)} ingredient queries")

    print("Generating KG-graded queries...")
    kg_queries = generate_kg_graded_queries(recipes, seed=seed)
    print(f"  {len(kg_queries)} KG-graded queries")

    print("Generating natural language queries...")
    nl_queries = generate_natural_language_queries(recipes, seed=seed)
    print(f"  {len(nl_queries)} natural language queries")

    ground_truth = {
        "metadata": {
            "num_recipes": len(recipes),
            "seed": seed,
            "query_counts": {
                "tag": len(tag_queries),
                "ingredient": len(ingredient_queries),
                "kg_graded": len(kg_queries),
                "natural_language": len(nl_queries),
                "total": len(tag_queries) + len(ingredient_queries) + len(kg_queries) + len(nl_queries),
            },
        },
        "queries": tag_queries + ingredient_queries + kg_queries + nl_queries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth to {output_path}")

    return ground_truth


if __name__ == "__main__":
    generate_all_ground_truth()
