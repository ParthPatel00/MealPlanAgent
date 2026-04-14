"""
Convert cleaned recipe dicts into LlamaIndex Document objects for indexing.
"""

import json
from pathlib import Path

from llama_index.core import Document


PROCESSED_PATH = Path("data/processed/recipes_clean.json")


def recipe_to_text(recipe: dict) -> str:
    """Flatten a recipe dict into a single indexable text string."""
    tags = ", ".join(recipe.get("tags", []))
    ingredients = ", ".join(recipe.get("ingredients", []))
    steps = " ".join(recipe.get("steps", []))
    nutrition = recipe.get("nutrition", {})
    nutrition_str = ", ".join(f"{k}: {v}" for k, v in nutrition.items())

    return (
        f"Recipe: {recipe['name']}. "
        f"Time: {recipe['minutes']} minutes. "
        f"Tags: {tags}. "
        f"Ingredients: {ingredients}. "
        f"Nutrition: {nutrition_str}. "
        f"Steps: {steps}"
    ).strip()


def recipes_to_documents(recipes: list[dict]) -> list[Document]:
    """Convert recipe dicts to LlamaIndex Documents with metadata.

    ChromaDB requires flat metadata (str/int/float/None only), so lists and
    dicts are JSON-encoded as strings. The retriever deserializes them back.
    """
    docs = []
    for r in recipes:
        text = recipe_to_text(r)
        metadata = {
            "recipe_id": r["id"],
            "name": r["name"],
            "minutes": r["minutes"],
            # Serialized as JSON strings for ChromaDB compatibility
            "tags_json": json.dumps(r.get("tags", [])),
            "ingredients_json": json.dumps(r.get("ingredients", [])),
            "nutrition_json": json.dumps(r.get("nutrition", {})),
        }
        docs.append(Document(text=text, metadata=metadata, id_=str(r["id"])))
    return docs


def load_documents(processed_path: Path = PROCESSED_PATH) -> list[Document]:
    """Load processed JSON and return Documents."""
    with open(processed_path) as f:
        recipes = json.load(f)
    return recipes_to_documents(recipes)
