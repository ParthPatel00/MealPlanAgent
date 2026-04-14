"""
Load and clean the Food.com RAW_recipes.csv dataset.

Download from:
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

Place the CSV at data/raw/RAW_recipes.csv then run:
    python -m src.data.loader
"""

import ast
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = Path(os.getenv("RECIPES_CSV_PATH", "data/RAW_recipes.csv"))
OUT_PATH = Path("data/processed/recipes_clean.json")
MAX_RECIPES = int(os.getenv("MAX_RECIPES", "10000"))

# Food.com nutrition columns (order matches the stored list)
NUTRITION_KEYS = [
    "calories_pdv",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
]


def _parse_list(raw: str) -> list:
    """Safely parse a Python-literal list stored as a string."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def load_recipes(csv_path: Path = RAW_PATH, max_rows: int = MAX_RECIPES) -> list[dict]:
    """Return a list of cleaned recipe dicts."""
    df = pd.read_csv(csv_path, nrows=max_rows)

    # Drop rows with no name or no ingredients
    df = df.dropna(subset=["name", "ingredients"])

    records = []
    for _, row in df.iterrows():
        nutrition_raw = _parse_list(str(row.get("nutrition", "[]")))
        nutrition = dict(zip(NUTRITION_KEYS, nutrition_raw)) if len(nutrition_raw) == 7 else {}

        recipe = {
            "id": int(row["id"]),
            "name": str(row["name"]).strip(),
            "minutes": int(row.get("minutes", 0)),
            "tags": _parse_list(str(row.get("tags", "[]"))),
            "nutrition": nutrition,
            "ingredients": _parse_list(str(row.get("ingredients", "[]"))),
            "steps": _parse_list(str(row.get("steps", "[]"))),
            "description": str(row.get("description", "")).strip(),
        }
        records.append(recipe)

    return records


def save_processed(records: list[dict], out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} recipes to {out_path}")


if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {RAW_PATH}.\n"
            "Download RAW_recipes.csv from Kaggle and place it at data/raw/RAW_recipes.csv"
        )
    recipes = load_recipes()
    save_processed(recipes)
