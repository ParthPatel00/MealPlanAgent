"""
Planner stage.

Given user constraints, the Planner calls the LLM to produce a structured
JSON plan describing which meals to prepare, a weekly schedule, and the
ordered tool-call steps the Executor should follow.
"""

from __future__ import annotations

import json

from src.agent.json_utils import extract_first_json
from src.models.client import LLMClient

SYSTEM_PROMPT = """You are a meal planning assistant. Given user constraints,
produce a structured JSON meal plan. Respond ONLY with valid JSON — no prose,
no markdown fences.

The JSON must follow this schema:
{
  "meal_queries": [
    {"query": "<search query for RAG>", "day": "<day>", "cook_hour": <int>, "max_minutes": <int>}
  ],
  "allergens": ["<allergen>", ...],
  "steps": [
    "<step description>",
    ...
  ],
  "notes": "<any additional notes>"
}

Rules:
- meal_queries length must equal the requested number of meals
- cook_hour is 24h format (e.g. 18 for 6 pm)
- max_minutes is per-meal cooking time limit
- steps should describe what the executor will do (search, check allergens, build grocery list, generate calendar)
"""


def run_planner(constraints: dict, client: LLMClient) -> dict:
    """
    Call the LLM to produce a structured execution plan.

    Args:
        constraints: Dict with keys like num_meals, max_minutes, tags, allergens,
                     cook_after_hour, dietary_notes.
        client: LLMClient instance.

    Returns:
        Parsed plan dict matching the schema above.

    Raises:
        ValueError: If the LLM returns unparseable JSON.
    """
    user_msg = (
        f"User constraints:\n{json.dumps(constraints, indent=2)}\n\n"
        "Generate a meal plan JSON following the schema."
    )

    response = client.chat(prompt=user_msg, system=SYSTEM_PROMPT, temperature=0.1)

    try:
        plan = extract_first_json(response.text)
    except ValueError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}") from e

    # Ensure required keys exist
    plan.setdefault("meal_queries", [])
    plan.setdefault("allergens", constraints.get("allergens", []))
    plan.setdefault("steps", [])
    plan.setdefault("notes", "")

    return plan
