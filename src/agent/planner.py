"""
Planner stage.

Given user constraints, the Planner calls the LLM to produce a structured
JSON plan describing which meals to prepare, a weekly schedule, and the
ordered tool-call steps the Executor should follow.

Includes few-shot examples for output format consistency and optional
memory context from past sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agent.few_shot_examples import format_few_shot_prompt
from src.agent.json_utils import extract_first_json
from src.models.client import LLMClient

SYSTEM_PROMPT = """You are a meal planning assistant. Given user constraints,
produce a structured JSON meal plan. Respond ONLY with valid JSON, no prose,
no markdown fences.

The JSON must follow this schema:
{
  "meal_queries": [
    {
      "query": "<search query for RAG>",
      "day": "<day>",
      "cook_hour": <int>,
      "max_minutes": <int>,
      "preferred_ingredients": ["<ingredient to prioritize>", ...],
      "preferred_tags": ["<tag to prioritize>", ...]
    }
  ],
  "allergens": ["<allergen>", ...],
  "steps": ["<step description>", ...],
  "notes": "<any additional notes>"
}

Rules:
- meal_queries length must equal the requested number of meals
- cook_hour is 24h format (e.g. 18 for 6 pm)
- max_minutes is per-meal cooking time limit
- If ingredients_on_hand is provided, incorporate those ingredients into the query string and list them in preferred_ingredients so recipes using them are prioritized
- If cuisine_preferences is provided, incorporate them into query strings and list them in preferred_tags
- If calorie_target_per_meal is provided, mention calorie preference in the query (e.g. "low calorie" or "high calorie")
- steps should describe what the executor will do (search, check allergens, build grocery list, estimate budget)
- If memory context is provided, use it to avoid recently served recipes and align with user preferences
"""


@dataclass
class PlannerTrace:
    system_prompt: str
    user_prompt: str
    raw_response: str
    plan: dict
    model: str
    latency_ms: float


def run_planner(
    constraints: dict,
    client: LLMClient,
    memory_context: str = "",
) -> dict:
    """
    Call the LLM to produce a structured execution plan.

    Returns:
        Parsed plan dict matching the schema above.
        Also stores trace on the returned dict as plan["_trace"].
    """
    few_shot = format_few_shot_prompt()

    parts = [few_shot, f"User constraints:\n{json.dumps(constraints, indent=2)}"]
    if memory_context:
        parts.append(f"\nUser memory (from past sessions):\n{memory_context}")
    parts.append("\nGenerate a meal plan JSON following the schema.")
    user_msg = "\n".join(parts)

    response = client.chat(prompt=user_msg, system=SYSTEM_PROMPT, temperature=0.1)

    try:
        plan = extract_first_json(response.text)
    except ValueError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}") from e

    plan.setdefault("meal_queries", [])
    plan.setdefault("allergens", constraints.get("allergens", []))
    plan.setdefault("steps", [])
    plan.setdefault("notes", "")

    plan["_trace"] = PlannerTrace(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_msg,
        raw_response=response.text,
        plan=plan,
        model=response.model,
        latency_ms=response.latency_ms,
    )

    return plan
