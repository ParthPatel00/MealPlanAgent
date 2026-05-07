"""
LLM-based memory summarization.

Reads a user's recent meal history and preferences, then calls the LLM
to produce a natural-language summary that the Planner can use as context.
"""

from __future__ import annotations

from src.memory.store import (
    get_memory_summary,
    get_session_count,
    retrieve_past_plans,
    retrieve_preferences,
    save_memory_summary,
)
from src.models.client import LLMClient

SUMMARIZE_PROMPT = """You are a meal planning memory assistant.
Given a user's past meal plans and stated preferences, write a concise summary
(3-5 sentences) of their patterns. Focus on:
- Preferred cuisines and ingredients
- Typical cooking time tolerance
- Recurring allergens or dislikes
- Any explicit feedback

Respond with ONLY the summary text, no JSON or formatting."""


def summarize_user_history(user_id: str, client: LLMClient) -> str:
    past_plans = retrieve_past_plans(user_id, limit=10)
    preferences = retrieve_preferences(user_id)

    if not past_plans and not preferences:
        return ""

    context_parts = []
    if preferences:
        context_parts.append(f"Stated preferences: {preferences}")
    for plan in past_plans:
        entry = f"Session {plan['session_id']}: recipes={plan['recipes']}, constraints={plan['constraints']}"
        if plan.get("feedback"):
            entry += f", feedback={plan['feedback']}"
        context_parts.append(entry)

    user_msg = "User history:\n" + "\n".join(context_parts)
    response = client.chat(prompt=user_msg, system=SUMMARIZE_PROMPT, temperature=0.3)
    summary = response.text.strip()

    source_sessions = [p["session_id"] for p in past_plans]
    save_memory_summary(user_id, summary, source_sessions)

    return summary


def build_memory_context(user_id: str, client: LLMClient | None = None) -> str:
    """
    Build a memory context string for the planner.

    Returns the latest summary plus recent recipe names to avoid repetition.
    Triggers re-summarization every 3 sessions if a client is provided.
    """
    session_count = get_session_count(user_id)
    summary = get_memory_summary(user_id)

    if client and session_count > 0 and session_count % 3 == 0:
        summary = summarize_user_history(user_id, client)

    past_plans = retrieve_past_plans(user_id, limit=3)
    preferences = retrieve_preferences(user_id)

    parts = []
    if summary:
        parts.append(f"User profile: {summary}")
    if preferences:
        pref_str = ", ".join(f"{k}: {v}" for k, v in preferences.items())
        parts.append(f"Preferences: {pref_str}")
    if past_plans:
        recent_recipes = []
        for plan in past_plans:
            recent_recipes.extend(plan["recipes"])
        if recent_recipes:
            parts.append(f"Recently served (avoid repeats): {', '.join(recent_recipes[:15])}")

    return "\n".join(parts) if parts else ""
