"""
User feedback collection and preference extraction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.memory.store import _get_conn, set_preference


def record_feedback(
    user_id: str,
    session_id: str,
    liked_recipes: list[str] | None = None,
    disliked_recipes: list[str] | None = None,
    notes: str = "",
) -> None:
    liked_recipes = liked_recipes or []
    disliked_recipes = disliked_recipes or []

    feedback_data = json.dumps({
        "liked": liked_recipes,
        "disliked": disliked_recipes,
        "notes": notes,
    })

    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE meal_history SET feedback = ? WHERE user_id = ? AND session_id = ?",
            (feedback_data, user_id, session_id),
        )
        conn.commit()
    finally:
        conn.close()

    if liked_recipes:
        set_preference(user_id, "liked_recipes", json.dumps(liked_recipes))
    if disliked_recipes:
        set_preference(user_id, "disliked_recipes", json.dumps(disliked_recipes))
    if notes:
        set_preference(user_id, "user_notes", notes)
