"""
Persistent memory store backed by SQLite.

Implements the write-summarize-retrieve pattern required by the assignment:
  - write_memory(): stores meal history and constraints after each pipeline run
  - retrieve_preferences(): returns learned user preferences
  - retrieve_past_plans(): returns recent meal plans for variety
  - get_memory_summary(): returns the latest LLM-generated summary
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/memory.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            recipes TEXT NOT NULL,
            constraints TEXT NOT NULL,
            feedback TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_sessions TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mh_user ON meal_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_up_user ON user_preferences(user_id);
        CREATE INDEX IF NOT EXISTS idx_ms_user ON memory_summaries(user_id);
    """)


def write_memory(
    user_id: str,
    session_id: str,
    recipes: list[dict],
    constraints: dict,
) -> None:
    conn = _get_conn()
    try:
        recipe_names = [r.get("name", "unknown") for r in recipes]
        conn.execute(
            "INSERT INTO meal_history (session_id, user_id, recipes, constraints, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                user_id,
                json.dumps(recipe_names),
                json.dumps(constraints),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def retrieve_past_plans(user_id: str, limit: int = 5) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT session_id, recipes, constraints, feedback, created_at "
            "FROM meal_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "recipes": json.loads(r["recipes"]),
                "constraints": json.loads(r["constraints"]),
                "feedback": r["feedback"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def retrieve_preferences(user_id: str) -> dict[str, str]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()


def set_preference(user_id: str, key: str, value: str) -> None:
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM user_preferences WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        if existing:
            conn.execute(
                "UPDATE user_preferences SET value = ?, updated_at = ? WHERE id = ?",
                (value, now, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO user_preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, key, value, now),
            )
        conn.commit()
    finally:
        conn.close()


def save_memory_summary(user_id: str, summary: str, source_sessions: list[str]) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO memory_summaries (user_id, summary, source_sessions, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                user_id,
                summary,
                json.dumps(source_sessions),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_memory_summary(user_id: str) -> str | None:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT summary FROM memory_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row["summary"] if row else None
    finally:
        conn.close()


def get_session_count(user_id: str) -> int:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM meal_history WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["cnt"]
    finally:
        conn.close()
