import json
import sqlite3
from unittest.mock import patch

import pytest

from src.memory.store import (
    DB_PATH,
    get_memory_summary,
    get_session_count,
    retrieve_past_plans,
    retrieve_preferences,
    save_memory_summary,
    set_preference,
    write_memory,
)


@pytest.fixture(autouse=True)
def use_tmp_db(tmp_path):
    tmp_db = tmp_path / "test_memory.db"
    with patch("src.memory.store.DB_PATH", tmp_db):
        yield tmp_db


class TestWriteAndRetrieve:
    def test_write_and_retrieve_round_trip(self):
        write_memory(
            user_id="user1",
            session_id="sess1",
            recipes=[{"name": "Chicken Rice"}, {"name": "Pasta"}],
            constraints={"num_meals": 2, "tags": ["high-protein"]},
        )
        plans = retrieve_past_plans("user1")
        assert len(plans) == 1
        assert plans[0]["session_id"] == "sess1"
        assert "Chicken Rice" in plans[0]["recipes"]

    def test_multiple_sessions(self):
        for i in range(5):
            write_memory(
                user_id="user1",
                session_id=f"sess{i}",
                recipes=[{"name": f"Recipe {i}"}],
                constraints={"num_meals": 1},
            )
        plans = retrieve_past_plans("user1", limit=3)
        assert len(plans) == 3

    def test_user_isolation(self):
        write_memory("user1", "s1", [{"name": "R1"}], {})
        write_memory("user2", "s2", [{"name": "R2"}], {})
        assert len(retrieve_past_plans("user1")) == 1
        assert len(retrieve_past_plans("user2")) == 1


class TestPreferences:
    def test_set_and_get(self):
        set_preference("user1", "cuisine", "italian")
        prefs = retrieve_preferences("user1")
        assert prefs["cuisine"] == "italian"

    def test_update_existing(self):
        set_preference("user1", "cuisine", "italian")
        set_preference("user1", "cuisine", "mexican")
        prefs = retrieve_preferences("user1")
        assert prefs["cuisine"] == "mexican"

    def test_multiple_keys(self):
        set_preference("user1", "cuisine", "italian")
        set_preference("user1", "diet", "vegetarian")
        prefs = retrieve_preferences("user1")
        assert len(prefs) == 2


class TestMemorySummary:
    def test_save_and_retrieve(self):
        save_memory_summary("user1", "Prefers Italian food.", ["s1", "s2"])
        summary = get_memory_summary("user1")
        assert summary == "Prefers Italian food."

    def test_latest_summary_returned(self):
        save_memory_summary("user1", "Old summary.", ["s1"])
        save_memory_summary("user1", "New summary.", ["s2"])
        summary = get_memory_summary("user1")
        assert summary == "New summary."

    def test_no_summary(self):
        assert get_memory_summary("nonexistent") is None


class TestSessionCount:
    def test_count(self):
        assert get_session_count("user1") == 0
        write_memory("user1", "s1", [{"name": "R"}], {})
        assert get_session_count("user1") == 1
        write_memory("user1", "s2", [{"name": "R"}], {})
        assert get_session_count("user1") == 2
