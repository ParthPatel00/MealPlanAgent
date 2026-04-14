"""
Structured JSONL logger for agent sessions.

Every agent run writes log entries to logs/session_<id>.jsonl.
Each entry has a `type` field so downstream analysis can filter by stage.

Entry types:
    UserInput       — raw user constraints
    PlannerOutput   — structured plan from Planner
    ToolCall        — single tool invocation with input/output/latency
    CriticCheck     — critic verdict and issues list
    FinalOutput     — complete AgentResult summary
    Error           — unexpected exception during a stage
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIR = Path("logs")


class StructuredLogger:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        LOG_DIR.mkdir(exist_ok=True)
        self._path = LOG_DIR / f"session_{self.session_id}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_user_input(self, constraints: dict) -> None:
        self._write("UserInput", {"constraints": constraints})

    def log_planner_output(self, plan: dict) -> None:
        self._write("PlannerOutput", {"plan": plan})

    def log_tool_call(
        self,
        tool_name: str,
        inputs: dict,
        output: Any,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        self._write(
            "ToolCall",
            {
                "tool": tool_name,
                "inputs": inputs,
                "output": output,
                "latency_ms": round(latency_ms, 2),
                "success": success,
            },
        )

    def log_critic_check(self, valid: bool, issues: list[str], fix_instructions: str) -> None:
        self._write(
            "CriticCheck",
            {"valid": valid, "issues": issues, "fix_instructions": fix_instructions},
        )

    def log_final_output(self, result: dict) -> None:
        self._write("FinalOutput", {"result": result})

    def log_error(self, stage: str, error: str) -> None:
        self._write("Error", {"stage": stage, "error": error})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, entry_type: str, payload: dict) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "type": entry_type,
            **payload,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def get_log_path(self) -> str:
        return str(self._path)

    def read_log(self) -> list[dict]:
        """Return all entries for this session as a list of dicts."""
        entries = []
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
