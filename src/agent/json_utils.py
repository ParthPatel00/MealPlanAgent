"""
Robust JSON extraction for LLM outputs.

Small models often produce extra text, multiple JSON objects, or markdown
fences around the JSON. This module extracts the first valid JSON object.
"""

from __future__ import annotations

import json
import re


def extract_first_json(text: str) -> dict:
    """
    Extract the first complete JSON object from an LLM response string.

    Handles:
    - Markdown code fences (```json ... ```)
    - Extra text before/after the JSON
    - Multiple JSON objects in the output (takes the first)

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try parsing the whole thing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first '{' and scan for the matching closing '}'
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output:\n{text[:500]}")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Found JSON-like block but it failed to parse: {e}\n{candidate[:500]}"
                    ) from e

    raise ValueError(f"Unbalanced braces — could not extract JSON from:\n{text[:500]}")
