"""
Critic stage.

Fully rule-based checks — deterministic Python, no LLM required:
  1. At least one recipe was found
  2. No allergy violations (all reports have safe=True)
  3. Every recipe has a valid citation (recipe_id present)
  4. No duplicate recipes in the plan
  5. All cooking blocks have a valid day name and cook_hour (0-23)

Using a small local LLM for the critic introduced too many false positives.
Rule-based checks are more reliable and perfectly adequate for these constraints.
The LLM is reserved for Planner and Executor stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.executor import ExecutorResult

VALID_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


@dataclass
class CriticResult:
    valid: bool
    issues: list[str]
    fix_instructions: str


def run_critic(executor_result: ExecutorResult, client=None) -> CriticResult:  # noqa: ANN001
    """
    Run rule-based checks over the executor output.

    Args:
        executor_result: ExecutorResult from the Executor stage.
        client: Unused — kept for API compatibility with the pipeline.

    Returns:
        CriticResult. valid=True only when all checks pass.
    """
    issues: list[str] = []

    # 1. At least one recipe
    if not executor_result.recipes:
        issues.append("No recipes were returned. Try broader search terms or a higher max_minutes.")

    # 2. Allergy violations
    for report in executor_result.allergy_reports:
        if not report.get("safe", True):
            violations = ", ".join(report.get("violations", []))
            name = report.get("recipe_name", "unknown")
            issues.append(f"Allergy violation in '{name}': contains {violations}.")

    # 3. Missing citations
    for recipe in executor_result.recipes:
        citation = recipe.get("citation", {})
        if not citation or citation.get("recipe_id") is None:
            issues.append(f"Missing citation (recipe_id) for '{recipe.get('name', '?')}'.")

    # 4. Duplicate recipes
    seen_ids: set = set()
    for recipe in executor_result.recipes:
        rid = recipe.get("citation", {}).get("recipe_id")
        if rid is not None:
            if rid in seen_ids:
                issues.append(f"Duplicate recipe '{recipe.get('name', '?')}' in plan.")
            seen_ids.add(rid)

    # 5. Cooking block validity
    for block in executor_result.cooking_blocks:
        day = str(block.get("day", "")).lower()
        hour = block.get("cook_hour")
        name = block.get("meal_name", "?")
        if day not in VALID_DAYS:
            issues.append(f"Invalid day '{day}' for meal '{name}'.")
        if hour is None or not (0 <= int(hour) <= 23):
            issues.append(f"Invalid cook_hour '{hour}' for meal '{name}'.")

    if issues:
        fix = "Re-plan to address: " + "; ".join(issues)
        return CriticResult(valid=False, issues=issues, fix_instructions=fix)

    return CriticResult(valid=True, issues=[], fix_instructions="")
