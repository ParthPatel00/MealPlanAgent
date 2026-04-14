"""
Executor stage.

Iterates the plan steps produced by the Planner, dispatches tool calls,
and collects results. All tool calls are logged via StructuredLogger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.logging_utils import StructuredLogger
from src.tools.allergy_checker import allergy_checker
from src.tools.grocery_list import build_grocery_list
from src.tools.ics_generator import generate_ics
from src.tools.nutrition import summarize_plan_nutrition, summarize_recipe_nutrition
from src.tools.recipe_search import recipe_search


@dataclass
class ExecutorResult:
    recipes: list[dict] = field(default_factory=list)
    allergy_reports: list[dict] = field(default_factory=list)
    grocery_list: dict[str, list[str]] = field(default_factory=dict)
    nutrition_summary: dict = field(default_factory=dict)
    ics_bytes: bytes = b""
    cooking_blocks: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


def run_executor(plan: dict, constraints: dict, logger: StructuredLogger) -> ExecutorResult:
    """
    Execute all tool calls described in the plan.

    Args:
        plan: Structured plan dict from Planner.
        constraints: Original user constraints (fallback values).
        logger: StructuredLogger for this session.

    Returns:
        ExecutorResult with all collected data.
    """
    result = ExecutorResult()
    allergens = plan.get("allergens", [])
    max_minutes = constraints.get("max_minutes", 60)
    selected_recipe_ids: set[int] = set()  # prevent duplicate meals

    _WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def _normalize_day(raw_day, slot_index: int) -> str:
        """Convert whatever the planner gave us into a proper weekday name."""
        if isinstance(raw_day, str) and raw_day.strip().capitalize() in _WEEKDAYS:
            return raw_day.strip().capitalize()
        # If it's an integer or a numeric string, treat it as a slot index
        try:
            idx = int(raw_day) - 1  # planner may use 1-based
            return _WEEKDAYS[idx % 7]
        except (TypeError, ValueError):
            pass
        # Fall back to position in the plan
        return _WEEKDAYS[slot_index % 7]

    # ------------------------------------------------------------------
    # Step 1: Recipe search for each meal slot
    # Pick the first candidate that passes a local allergy pre-check.
    # Fetch top_k=5 so we have fallbacks if the best hit is unsafe.
    # ------------------------------------------------------------------
    for slot_idx, meal_req in enumerate(plan.get("meal_queries", [])):
        t0 = time.time()
        try:
            hits = recipe_search(
                query=meal_req.get("query", ""),
                max_minutes=meal_req.get("max_minutes", max_minutes),
                forbidden_ingredients=allergens,
                top_k=5,
            )

            # Pick first candidate that is locally allergy-safe and not a duplicate
            best = None
            for candidate in hits:
                cid = candidate.get("citation", {}).get("recipe_id", -1)
                if cid in selected_recipe_ids:
                    continue
                pre_check = allergy_checker(
                    ingredients=candidate.get("ingredients", []),
                    allergens=allergens,
                    use_api=False,  # fast local check only at selection time
                )
                if pre_check["safe"]:
                    best = candidate
                    break
            # Fall back to top hit if all candidates failed local check
            if best is None and hits:
                best = hits[0]

            if best:
                rid = best.get("citation", {}).get("recipe_id", -1)
                selected_recipe_ids.add(rid)

            latency = (time.time() - t0) * 1000

            logger.log_tool_call(
                tool_name="recipe_search",
                inputs=meal_req,
                output=best,
                latency_ms=latency,
                success=best is not None,
            )

            if best:
                best["_day"] = _normalize_day(meal_req.get("day", ""), slot_idx)
                raw_hour = meal_req.get("cook_hour", constraints.get("cook_after_hour", 18))
                try:
                    best["_cook_hour"] = max(0, min(23, int(raw_hour)))
                except (TypeError, ValueError):
                    best["_cook_hour"] = constraints.get("cook_after_hour", 18)
                result.recipes.append(best)
                result.tool_calls.append(
                    {"tool": "recipe_search", "query": meal_req.get("query"), "result": best["name"]}
                )
        except Exception as exc:
            logger.log_error("executor.recipe_search", str(exc))

    # ------------------------------------------------------------------
    # Step 1b: Deduplication safety net (runs before fill phase so fill
    # can top up to the correct count after removing any duplicates).
    # ------------------------------------------------------------------
    def _dedup(recipes: list[dict]) -> list[dict]:
        seen_keys: set = set()
        out: list[dict] = []
        for r in recipes:
            rid = r.get("citation", {}).get("recipe_id", None)
            key = rid if (rid is not None and rid != -1) else r.get("name", "")
            if key not in seen_keys:
                seen_keys.add(key)
                out.append(r)
        return out

    result.recipes = _dedup(result.recipes)
    # Sync selected_recipe_ids with the post-dedup set
    selected_recipe_ids = {
        r.get("citation", {}).get("recipe_id", -1)
        for r in result.recipes
    }

    # ------------------------------------------------------------------
    # Step 1c: Fill phase — if fewer recipes than requested, do broad
    # fallback searches to reach the target count.
    # ------------------------------------------------------------------
    num_meals = constraints.get("num_meals", len(plan.get("meal_queries", [])))
    fallback_queries = [
        "quick easy dinner",
        "simple healthy meal",
        "easy weeknight meal",
        "family dinner recipe",
        "simple lunch recipe",
        "easy breakfast recipe",
        "vegetable side dish",
        "30 minute dinner",
        "easy chicken recipe",
        "simple pasta dish",
    ]
    fq_idx = 0
    while len(result.recipes) < num_meals and fq_idx < len(fallback_queries):
        try:
            hits = recipe_search(
                query=fallback_queries[fq_idx],
                max_minutes=max_minutes,
                forbidden_ingredients=allergens,
                top_k=10,
            )
            for candidate in hits:
                cid = candidate.get("citation", {}).get("recipe_id", -1)
                if cid in selected_recipe_ids:
                    continue
                name_key = candidate.get("name", "")
                if name_key in {r.get("name", "") for r in result.recipes}:
                    continue
                pre_check = allergy_checker(
                    ingredients=candidate.get("ingredients", []),
                    allergens=allergens,
                    use_api=False,
                )
                if pre_check["safe"]:
                    slot = len(result.recipes)
                    candidate["_day"] = _WEEKDAYS[slot % 7]
                    candidate["_cook_hour"] = constraints.get("cook_after_hour", 18)
                    selected_recipe_ids.add(cid)
                    result.recipes.append(candidate)
                    result.tool_calls.append(
                        {"tool": "recipe_search_fill", "query": fallback_queries[fq_idx], "result": candidate["name"]}
                    )
                    break
        except Exception:
            pass
        fq_idx += 1

    # ------------------------------------------------------------------
    # Step 2: Allergy check for each selected recipe
    # ------------------------------------------------------------------
    for recipe in result.recipes:
        t0 = time.time()
        try:
            report = allergy_checker(
                ingredients=recipe.get("ingredients", []),
                allergens=allergens,
                use_api=False,  # local check is deterministic; API adds noise from cross-contamination labels
            )
            report["recipe_name"] = recipe["name"]
            latency = (time.time() - t0) * 1000

            logger.log_tool_call(
                tool_name="allergy_checker",
                inputs={"recipe": recipe["name"], "allergens": allergens},
                output=report,
                latency_ms=latency,
                success=True,
            )

            result.allergy_reports.append(report)
            result.tool_calls.append(
                {"tool": "allergy_checker", "recipe": recipe["name"], "safe": report["safe"]}
            )
        except Exception as exc:
            logger.log_error("executor.allergy_checker", str(exc))

    # ------------------------------------------------------------------
    # Step 3: Nutrition per recipe + plan-wide summary
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        for recipe in result.recipes:
            recipe["_nutrition_abs"] = summarize_recipe_nutrition(recipe)
        result.nutrition_summary = summarize_plan_nutrition(result.recipes)
        latency = (time.time() - t0) * 1000

        logger.log_tool_call(
            tool_name="nutrition",
            inputs={"num_recipes": len(result.recipes)},
            output=result.nutrition_summary,
            latency_ms=latency,
            success=True,
        )
        result.tool_calls.append({"tool": "nutrition", "summary": result.nutrition_summary})
    except Exception as exc:
        logger.log_error("executor.nutrition", str(exc))

    # ------------------------------------------------------------------
    # Step 4: Grocery list
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        result.grocery_list = build_grocery_list(result.recipes)
        latency = (time.time() - t0) * 1000

        logger.log_tool_call(
            tool_name="grocery_list",
            inputs={"num_recipes": len(result.recipes)},
            output={k: len(v) for k, v in result.grocery_list.items()},
            latency_ms=latency,
            success=True,
        )
        result.tool_calls.append(
            {"tool": "grocery_list", "categories": list(result.grocery_list.keys())}
        )
    except Exception as exc:
        logger.log_error("executor.grocery_list", str(exc))

    # ------------------------------------------------------------------
    # Step 5: ICS calendar file
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        result.cooking_blocks = [
            {
                "meal_name": r["name"],
                "day": r.get("_day", "Monday"),
                "cook_hour": r.get("_cook_hour", 18),
                "duration_minutes": r.get("minutes", 30),
            }
            for r in result.recipes
        ]
        result.ics_bytes = generate_ics(result.cooking_blocks)
        latency = (time.time() - t0) * 1000

        logger.log_tool_call(
            tool_name="ics_generator",
            inputs={"num_blocks": len(result.cooking_blocks)},
            output={"ics_size_bytes": len(result.ics_bytes)},
            latency_ms=latency,
            success=True,
        )
        result.tool_calls.append(
            {"tool": "ics_generator", "blocks": len(result.cooking_blocks)}
        )
    except Exception as exc:
        logger.log_error("executor.ics_generator", str(exc))

    return result
