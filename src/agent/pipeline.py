"""
Agent pipeline: Planner → Executor → Critic (with retry).

Orchestrates all three stages and returns a final AgentResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.critic import CriticResult, run_critic
from src.agent.executor import ExecutorResult, run_executor
from src.agent.planner import run_planner
from src.logging_utils import StructuredLogger
from src.models.client import LLMClient

MAX_RETRIES = 2


@dataclass
class AgentResult:
    # Core outputs
    recipes: list[dict] = field(default_factory=list)
    grocery_list: dict[str, list[str]] = field(default_factory=dict)
    nutrition_summary: dict = field(default_factory=dict)
    ics_bytes: bytes = b""
    cooking_blocks: list[dict] = field(default_factory=list)

    # Audit
    plan: dict = field(default_factory=dict)
    critic: CriticResult | None = None
    tool_calls: list[dict] = field(default_factory=list)
    allergy_reports: list[dict] = field(default_factory=list)
    session_id: str = ""
    log_path: str = ""
    retries: int = 0


def run_pipeline(constraints: dict, model_name: str = "gemini") -> AgentResult:
    """
    Run the full Planner-Executor-Critic pipeline.

    Args:
        constraints: User constraints dict. Expected keys:
            - num_meals (int): Number of meals to plan
            - max_minutes (int): Max cooking time per meal
            - tags (list[str]): Dietary tags, e.g. ["high-protein", "vegetarian"]
            - allergens (list[str]): Allergen list, e.g. ["peanuts", "gluten"]
            - cook_after_hour (int): Earliest cooking start hour (24h)
            - dietary_notes (str): Free-text notes
        model_name: One of "gemini", "groq-llama", "groq-mistral".

    Returns:
        AgentResult with all outputs and audit info.
    """
    client = LLMClient(model_name)

    with StructuredLogger() as logger:
        logger.log_user_input(constraints)

        # ---- Planner ----
        plan = run_planner(constraints, client)
        # Guarantee the plan has exactly num_meals queries; pad with generic if short
        num_meals = constraints.get("num_meals", 5)
        tags = constraints.get("tags", [])
        cook_hour = constraints.get("cook_after_hour", 18)
        max_min = constraints.get("max_minutes", 60)
        weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        while len(plan.get("meal_queries", [])) < num_meals:
            slot = len(plan["meal_queries"])
            tag_str = " ".join(tags) if tags else "healthy"
            plan["meal_queries"].append({
                "query": f"{tag_str} meal",
                "day": weekdays[slot % 7],
                "cook_hour": cook_hour,
                "max_minutes": max_min,
            })
        logger.log_planner_output(plan)

        # ---- Executor + Critic loop ----
        retries = 0
        executor_result: ExecutorResult | None = None
        critic_result: CriticResult | None = None

        for attempt in range(MAX_RETRIES + 1):
            executor_result = run_executor(plan, constraints, logger)

            critic_result = run_critic(executor_result, client)
            logger.log_critic_check(
                valid=critic_result.valid,
                issues=critic_result.issues,
                fix_instructions=critic_result.fix_instructions,
            )

            if critic_result.valid:
                break

            if attempt < MAX_RETRIES:
                # Inject fix instructions into constraints for re-planning
                retries += 1
                constraints["_fix_instructions"] = critic_result.fix_instructions
                plan = run_planner(constraints, client)
                logger.log_planner_output(plan)

        result = AgentResult(
            recipes=executor_result.recipes if executor_result else [],
            grocery_list=executor_result.grocery_list if executor_result else {},
            nutrition_summary=executor_result.nutrition_summary if executor_result else {},
            ics_bytes=executor_result.ics_bytes if executor_result else b"",
            cooking_blocks=executor_result.cooking_blocks if executor_result else [],
            plan=plan,
            critic=critic_result,
            tool_calls=executor_result.tool_calls if executor_result else [],
            allergy_reports=executor_result.allergy_reports if executor_result else [],
            session_id=logger.session_id,
            log_path=logger.get_log_path(),
            retries=retries,
        )

        logger.log_final_output(
            {
                "num_recipes": len(result.recipes),
                "grocery_categories": list(result.grocery_list.keys()),
                "critic_valid": critic_result.valid if critic_result else None,
                "retries": retries,
            }
        )

    return result
