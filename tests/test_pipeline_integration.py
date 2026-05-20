"""
Integration tests for the agent pipeline with mocked LLM and retrieval.

Verifies the Planner-Executor-Critic loop produces valid results
without requiring API keys or a running Ollama instance.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.agent.pipeline import AgentResult, run_pipeline


MOCK_PLAN_JSON = json.dumps({
    "meal_queries": [
        {"query": "high-protein chicken dinner", "day": "Monday", "cook_hour": 18, "max_minutes": 30},
        {"query": "healthy salmon meal", "day": "Tuesday", "cook_hour": 18, "max_minutes": 30},
        {"query": "quick vegetable stir fry", "day": "Wednesday", "cook_hour": 19, "max_minutes": 25},
    ],
    "allergens": [],
    "steps": ["search recipes", "check allergies", "build grocery list"],
    "notes": "",
})


MOCK_RECIPE_HITS = [
    {
        "name": "Garlic Chicken",
        "minutes": 25,
        "tags": ["high-protein", "dinner"],
        "ingredients": ["chicken breast", "garlic", "olive oil", "salt"],
        "nutrition": {"calories_pdv": 400, "total_fat_pdv": 25, "sugar_pdv": 5,
                      "sodium_pdv": 20, "protein_pdv": 55, "saturated_fat_pdv": 8,
                      "carbohydrates_pdv": 15},
        "citation": {"recipe_id": 1001, "source": "Food.com"},
        "relevance_score": 0.95,
    },
    {
        "name": "Lemon Salmon",
        "minutes": 20,
        "tags": ["healthy", "high-protein"],
        "ingredients": ["salmon", "lemon", "dill", "olive oil"],
        "nutrition": {"calories_pdv": 350, "total_fat_pdv": 30, "sugar_pdv": 3,
                      "sodium_pdv": 15, "protein_pdv": 50, "saturated_fat_pdv": 6,
                      "carbohydrates_pdv": 10},
        "citation": {"recipe_id": 1002, "source": "Food.com"},
        "relevance_score": 0.90,
    },
    {
        "name": "Quick Veggie Stir Fry",
        "minutes": 15,
        "tags": ["vegetarian", "quick"],
        "ingredients": ["bell pepper", "broccoli", "soy sauce", "garlic", "rice"],
        "nutrition": {"calories_pdv": 280, "total_fat_pdv": 10, "sugar_pdv": 8,
                      "sodium_pdv": 35, "protein_pdv": 15, "saturated_fat_pdv": 2,
                      "carbohydrates_pdv": 45},
        "citation": {"recipe_id": 1003, "source": "Food.com"},
        "relevance_score": 0.85,
    },
]


@dataclass
class MockLLMResponse:
    text: str
    model: str = "mock-model"
    prompt_tokens: int = 100
    completion_tokens: int = 200
    latency_ms: float = 50.0
    cost_usd: float = 0.0
    raw: dict = None

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


def mock_recipe_search(query, max_minutes=60, forbidden_ingredients=None,
                       preferred_ingredients=None, preferred_tags=None,
                       calorie_target=None, top_k=5):
    results = []
    for hit in MOCK_RECIPE_HITS:
        if hit["minutes"] <= max_minutes:
            results.append(dict(hit))
    return results[:top_k]


class TestPipelineIntegration:
    @patch("src.agent.pipeline.build_memory_context", return_value="")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_full_pipeline_produces_valid_result(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        constraints = {
            "num_meals": 3,
            "max_minutes": 30,
            "tags": ["high-protein"],
            "allergens": [],
            "cook_after_hour": 18,
        }

        result = run_pipeline(constraints, model_name="gemini", user_id="test_user")

        assert isinstance(result, AgentResult)
        assert len(result.recipes) >= 1
        assert result.session_id != ""
        assert result.critic is not None

    @patch("src.agent.pipeline.build_memory_context", return_value="")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_pipeline_generates_grocery_list(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        result = run_pipeline(
            {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": [], "cook_after_hour": 18},
            model_name="gemini",
        )
        assert len(result.grocery_list) > 0

    @patch("src.agent.pipeline.build_memory_context", return_value="")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_pipeline_generates_nutrition(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        result = run_pipeline(
            {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": [], "cook_after_hour": 18},
            model_name="gemini",
        )
        assert len(result.nutrition_summary) > 0

    @patch("src.agent.pipeline.build_memory_context", return_value="")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_pipeline_tool_calls_logged(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        result = run_pipeline(
            {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": [], "cook_after_hour": 18},
            model_name="gemini",
        )
        tool_names = [tc["tool"] for tc in result.tool_calls]
        assert "recipe_search" in tool_names

    @patch("src.agent.pipeline.build_memory_context", return_value="")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_pipeline_with_allergens(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        result = run_pipeline(
            {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": ["dairy"], "cook_after_hour": 18},
            model_name="gemini",
        )
        assert len(result.allergy_reports) > 0
        for report in result.allergy_reports:
            assert "safe" in report

    @patch("src.agent.pipeline.build_memory_context", return_value="User prefers Italian food.")
    @patch("src.agent.pipeline.write_memory")
    @patch("src.agent.pipeline.LLMClient")
    @patch("src.agent.executor.recipe_search", side_effect=mock_recipe_search)
    def test_pipeline_uses_memory_context(
        self, mock_search, mock_client_cls, mock_write_mem, mock_build_mem
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MockLLMResponse(text=MOCK_PLAN_JSON)
        mock_client_cls.return_value = mock_client

        result = run_pipeline(
            {"num_meals": 3, "max_minutes": 30, "tags": [], "allergens": [], "cook_after_hour": 18},
            model_name="gemini",
        )
        assert result.memory_context == "User prefers Italian food."
