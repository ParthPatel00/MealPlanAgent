"""
MealPlanAgent - FastAPI Backend

Run with:
    uvicorn api.main:app --reload --port 8001
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.pipeline import run_pipeline
from src.models.client import LLMClient
from src.agent.json_utils import extract_first_json
from src.memory.store import get_memory_summary, retrieve_past_plans, retrieve_preferences
from src.memory.feedback import record_feedback

app = FastAPI(title="MealPlanAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NL_PARSE_PROMPT = """Extract meal planning constraints from the user's message.
Return ONLY valid JSON with these keys:
- num_meals (int): how many meals they want. Pay close attention to singular vs plural: "a meal" or "a dinner" means 1, "some meals" or "meals for the week" means 5. If they say a specific number, use that number. Only default to 5 if quantity is truly ambiguous.
- max_minutes (int): maximum cooking time per meal. If not mentioned, use 60.
- tags (list of strings): dietary preferences or cuisine types they want (e.g. "high-protein", "vegetarian", "keto", "italian"). Only include if explicitly stated.
- allergens (list of strings): foods/ingredients to AVOID (e.g. "peanuts", "gluten", "dairy", "lettuce"). Include anything they say they're allergic to or want to avoid.
- cook_after_hour (int 0-23): when they want to start cooking. If not mentioned, use 18.
- dietary_notes (string): any other relevant context that doesn't fit above fields. Empty string if nothing extra.

IMPORTANT: Only extract what the user explicitly mentions. If a field is not mentioned, use the default value stated above. Never use 0 for num_meals or max_minutes.

User message: {user_input}

JSON:"""

BLOCKED_KEYWORDS = {
    "alcohol", "raw meat", "unpasteurized", "uncooked egg",
    "poison", "toxic", "bleach", "detergent",
}


class GenerateRequest(BaseModel):
    user_input: str
    model: str = "ollama-llama3b"
    user_id: str = "default_user"


class IcsRequest(BaseModel):
    cooking_blocks: list[dict]
    cook_hour: int = 18
    cook_minute: int = 0


class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    liked_recipes: list[str] = []
    disliked_recipes: list[str] = []
    notes: str = ""


@app.post("/api/generate")
def generate_meal_plan(req: GenerateRequest):
    lower = req.user_input.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lower:
            return {"error": f"Request blocked: '{kw}' is not supported for safety reasons."}

    client = LLMClient(req.model)

    # Parse natural language
    prompt = NL_PARSE_PROMPT.format(user_input=req.user_input)
    parse_response = client.chat(prompt=prompt, temperature=0.0, max_tokens=512)
    constraints = extract_first_json(parse_response.text)
    if not constraints:
        constraints = {}
    constraints.setdefault("num_meals", 5)
    constraints.setdefault("max_minutes", 60)
    constraints.setdefault("tags", [])
    constraints.setdefault("allergens", [])
    constraints.setdefault("cook_after_hour", 18)
    constraints.setdefault("dietary_notes", "")

    # Validate: fix nonsensical values from bad LLM parses
    if not constraints["num_meals"] or constraints["num_meals"] < 1:
        constraints["num_meals"] = 5
    if not constraints["max_minutes"] or constraints["max_minutes"] < 5:
        constraints["max_minutes"] = 60
    if not isinstance(constraints["cook_after_hour"], int) or constraints["cook_after_hour"] < 0 or constraints["cook_after_hour"] > 23:
        constraints["cook_after_hour"] = 18

    constraints["_original_input"] = req.user_input

    # Run pipeline
    start = time.time()
    result = run_pipeline(constraints, model_name=req.model, user_id=req.user_id)
    elapsed = time.time() - start

    # Build trace
    trace_data = None
    if result.planner_trace:
        t = result.planner_trace
        trace_data = {
            "model": t.model,
            "latency_ms": t.latency_ms,
            "system_prompt": t.system_prompt,
            "user_prompt": t.user_prompt,
            "raw_response": t.raw_response,
        }

    critic_data = None
    if result.critic:
        critic_data = {
            "valid": result.critic.valid,
            "issues": result.critic.issues,
            "fix_instructions": result.critic.fix_instructions,
        }

    # Encode ICS as base64
    ics_b64 = base64.b64encode(result.ics_bytes).decode() if result.ics_bytes else None

    return {
        "parsed_constraints": {
            "num_meals": constraints.get("num_meals"),
            "max_minutes": constraints.get("max_minutes"),
            "tags": constraints.get("tags"),
            "allergens": constraints.get("allergens"),
            "cook_after_hour": constraints.get("cook_after_hour"),
            "dietary_notes": constraints.get("dietary_notes"),
        },
        "nl_parse_raw": parse_response.text,
        "original_input": req.user_input,
        "recipes": result.recipes,
        "grocery_list": result.grocery_list,
        "nutrition_summary": result.nutrition_summary,
        "budget_estimate": result.budget_estimate,
        "ics_base64": ics_b64,
        "cooking_blocks": result.cooking_blocks,
        "allergy_reports": result.allergy_reports,
        "tool_calls": result.tool_calls,
        "critic": critic_data,
        "planner_trace": trace_data,
        "memory_context": result.memory_context,
        "session_id": result.session_id,
        "retries": result.retries,
        "elapsed_seconds": elapsed,
    }


@app.post("/api/generate-ics")
def generate_ics(req: IcsRequest):
    """Generate ICS with user-chosen time."""
    from src.tools.ics_generator import generate_ics_bytes
    blocks = []
    for b in req.cooking_blocks:
        blocks.append({
            "day": b.get("day", "Monday"),
            "meal_name": b.get("meal_name", "Meal"),
            "duration_minutes": b.get("duration_minutes", 30),
            "cook_hour": req.cook_hour,
            "cook_minute": req.cook_minute,
        })
    ics_bytes = generate_ics_bytes(blocks)
    return {"ics_base64": base64.b64encode(ics_bytes).decode()}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    import io
    import speech_recognition as sr

    audio_bytes = await file.read()
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return {"text": text}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/memory/{user_id}")
def get_memory(user_id: str):
    summary = get_memory_summary(user_id)
    preferences = retrieve_preferences(user_id)
    past = retrieve_past_plans(user_id, limit=5)
    # Return constraints-focused view, not recipe names
    past_formatted = []
    for p in past:
        c = p.get("constraints", {})
        past_formatted.append({
            "session_id": p["session_id"],
            "created_at": p.get("created_at", ""),
            "original_input": c.get("_original_input", ""),
            "tags": c.get("tags", []),
            "allergens": c.get("allergens", []),
            "num_meals": c.get("num_meals"),
            "max_minutes": c.get("max_minutes"),
        })
    return {
        "summary": summary,
        "preferences": preferences,
        "past_requests": past_formatted,
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    record_feedback(
        user_id=req.user_id,
        session_id=req.session_id,
        liked_recipes=req.liked_recipes,
        disliked_recipes=req.disliked_recipes,
        notes=req.notes,
    )
    return {"status": "ok"}


@app.get("/api/health")
def health():
    return {"status": "ok"}
