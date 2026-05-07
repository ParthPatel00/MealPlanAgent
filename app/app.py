"""
MealPlanAgent - Streamlit Web UI

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.pipeline import run_pipeline
from src.memory.feedback import record_feedback
from src.memory.store import get_memory_summary, retrieve_past_plans, retrieve_preferences

# ---- Page Config ----
st.set_page_config(page_title="MealPlanAgent", page_icon="🥗", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    [data-testid="stSidebar"] { min-width: 340px; max-width: 380px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

# ---- Session State ----
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---- NL Parser ----
NL_PARSE_PROMPT = """Extract meal planning constraints from the user's message.
Return ONLY valid JSON with these keys:
- num_meals (int, default 5)
- max_minutes (int, default 30)
- tags (list of strings: dietary preferences like "high-protein", "vegetarian", "keto", cuisine types)
- allergens (list of strings: things to avoid like "peanuts", "gluten", "dairy")
- cook_after_hour (int 0-23, default 18)
- dietary_notes (string: anything else relevant)

User message: {user_input}

JSON:"""


def parse_natural_input(text: str, client) -> tuple[dict, str]:
    from src.agent.json_utils import extract_first_json
    prompt = NL_PARSE_PROMPT.format(user_input=text)
    response = client.chat(prompt=prompt, temperature=0.0, max_tokens=512)
    parsed = extract_first_json(response.text)
    if not parsed:
        parsed = {}
    parsed.setdefault("num_meals", 5)
    parsed.setdefault("max_minutes", 30)
    parsed.setdefault("tags", [])
    parsed.setdefault("allergens", [])
    parsed.setdefault("cook_after_hour", 18)
    parsed.setdefault("dietary_notes", "")
    return parsed, response.text


# ---- Safety ----
BLOCKED_KEYWORDS = {
    "alcohol", "raw meat", "unpasteurized", "uncooked egg",
    "poison", "toxic", "bleach", "detergent",
}


def safety_check(text: str) -> str | None:
    lower = text.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lower:
            return f"Request blocked: '{kw}' is not supported for safety reasons."
    if len(text) > 2000:
        return "Request blocked: input exceeds maximum length."
    return None


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## MealPlanAgent")

    model_name = st.selectbox(
        "Model",
        options=["ollama-llama3b", "ollama-granite2b", "groq-llama"],
        index=0,
        help="ollama = local (no API key) | groq = cloud (needs GROQ_API_KEY)",
    )

    user_id = st.text_input("User ID", value="default_user",
                            help="Tracks your preferences across sessions")

    st.divider()

    SAMPLE_REQUESTS = [
        "(custom)",
        "Plan me 3 high-protein dinners this week, no peanuts. Under 30 minutes each.",
        "I want 5 vegetarian meals, avoid gluten and dairy. I cook after 7pm.",
        "Quick keto lunches for 4 days, nothing with shellfish or soy.",
    ]

    sample = st.selectbox("Try a sample request", SAMPLE_REQUESTS, index=0)

    user_input = st.text_area(
        "What do you want to eat this week?",
        value=sample if sample != "(custom)" else "",
        placeholder="Describe your meals in plain English...",
        height=120,
    )

    audio = st.audio_input("Or speak your request")
    if audio and not user_input:
        try:
            import io
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            audio_bytes = audio.read()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = recognizer.record(source)
            user_input = recognizer.recognize_google(audio_data)
            st.success(f'Heard: "{user_input}"')
        except Exception as exc:
            st.warning(f"Could not transcribe: {exc}")

    with st.expander("Advanced options"):
        num_meals = st.slider("Meals", 1, 7, 5)
        max_minutes = st.slider("Max cook time (min)", 10, 120, 30)
        cook_after_hour = st.slider("Earliest cook hour", 6, 22, 18)
        tags_input = st.text_input("Tags", placeholder="vegetarian, keto")
        allergens_input = st.text_input("Allergens", placeholder="peanuts, dairy")

    run_btn = st.button("Generate Meal Plan", type="primary", use_container_width=True)

# ---- Main Area Header ----
if not st.session_state.result and not run_btn:
    st.markdown("# MealPlanAgent")
    st.markdown("Describe what you want to eat in the sidebar, pick a model, and hit **Generate**.")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. You describe your meal preferences in natural language
2. The LLM parses your request into structured constraints
3. A Planner agent designs meal queries using few-shot examples and your history
4. An Executor searches recipes via hybrid RAG (BM25 + vector + knowledge graph), checks allergies, calculates nutrition, builds a grocery list and budget
5. A Critic agent verifies the plan meets all constraints (retries if not)

**Models available:**
- `ollama-llama3b` / `ollama-granite2b` - run locally, no API key needed
- `groq-llama` - Llama 3.3 70B via Groq cloud (needs `GROQ_API_KEY`)
""")
    st.stop()

# ---- Run Pipeline ----
if run_btn:
    if user_input and user_input.strip():
        block_msg = safety_check(user_input)
        if block_msg:
            st.error(block_msg)
            st.stop()

        with st.spinner("Understanding your request..."):
            from src.models.client import LLMClient
            try:
                parse_client = LLMClient(model_name)
                constraints, raw_parse = parse_natural_input(user_input, parse_client)
                constraints["_original_input"] = user_input
                st.session_state.nl_input = user_input
                st.session_state.nl_parse = raw_parse
            except Exception as exc:
                st.error(f"Could not parse input: {exc}")
                st.stop()
    else:
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
        allergens = [a.strip() for a in allergens_input.split(",") if a.strip()] if allergens_input else []
        constraints = {
            "num_meals": num_meals,
            "max_minutes": max_minutes,
            "tags": tags,
            "allergens": allergens,
            "cook_after_hour": cook_after_hour,
            "dietary_notes": "",
        }
        block_msg = safety_check(" ".join(tags + allergens))
        if block_msg:
            st.error(block_msg)
            st.stop()

    with st.spinner("Generating meal plan..."):
        try:
            result = run_pipeline(constraints, model_name=model_name, user_id=user_id)
            st.session_state.result = result
            st.session_state.history.append({
                "model": model_name,
                "num_recipes": len(result.recipes),
                "critic_valid": result.critic.valid if result.critic else None,
            })
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

# ---- Display Results ----
result = st.session_state.result
if result is None:
    st.stop()

if result.critic and not result.critic.valid:
    st.warning("Critic flagged issues: " + " | ".join(result.critic.issues))

tabs = st.tabs(["Meal Plan", "Grocery & Budget", "Nutrition", "Calendar", "Agent Trace", "Memory"])

# ---- Tab: Meal Plan ----
with tabs[0]:
    if not result.recipes:
        st.info("No recipes found. Try different constraints.")
    else:
        for i, recipe in enumerate(result.recipes, 1):
            day = recipe.get("_day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][min(i - 1, 6)])
            with st.expander(f"**{day}** - {recipe['name']} ({recipe['minutes']} min)", expanded=(i == 1)):
                st.markdown(f"**Ingredients:** {', '.join(recipe.get('ingredients', []))}")
                citation = recipe.get("citation", {})
                st.caption(f"Source: {citation.get('source', 'Food.com')} | ID: {citation.get('recipe_id', 'N/A')}")

                allergy = next((r for r in result.allergy_reports if r.get("recipe_name") == recipe["name"]), None)
                if allergy:
                    if allergy["safe"]:
                        st.success("Allergy check: SAFE")
                    else:
                        st.error(f"Allergy check: UNSAFE - {', '.join(allergy['violations'])}")

        try:
            from src.tools.pdf_export import generate_meal_plan_pdf
            pdf_bytes = generate_meal_plan_pdf(
                result.recipes, result.grocery_list,
                result.nutrition_summary, result.budget_estimate or None,
            )
            st.download_button("Download PDF", data=pdf_bytes, file_name="meal_plan.pdf", mime="application/pdf")
        except Exception:
            pass

# ---- Tab: Grocery & Budget ----
with tabs[1]:
    if not result.grocery_list:
        st.info("No grocery list generated.")
    else:
        cols = st.columns(2)
        for i, (category, items) in enumerate(result.grocery_list.items()):
            with cols[i % 2]:
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"- {item}")

        if result.budget_estimate:
            st.divider()
            total = result.budget_estimate.get("total_estimated_cost", 0)
            st.metric("Estimated Total", f"${total:.2f}")

            per_cat = result.budget_estimate.get("per_category", {})
            if per_cat:
                import plotly.graph_objects as go
                fig = go.Figure(go.Pie(labels=list(per_cat.keys()), values=list(per_cat.values()), hole=0.4))
                fig.update_layout(height=300, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ---- Tab: Nutrition ----
with tabs[2]:
    if result.nutrition_summary:
        import plotly.graph_objects as go
        labels = list(result.nutrition_summary.keys())
        values = list(result.nutrition_summary.values())
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#4A90D9"))
        fig.update_layout(height=350, margin=dict(t=30, b=10), xaxis_title="Nutrient", yaxis_title="Amount")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No nutrition data available.")

# ---- Tab: Calendar ----
with tabs[3]:
    if result.ics_bytes:
        st.download_button("Download .ics", data=result.ics_bytes, file_name="meal_plan.ics", mime="text/calendar")
        for block in result.cooking_blocks:
            hour = block.get("cook_hour", 18)
            ampm = f"{hour % 12 or 12}:00 {'PM' if hour >= 12 else 'AM'}"
            st.markdown(f"- **{block.get('day', '?')}** at {ampm} - {block.get('meal_name', '?')} ({block.get('duration_minutes', 0)} min)")
    else:
        st.info("No calendar data generated.")

# ---- Tab: Agent Trace ----
with tabs[4]:
    st.caption(f"Session: `{result.session_id}` | Retries: {result.retries}")

    if "nl_input" in st.session_state and st.session_state.nl_input:
        st.markdown("#### Step 0: Input Parsing")
        st.markdown(f"> {st.session_state.nl_input}")
        if "nl_parse" in st.session_state:
            with st.expander("Raw LLM parse output"):
                st.code(st.session_state.nl_parse, language="json")

    if result.planner_trace:
        trace = result.planner_trace
        st.markdown("#### Step 1: Planner")
        st.markdown(f"`{trace.model}` responded in **{trace.latency_ms:.0f}ms**")
        with st.expander("System prompt"):
            st.code(trace.system_prompt, language="text")
        with st.expander("User prompt (few-shot + constraints + memory)"):
            st.code(trace.user_prompt, language="text")
        with st.expander("LLM response", expanded=True):
            st.code(trace.raw_response, language="json")

    if result.memory_context:
        with st.expander("Memory context injected"):
            st.code(result.memory_context, language="text")

    st.markdown("#### Step 2: Executor")
    if result.tool_calls:
        import plotly.graph_objects as go
        tool_names = [tc.get("tool", "?") for tc in result.tool_calls]
        tool_counts = {}
        for t in tool_names:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        fig = go.Figure(go.Bar(x=list(tool_counts.keys()), y=list(tool_counts.values()), marker_color="#7B68EE"))
        fig.update_layout(height=250, margin=dict(t=20, b=10), title="Tool calls")
        st.plotly_chart(fig, use_container_width=True)

        for i, tc in enumerate(result.tool_calls):
            with st.expander(f"#{i+1} {tc.get('tool', '?')}"):
                st.json(tc)

    st.markdown("#### Step 3: Critic")
    if result.critic:
        if result.critic.valid:
            st.success(f"Plan approved (retries: {result.retries})")
        else:
            st.error(f"Issues: {', '.join(result.critic.issues)}")

# ---- Tab: Memory ----
with tabs[5]:
    st.caption(f"User: {user_id}")

    summary = get_memory_summary(user_id)
    if summary:
        st.markdown("**Learned preferences:**")
        st.info(summary)

    preferences = retrieve_preferences(user_id)
    if preferences:
        st.markdown("**Stored:**")
        for k, v in preferences.items():
            st.markdown(f"- **{k}:** {v}")

    past = retrieve_past_plans(user_id, limit=5)
    if past:
        st.markdown("**Recent plans:**")
        for plan in past:
            st.markdown(f"- `{plan['session_id']}`: {', '.join(plan['recipes'][:3])}")

    st.divider()
    st.markdown("**Feedback**")
    if result.recipes:
        if "feedback_liked" not in st.session_state:
            st.session_state.feedback_liked = set()
        if "feedback_disliked" not in st.session_state:
            st.session_state.feedback_disliked = set()

        for recipe in result.recipes:
            name = recipe["name"]
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.text(name)
            with c2:
                if st.button("👍", key=f"like_{name}"):
                    st.session_state.feedback_liked.add(name)
                    st.session_state.feedback_disliked.discard(name)
                    st.rerun()
            with c3:
                if st.button("👎", key=f"dislike_{name}"):
                    st.session_state.feedback_disliked.add(name)
                    st.session_state.feedback_liked.discard(name)
                    st.rerun()

        if st.button("Submit Feedback"):
            record_feedback(
                user_id=user_id,
                session_id=result.session_id,
                liked_recipes=list(st.session_state.feedback_liked),
                disliked_recipes=list(st.session_state.feedback_disliked),
                notes="",
            )
            st.session_state.feedback_liked = set()
            st.session_state.feedback_disliked = set()
            st.success("Feedback saved.")
