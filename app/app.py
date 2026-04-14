"""
MealPlanAgent — Streamlit Web UI

Run with:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.pipeline import run_pipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MealPlanAgent",
    page_icon="🥗",
    layout="wide",
)

st.title("MealPlanAgent")
st.caption("AI-powered weekly meal planning with citations, grocery lists, and calendar export.")

# ─── Session State ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of AgentResult dicts (serializable summary)
if "result" not in st.session_state:
    st.session_state.result = None

# ─── Sidebar: Constraints ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Constraints")

    model_name = st.selectbox(
        "Model",
        options=["gemini", "groq-llama", "groq-mistral"],
        index=0,
        help="gemini = Gemini 2.0 Flash | groq-llama = Llama 3.1 70B | groq-mistral = Mixtral 8x7B (open-source)",
    )

    num_meals = st.slider("Number of meals", min_value=1, max_value=7, value=5)

    max_minutes = st.slider("Max cooking time per meal (minutes)", min_value=10, max_value=120, value=30)

    cook_after_hour = st.slider(
        "Cook no earlier than (24h)", min_value=6, max_value=22, value=18,
        help="e.g. 18 = 6 pm"
    )

    tags_input = st.text_input(
        "Dietary tags (comma-separated)",
        value="high-protein",
        help='e.g. "vegetarian, low-calorie, asian"',
    )
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    allergens_input = st.text_input(
        "Allergens to avoid (comma-separated)",
        value="",
        help='e.g. "peanuts, dairy, gluten"',
    )
    allergens = [a.strip() for a in allergens_input.split(",") if a.strip()]

    dietary_notes = st.text_area(
        "Additional notes (optional)",
        placeholder="e.g. I prefer Mediterranean-style food and cook for 2 people.",
        height=80,
    )

    run_btn = st.button("Generate Meal Plan", type="primary", use_container_width=True)

# ─── Safety Guardrail ────────────────────────────────────────────────────────
BLOCKED_KEYWORDS = {"alcohol", "raw meat", "unpasteurized", "uncooked egg"}


def safety_check(tags: list[str], notes: str) -> str | None:
    combined = " ".join(tags + [notes]).lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in combined:
            return f"Request blocked: '{kw}' is not supported for safety reasons."
    return None


# ─── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    block_msg = safety_check(tags, dietary_notes)
    if block_msg:
        st.error(block_msg)
    else:
        constraints = {
            "num_meals": num_meals,
            "max_minutes": max_minutes,
            "tags": tags,
            "allergens": allergens,
            "cook_after_hour": cook_after_hour,
            "dietary_notes": dietary_notes,
        }

        with st.spinner("Planner is thinking..."):
            try:
                result = run_pipeline(constraints, model_name=model_name)
                st.session_state.result = result

                # Save summary to history
                st.session_state.history.append(
                    {
                        "model": model_name,
                        "constraints": constraints,
                        "num_recipes": len(result.recipes),
                        "critic_valid": result.critic.valid if result.critic else None,
                        "session_id": result.session_id,
                    }
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

# ─── Display Results ──────────────────────────────────────────────────────────
result = st.session_state.result

if result is not None:
    if result.critic and not result.critic.valid:
        st.warning(
            "Critic flagged issues (plan may be imperfect): "
            + " | ".join(result.critic.issues)
        )

    tabs = st.tabs(["Meal Plan", "Grocery List", "Nutrition", "Calendar", "Agent Trace"])

    # ── Tab 1: Meal Plan ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Weekly Meal Plan")
        if not result.recipes:
            st.info("No recipes found. Try relaxing your constraints.")
        else:
            for i, recipe in enumerate(result.recipes, 1):
                with st.expander(
                    f"{'Mon Tue Wed Thu Fri Sat Sun'.split()[min(i-1,6)]}  —  {recipe['name']}  ({recipe['minutes']} min)",
                    expanded=(i == 1),
                ):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Ingredients:** {', '.join(recipe.get('ingredients', []))}")
                        citation = recipe.get("citation", {})
                        st.caption(
                            f"Source: {citation.get('source', 'Food.com')} | "
                            f"Recipe ID: {citation.get('recipe_id', 'N/A')} | "
                            f"Tags: {', '.join(recipe.get('tags', [])[:5])}"
                        )
                    with col2:
                        allergy = next(
                            (r for r in result.allergy_reports if r.get("recipe_name") == recipe["name"]),
                            None,
                        )
                        if allergy:
                            if allergy["safe"]:
                                st.success("Allergy: SAFE")
                            else:
                                st.error(f"Allergy: UNSAFE ({', '.join(allergy['violations'])})")

    # ── Tab 2: Grocery List ─────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Grocery List")
        if not result.grocery_list:
            st.info("No grocery list generated.")
        else:
            cols = st.columns(2)
            for i, (category, items) in enumerate(result.grocery_list.items()):
                with cols[i % 2]:
                    st.markdown(f"**{category}**")
                    for item in items:
                        st.markdown(f"- {item}")

    # ── Tab 3: Nutrition ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Weekly Nutrition Summary")
        if result.nutrition_summary:
            import plotly.graph_objects as go

            labels = list(result.nutrition_summary.keys())
            values = list(result.nutrition_summary.values())

            fig = go.Figure(
                go.Bar(x=labels, y=values, marker_color="steelblue")
            )
            fig.update_layout(
                title="Total Nutrients Across All Meals",
                xaxis_title="Nutrient",
                yaxis_title="Amount",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                {"Nutrient": labels, "Total": values},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No nutrition data available.")

    # ── Tab 4: Calendar / ICS ───────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Cooking Calendar")
        if result.ics_bytes:
            st.download_button(
                label="Download .ics (import into Google/Apple/Outlook Calendar)",
                data=result.ics_bytes,
                file_name="meal_plan.ics",
                mime="text/calendar",
            )
            st.markdown("**Scheduled cooking blocks:**")
            for block in result.cooking_blocks:
                hour = block.get("cook_hour", 18)
                ampm = f"{hour % 12 or 12}:00 {'AM' if hour < 12 else 'PM'}"
                st.markdown(
                    f"- **{block.get('day', '?')}** at {ampm} — "
                    f"{block.get('meal_name', '?')} ({block.get('duration_minutes', 0)} min)"
                )
        else:
            st.info("No calendar blocks generated.")

    # ── Tab 5: Agent Trace ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Agent Trace")
        st.caption(f"Session: `{result.session_id}` | Log: `{result.log_path}` | Retries: {result.retries}")

        st.markdown("**Planner Output (raw plan):**")
        st.json(result.plan)

        st.markdown("**Tool Calls:**")
        st.json(result.tool_calls)

        if result.critic:
            st.markdown("**Critic Verdict:**")
            st.json(
                {
                    "valid": result.critic.valid,
                    "issues": result.critic.issues,
                    "fix_instructions": result.critic.fix_instructions,
                }
            )

# ─── History Sidebar ─────────────────────────────────────────────────────────
if st.session_state.history:
    with st.sidebar:
        st.divider()
        st.subheader("Session History")
        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"- **{entry['model']}** | {entry['num_recipes']} meals | "
                f"{'valid' if entry.get('critic_valid') else 'flagged'}"
            )
