# MealPlanAgent

An AI-powered weekly meal planning agent built for CMPE258. Given your time, dietary, and allergy constraints, it produces a weekly meal plan with recipe citations, a categorized grocery list, nutritional summaries, and a downloadable calendar (.ics) file.

---

## Architecture

```
User Constraints
      |
  [Planner]  LLM generates a structured execution plan
      |
  [Executor] Dispatches tool calls in sequence, logs each one
   |  |  |  |  |
   |  |  |  |  +-- ics_generator     → .ics calendar file
   |  |  |  +----- grocery_list      → categorized ingredient list
   |  |  +-------- nutrition         → per-recipe + plan-wide totals
   |  +----------- allergy_checker   → local match + Open Food Facts API
   +-------------- recipe_search     → Hybrid RAG (BM25 + vector similarity)
      |
  [Critic]   LLM verifies allergy safety, citations, calendar validity
      |       Triggers up to 2 re-planning retries if issues found
      |
  [Output]   Streamlit web UI
```

---

## Datasets

### 1. Food.com Recipes and User Interactions
**Source:** [Kaggle — shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv)

The primary dataset powering recipe search, nutrition, and citations. The full `RAW_recipes.csv` is 281 MB and is not committed to this repo. A 200-row sample is included at `data/sample_recipes.csv` so the code structure and field format are clear without requiring a download.

**Fields used:**

| Field | Purpose |
|---|---|
| `name` | Recipe display name and search |
| `id` | Unique citation key (`recipe_id` in all outputs) |
| `minutes` | Cooking time filter |
| `tags` | Dietary tag matching (e.g. `vegetarian`, `high-protein`) |
| `nutrition` | List of 7 PDV values: calories, fat, sugar, sodium, protein, sat-fat, carbs |
| `ingredients` | Allergy checking and grocery list generation |
| `steps` | Indexed in the RAG vector store for semantic search |

To use the full dataset, download `RAW_recipes.csv` from Kaggle and place it at `data/RAW_recipes.csv`.

### 2. Open Food Facts
**Source:** [world.openfoodfacts.org API](https://world.openfoodfacts.org)

Queried at runtime via REST API to cross-check ingredient-level allergen labels (`allergens`, `traces` fields). No file download required.

### 3. Instacart Online Grocery Shopping Dataset 2017 *(next steps)*
**Source:** [github.com/subwaymatch/instacart-dataset-2017](https://github.com/subwaymatch/instacart-dataset-2017)

Will replace the current keyword-based grocery grouping with real aisle and department IDs (`product_name`, `aisle_id`, `department_id`).

---

## Models Compared

| Name | Provider | Type | Notes |
|---|---|---|---|
| Gemini 2.0 Flash | Google AI Studio | Closed-source | Free tier via AI Studio |
| Llama 3.3 70B | Groq | Open-source | Free tier via Groq |
| Mixtral 8x7B | Groq | Open-source | Free tier via Groq |
| granite3.1-dense:2b | Ollama (local) | Open-source | Runs fully offline, no API key |

All models are evaluated on the same 60-case test set. Performance, latency, and cost tradeoffs are reported in the Evaluation Results section.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd MealPlanAgent
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY and GROQ_API_KEY
```

- Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com/apikey)
- Get a free Groq key at [console.groq.com](https://console.groq.com)

### 3. Get the dataset

A 200-row sample (`data/sample_recipes.csv`) is included in the repo for reference.

For the full pipeline, download `RAW_recipes.csv` (281 MB) from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) and place it at:

```
data/RAW_recipes.csv
```

### 4. Build the vector index

```bash
# Clean and process recipes (produces data/processed/recipes_clean.json)
python -m src.data.loader

# Build ChromaDB vector index (one-time, takes a few minutes)
python -m src.rag.indexer
```

### 5. Run the web app

```bash
streamlit run app/app.py
```

---

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Set your constraints in the sidebar: number of meals, max cooking time, dietary tags, allergens
3. Click **Generate Meal Plan**
4. Explore the output tabs: Meal Plan, Grocery List, Nutrition, Calendar, Agent Trace

---

## Progress

### Completed
- Data pipeline: loads and cleans 10,000 Food.com recipes
- Hybrid RAG: ChromaDB vector store + BM25 keyword retrieval with Reciprocal Rank Fusion
- 5 tools: recipe search, allergy checker (local + Open Food Facts API), nutrition summary, grocery list, ICS calendar generator
- Planner-Executor-Critic agent with up to 2 retry loops and structured JSONL logging
- Unified LLM client supporting Ollama (local), Groq, and Gemini
- 60-case evaluation set with constraint, allergy, citation, and tool-success metrics
- Streamlit web UI with 5 output tabs: Meal Plan, Grocery List, Nutrition, Calendar, Agent Trace
- Safety guardrails: allergen filtering at recipe selection, keyword blocklist in the UI

### Evaluation Results (10-case sample)

| Model | Constraint Pass | Allergy Violation | Citation Pass | Tool Success | Avg Latency |
|---|---|---|---|---|---|
| Llama 3.3 70B (Groq) | 100% | 0% | 100% | 100% | 1,310 ms |
| granite3.1-dense:2b (Ollama) | 100% | 0% | 100% | 100% | 8,057 ms |
| Gemini 2.0 Flash | TBD | TBD | TBD | TBD | TBD |
| Mixtral 8x7B (Groq) | TBD | TBD | TBD | TBD | TBD |

Full 60-case results across all models are a next step.

---

## Running Evaluation

```bash
# Run 10 test cases locally with no API key required
python -m src.evaluation.evaluator --model ollama-granite2b --limit 10

# Run all 60 cases against all models
python -m src.evaluation.evaluator --model ollama-granite2b
python -m src.evaluation.evaluator --model groq-llama
python -m src.evaluation.evaluator --model groq-mistral
python -m src.evaluation.evaluator --model gemini
```

Results are saved to `data/eval/results_<model>_<timestamp>.json`.

### Metrics

| Metric | Description |
|---|---|
| Constraint Pass Rate | % of meals within time + count requirements |
| Allergy Violation Rate | % of recipes flagged as unsafe (target: 0%) |
| Citation Pass Rate | % of recipes with a valid recipe_id citation |
| Tool Success Rate | % of tool calls that completed without error |
| Avg Latency (ms) | End-to-end pipeline latency |

---

## Project Structure

```
MealPlanAgent/
├── PLAN.md                        Implementation plan and checklist
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                       RAW_recipes.csv (download from Kaggle)
│   ├── processed/                 recipes_clean.json (generated)
│   ├── eval/test_cases.json       60 evaluation test cases
│   └── chroma_db/                 Vector index (generated)
├── src/
│   ├── logging_utils.py           Structured JSONL logger
│   ├── data/
│   │   ├── loader.py              Load + clean Food.com CSV
│   │   └── preprocessor.py       Convert to LlamaIndex Documents
│   ├── rag/
│   │   ├── indexer.py             Build ChromaDB vector index
│   │   └── retriever.py           Hybrid BM25 + vector retrieval
│   ├── tools/
│   │   ├── recipe_search.py       RAG-powered recipe lookup
│   │   ├── allergy_checker.py     Local + Open Food Facts allergy check
│   │   ├── nutrition.py           PDV to absolute nutrition values
│   │   ├── grocery_list.py        Aggregate + categorize ingredients
│   │   └── ics_generator.py       Generate .ics calendar file
│   ├── agent/
│   │   ├── planner.py             Planner stage (LLM)
│   │   ├── executor.py            Executor stage (tool dispatch)
│   │   ├── critic.py              Critic stage (LLM verification)
│   │   └── pipeline.py            Orchestrates all three stages
│   ├── models/
│   │   └── client.py              Unified LLM client (Gemini + Groq)
│   └── evaluation/
│       ├── metrics.py             Metric functions
│       └── evaluator.py           Run eval across models
└── app/
    └── app.py                     Streamlit web UI
```

---

## Next Steps

1. **Google Calendar API** — write events directly to Google Calendar (OAuth flow)
2. **Gmail integration** — draft and send weekly meal plan email
3. **Instacart dataset** — replace keyword grocery grouping with real aisle/department IDs
4. **Open Food Facts full integration** — query product database for allergen labels
5. **Complete multi-model evaluation** — fill in results table above with charts
6. **GraphRAG** — build a recipe-ingredient-nutrition knowledge graph for richer retrieval
7. **Budget estimator** — add price estimation tool using average grocery prices
8. **Evaluation UI page** — in-app model comparison with Plotly charts
9. **Mobile-responsive UI**
10. **Video demo**
