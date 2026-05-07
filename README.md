# MealPlanAgent

An AI-powered weekly meal planning agent built for CMPE258. Given your time, dietary, and allergy constraints, it produces a weekly meal plan with recipe citations, a categorized grocery list with budget estimates, nutritional summaries, a downloadable calendar (.ics) file, and a PDF export.

The system learns your preferences over time through a stateful memory system that writes, summarizes, and retrieves user patterns across sessions.

---

## Architecture

```
User Constraints + Memory Context
      |
  [Planner]  LLM generates a structured plan (with few-shot examples)
      |
  [Executor] Dispatches tool calls in sequence, logs each one
   |  |  |  |  |  |
   |  |  |  |  |  +-- budget_estimator  -> estimated grocery cost
   |  |  |  |  +----- ics_generator     -> .ics calendar file
   |  |  |  +-------- grocery_list      -> categorized ingredient list
   |  |  +----------- nutrition         -> per-recipe + plan-wide totals
   |  +-------------- allergy_checker   -> local match + Open Food Facts API
   +----------------- recipe_search     -> Hybrid RAG (BM25 + vector + graph)
      |
  [Critic]   Rule-based verification of allergy safety, citations, calendar
      |       Triggers up to 2 re-planning retries if issues found
      |
  [Memory]   Writes session to SQLite, summarizes patterns via LLM
      |
  [Output]   Streamlit web UI (7 tabs)
```

### Key Technical Features

- **3-Way Hybrid RAG**: BM25 keyword search + ChromaDB vector similarity + NetworkX knowledge graph re-ranking
- **Stateful Memory**: SQLite-backed write/summarize/retrieve system that learns user preferences across sessions
- **Few-Shot Prompting**: 3 curated examples injected into the planner prompt for consistent JSON output
- **Planner-Executor-Critic Loop**: Up to 2 retry cycles with fix instructions fed back to the planner
- **6 Tools**: recipe_search, allergy_checker, nutrition, grocery_list, budget_estimator, ics_generator
- **Multi-Model Support**: 7 models across 3 providers (Ollama local, Groq cloud, Google Gemini)

---

## Datasets

### 1. Food.com Recipes and User Interactions
**Source:** [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv)

The primary dataset powering recipe search, nutrition, and citations. The full `RAW_recipes.csv` is 281 MB and is not committed to this repo. A 200-row sample is included at `data/sample_recipes.csv`.

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

### 2. Open Food Facts
**Source:** [world.openfoodfacts.org API](https://world.openfoodfacts.org)

Queried at runtime via REST API to cross-check ingredient-level allergen labels.

---

## Models Compared

| Name | Provider | Type | Notes |
|---|---|---|---|
| Gemini 2.5 Flash Lite | Google AI Studio | Closed-source | Free tier, fast |
| Llama 3.3 70B | Groq | Open-source | Free tier, highest quality |
| Llama 3.2 3B | Ollama (local) | Open-source | Runs fully offline, no API key |
| Granite 3.1 2B | Ollama (local) | Open-source | Backup model, smallest/fastest local |

All models are evaluated on the same 60-case test set. Performance, latency, and cost tradeoffs are reported in the Model Comparison tab of the UI.

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

### 3. Local models (optional, for offline use)

Install [Ollama](https://ollama.com) and pull the models:

```bash
ollama pull llama3.2:3b
ollama pull granite3.1-dense:2b
```

### 4. Get the dataset

Download `RAW_recipes.csv` (281 MB) from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) and place it at `data/RAW_recipes.csv`.

### 5. Build the indexes

```bash
# Clean and process recipes
python -m src.data.loader

# Build ChromaDB vector index (one-time)
python -m src.rag.indexer

# Build knowledge graph (optional, enables graph re-ranking)
python -m src.rag.graph_builder
```

### 6. Run the web app

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to select a model, set dietary constraints, and generate a meal plan.

---

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Enter your User ID for personalized memory
3. Set constraints: number of meals, max cooking time, dietary tags, allergens
4. Click **Generate Meal Plan**
5. Explore the 7 output tabs:
   - **Meal Plan**: Weekly recipes with ingredients, citations, allergy status, PDF download
   - **Grocery List**: Categorized ingredients with estimated budget and cost breakdown
   - **Nutrition**: Bar chart of weekly nutrient totals
   - **Calendar**: Downloadable .ics file with cooking time blocks
   - **Agent Trace**: Full pipeline trace with tool call distribution chart
   - **Memory**: Learned preferences, past plans, feedback form
   - **Model Comparison**: Interactive charts comparing all evaluated models

---

## Stateful Memory System

The memory system implements the write-summarize-retrieve pattern:

- **Write**: After every pipeline run, the session's recipes and constraints are stored in SQLite (`data/memory.db`)
- **Summarize**: Every 3rd session, the LLM generates a natural-language summary of user patterns (preferred cuisines, time tolerance, allergen history)
- **Retrieve**: Before planning, the system retrieves the user's memory context (summary + preferences + recent recipes) and injects it into the planner prompt

Users can provide explicit feedback (thumbs up/down per recipe) in the Memory tab, which further refines learned preferences.

---

## Evaluation

### Running evaluations

```bash
# Run 10 cases with a single model
python -m src.evaluation.evaluator --model gemini --limit 10

# Run all 60 cases across all target models
python -m src.evaluation.run_all

# Run specific models
python -m src.evaluation.run_all --models gemini groq-llama --limit 10
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
| Cost per Request ($) | Estimated cost based on token usage and provider pricing |

### Evaluation Results (60-case test set)

| Model | Cases | Constraint Pass | Allergy Violation | Citation Pass | Tool Success | Avg Latency |
|---|---|---|---|---|---|---|
| Llama 3.3 70B (Groq) | 23/60 | 100% | 0% | 100% | 100% | 12,782 ms |
| Llama 3.2 3B (Ollama) | 60/60 | 100% | 0.9% | 100% | 100% | 30,871 ms |
| Granite 3.1 2B (Ollama) | 60/60 | 100% | 1.0% | 100% | 100% | 25,897 ms |
| Gemini 2.5 Flash Lite | pending | -- | -- | -- | -- | -- |

Groq and Gemini evaluations are limited by free-tier daily API quotas (100K tokens/day and 20 requests/day respectively). Full interactive results with cost/latency tradeoff charts are available in the Model Comparison tab of the UI.

---

## Project Structure

```
MealPlanAgent/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── RAW_recipes.csv            Download from Kaggle (281 MB, gitignored)
│   ├── processed/                 recipes_clean.json (generated)
│   ├── eval/                      test_cases.json + results JSON files
│   ├── chroma_db/                 Vector index (generated)
│   ├── recipe_graph.pkl           Knowledge graph (generated)
│   └── memory.db                  User memory database (generated)
├── src/
│   ├── logging_utils.py           Structured JSONL logger
│   ├── data/
│   │   ├── loader.py              Load + clean Food.com CSV
│   │   └── preprocessor.py        Convert to LlamaIndex Documents
│   ├── rag/
│   │   ├── indexer.py             Build ChromaDB vector index
│   │   ├── retriever.py           3-way hybrid retrieval (BM25 + vector + graph)
│   │   ├── knowledge_graph.py     NetworkX recipe-ingredient-tag graph
│   │   └── graph_builder.py       CLI to build the knowledge graph
│   ├── tools/
│   │   ├── recipe_search.py       RAG-powered recipe lookup
│   │   ├── allergy_checker.py     Local + Open Food Facts allergy check
│   │   ├── nutrition.py           PDV to absolute nutrition values
│   │   ├── grocery_list.py        Aggregate + categorize ingredients
│   │   ├── budget_estimator.py    Grocery cost estimation
│   │   ├── ics_generator.py       Generate .ics calendar file
│   │   └── pdf_export.py          Generate formatted PDF meal plan
│   ├── agent/
│   │   ├── planner.py             Planner stage (LLM + few-shot examples)
│   │   ├── executor.py            Executor stage (6 tool dispatch)
│   │   ├── critic.py              Critic stage (rule-based verification)
│   │   ├── pipeline.py            Pipeline orchestrator with memory integration
│   │   ├── json_utils.py          Robust JSON extraction from LLM output
│   │   └── few_shot_examples.py   Curated few-shot examples for planner
│   ├── models/
│   │   └── client.py              Unified LLM client (Ollama + Groq + Gemini)
│   ├── memory/
│   │   ├── store.py               SQLite-backed memory persistence
│   │   ├── summarizer.py          LLM-based user history summarization
│   │   └── feedback.py            User feedback collection
│   └── evaluation/
│       ├── metrics.py             Metric functions
│       ├── evaluator.py           Single-model evaluation runner
│       ├── run_all.py             Batch multi-model evaluation
│       └── comparison.py          Cross-model comparison tables + charts
├── logs/                          Session JSONL logs
└── app/
    └── app.py                     Streamlit web UI (7 tabs)
```

---

## Next Steps

1. Complete Gemini and Groq evaluations (pending daily quota resets)
2. Google Calendar API integration (write events via OAuth)
3. Gmail integration (send meal plan email)
4. Video demo recording
5. Final project report and presentation
