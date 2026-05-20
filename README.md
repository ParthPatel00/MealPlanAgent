# MealPlanAgent

An AI-powered weekly meal planning agent built for CMPE 258. Given your time, dietary, and allergy constraints, it produces a weekly meal plan with recipe citations, a categorized grocery list with budget estimates, nutritional summaries, and a PDF export.

The system learns your preferences over time through a stateful memory system that writes, summarizes, and retrieves user patterns across sessions.

---

## Architecture

```
User Constraints + Memory Context
      |
  [Planner]  LLM generates a structured plan (with few-shot examples)
      |
  [Executor] Dispatches tool calls in sequence, logs each one
   |  |  |  |  |
   |  |  |  |  +-- budget_estimator  -> estimated grocery cost
   |  |  |  +----- grocery_list      -> categorized ingredient list
   |  |  +-------- nutrition         -> per-recipe + plan-wide totals
   |  +----------- allergy_checker   -> local match + Open Food Facts API
   +-------------- recipe_search     -> Hybrid RAG (BM25 + vector + graph)
      |
  [Critic]   Rule-based verification of allergy safety, citations, duplicates
      |       Triggers up to 2 re-planning retries if issues found
      |
  [Memory]   Writes session to SQLite, summarizes patterns via LLM
      |
  [Output]   Next.js web UI (6 tabs)
```

### Key Technical Features

- **3-Way Hybrid RAG**: BM25 keyword search + ChromaDB vector similarity + NetworkX knowledge graph re-ranking, fused via Reciprocal Rank Fusion (k=60)
- **Stateful Memory**: SQLite-backed write/summarize/retrieve system that learns user preferences across sessions
- **Few-Shot Prompting**: 3 curated examples injected into the planner prompt for consistent JSON output
- **Planner-Executor-Critic Loop**: Up to 2 retry cycles with fix instructions fed back to the planner
- **5 Tools**: recipe_search, allergy_checker, nutrition, grocery_list, budget_estimator
- **Multi-Model Support**: 4 models across 3 providers (Ollama local, Groq cloud, Google Gemini)

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

### 3. Evaluation Data
- **60 structured test cases** (`data/eval/test_cases.json`): cover diverse dietary constraints, allergens, and meal counts
- **205 dynamic scenarios** (`data/eval/dynamic_test_cases.json`): programmatically generated edge cases
- **330 retrieval ground truth queries** (`data/eval/retrieval_ground_truth.json`): 150 tag, 100 ingredient, 50 KG-graded, 30 natural language

---

## Models Compared

| Name | Provider | Type | Notes |
|---|---|---|---|
| Gemini 2.5 Flash Lite | Google AI Studio | Closed-source | Free tier, fast |
| Llama 3.3 70B | Groq | Open-source | Free tier, highest quality |
| Llama 3.2 3B | Ollama (local) | Open-source | Runs fully offline, no API key |
| Granite 3.1 2B | Ollama (local) | Open-source | Backup model, smallest/fastest local |

All models are evaluated on the same structured test set. Full results including cost and latency tradeoffs are reported in the project report.

---

## Evaluation Results

### Pipeline Evaluation (End-to-End)

| Model | Cases | Constraint Pass | Allergy Violation | Citation Pass | Tool Success | Avg Latency |
|---|---|---|---|---|---|---|
| Llama 3.3 70B (Groq) | 23/60 | 100% | 0% | 100% | 100% | 12,782 ms |
| Llama 3.2 3B (Ollama) | 60/60 | 100% | 0.9% | 100% | 100% | 30,871 ms |
| Granite 3.1 2B (Ollama) | 60/60 | 100% | 1.0% | 100% | 100% | 25,897 ms |
| Gemini 2.5 Flash Lite | -- | -- | -- | -- | -- | -- |

Groq evaluations are limited by free-tier daily API quotas (100K tokens/day). Gemini evaluation could not be completed due to free-tier rate limits.

### Retrieval Evaluation (Ablation Study)

| Configuration | P@5 | P@10 | R@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| **Full Hybrid** (BM25 + Vector + KG) | **0.655** | **0.568** | **0.338** | **0.688** | **0.438** |
| BM25 + KG | 0.580 | 0.550 | 0.323 | 0.673 | 0.370 |
| BM25 Only | 0.575 | 0.535 | 0.314 | 0.648 | 0.361 |
| Vector + BM25 | 0.365 | 0.415 | 0.241 | 0.502 | 0.234 |
| Vector + KG | 0.365 | 0.308 | 0.092 | 0.466 | 0.189 |
| Vector Only | 0.255 | 0.233 | 0.075 | 0.373 | 0.154 |

Statistical significance confirmed via paired t-test, Wilcoxon signed-rank test, bootstrap confidence intervals, and Cohen's d effect sizes. Full results in `data/eval/statistical_analysis.json`.

### Test Suite

146 unit and integration tests covering all components. Run with:

```bash
pytest tests/ -v
```

---

## Setup

See [report/Setup.md](report/Setup.md) for full setup instructions.

Quick start:

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
cp .env.example .env
python -m src.data.loader
python -m src.rag.indexer
python -m src.rag.graph_builder

# Terminal 1: Backend
uvicorn server:app --reload --port 8001

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open http://localhost:3000.

---

## Project Structure

```
MealPlanAgent/
├── README.md
├── requirements.txt
├── .env.example
├── server.py                      FastAPI backend
├── frontend/                      Next.js web UI (6 tabs)
│   ├── app/page.tsx               Main application page
│   └── package.json
├── src/                           Core library
│   ├── agent/                     Planner-Executor-Critic pipeline
│   │   ├── planner.py             LLM planner with few-shot prompting
│   │   ├── executor.py            5-tool dispatch engine
│   │   ├── critic.py              Rule-based verification
│   │   ├── pipeline.py            Orchestrator with memory integration
│   │   ├── json_utils.py          Robust JSON extraction from LLM output
│   │   └── few_shot_examples.py   Curated few-shot examples
│   ├── rag/                       Hybrid retrieval system
│   │   ├── indexer.py             ChromaDB vector index builder
│   │   ├── retriever.py           BM25 + Vector + KG fusion
│   │   ├── knowledge_graph.py     NetworkX recipe-ingredient-tag graph
│   │   └── graph_builder.py       Knowledge graph builder
│   ├── tools/                     Deterministic tool implementations
│   │   ├── recipe_search.py       RAG-powered recipe lookup
│   │   ├── allergy_checker.py     Local + Open Food Facts allergy check
│   │   ├── nutrition.py           Nutrition value conversion
│   │   ├── grocery_list.py        Ingredient aggregation + categorization
│   │   ├── budget_estimator.py    Grocery cost estimation
│   │   └── pdf_export.py          PDF meal plan export
│   ├── evaluation/                Evaluation and analysis
│   │   ├── ablation_study.py      6-config retrieval ablation
│   │   ├── retrieval_eval.py      Retrieval quality evaluation
│   │   ├── retrieval_metrics.py   IR metrics (P@k, MRR, NDCG, MAP)
│   │   ├── statistical_analysis.py Statistical significance testing
│   │   ├── hyperparam_study.py    Hyperparameter sensitivity sweeps
│   │   ├── visualization.py       Chart generation for report
│   │   └── ...                    Evaluator, comparison, ground truth
│   ├── memory/                    Stateful memory system
│   │   ├── store.py               SQLite-backed persistence
│   │   ├── summarizer.py          LLM-based history summarization
│   │   └── feedback.py            User feedback collection
│   ├── models/
│   │   └── client.py              Unified LLM client (Ollama/Groq/Gemini)
│   └── data/
│       ├── loader.py              Food.com CSV loader + cleaner
│       └── preprocessor.py        Document preprocessing
├── tests/                         146 pytest tests
├── data/
│   ├── sample_recipes.csv         200-row sample dataset
│   └── eval/                      Test cases, results, ablation data
├── scripts/                       Report and evaluation generation
│   ├── run_evaluation.py          Master evaluation pipeline
│   ├── generate_report.py         PDF report generator
│   ├── create_presentation.py     PowerPoint slide generator
│   └── generate_graph_viz.py      Knowledge graph visualizations
└── report/                        Documentation and deliverables
    ├── report.tex                 LaTeX project report
    ├── report.pdf                 Compiled report (10 pages)
    ├── charts/                    Evaluation charts (17 PNGs)
    ├── Setup.md                   Full setup guide
    ├── Presentation.pptx          12-slide presentation
    ├── OVERVIEW.md                Detailed system overview
    └── POWERPOINT.md              Presentation speaking notes
```

---

## Report

The project report is available at `report/report.pdf`. To recompile:

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```
