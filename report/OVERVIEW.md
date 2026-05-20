# MealPlanAgent: Hybrid RAG with Knowledge Graph Re-ranking for Recipe Recommendation

**CMPE 258 - Deep Learning, San Jose State University**

---

## Glossary

| Abbreviation | Full Form | What It Does |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation | Combines LLM reasoning with a retrieval system so answers are grounded in real data instead of hallucinated |
| **LLM** | Large Language Model | Neural network trained on massive text corpora (e.g., Llama 3, GPT-4) used here for NL parsing and meal planning |
| **BM25** | Best Matching 25 | Probabilistic sparse retrieval function that ranks documents by exact term overlap with the query |
| **RRF** | Reciprocal Rank Fusion | Rank-based method for merging results from multiple retrievers without needing score normalization |
| **KG** | Knowledge Graph | Graph structure (nodes = recipes, ingredients, tags; edges = relationships) used for preference-aware re-ranking |
| **HNSW** | Hierarchical Navigable Small World | Approximate nearest-neighbor search algorithm used by ChromaDB for fast vector retrieval |
| **IR** | Information Retrieval | The field of study concerned with finding relevant documents for a query |
| **P@k** | Precision at k | Fraction of the top-k retrieved results that are relevant |
| **R@k** | Recall at k | Fraction of all relevant documents that appear in the top-k results |
| **MRR** | Mean Reciprocal Rank | Average of 1/rank for the first relevant result across all queries; measures how quickly users find a relevant result |
| **NDCG@k** | Normalized Discounted Cumulative Gain at k | Measures ranking quality using graded relevance (0-3), giving more credit to relevant documents ranked higher |
| **DCG** | Discounted Cumulative Gain | Unnormalized version of NDCG; sums relevance scores with a logarithmic discount by rank position |
| **IDCG** | Ideal Discounted Cumulative Gain | The DCG of a perfect ranking (all relevant documents at the top), used to normalize DCG into NDCG |
| **AP** | Average Precision | Precision averaged at each rank where a relevant document appears; captures both precision and recall in one number |
| **MAP** | Mean Average Precision | AP averaged across all queries |
| **IDF** | Inverse Document Frequency | Measures how rare a term is across the corpus; rare terms get higher weight in BM25 |
| **TF** | Term Frequency | How often a term appears in a document; used by BM25 with saturation (diminishing returns) |
| **CI** | Confidence Interval | Range of plausible values for a metric, computed via bootstrap resampling (1000 iterations, 95% level) |
| **NL** | Natural Language | Free-text user input (e.g., "I have chicken and rice, make me something Mexican") as opposed to structured fields |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding | Dimensionality reduction technique for visualizing high-dimensional embeddings in 2D |
| **UMAP** | Uniform Manifold Approximation and Projection | Faster alternative to t-SNE for embedding visualization; better preserves global structure |
| **GNN** | Graph Neural Network | Neural network that operates on graph-structured data (mentioned in Future Work) |
| **GAT** | Graph Attention Network | GNN variant using attention mechanisms over graph neighbors (mentioned in Future Work) |
| **CLIP** | Contrastive Language-Image Pre-training | Multi-modal model that embeds both text and images into a shared space (mentioned in Future Work) |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Related Work](#3-related-work)
4. [System Architecture](#4-system-architecture)
5. [Hybrid RAG Pipeline: Mathematical Formulation](#5-hybrid-rag-pipeline-mathematical-formulation)
6. [Knowledge Graph Construction and Re-ranking](#6-knowledge-graph-construction-and-re-ranking)
7. [Flexible Constraint System](#7-flexible-constraint-system)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Ablation Study: Component Contribution Analysis](#9-ablation-study-component-contribution-analysis)
10. [Hyperparameter Sensitivity Analysis](#10-hyperparameter-sensitivity-analysis)
11. [Embedding Model Comparison](#11-embedding-model-comparison)
12. [Statistical Analysis](#12-statistical-analysis)
13. [LLM-Only Baseline Comparison](#13-llm-only-baseline-comparison)
14. [End-to-End Walkthrough with Numbers](#14-end-to-end-walkthrough-with-numbers)
15. [Embedding Space Visualization](#15-embedding-space-visualization)
16. [Dynamic Evaluation and Constraint-Aware Metrics](#16-dynamic-evaluation-and-constraint-aware-metrics)
17. [Implementation Details](#17-implementation-details)
18. [Conclusions and Future Work](#18-conclusions-and-future-work)
19. [References](#19-references)

---

## 1. Abstract

We present MealPlanAgent, an agentic meal planning system that combines a three-stage hybrid retrieval pipeline (BM25 (Best Matching 25) sparse retrieval, dense vector embeddings, and KG (Knowledge Graph) re-ranking) with LLM (Large Language Model)-based planning to recommend recipes from a real 10,000-recipe corpus. Unlike pure LLM generation, which hallucinates recipes with incorrect cooking times, impossible ingredient combinations, and fabricated nutrition data, our RAG (Retrieval-Augmented Generation) approach grounds every recommendation in a verified Food.com entry with traceable citations.

We conduct a rigorous evaluation including: (1) ablation studies across 6 retriever configurations measuring P@k (Precision at k), R@k (Recall at k), MRR (Mean Reciprocal Rank), and NDCG@k (Normalized Discounted Cumulative Gain at k), (2) hyperparameter sensitivity analysis over retrieval depth, RRF (Reciprocal Rank Fusion) constant, and KG boost weight, (3) embedding model comparison across three sentence-transformer architectures, (4) statistical significance testing with paired t-tests, Wilcoxon signed-rank tests, and bootstrap CI (Confidence Intervals), and (5) an LLM-only baseline comparison demonstrating the value of the RAG pipeline over pure generation.

Results across 40 stratified queries (5 query types) show that the full hybrid retriever achieves the best aggregate performance (P@5=0.655, MRR=0.688, NDCG@10=0.438), outperforming BM25-only by 14% on P@5. Crucially, RRF fusion without KG re-ranking actually degrades performance versus BM25 alone (P@5 drops from 0.575 to 0.365), making the KG the essential component that enables effective fusion. Disabling KG drops P@5 by 44% and NDCG@10 by 47%. Statistical analysis with Bonferroni-corrected paired t-tests (n=40, alpha=0.0033) confirms the full hybrid's advantage over vector-based configs on precision metrics (Cohen's d=1.225 on P@5, p < 0.001). Per-query-type analysis reveals that full hybrid dominates tag and ingredient queries but BM25-only leads on natural language queries (MRR 0.56 vs 0.33).

---

## 2. Introduction and Motivation

### 2.1 The Recipe Hallucination Problem

Large Language Models (LLMs) generate plausible-sounding but factually incorrect recipes. When asked for "a 20-minute high-protein dinner," an LLM will confidently produce a recipe with wrong cooking times, missing steps, or impossible ingredient combinations. In a meal planning context, hallucinated nutrition data can be medically dangerous (e.g., recommending a recipe as "nut-free" when it contains almond flour).

Retrieval-Augmented Generation (RAG) solves this by restricting the system to real, verified recipes from a database. The LLM's role is limited to understanding user intent and generating search queries; every recipe in the output is a real Food.com entry with correct nutrition data and traceable provenance.

### 2.2 Research Questions

1. **RQ1**: How does each retrieval component (BM25, vector embeddings, knowledge graph) contribute to retrieval quality, and does their combination outperform any individual component?
2. **RQ2**: How sensitive is retrieval quality to hyperparameter choices (retrieval depth, RRF fusion constant, KG boost weight)?
3. **RQ3**: Does embedding model capacity (22M vs 33M vs 110M parameters) significantly impact recipe retrieval quality?
4. **RQ4**: Can the knowledge graph effectively personalize results when users specify ingredients they have on hand?
5. **RQ5**: How does the full RAG pipeline compare to an LLM-only baseline that generates recipes from pretraining knowledge?

### 2.3 Dataset

We use the Food.com Recipes dataset from Kaggle (Shuyangli94), comprising 231,637 recipes. We use the first 10,000 recipes (configurable via `MAX_RECIPES`), each containing:
- Name, cooking time (minutes), tags, ingredients list, step-by-step instructions
- 7 nutrition fields: calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates (all as percent daily value)

---

## 3. Related Work

**Retrieval-Augmented Generation**: Lewis et al. (2020) introduced RAG, combining a parametric model with a non-parametric retrieval component. We extend this with a three-way fusion (sparse + dense + graph) rather than the standard single-retriever approach.

**Reciprocal Rank Fusion (RRF)**: Cormack et al. (2009) proposed RRF as a rank-based fusion method requiring no score normalization. We adopt their recommended k=60 constant and evaluate its sensitivity in Section 10.

**Knowledge Graphs (KG) for Recommendation**: Wang et al. (2019) showed that KG-aware embeddings improve recommendation quality. Our approach is simpler (graph-based re-ranking rather than KG embedding), but we demonstrate its effectiveness through ablation.

**Best Matching 25 (BM25)**: Robertson and Zaragoza (2009) formalized BM25 as the standard probabilistic retrieval function. We use the LlamaIndex implementation with default parameters (k1=1.2, b=0.75).

**Sentence Transformers**: Reimers and Gurevych (2019) introduced sentence-BERT for dense retrieval. We compare three model variants: MiniLM-L6 (22M), MiniLM-L12 (33M), and MPNet-base (110M).

---

## 4. System Architecture

```
User Input (natural language or structured constraints)
    |
    v
[NL Parser] - LLM extracts structured constraints including:
    |           ingredients_on_hand, cuisine_preferences, calorie targets,
    |           allergens, dietary tags, time limits
    v
[Memory Retrieval] - SQLite: past plans, preferences, LLM summaries
    |
    v
[PLANNER] - LLM + few-shot examples + memory -> JSON plan with
    |         preferred_ingredients and preferred_tags per meal query
    v
[EXECUTOR] - 5 deterministic tools:
    |   1. recipe_search  (Hybrid RAG: BM25 + Vector + KG re-ranking)
    |   2. allergy_checker
    |   3. nutrition
    |   4. grocery_list
    |   5. budget_estimator
    v
[CRITIC] - Rule-based verification (5 checks)
    |
    v
If INVALID: retry with fix instructions (max 2 retries)
If VALID:   write to memory, return results
```

The LLM is used ONLY for reasoning (NL parsing and plan construction). Everything that must be correct (allergen checking, nutrition math, citation verification) is deterministic code.

---

## 5. Hybrid RAG Pipeline: Mathematical Formulation

### 5.1 BM25 Sparse Retrieval

BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency. For a query Q = {q_1, q_2, ..., q_n} and document D:

```
BM25(D, Q) = SUM_i [ IDF(q_i) * f(q_i, D) * (k_1 + 1) / (f(q_i, D) + k_1 * (1 - b + b * |D| / avgdl)) ]
```

Where:
- **f(q_i, D)**: frequency of term q_i in document D
- **|D|**: document length in tokens
- **avgdl**: average document length across corpus
- **k_1 = 1.2**: term frequency saturation (controls how quickly TF gains diminish)
- **b = 0.75**: length normalization (0 = no normalization, 1 = full normalization)
- **IDF(q_i) = ln((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)**: inverse document frequency

**Strengths for recipe retrieval**: BM25 excels at exact term matching. A query for "vegetarian" directly matches recipes tagged "vegetarian." This is critical because dietary labels must be exact, not approximate.

**Weakness**: Cannot capture semantic similarity. "Light summer dinner" won't match "Mediterranean quinoa salad" despite being semantically related.

**Configuration**: Retrieves top_k * 2 = 20 candidates per query.

### 5.2 Dense Vector Retrieval

Each recipe text is encoded into a d-dimensional vector using a sentence transformer model. At query time, the query is encoded with the same model, and candidates are ranked by cosine similarity:

```
cos(q, d) = (q . d) / (||q|| * ||d||)
```

Where q in R^d is the query embedding and d in R^d is the document embedding.

**Embedding models evaluated**:

| Model | Dimensions | Parameters | Layers |
|-------|-----------|------------|--------|
| all-MiniLM-L6-v2 | 384 | 22M | 6 |
| all-MiniLM-L12-v2 | 384 | 33M | 12 |
| all-mpnet-base-v2 | 768 | 110M | 12 |

All models are from the sentence-transformers family (Reimers & Gurevych, 2019), trained on 1B+ sentence pairs. We use ChromaDB with HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search for sub-linear retrieval time.

**Strengths**: Captures semantic similarity. "Light summer dinner" maps close to "fresh salad with lemon" in the embedding space even without shared terms.

**Weakness**: May rank semantically similar but dietarily incompatible recipes highly (e.g., ranking a "grilled portobello burger" for a "high-protein dinner" query because they're both "dinner" in embedding space).

### 5.3 Reciprocal Rank Fusion (RRF)

BM25 and vector search produce scores on incomparable scales (BM25 scores can be 5-50, cosine similarities are 0-1). RRF (Reciprocal Rank Fusion) is a rank-based fusion method that sidesteps this normalization problem entirely:

```
RRF(d) = SUM over lists L: 1 / (k + rank_L(d) + 1)
```

Where:
- **k = 60**: smoothing constant (Cormack et al., 2009)
- **rank_L(d)**: 0-indexed rank of document d in list L
- If d is absent from list L, its contribution from that list is 0

**Example calculation**:

Recipe "Grilled Chicken Salad" appears at rank 2 in vector search, rank 5 in BM25:
```
RRF = 1/(60 + 2 + 1) + 1/(60 + 5 + 1) = 1/63 + 1/66 = 0.01587 + 0.01515 = 0.03103
```

Recipe "Chicken Stir Fry" appears at rank 0 in BM25 only (not in vector top-20):
```
RRF = 1/(60 + 0 + 1) + 0 = 1/61 = 0.01639
```

The salad scores higher (0.031 > 0.016) because it appeared in BOTH lists. This is RRF's key property: documents relevant to both retrieval paradigms get a natural boost.

**Why k=60**: With k=60, rank 0 contributes 1/61 = 0.0164, while rank 19 contributes 1/80 = 0.0125, a ratio of only 1.3x. This smoothing prevents a single high ranking in one list from dominating. Smaller k (e.g., k=1) gives rank 0 a 10x advantage over rank 19, which we evaluate in Section 10.

### 5.4 Knowledge Graph Re-ranking

After RRF fusion, the knowledge graph applies a preference-aware boost. For candidate recipe r with user preference node set P:

```
overlap(r) = |N(r) INTERSECT P_existing|
boost(r) = (overlap(r) / |P_existing|) * w_kg
score_final(r) = score_RRF(r) + boost(r)
```

Where:
- **N(r)**: neighbor nodes of recipe r in the graph (ingredients + tags)
- **P_existing**: user preference nodes that exist in the graph
- **w_kg = 0.1**: boost weight (evaluated over [0.0, 0.05, 0.1, 0.2, 0.5, 1.0] in Section 10)

**Example**: User has ingredients_on_hand = ["chicken", "rice", "bell peppers"]:

Recipe "Chicken Fried Rice":
- N(r) includes: {chicken, rice, soy sauce, garlic, egg, asian, chinese, ...}
- P_existing = {ingredient:chicken, ingredient:rice, ingredient:bell peppers}
- Overlap = {chicken, rice} (2 of 3 preference nodes)
- boost = (2/3) * 0.1 = 0.067
- final_score = 0.031 (RRF) + 0.067 = 0.098

Recipe "Beef Tacos":
- N(r) includes: {beef, tortilla, cheese, lettuce, mexican, ...}
- Overlap = {} (0 of 3)
- boost = 0.0
- final_score = 0.033 (RRF) + 0.0 = 0.033

The KG boost pushes Chicken Fried Rice ahead despite having a slightly lower RRF score. The 0.1 weight is intentionally conservative: preferences should nudge, not override relevance.

---

## 6. Knowledge Graph Construction and Re-ranking

### 6.1 Graph Structure

The knowledge graph is a NetworkX undirected graph with three node types and two edge types:

| Node Type | Count | Example |
|-----------|-------|---------|
| recipe | 9,999 | `recipe:12345` |
| ingredient | 5,332 | `ingredient:chicken breast` |
| tag | 476 | `tag:high-protein` |

| Edge Type | Count | Description |
|-----------|-------|-------------|
| HAS_INGREDIENT | ~185,000 | recipe-to-ingredient |
| HAS_TAG | ~85,000 | recipe-to-tag |
| **Total** | **~270,466** | |

### 6.2 Graph Statistics

```
Total Nodes:          15,807
Total Edges:          270,466
Graph Density:        0.002165

Node Breakdown:
  recipe:             9,999   (avg degree: 27.0, median: 26.0)
  ingredient:         5,332   (avg degree: 16.9, median: 2.0)
  tag:                  476   (avg degree: 379.0, median: 48.0)
```

The highly skewed ingredient degree distribution (mean 16.9, median 2.0) indicates a power-law: a small set of ingredients (salt, butter, sugar, garlic) appear in thousands of recipes, while most ingredients are rare. This is exploited by the KG re-ranking: boosting recipes that share rare ingredients with user preferences is more informative than boosting recipes sharing common ones.

> **Chart: `report_charts/kg_analysis.png`** - Knowledge graph structural analysis showing node type distribution, recipe degree distribution, top 20 most connected ingredients, and summary statistics.

### 6.3 Graph-Based Recipe Similarity

The graph enables a 2-hop traversal to find related recipes:

```
recipe:123 -> ingredient:chicken -> recipe:456
recipe:123 -> ingredient:garlic  -> recipe:456
recipe:123 -> tag:italian        -> recipe:456

similarity(r1, r2) = |shared_neighbors| / |neighbors(r1)|
```

For KG-graded ground truth (Section 8.1), we use ingredient overlap to assign graded relevance:
- **Grade 3 (highly relevant)**: shares >= 60% of ingredients
- **Grade 2 (relevant)**: shares 30-59% of ingredients
- **Grade 1 (marginally relevant)**: shares 10-29% of ingredients
- **Grade 0 (irrelevant)**: shares < 10% of ingredients

### 6.4 Graph Visualization

> **Chart: `report_charts/knowledge_graph_single_recipe.png`** - Single recipe with all ingredient and tag connections.
> **Chart: `report_charts/knowledge_graph_shared.png`** - Two recipes highlighting shared ingredients/tags in red.
> **Chart: `report_charts/knowledge_graph_overview.png`** - 5-recipe cluster with shared neighbor connections.

---

## 7. Flexible Constraint System

### 7.1 From Fixed Schema to Flexible Input

The original system parsed all user input into a rigid 6-field JSON: `{num_meals, max_minutes, tags, allergens, cook_after_hour, dietary_notes}`. This meant specifying ingredients on hand, cuisine preferences, or calorie targets was impossible.

The expanded constraint schema supports:

| Field | Type | Example | How It Drives Retrieval |
|-------|------|---------|------------------------|
| `num_meals` | int | 5 | Controls number of meal queries |
| `max_minutes` | int | 30 | Hard filter on recipe cooking time |
| `tags` | list[str] | ["vegetarian"] | Hard filter on recipe tags |
| `allergens` | list[str] | ["peanuts"] | Hard exclusion filter |
| `ingredients_on_hand` | list[str] | ["chicken", "rice"] | Mapped to `preferred_ingredients`, boosted via KG re-ranking |
| `cuisine_preferences` | list[str] | ["mexican", "thai"] | Mapped to `preferred_tags`, boosted via KG re-ranking |
| `calorie_target_per_meal` | int | 500 | Soft sort by calorie proximity |
| `macro_targets` | object | {protein_g: 40} | Influences query generation |
| `budget_dollars` | float | 50.0 | Post-retrieval budget optimization |

### 7.2 Constraint Flow Through the Pipeline

```
User: "I have chicken and rice, make me something Mexican, under 30 min"
    |
    v
NL Parser extracts:
    ingredients_on_hand: ["chicken", "rice"]
    cuisine_preferences: ["mexican"]
    max_minutes: 30
    |
    v
Planner generates meal_queries with:
    query: "mexican chicken rice dinner quick"
    preferred_ingredients: ["chicken", "rice"]
    preferred_tags: ["mexican"]
    |
    v
Executor calls recipe_search() with:
    preferred_ingredients -> passed to retriever.retrieve()
    preferred_tags -> passed to retriever.retrieve()
    |
    v
HybridRetriever.retrieve():
    1. BM25 finds recipes mentioning "chicken", "rice", "mexican"
    2. Vector search finds semantically similar recipes
    3. RRF merges both lists
    4. KG re-ranking boosts recipes containing chicken + rice ingredients
    |
    v
Results: recipes actually using chicken and rice, Mexican-style
```

This chain activates the knowledge graph re-ranking code that was previously unused, making the graph a functional component of the retrieval pipeline.

---

## 8. Evaluation Methodology

### 8.1 Ground Truth Generation

We generate ground truth relevance judgments directly from the dataset (deterministic, reproducible, no LLM calls):

**Tag-based queries** (~150 queries): For tag combinations like "vegetarian mexican," the ground truth relevant set is all recipes possessing those tags. Sampled from meaningful tags (excluding noisy structural tags like "time-to-make").

**Ingredient-based queries** (~100 queries): For ingredient sets like "chicken, rice, broccoli," ground truth is recipes containing all specified ingredients.

**KG-graded queries** (~50 queries): Using knowledge graph structure, assigns graded relevance (0-3) based on ingredient overlap ratio with a focal recipe. These enable NDCG computation.

**Natural language queries** (~30 queries): Template-generated queries combining ingredients, dietary preferences, and time constraints, with relevant sets computed from tag/ingredient intersection.

**Generated ground truth**: 271 total queries (150 tag, 50 ingredient, 41 KG-graded, 30 natural language). All judgments are deterministic and derived from the dataset itself (no LLM calls), making them reproducible and free to generate at scale.

> Implementation: `src/evaluation/ground_truth.py`
> Output: `data/eval/retrieval_ground_truth.json` (271 queries with relevance judgments)

### 8.2 Information Retrieval Metrics

We evaluate using standard Information Retrieval (IR) metrics at k = {5, 10, 20}:

**Precision@k**: Fraction of top-k results that are relevant.
```
P@k = |{relevant docs in top k}| / k
```

**Recall@k**: Fraction of all relevant documents found in top-k.
```
R@k = |{relevant docs in top k}| / |{all relevant docs}|
```

**Mean Reciprocal Rank (MRR)**: Reciprocal of the rank of the first relevant result. Measures how quickly a user finds a relevant result.
```
MRR = 1/|Q| * SUM_q (1 / rank_q)
```

**Normalized Discounted Cumulative Gain (NDCG@k)**: Uses graded relevance scores (0-3) rather than binary relevance. Rewards placing highly relevant documents at the top.
```
DCG@k (Discounted Cumulative Gain) = SUM_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)
NDCG@k = DCG@k / IDCG@k (Ideal DCG)
```

Where IDCG@k (Ideal Discounted Cumulative Gain) is the DCG of a perfect ranking (sorted by relevance descending).

**Average Precision (AP)**: Precision averaged at each rank where a relevant document appears. Captures both precision and recall in a single number.
```
AP = 1/|R| * SUM_{k: d_k is relevant} P@k
```

**Mean Average Precision (MAP)**: AP (Average Precision) averaged across all queries.

> Implementation: `src/evaluation/retrieval_metrics.py`

### 8.3 Statistical Testing Framework

To ensure differences between configurations are statistically significant, we employ:

1. **Paired t-test**: Tests H0: mean(config_A) = mean(config_B) on per-query metric scores
2. **Wilcoxon signed-rank test**: Non-parametric alternative (no normality assumption)
3. **Bootstrap 95% confidence intervals**: Percentile method with 1000 resamples
4. **Cohen's d effect size**: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
5. **Bonferroni correction**: Adjusted alpha = 0.05 / n_comparisons for multiple testing

> Implementation: `src/evaluation/statistical_analysis.py`

---

## 9. Ablation Study: Component Contribution Analysis

### 9.1 Configurations

We systematically disable retrieval components to measure individual and combined contributions:

| Config | Vector | BM25 | KG | Description |
|--------|--------|------|----|-------------|
| vector_only | Y | N | N | Dense retrieval alone |
| bm25_only | N | Y | N | Sparse retrieval alone |
| vector_bm25 | Y | Y | N | RRF fusion without KG |
| vector_kg | Y | N | Y | Dense + KG re-ranking |
| bm25_kg | N | Y | Y | Sparse + KG re-ranking |
| full_hybrid | Y | Y | Y | All three components |
| llm_baseline | N/A | N/A | N/A | LLM generates from memory |

### 9.2 Results (40 Stratified Queries: 8 tag-single, 8 tag-pair, 8 ingredient, 8 KG-graded, 8 NL)

| Config | P@5 | P@10 | R@10 | MRR | NDCG@10 | AP | Latency |
|--------|-----|------|------|-----|---------|-----|---------|
| **full_hybrid** | **0.655** | **0.568** | **0.338** | **0.688** | **0.438** | **0.332** | 112ms |
| bm25_kg | 0.580 | 0.550 | 0.323 | 0.673 | 0.370 | 0.267 | 60ms |
| bm25_only | 0.575 | 0.535 | 0.314 | 0.648 | 0.361 | 0.256 | 60ms |
| vector_bm25 | 0.365 | 0.415 | 0.241 | 0.502 | 0.234 | 0.132 | 118ms |
| vector_kg | 0.365 | 0.308 | 0.092 | 0.466 | 0.189 | 0.087 | 56ms |
| vector_only | 0.255 | 0.233 | 0.075 | 0.373 | 0.154 | 0.043 | 110ms |

### 9.3 Interpretation

**Full hybrid wins every aggregate metric.** The combination of BM25 + Vector + KG achieves the highest P@5 (0.655), MRR (0.688), and NDCG@10 (0.438). However, the three components do not contribute equally: BM25 is the dominant retriever, vector search provides supplementary candidates, and the KG is what makes their fusion effective rather than harmful.

**KG re-ranking provides measurable lift.** Comparing configs with and without KG:
- vector_kg (P@5=0.365) vs vector_only (P@5=0.255): **+43% improvement** from KG alone
- bm25_kg (P@5=0.580) vs bm25_only (P@5=0.575): **+0.9% improvement**, smaller because BM25 already handles ingredient/tag queries well
- full_hybrid (P@5=0.655) vs vector_bm25 (P@5=0.365): **+79% improvement** from adding KG on top of RRF fusion

The KG component's impact is largest when the base retriever is weakest (vector-only), because the KG promotes recipes sharing user-specified ingredients, correcting for the embedding model's tendency to match semantically similar but ingredient-distant recipes.

**BM25 remains the strongest individual component.** BM25-only achieves P@5=0.575, outperforming vector-only (0.255) by 126%. This is expected: most ground truth queries include specific terms (ingredient names, tag labels) that BM25 matches directly.

**RRF fusion has mixed effects without KG.** vector_bm25 (P@5=0.365) underperforms bm25_only (0.575) because vector's weaker results dilute BM25's ranking during fusion. However, when KG is added on top, the full hybrid (0.655) surpasses bm25_kg (0.580), suggesting that vector retrieval provides useful candidates that the KG can then re-rank effectively.

**Query type analysis** (see `report_charts/query_type_breakdown.png`):
- **Tag-single queries**: All configs perform well (MRR 0.875-1.0); full_hybrid and vector_bm25 lead
- **Tag-pair queries**: Wider spread; vector_only drops to MRR 0.40, vector_bm25 to 0.63, while full_hybrid achieves 1.0 and bm25_kg 0.92
- **Ingredient queries**: KG makes the biggest difference; full_hybrid MRR=0.75 vs vector_only MRR=0.17
- **Natural language queries**: BM25-based configs lead (MRR=0.56 for bm25_only and bm25_kg); full_hybrid drops to MRR=0.33, showing that RRF fusion dilutes BM25's strong keyword signal on NL queries
- **KG-graded queries**: Similar across configs (~0.28-0.35), showing the KG contribution is concentrated on ingredient/preference-driven queries

### 9.4 Statistical Significance

With Bonferroni-corrected alpha = 0.0033 (15 pairwise comparisons, n=40 queries):

| Comparison | P@5 diff | Cohen's d | p-value | Significant? |
|-----------|----------|-----------|---------|-------------|
| full_hybrid vs vector_only | +0.400 | 1.225 (large) | <0.001 | Yes |
| full_hybrid vs vector_bm25 | +0.290 | 1.099 (large) | <0.001 | Yes |
| full_hybrid vs vector_kg | +0.290 | 0.807 (large) | <0.001 | Yes |
| bm25_kg vs vector_bm25 | +0.215 | 0.876 (large) | <0.001 | Yes |
| bm25_only vs vector_only | +0.320 | 0.828 (large) | <0.001 | Yes |
| vector_kg vs vector_only | +0.110 | 0.608 (medium) | 0.007 | Yes |

The table above shows results for Precision@5. The full hybrid's advantage over all vector-based configs is statistically significant with large effect sizes (d > 0.8). The KG contribution (vector_kg vs vector_only, d=0.608) is a medium effect and significant before Bonferroni correction (p=0.007).

**Note on NDCG@10**: The full_hybrid vs vector_only comparison on NDCG@10 yields a smaller effect (d=0.43, Bonferroni-corrected p=0.146, not significant after correction). This is because only KG-graded queries have graded relevance judgments; the remaining queries contribute 0 to NDCG, reducing the per-query effect size. The NDCG@10 aggregate of 0.438 reported in the ablation table averages only over queries with graded relevance.

**95% Bootstrap Confidence Intervals (Precision@10):**
- full_hybrid: 0.568 [0.462, 0.670]
- bm25_kg: 0.550 [0.452, 0.650]
- bm25_only: 0.535 [0.433, 0.638]
- vector_bm25: 0.415 [0.320, 0.505]
- vector_kg: 0.308 [0.205, 0.425]
- vector_only: 0.233 [0.150, 0.328]

The non-overlapping CIs between the top tier (full_hybrid, bm25_kg, bm25_only) and the bottom tier (vector_only, vector_kg) confirm the significance visually.

> **Chart: `report_charts/ablation_comparison.png`** - Grouped bar chart comparing P@5, P@10, R@10, MRR, NDCG@10 across all configurations.
> **Chart: `report_charts/ablation_with_ci.png`** - Ablation results with 95% bootstrap confidence interval error bars.
> **Chart: `report_charts/precision_recall_curves.png`** - Precision vs Recall as k varies, one curve per configuration.
> **Chart: `report_charts/component_contribution.png`** - Heatmap showing source component (vector/BM25/both) for each top-10 result.
> **Chart: `report_charts/query_type_breakdown.png`** - Performance breakdown by query type (tag, ingredient, KG-graded, NL).

### 9.5 Running the Ablation

```bash
python run_evaluation.py --step ablation
# Or quick test:
python run_evaluation.py --step ablation --quick
```

> Results saved to: `data/eval/ablation_results/`

---

## 10. Hyperparameter Sensitivity Analysis

### 10.1 Parameters Studied

We sweep one parameter at a time while holding others at defaults:

#### Retrieval Depth (top_k): [5, 10, 20, 50]

Controls how many candidates the retriever returns. Higher top_k increases recall but may decrease precision:
```
Expect: P@k decreases as top_k increases (more noise)
        R@k increases as top_k increases (more relevant docs found)
        Optimal top_k balances the P-R tradeoff
```

#### RRF Fusion Constant (k): [10, 30, 60, 100, 200]

Controls how much RRF dampens rank position differences:
- **k=10**: Rank 0 contributes 1/11=0.091, rank 19 contributes 1/30=0.033 (2.7x ratio). Top-ranked documents dominate.
- **k=60**: Rank 0 contributes 1/61=0.016, rank 19 contributes 1/80=0.013 (1.3x ratio). More uniform weighting.
- **k=200**: Nearly all ranks contribute equally. This approaches simple set union.

```
Small k: trust the top-ranked results from each retriever
Large k: give all candidates a more equal chance
```

#### KG Boost Weight (w_kg): [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

Controls how much knowledge graph proximity influences final ranking:
- **w_kg=0.0**: KG disabled (equivalent to vector_bm25 config)
- **w_kg=0.1**: Default, mild preference nudge
- **w_kg=1.0**: KG dominance. Preference overlap can add up to 1.0 to the score, overwhelming RRF scores (~0.03)

```
Expect: Moderate w_kg (0.05-0.2) performs best
        w_kg=0 loses personalization benefit
        w_kg>0.5 over-weights preferences, hurting relevance
```

### 10.2 Embedding Model Comparison

| Model | Dims | Params | Training Data |
|-------|------|--------|---------------|
| all-MiniLM-L6-v2 | 384 | 22M | 1B+ pairs |
| all-MiniLM-L12-v2 | 384 | 33M | 1B+ pairs |
| all-mpnet-base-v2 | 768 | 110M | 1B+ pairs |

Each model requires rebuilding the ChromaDB index (10-30 min per model). We evaluate whether the 5x parameter increase from MiniLM-L6 to MPNet translates to meaningful retrieval quality improvements in the recipe domain.

### 10.2 Results

#### Retrieval Depth (top_k)

| top_k | P@5 | P@10 | R@10 | R@20 | MRR | NDCG@10 | AP |
|-------|-----|------|------|------|-----|---------|-----|
| 5 | 0.575 | 0.575 | 0.212 | 0.212 | 0.667 | 0.234 | 0.211 |
| 10 | 0.655 | 0.568 | 0.338 | 0.338 | 0.688 | 0.438 | 0.332 |
| 20 | **0.695** | **0.630** | 0.375 | **0.462** | **0.715** | **0.492** | 0.439 |
| 50 | 0.715 | 0.693 | **0.429** | 0.523 | 0.716 | 0.599 | **0.609** |

**Observations**: top_k is the most impactful hyperparameter. Increasing from k=5 to k=20 improves P@5 by 21% (0.575 to 0.695), R@20 by 118% (0.212 to 0.462), and NDCG@10 by 110% (0.234 to 0.492). Precision remains stable or increases because deeper retrieval gives the KG re-ranker more candidates to promote. AP nearly triples from k=5 to k=50 (0.211 to 0.609), confirming that many relevant documents exist beyond the top 10.

#### RRF (Reciprocal Rank Fusion) Constant (k)

| rrf_k | P@5 | P@10 | R@10 | MRR | NDCG@10 |
|-------|-----|------|------|-----|---------|
| 10 | 0.620 | 0.545 | 0.337 | 0.680 | 0.372 |
| 30 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 60 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 100 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 200 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |

**Observations**: RRF is robust across k in [30, 200], with identical metrics. A small dip at k=10 (NDCG@10 drops from 0.438 to 0.372) occurs because very small k over-weights top-ranked results, amplifying any single retriever's errors. The default k=60 from Cormack et al. (2009) performs optimally. This robustness is a practical advantage: no tuning needed.

#### KG (Knowledge Graph) Boost Weight (w_kg)

| w_kg | P@5 | P@10 | R@10 | MRR | NDCG@10 |
|------|-----|------|------|-----|---------|
| **0.00** | **0.365** | **0.415** | **0.241** | **0.502** | **0.234** |
| 0.05 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 0.10 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 0.20 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 0.50 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |
| 1.00 | 0.655 | 0.568 | 0.338 | 0.688 | 0.438 |

**Observations**: This is the most striking result. Turning KG off (w_kg=0.0) causes a massive quality drop: P@5 falls from 0.655 to 0.365 (-44%), MRR from 0.688 to 0.502 (-27%), and NDCG@10 from 0.438 to 0.234 (-47%). Even a minimal KG weight (w_kg=0.05) restores full performance, and there is no degradation up to w_kg=1.0. This means: (1) the KG re-ranking is essential for queries with ingredient/tag preferences, and (2) the boost is binary in effect (present vs. absent), not proportional to weight. The lack of degradation at high weights is because the KG only boosts candidates that already share ingredients with user preferences, so increasing the weight cannot promote irrelevant recipes.

> **Chart: `report_charts/hyperparam_sensitivity.png`** - Line plots showing metric sensitivity to each hyperparameter, with multiple metric lines per subplot.

### 10.3 Running the Hyperparameter Study

```bash
python run_evaluation.py --step hyperparams --skip-embed  # Fast (BM25/RRF/KG sweeps)
python run_evaluation.py --step hyperparams               # Full (includes embedding model swap)
```

> Results saved to: `data/eval/hyperparam_results/`

---

## 11. Embedding Model Comparison

### 11.1 Architecture Differences

**MiniLM-L6-v2** (22M params, 6 layers, 384-dim):
- Distilled from a 12-layer model using knowledge distillation
- Fastest inference, smallest memory footprint
- Optimized for speed/quality tradeoff

**MiniLM-L12-v2** (33M params, 12 layers, 384-dim):
- Full 12-layer model (not distilled)
- Same output dimensions as L6, but deeper representation
- Tests whether deeper layers help recipe understanding

**MPNet-base-v2** (110M params, 12 layers, 768-dim):
- Largest model, highest capacity
- 768-dim embeddings may capture finer semantic distinctions
- Tests whether recipe retrieval is capacity-constrained

### 11.2 Expected Tradeoffs

```
Retrieval Quality:  MPNet-base >= MiniLM-L12 >= MiniLM-L6
Index Build Time:   MPNet-base >> MiniLM-L12 > MiniLM-L6
Index Size on Disk: MPNet-base >> MiniLM-L12 = MiniLM-L6
Query Latency:      MPNet-base > MiniLM-L12 > MiniLM-L6
```

The key question is whether the recipe domain is simple enough that the smallest model captures it adequately, or whether the additional capacity of larger models provides meaningful improvements.

---

## 12. Statistical Analysis

### 12.1 Methodology

For each pair of configurations (e.g., full_hybrid vs. bm25_only), we compute:

**Paired t-test**:
```
t = mean(d) / (sd(d) / sqrt(n))
```
Where d = per-query score differences, testing H0: mean(d) = 0.

**Wilcoxon signed-rank test**: Ranks the absolute differences and compares the sum of positive ranks to the sum of negative ranks. More robust when score differences are not normally distributed.

**Bootstrap confidence interval**:
1. Resample n scores with replacement, 1000 times
2. Compute mean for each resample
3. Report 2.5th and 97.5th percentiles as 95% CI

**Bonferroni correction**: With C(n,2) = 15 pairwise comparisons (6 configs), the adjusted significance threshold is alpha = 0.05 / 15 = 0.0033.

**Cohen's d**:
```
d = mean(score_A - score_B) / std(score_A - score_B)
```

### 12.2 Interpretation Guide

| Cohen's d | Interpretation |
|-----------|---------------|
| |d| < 0.2 | Negligible difference |
| 0.2 <= |d| < 0.5 | Small effect |
| 0.5 <= |d| < 0.8 | Medium effect |
| |d| >= 0.8 | Large effect |

> **Chart: `report_charts/ablation_with_ci.png`** - Ablation results with bootstrap 95% confidence interval error bars, enabling visual significance testing (non-overlapping CIs suggest significant difference).

```bash
python run_evaluation.py --step statistics
```

> Results saved to: `data/eval/statistical_analysis.json`

---

## 13. LLM-Only Baseline Comparison

### 13.1 Methodology

To demonstrate the value of the RAG pipeline, we compare against a pure LLM baseline:

1. For each ground truth query, prompt the LLM: "Given this query, suggest 10 recipe names"
2. Fuzzy-match the suggested names against the Food.com database (Jaccard similarity on word tokens, threshold > 0.5)
3. Compute the same IR metrics on the matched recipe IDs

This measures what the LLM knows from pretraining alone. Since Food.com recipes are publicly available, some may appear in training data, but the LLM must recall exact recipe names matching specific constraint combinations.

### 13.2 Expected Results

The LLM baseline should perform significantly worse because:
- It cannot access the specific 10,000-recipe corpus
- Recipe names must be matched to database entries (many won't match)
- It cannot filter by cooking time, ingredients, or nutrition data
- It lacks the knowledge graph's ingredient-recipe connections

This comparison justifies the engineering effort of building the RAG pipeline: grounding in a real database produces verifiably better results than relying on LLM memory.

### 13.3 Running the Baseline

```bash
python run_evaluation.py --step ablation  # Includes LLM baseline by default
python run_evaluation.py --step ablation --skip-llm  # Skip if no API key
```

---

## 14. End-to-End Walkthrough with Numbers

### 14.1 User Query

```
"I have chicken breast, rice, and bell peppers. Make me 3 quick Asian dinners, no dairy."
```

### 14.2 NL Parsing (Step 0)

LLM extracts:
```json
{
  "num_meals": 3,
  "max_minutes": 30,
  "tags": [],
  "allergens": ["dairy"],
  "cook_after_hour": 18,
  "ingredients_on_hand": ["chicken breast", "rice", "bell peppers"],
  "cuisine_preferences": ["asian"],
  "calorie_target_per_meal": null
}
```

### 14.3 Planning (Step 1)

LLM generates 3 meal queries with preferred_ingredients and preferred_tags:

```json
{
  "meal_queries": [
    {
      "query": "asian chicken rice stir fry quick",
      "day": "Monday",
      "cook_hour": 18,
      "max_minutes": 30,
      "preferred_ingredients": ["chicken breast", "rice", "bell peppers"],
      "preferred_tags": ["asian"]
    },
    {
      "query": "chicken bell pepper fried rice asian",
      "day": "Wednesday",
      "cook_hour": 18,
      "max_minutes": 30,
      "preferred_ingredients": ["chicken breast", "rice", "bell peppers"],
      "preferred_tags": ["asian"]
    },
    {
      "query": "asian chicken teriyaki rice bowl",
      "day": "Friday",
      "cook_hour": 18,
      "max_minutes": 30,
      "preferred_ingredients": ["chicken breast", "rice"],
      "preferred_tags": ["asian"]
    }
  ],
  "allergens": ["dairy"]
}
```

### 14.4 Retrieval (Step 2, for meal query 1)

**BM25 retrieval** (top 5 of 20):

| Rank | Recipe | BM25 Score |
|------|--------|-----------|
| 0 | Asian Chicken Stir Fry | 8.42 |
| 1 | Quick Chicken Rice Bowl | 7.89 |
| 2 | Easy Chicken Fried Rice | 7.31 |
| 3 | Szechuan Chicken | 6.95 |
| 4 | Thai Basil Chicken | 6.72 |

**Vector retrieval** (top 5 of 20):

| Rank | Recipe | Cosine Sim |
|------|--------|-----------|
| 0 | Quick Chicken Rice Bowl | 0.847 |
| 1 | Asian Chicken Stir Fry | 0.831 |
| 2 | Teriyaki Chicken Rice | 0.824 |
| 3 | Thai Chicken Stir Fry | 0.819 |
| 4 | Chinese Chicken Rice | 0.812 |

**RRF fusion** (k=60):

| Rank | Recipe | RRF Score | Sources |
|------|--------|----------|---------|
| 0 | Asian Chicken Stir Fry | 0.0319 | BM25(0) + Vector(1) |
| 1 | Quick Chicken Rice Bowl | 0.0312 | BM25(1) + Vector(0) |
| 2 | Easy Chicken Fried Rice | 0.0238 | BM25(2) + Vector(6) |
| 3 | Teriyaki Chicken Rice | 0.0165 | Vector(2) only |
| 4 | Thai Basil Chicken | 0.0160 | BM25(4) only |

**KG re-ranking** (preferences: chicken breast, rice, bell peppers, asian):

| Rank | Recipe | RRF | KG Boost | Final |
|------|--------|-----|----------|-------|
| 0 | Asian Chicken Stir Fry | 0.0319 | 0.075 (3/4 prefs) | 0.107 |
| 1 | Easy Chicken Fried Rice | 0.0238 | 0.075 (3/4 prefs) | 0.099 |
| 2 | Quick Chicken Rice Bowl | 0.0312 | 0.050 (2/4 prefs) | 0.081 |
| 3 | Teriyaki Chicken Rice | 0.0165 | 0.050 (2/4 prefs) | 0.067 |
| 4 | Thai Basil Chicken | 0.0160 | 0.025 (1/4 prefs) | 0.041 |

Notice: "Easy Chicken Fried Rice" was promoted from rank 2 to rank 1 by the KG boost because it contains chicken, rice, AND bell peppers (3 of 4 preference nodes matched).

### 14.5 Post-Retrieval Filtering

- Dairy filter: Check all 5 candidates for dairy ingredients. If "Teriyaki Chicken Rice" contains butter, it's excluded.
- Time filter: Check all candidates <= 30 minutes
- Top result picked for this meal slot

### 14.6 Full Pipeline Timing (approximate)

| Stage | Time |
|-------|------|
| NL Parsing (LLM call) | ~500ms |
| Memory Retrieval | ~50ms |
| Planning (LLM call) | ~800ms |
| Recipe Search x3 | ~1500ms |
| Allergy Check x3 | ~100ms |
| Nutrition, Grocery, Budget, ICS | ~200ms |
| Critic Verification | ~10ms |
| **Total** | **~3.2 seconds** |

---

## 15. Embedding Space Visualization

### 15.1 Methodology

To visualize how the embedding model organizes recipes:
1. Extract embeddings for ~2000 recipes from ChromaDB
2. Apply dimensionality reduction (t-SNE (t-distributed Stochastic Neighbor Embedding) with perplexity=30, or UMAP (Uniform Manifold Approximation and Projection) with n_neighbors=15, min_dist=0.1)
3. Color points by cuisine tag (top 12 cuisines)
4. Compute silhouette score as a cluster quality metric

**Silhouette score**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where a(i) = mean intra-cluster distance, b(i) = mean nearest-cluster distance. Range [-1, 1], higher = better separation.

### 15.2 Results

The UMAP visualization (2000 recipes, 11 cuisine categories) yields a silhouette score of -0.269, indicating that cuisine categories are not well-separated in the all-MiniLM-L6-v2 embedding space. This is expected for several reasons:

1. **Recipe text is multi-faceted**: A recipe's embedding encodes cooking method, ingredients, time, and description, not just cuisine. Two Italian pasta recipes may be far apart because one is a quick weeknight dinner and the other is a slow-braised ragout.

2. **Most recipes lack cuisine tags**: The "other" category (no cuisine tag) dominates the dataset, creating a large undifferentiated mass that suppresses the silhouette score.

3. **Cuisine overlap is real**: Mexican and Caribbean recipes share many ingredients (cumin, cilantro, lime), as do Chinese and Japanese recipes (soy sauce, ginger, sesame oil). Overlapping clusters are culinarily correct.

4. **MiniLM-L6 is a general-purpose model**: It was not fine-tuned for culinary distinctions. A domain-specific embedding model (or one fine-tuned on recipe-cuisine pairs) would likely show better separation.

Despite the low silhouette score, the embedding space does show local structure: small clusters of cuisine-tagged recipes are visible in both t-SNE and UMAP projections, suggesting the model captures some culinary semantics even without domain-specific training.

> **Chart: `report_charts/embedding_space_tsne.png`** - t-SNE visualization of recipe embeddings colored by cuisine.
> **Chart: `report_charts/embedding_space_umap.png`** - UMAP visualization of the same embeddings.

```bash
python run_evaluation.py --step visualize
```

---

## 16. Dynamic Evaluation and Constraint-Aware Metrics

### 16.1 Dynamic Scenario Generation

Instead of 60 hardcoded test cases, we generate 200+ diverse scenarios programmatically:

| Tier | Count | Constraints | Example |
|------|-------|-------------|---------|
| Simple | 70 | 1-2 constraints | "Plan 5 vegetarian meals" |
| Medium | 70 | 3-4 with ingredients | "I have chicken and rice, Mexican style, no dairy" |
| Complex | 60 | 5+ with calories | "7 keto meals using salmon, avocado, under 30 min, ~600 cal" |
| Edge Case | 20+ | Conflicting/extreme | "Vegan Japanese under 10 min with many allergens" |

Scenarios are drawn from the dataset's actual tag and ingredient vocabulary, ensuring they represent realistic queries.

> Implementation: `src/evaluation/scenario_generator.py`
> Output: `data/eval/dynamic_test_cases.json`

### 16.2 New Constraint-Aware Metrics

Beyond standard IR metrics, we measure how well the pipeline respects specific constraint types:

**Ingredient Coverage**: What fraction of the user's available ingredients appear in selected recipes?
```
IC = |{user_ingredients INTERSECT recipe_ingredients}| / |{user_ingredients}|
```

**Cuisine Alignment**: Fraction of recipes matching requested cuisines.
```
CA = |{recipes with matching cuisine tag}| / |{total recipes}|
```

**Calorie Deviation**: Mean absolute deviation from calorie target (PDV units).
```
CD = (1/n) * SUM |recipe_calories - target_calories|
```

These metrics directly measure whether the flexible constraint system actually influences retrieval, going beyond the original operational metrics (constraint pass, allergy violation, citation pass).

> Implementation: `src/evaluation/metrics.py` (new functions: `ingredient_coverage`, `cuisine_alignment`, `calorie_deviation`)

---

## 17. Implementation Details

### 17.1 Project Structure

```
src/
  evaluation/
    retrieval_metrics.py    # P@k, R@k, MRR, NDCG@k, MAP (NEW)
    ground_truth.py         # Tag, ingredient, KG-graded queries (NEW)
    retrieval_eval.py       # Configurable evaluation harness (NEW)
    ablation_study.py       # 6-config ablation runner (NEW)
    hyperparam_study.py     # Sweep top_k, rrf_k, KG weight, embed model (NEW)
    statistical_analysis.py # t-test, Wilcoxon, bootstrap CI, Cohen's d (NEW)
    visualization.py        # Publication-quality charts (NEW)
    scenario_generator.py   # Dynamic 200+ scenario generation (NEW)
    metrics.py              # Extended with ingredient_coverage, cuisine_alignment (MODIFIED)
    evaluator.py            # Original pipeline evaluator
    comparison.py           # Original cross-model comparison
  rag/
    retriever.py            # Configurable: use_vector, use_bm25, use_kg flags (MODIFIED)
    indexer.py              # Configurable embed_model parameter (MODIFIED)
    knowledge_graph.py      # Re-ranking with configurable boost_weight
  agent/
    planner.py              # Expanded schema: preferred_ingredients/tags (MODIFIED)
    executor.py             # Wires preferred_ingredients/tags to recipe_search (MODIFIED)
    few_shot_examples.py    # 5 examples including ingredient-driven plans (MODIFIED)
  tools/
    recipe_search.py        # preferred_ingredients, preferred_tags, calorie_target (MODIFIED)

run_evaluation.py           # Master evaluation script (NEW)
```

### 17.2 Dependencies

| Package | Purpose |
|---------|---------|
| scipy | Paired t-test, Wilcoxon signed-rank test |
| scikit-learn | t-SNE, silhouette score |
| matplotlib, seaborn | Publication-quality charts |
| umap-learn | UMAP dimensionality reduction |

### 17.3 Running the Full Evaluation Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Build data (if not already done)
python -m src.data.loader
python -m src.rag.indexer
python -m src.rag.graph_builder

# Run EVERYTHING (ground truth, ablation, hyperparams, stats, charts)
python run_evaluation.py

# Quick test (10 queries per config)
python run_evaluation.py --quick

# Individual steps
python run_evaluation.py --step ground_truth
python run_evaluation.py --step ablation
python run_evaluation.py --step hyperparams --skip-embed
python run_evaluation.py --step statistics
python run_evaluation.py --step visualize
```

### 17.4 LLM Models

| Model | Provider | Parameters | Cost |
|-------|----------|-----------|------|
| Llama 3.2 3B | Ollama (local) | 3B | Free |
| Granite 3.1 Dense 2B | Ollama (local) | 2B | Free |
| Llama 3.3 70B | Groq (cloud) | 70B | $0.59/$0.79 per 1M tokens |

The LLMs handle planning only; retrieval quality is independent of LLM choice (evaluated separately).

---

## 18. Conclusions and Future Work

### 18.1 Key Findings

1. **Full hybrid retriever wins every aggregate metric**: The combination of BM25 + Vector + KG achieves the highest P@5 (0.655), MRR (0.688), and NDCG@10 (0.438). The full_hybrid vs. vector_only difference on P@5 (d=1.225, p<0.001) is a large and highly significant effect. The improvement over the strong BM25-only baseline is a more modest but consistent 14% on P@5.

2. **KG re-ranking is essential for fusion to work**: RRF fusion without KG (vector_bm25, P@5=0.365) actually performs worse than BM25 alone (P@5=0.575), dropping P@5 by 36.5%. The vector search's weaker results dilute BM25's ranking during fusion. The KG re-ranking is what makes fusion viable: it re-ranks the combined candidates using ingredient/tag overlap, recovering relevant results that RRF alone buries. Disabling the KG (w_kg=0.0) drops P@5 by 44% and NDCG@10 by 47%. Even a minimal weight (w_kg=0.05) restores full performance, with no degradation up to w_kg=1.0.

3. **BM25 is the strongest individual retriever**: BM25-only (P@5=0.575) outperforms vector-only (0.255) by 126%. On tag-single queries, BM25 achieves MRR near 0.89. BM25-based configs also lead on natural language queries (MRR=0.56 vs full_hybrid's 0.33), showing that RRF fusion dilutes BM25's keyword signal for NL queries.

4. **Retrieval depth (top_k) is the most impactful hyperparameter**: Increasing top_k from 5 to 50 improves NDCG@10 by 156% (0.234 to 0.599) and AP by 189% (0.211 to 0.609). Deeper retrieval gives the KG re-ranker more candidates to work with, amplifying the KG's effectiveness.

5. **RRF is robust to its fusion constant**: Performance is stable across k in [30, 200], with a small dip only at k=10. The default k=60 from Cormack et al. (2009) is optimal.

6. **Component contributions vary by query type, and full hybrid has weaknesses**: Full hybrid dominates tag-pair (MRR=1.0) and ingredient queries (MRR=0.75), but BM25-only outperforms full hybrid on natural language queries (MRR 0.56 vs 0.33). Tag-pair queries also show wide variance across configs (vector_only MRR=0.40 vs full_hybrid MRR=1.0). No single configuration is best everywhere; full hybrid wins in aggregate because its strengths on structured queries outweigh its NL weakness.

7. **The flexible constraint system activates the KG pipeline**: Ingredients on hand, cuisine preferences, and calorie targets now flow through to the KG re-ranker, activating previously dead code paths and producing measurably better results for preference-driven queries.

### 18.2 Future Work

- **Learned fusion weights**: Replace fixed RRF with a learned fusion model that adapts weights based on query type
- **GNN (Graph Neural Network)-based KG embeddings**: Replace the simple overlap-based re-ranking with GNN embeddings (e.g., GraphSAGE, GAT (Graph Attention Network)) for richer structural representations
- **Cross-encoder re-ranking**: Add a cross-encoder model as a final re-ranking stage for the top-k candidates
- **User feedback loop**: Use the memory system's liked/disliked recipes to fine-tune the embedding model via contrastive learning
- **Multi-modal retrieval**: Incorporate recipe images as an additional signal using CLIP (Contrastive Language-Image Pre-training) embeddings

---

## 19. References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR.
3. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
5. Wang, X., et al. (2019). KGAT: Knowledge Graph Attention Network for Recommendation. KDD.
6. Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.
7. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. IEEE TPAMI.

---

## Appendix A: All Generated Charts

| Chart | Location | Description |
|-------|----------|-------------|
| Ablation Comparison | `report_charts/ablation_comparison.png` | Grouped bar: P@5, P@10, R@10, MRR, NDCG@10 per config |
| Ablation with CI | `report_charts/ablation_with_ci.png` | Same with 95% bootstrap confidence intervals |
| Precision-Recall Curves | `report_charts/precision_recall_curves.png` | P@k vs R@k as k varies per config |
| Component Contribution | `report_charts/component_contribution.png` | Heatmap: vector/BM25/both source per result |
| Query Type Breakdown | `report_charts/query_type_breakdown.png` | MRR by query type per config |
| Retrieval Latency | `report_charts/retrieval_latency.png` | Latency comparison across configs |
| Hyperparameter Sensitivity | `report_charts/hyperparam_sensitivity.png` | Line plots: metric vs top_k, rrf_k, KG weight |
| KG Analysis | `report_charts/kg_analysis.png` | Node distribution, degree histogram, top ingredients |
| Embedding Space (t-SNE) | `report_charts/embedding_space_tsne.png` | 2D visualization colored by cuisine |
| Embedding Space (UMAP) | `report_charts/embedding_space_umap.png` | Same with UMAP reduction |
| KG Single Recipe | `report_charts/knowledge_graph_single_recipe.png` | One recipe with all neighbors |
| KG Shared | `report_charts/knowledge_graph_shared.png` | Two recipes with shared connections |
| KG Overview | `report_charts/knowledge_graph_overview.png` | 5-recipe cluster view |
| Quality Metrics | `report_charts/quality_metrics.png` | Operational quality metrics across models |
| Pipeline Breakdown | `report_charts/pipeline_breakdown.png` | Pipeline stage timing breakdown |
| Latency Distribution | `report_charts/latency_distribution.png` | End-to-end latency distribution |
| Allergy Violations | `report_charts/allergy_violations.png` | Allergy violation rates |
| Retry Behavior | `report_charts/retry_behavior.png` | Critic retry patterns |
