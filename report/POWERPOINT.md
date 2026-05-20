# Presentation Script (12 Slides)

Use this as a speaking script for Presentation.pptx. Read the talking points for each slide in your own words. Each slide has: what's on the slide, what to say, and what the charts/numbers mean.

---

## Slide 1: Title

**On the slide**: Project name, subtitle, course info.

**Say**: "Our project is called MealPlanAgent. It's a meal planning system that uses a combination of keyword search, semantic search, and a knowledge graph to recommend real recipes from a database of 10,000 Food.com recipes. The core idea is Hybrid RAG, which stands for Retrieval-Augmented Generation, meaning the LLM doesn't make up recipes. It retrieves them from a verified database."

---

## Slide 2: The Problem

**On the slide**: Left side lists problems with LLMs, right side shows a knowledge graph visualization of a single recipe.

**Say**: "The problem we're solving is that LLMs hallucinate recipes. They invent wrong cooking times, impossible ingredient combinations, and fake nutrition data. They might say a recipe is nut-free when it contains almond flour. Our solution is RAG: we ground every recipe in a real database. The LLM reasons about what to search for, but the actual recipes come from 10,000 verified Food.com entries with real nutrition data and traceable citations. The image on the right shows one recipe in our knowledge graph, connected to all its ingredients and tags."

---

## Slide 3: Architecture

**On the slide**: Text-based flowchart showing the pipeline from user input to results.

**Say**: "Here's how the system works end to end. The user types a natural language request like 'plan me 5 high-protein dinners, no peanuts, under 30 minutes.' A parser extracts structured fields: ingredients, cuisine, allergens, time limits, calories. The Planner, which is an LLM, generates search queries for each meal, including preferred ingredients and tags. The Executor runs 5 deterministic tools: recipe search using our hybrid retrieval, allergy checking, nutrition calculation, grocery list generation, and budget estimation. Finally, a rule-based Critic runs 5 verification checks. If anything fails, such as wrong meal count, time limit exceeded, or allergen found, it retries up to twice. The key design principle is: the LLM reasons, the code verifies."

---

## Slide 4: Three-Stage Hybrid Retrieval

**On the slide**: Three cards explaining BM25, Vector, and RRF + KG. A pipeline arrow below. A heatmap chart at the bottom.

**Purpose of this study**: "The core question of our project is: how should we search a recipe database? We built a three-stage retrieval system and then rigorously tested whether each stage actually helps."

**What each stage does**:

- **BM25 (Sparse search)**: "BM25 is a traditional keyword matching algorithm. It's the same method search engines used before neural networks. It counts how often query words appear in a document, weighted by how rare those words are across all documents. So if you search 'vegetarian Mexican,' it finds recipes literally tagged with those words. It's great for exact matches but misses paraphrases. On its own, it gets a Precision@5 of 0.575, meaning about 58% of the top 5 results are relevant."

- **Vector (Dense search)**: "Vector search uses a neural network called all-MiniLM-L6-v2, a 22-million parameter sentence transformer, to convert recipes and queries into 384-dimensional vectors. Similar meanings end up close together in this space, so 'light summer dinner' can match salads even if those exact words don't appear. On its own, P@5 is only 0.255 because it's imprecise for exact constraints."

- **RRF + KG Re-rank**: "RRF stands for Reciprocal Rank Fusion. It merges the two ranked lists from BM25 and vector search using the formula: score equals 1 divided by (k plus rank plus 1), where k is a constant (default 60). Documents appearing in both lists get boosted. Then the Knowledge Graph re-ranks by checking how many of the user's preferred ingredients and tags overlap with each recipe's graph neighbors. Together, the full hybrid achieves P@5 of 0.655."

**What the heatmap shows**: "The chart at the bottom is a component contribution heatmap. For the full hybrid config, 44% of results came from BM25 alone, 24% from vector search alone, and only 32% were found by both. That means 68% of results would be lost if you used only one system. The two search methods find genuinely different recipes, which is exactly why fusing them improves performance."

---

## Slide 5: Knowledge Graph

**On the slide**: KG analysis charts showing degree distribution, node types, and connectivity.

**Say**: "Our knowledge graph has 15,807 nodes and 270,466 edges. There are three types of nodes: 9,999 recipes, 5,332 ingredients, and 476 tags. The charts show the structure of this graph. The degree distribution is heavily skewed: a few common ingredients like salt, sugar, and butter connect to thousands of recipes, while most ingredients connect to just a handful. This skewed structure is what makes the KG useful for re-ranking: it can identify recipes that share rare, specific ingredients with the user's preferences, not just common ones like salt."

---

## Slide 6: Ablation Study Results

**On the slide**: Grouped bar chart comparing 6 retriever configurations across multiple metrics.

**Purpose**: "An ablation study tests what happens when you remove components from a system. We wanted to know: does every part of our hybrid retrieval actually help, or are some components redundant?"

**The 6 configurations**:
- "vector_only: just the neural network embeddings"
- "bm25_only: just keyword matching"
- "vector_bm25: both searches fused with RRF, but no knowledge graph"
- "vector_kg: vector search plus KG re-ranking, no BM25"
- "bm25_kg: BM25 plus KG re-ranking, no vector"
- "full_hybrid: all three components together"

**What the metrics mean**:
- "P@5 (Precision at 5): out of the top 5 results, what fraction is actually relevant? Higher is better."
- "MRR (Mean Reciprocal Rank): how high up is the first relevant result? A score of 1.0 means the first result is always relevant. 0.5 means the first relevant result is typically second."
- "NDCG@10 (Normalized Discounted Cumulative Gain at 10): like precision, but it gives more credit for relevant results appearing higher in the list and supports graded relevance (not just yes/no, but how relevant). Ranges from 0 to 1."
- "AP (Average Precision): precision averaged at each point where a relevant document is found."

**What to highlight**: "In aggregate, full hybrid wins every metric: P@5 = 0.655, MRR = 0.688, NDCG@10 = 0.438. But the most interesting finding is what happens without the knowledge graph. vector_bm25, which fuses BM25 and vector search with RRF but has no KG, actually performs WORSE than bm25_only on every metric. P@5 drops from 0.575 to 0.365. That means RRF fusion alone dilutes BM25's strong signal with vector search's weaker results. The KG is what makes fusion work: it re-ranks the combined candidates using ingredient and tag overlap, recovering the relevant results that RRF alone buries."

**We tested on 40 queries**: "These are stratified across 5 query types: single-tag, multi-tag, ingredient-based, natural language, and KG-graded queries, so the results aren't biased toward any one type."

---

## Slide 7: Statistical Significance

**On the slide**: Ablation bar chart with confidence interval error bars. Text box with key statistics.

**Purpose**: "Good results aren't enough for an academic paper. We need to prove the differences aren't due to random chance."

**Say**: "This is the same ablation data but with 95% confidence intervals shown as error bars. The intervals were computed using bootstrap resampling with 1,000 iterations. The key comparison is full_hybrid versus vector_only on Precision@5: Cohen's d = 1.225 (a large effect size, anything above 0.8 is large), p < 0.001, and it remains significant after Bonferroni correction. Precision@10 is also strongly significant (d = 1.359). MRR and Recall@10 show medium effect sizes (d = 0.73 and 0.69) and pass Bonferroni. NDCG@10 has a smaller effect (d = 0.43) and does not pass Bonferroni correction, partly because only a subset of queries have graded relevance judgments. When the KG is completely disabled (weight set to 0), P@5 drops by 44% and NDCG@10 drops by 47%."

---

## Slide 8: Hyperparameter Sensitivity

**On the slide**: Line plots showing how metrics change as each parameter varies. Text bullets below.

**Purpose**: "Hyperparameters are the tunable knobs in our system. We need to understand how sensitive the results are to these choices, so users know which ones matter most and which are safe to leave at defaults."

**What each parameter does and what we found**:

- **top_k** (how many final results to return): "This is the most impactful parameter. Going from k=5 to k=50 improves NDCG@10 by 156%. This makes sense: deeper retrieval gives the KG more candidates to re-rank, amplifying its effect."

- **rrf_k** (the constant in the RRF fusion formula): "This controls how much weight goes to top-ranked versus lower-ranked documents. Low k means only the very top matters; high k spreads weight more evenly. We found it's robust across the range [30, 200]. Our default of 60 is near-optimal, so you don't need to tune this."

- **kg_boost_weight** (how much the knowledge graph influences the final score): "This has a binary effect. Any weight above 0, even as small as 0.05, restores full performance. Setting it to 0 drops P@5 by 44%. But there's no degradation at high weights either, so anywhere from 0.05 to 1.0 works fine."

---

## Slide 9: Performance by Query Type

**On the slide**: Grouped chart on the left showing metrics broken down by query type. Bullet explanations on the right.

**Purpose**: "Different users search differently. Some type tags like 'vegetarian,' some list ingredients like 'chicken rice broccoli,' and some write natural language like 'something light for summer.' We need to know which retrieval components help which query types."

**Say**: "Single-tag queries are easy for everyone, with MRR from 0.875 to 1.0. But tag-pair queries show more spread: vector_only drops to 0.40 while full_hybrid hits 1.0. The knowledge graph makes the biggest difference for ingredient queries, boosting MRR from 0.17 (vector_only) to 0.75 (full_hybrid). For natural language queries, BM25-based configs actually lead at 0.56 MRR, while full_hybrid drops to 0.33. This is because RRF fusion dilutes BM25's strong keyword signal with weaker vector results on NL queries. The takeaway is nuanced: full_hybrid wins in aggregate because it dominates tag and ingredient queries by a large margin, but it has a real weakness on natural language queries where pure BM25 is better. Each component genuinely excels at different query types."

---

## Slide 10: Embedding Space Visualization

**On the slide**: UMAP plot on the left, t-SNE plot on the right. Both show 2,000 recipe embeddings colored by cuisine.

**What these are**: "UMAP and t-SNE are dimensionality reduction techniques. Our recipe embeddings live in 384-dimensional space, which we can't visualize. These algorithms compress that down to 2D while trying to preserve the neighborhood structure, so similar recipes stay close together."

**Say**: "Each dot is a recipe, colored by its cuisine tag (Mexican, Italian, Indian, etc.). You can see some clustering, meaning the embedding model does capture cuisine similarity. But the silhouette score is -0.269, which means the clusters overlap significantly. This makes sense: a 'chicken stir-fry' could be Chinese, Thai, or American, and ingredients overlap across cuisines. This is actually why we need the knowledge graph on top of vector search. The embeddings alone can't cleanly separate cuisines, but the KG can boost results that share specific ingredients with the user's preferences."

---

## Slide 11: Precision-Recall & Latency

**On the slide**: Precision-recall curves on the left, latency box plot on the right.

**Precision-Recall curves**: "These show the tradeoff between precision (what fraction of results are relevant) and recall (what fraction of all relevant items did we find) as we vary k from 1 to 20. The full hybrid curve dominates, staying higher and further right than other configs. A curve closer to the top-right corner is better."

**Latency**: "The box plot shows retrieval time per query for each configuration. The full hybrid is slightly slower because it runs all three components, but the difference is small, typically under 200 milliseconds total. For a meal planning use case, this latency is negligible."

---

## Slide 12: Key Takeaways

**On the slide**: Four conclusion cards with headline and supporting data.

**Say**:

1. "First, in aggregate, full hybrid wins every metric: P@5 = 0.655, MRR = 0.688, NDCG@10 = 0.438. The improvement over vector-only is large and statistically significant for precision (d=1.225, p<0.001 on P@5). Compared to the strong BM25 baseline, the improvement is a more modest but consistent 13.9%."

2. "Second, the knowledge graph is essential, not optional. Without it, RRF fusion of BM25 and vector search actually performs worse than BM25 alone, dropping P@5 by 36%. The KG is what makes fusion work: it re-ranks the combined candidates using ingredient overlap, recovering the relevant results that RRF alone buries. Disabling KG drops P@5 by 44%."

3. "Third, top_k is the most impactful hyperparameter. Going from 5 to 50 results improves NDCG@10 by 156% and AP by 189%, because deeper retrieval gives the KG more candidates to re-rank effectively."

4. "Fourth, each component has genuine strengths and weaknesses. BM25 dominates natural language queries (MRR 0.56 vs full hybrid's 0.33). The KG drives ingredient queries (MRR 0.17 to 0.75). Full hybrid dominates tag-pair and ingredient queries by large margins. No single configuration is best everywhere, but full hybrid wins in aggregate because its strengths outweigh its NL weakness."

**Closing**: "In summary, the knowledge graph is the key innovation. It transforms RRF fusion from a net negative into a net positive, and it provides the largest single-component lift across 40 stratified queries. The results are statistically robust, with 4 of 5 metrics passing Bonferroni-corrected significance testing."
