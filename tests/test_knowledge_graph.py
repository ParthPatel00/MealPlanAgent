import pytest

from src.rag.knowledge_graph import build_graph, find_related_recipes, graph_rerank


class TestBuildGraph:
    def test_creates_recipe_nodes(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        recipe_nodes = [n for n in G.nodes() if n.startswith("recipe:")]
        assert len(recipe_nodes) == 4

    def test_creates_ingredient_nodes(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        ing_nodes = [n for n in G.nodes() if n.startswith("ingredient:")]
        assert len(ing_nodes) > 0
        assert "ingredient:chicken" in G.nodes()
        assert "ingredient:rice" in G.nodes()

    def test_creates_tag_nodes(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        tag_nodes = [n for n in G.nodes() if n.startswith("tag:")]
        assert "tag:asian" in G.nodes()
        assert "tag:high-protein" in G.nodes()

    def test_edges_exist(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        assert G.has_edge("recipe:1", "ingredient:chicken")
        assert G.has_edge("recipe:1", "tag:asian")

    def test_edge_relations(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        edge = G.edges["recipe:1", "ingredient:chicken"]
        assert edge["relation"] == "HAS_INGREDIENT"
        edge = G.edges["recipe:1", "tag:asian"]
        assert edge["relation"] == "HAS_TAG"


class TestFindRelatedRecipes:
    def test_finds_related_by_ingredients(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        related = find_related_recipes(G, 1, "HAS_INGREDIENT", top_k=5)
        related_ids = [rid for rid, _ in related]
        assert 2 in related_ids  # shares chicken, garlic
        assert 3 in related_ids  # shares rice, garlic, soy sauce

    def test_similarity_scores(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        related = find_related_recipes(G, 1, "HAS_INGREDIENT", top_k=5)
        for rid, score in related:
            assert 0 <= score <= 1.0

    def test_nonexistent_recipe(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        related = find_related_recipes(G, 999, "HAS_INGREDIENT")
        assert related == []

    def test_related_by_tags(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)
        related = find_related_recipes(G, 1, "HAS_TAG", top_k=5)
        related_ids = [rid for rid, _ in related]
        assert 2 in related_ids  # shares high-protein
        assert 3 in related_ids  # shares asian


class TestGraphRerank:
    def test_boosts_matching_ingredients(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)

        class FakeHit:
            def __init__(self, rid, score):
                self.recipe_id = rid
                self.score = score

        hits = [FakeHit(1, 0.5), FakeHit(4, 0.6)]
        reranked = graph_rerank(G, hits, preferred_ingredients=["chicken"])
        assert reranked[0].recipe_id == 1  # chicken recipe boosted

    def test_no_preferences_unchanged(self, sample_graph_recipes):
        G = build_graph(sample_graph_recipes)

        class FakeHit:
            def __init__(self, rid, score):
                self.recipe_id = rid
                self.score = score

        hits = [FakeHit(1, 0.5), FakeHit(2, 0.6)]
        reranked = graph_rerank(G, hits)
        scores = [h.score for h in reranked]
        assert scores == [0.5, 0.6]  # scores unchanged when no preferences

    def test_none_graph_unchanged(self):
        class FakeHit:
            def __init__(self, rid, score):
                self.recipe_id = rid
                self.score = score

        hits = [FakeHit(1, 0.5)]
        reranked = graph_rerank(None, hits, preferred_ingredients=["chicken"])
        assert len(reranked) == 1
