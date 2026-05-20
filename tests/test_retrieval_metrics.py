import pytest

from src.evaluation.retrieval_metrics import (
    aggregate_query_metrics,
    average_precision,
    compute_all_metrics,
    f1_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_precision(self):
        retrieved = [10, 20, 30]
        relevant = {1, 2, 3}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self):
        retrieved = [1, 10, 2, 20]
        relevant = {1, 2, 3}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_larger_than_retrieved(self):
        retrieved = [1, 2]
        relevant = {1, 2, 3}
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_k_zero(self):
        assert precision_at_k([1], {1}, 0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = [1, 10, 20]
        relevant = {1, 2, 3}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert recall_at_k([1, 2], set(), 2) == 0.0


class TestF1AtK:
    def test_perfect_f1(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}
        assert f1_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_f1(self):
        retrieved = [10, 20]
        relevant = {1, 2}
        assert f1_at_k(retrieved, relevant, 2) == 0.0

    def test_nonzero(self):
        retrieved = [1, 10]
        relevant = {1, 2}
        f1 = f1_at_k(retrieved, relevant, 2)
        assert 0 < f1 < 1


class TestMRR:
    def test_first_hit_at_rank_1(self):
        assert mean_reciprocal_rank([1, 2, 3], {1}) == 1.0

    def test_first_hit_at_rank_3(self):
        assert mean_reciprocal_rank([10, 20, 1], {1}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mean_reciprocal_rank([10, 20, 30], {1}) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], {1}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = [1, 2, 3]
        graded = {1: 3, 2: 2, 3: 1}
        assert ndcg_at_k(retrieved, graded, 3) == pytest.approx(1.0)

    def test_reverse_ranking(self):
        retrieved = [3, 2, 1]
        graded = {1: 3, 2: 2, 3: 1}
        score = ndcg_at_k(retrieved, graded, 3)
        assert 0 < score < 1

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2], {}, 2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k([1], {1: 3}, 0) == 0.0


class TestAveragePrecision:
    def test_perfect_ap(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}
        assert average_precision(retrieved, relevant) == 1.0

    def test_interleaved(self):
        retrieved = [1, 10, 2, 20, 3]
        relevant = {1, 2, 3}
        ap = average_precision(retrieved, relevant)
        expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
        assert ap == pytest.approx(expected)

    def test_no_relevant(self):
        assert average_precision([1, 2], set()) == 0.0


class TestMAP:
    def test_map(self):
        queries = [
            ([1, 2, 3], {1, 2, 3}),
            ([10, 20, 30], {1}),
        ]
        result = mean_average_precision(queries)
        assert 0 < result < 1

    def test_empty(self):
        assert mean_average_precision([]) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        metrics = compute_all_metrics(retrieved, relevant, k_values=[5, 10])
        assert "precision@5" in metrics
        assert "recall@5" in metrics
        assert "f1@5" in metrics
        assert "mrr" in metrics
        assert "ap" in metrics

    def test_with_graded(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2}
        graded = {1: 3, 2: 2, 3: 0}
        metrics = compute_all_metrics(retrieved, relevant, graded, k_values=[3])
        assert "ndcg@3" in metrics


class TestAggregateQueryMetrics:
    def test_aggregation(self):
        per_query = [
            {"precision@5": 0.8, "mrr": 1.0},
            {"precision@5": 0.4, "mrr": 0.5},
        ]
        agg = aggregate_query_metrics(per_query)
        assert agg["mean_precision@5"] == pytest.approx(0.6)
        assert agg["mean_mrr"] == pytest.approx(0.75)

    def test_empty(self):
        assert aggregate_query_metrics([]) == {}
