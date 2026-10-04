"""
tests/test_retrieval_metrics.py
================================
Unit tests for information retrieval evaluation metrics.
"""

import pytest

from app.evaluation.retrieval_metrics import (
    evaluate_query_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["DOC-1", "DOC-2", "DOC-3"]
        expected = ["DOC-1", "DOC-2"]
        assert recall_at_k(retrieved, expected, k=2) == 1.0

    def test_partial_recall(self):
        retrieved = ["DOC-1", "DOC-3", "DOC-4"]
        expected = ["DOC-1", "DOC-2"]
        assert recall_at_k(retrieved, expected, k=2) == 0.5

    def test_zero_recall(self):
        retrieved = ["DOC-3", "DOC-4"]
        expected = ["DOC-1", "DOC-2"]
        assert recall_at_k(retrieved, expected, k=2) == 0.0

    def test_empty_expected(self):
        assert recall_at_k(["DOC-1"], [], k=5) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["DOC-1"], k=5) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["DOC-1", "DOC-2"]
        expected = ["DOC-1", "DOC-2", "DOC-3"]
        assert precision_at_k(retrieved, expected, k=2) == 1.0

    def test_half_precision(self):
        retrieved = ["DOC-1", "DOC-X"]
        expected = ["DOC-1"]
        assert precision_at_k(retrieved, expected, k=2) == 0.5

    def test_zero_precision(self):
        retrieved = ["DOC-A", "DOC-B"]
        expected = ["DOC-1"]
        assert precision_at_k(retrieved, expected, k=2) == 0.0


class TestMRR:
    def test_first_rank_hit(self):
        retrieved = ["DOC-1", "DOC-2", "DOC-3"]
        expected = ["DOC-1"]
        assert reciprocal_rank(retrieved, expected) == 1.0

    def test_third_rank_hit(self):
        retrieved = ["DOC-X", "DOC-Y", "DOC-1"]
        expected = ["DOC-1"]
        assert pytest.approx(reciprocal_rank(retrieved, expected), 0.001) == 1.0 / 3.0

    def test_no_hit_returns_zero(self):
        retrieved = ["DOC-X", "DOC-Y"]
        expected = ["DOC-1"]
        assert reciprocal_rank(retrieved, expected) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        retrieved = ["DOC-1", "DOC-2"]
        expected = ["DOC-1", "DOC-2"]
        assert ndcg_at_k(retrieved, expected, k=2) == 1.0

    def test_lower_ndcg_for_later_hit(self):
        perfect = ndcg_at_k(["DOC-1", "DOC-X"], ["DOC-1"], k=2)
        swapped = ndcg_at_k(["DOC-X", "DOC-1"], ["DOC-1"], k=2)
        assert perfect > swapped
        assert perfect == 1.0


class TestEvaluateQueryRetrieval:
    def test_returns_all_expected_keys(self):
        metrics = evaluate_query_retrieval(
            retrieved_doc_ids=["DOC-1", "DOC-2", "DOC-3"],
            expected_doc_ids=["DOC-1"],
            k_values=[1, 3, 5],
        )
        assert "mrr" in metrics
        assert "recall_at_1" in metrics
        assert "recall_at_3" in metrics
        assert "recall_at_5" in metrics
        assert "precision_at_1" in metrics
        assert "ndcg_at_1" in metrics
