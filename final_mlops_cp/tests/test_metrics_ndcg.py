from app.evaluation.retrieval_metrics import ndcg_at_k


def test_ndcg_never_exceeds_one_with_duplicate_chunks():
    # chunk-level retrieval returns the same document several times
    assert ndcg_at_k(["A", "A", "A", "B"], ["A"], 5) <= 1.0
    assert ndcg_at_k(["A", "A", "A", "B"], ["A", "B"], 5) <= 1.0
