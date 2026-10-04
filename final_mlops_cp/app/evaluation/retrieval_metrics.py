"""
app/evaluation/retrieval_metrics.py
===================================
Standard information retrieval evaluation metrics:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- NDCG@K
- Aggregated evaluation runner
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K: fraction of expected relevant documents found in the top-K retrieved docs.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        expected_doc_ids: Sequence of ground-truth relevant document IDs.
        k: Cut-off rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if not expected_doc_ids:
        return 1.0
    if k <= 0 or not retrieved_doc_ids:
        return 0.0

    retrieved_set = set(retrieved_doc_ids[:k])
    expected_set = set(expected_doc_ids)
    hits = len(retrieved_set.intersection(expected_set))
    return float(hits / len(expected_set))


def precision_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Precision@K: fraction of top-K retrieved documents that are relevant.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        expected_doc_ids: Sequence of ground-truth relevant document IDs.
        k: Cut-off rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    if not expected_doc_ids:
        return 0.0

    top_k_docs = retrieved_doc_ids[:k]
    expected_set = set(expected_doc_ids)
    hits = sum(1 for d in top_k_docs if d in expected_set)
    return float(hits / len(top_k_docs))


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of the first relevant retrieved document.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        expected_doc_ids: Sequence of ground-truth relevant document IDs.

    Returns:
        Float in [0.0, 1.0].
    """
    if not retrieved_doc_ids or not expected_doc_ids:
        return 0.0

    expected_set = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_set:
            return float(1.0 / rank)
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank K with binary relevance.

    Args:
        retrieved_doc_ids: Ordered sequence of retrieved document IDs.
        expected_doc_ids: Sequence of ground-truth relevant document IDs.
        k: Cut-off rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_doc_ids or not expected_doc_ids:
        return 0.0

    expected_set = set(expected_doc_ids)
    # Retrieval is chunk-level, so one document can appear several times. Count each
    # document once (first occurrence) - otherwise NDCG can exceed 1.0 (a bug in v1).
    seen: set = set()
    unique_docs = [d for d in retrieved_doc_ids if not (d in seen or seen.add(d))]
    top_k_docs = unique_docs[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k_docs, start=1):
        rel = 1.0 if doc_id in expected_set else 0.0
        dcg += rel / math.log2(i + 1)

    # IDCG (ideal ranking puts all relevant documents first)
    ideal_hits = min(k, len(expected_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def evaluate_query_retrieval(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute all standard retrieval metrics for a single query.
    """
    metrics: dict[str, float] = {
        "mrr": round(reciprocal_rank(retrieved_doc_ids, expected_doc_ids), 4),
    }

    for k in k_values:
        metrics[f"recall_at_{k}"] = round(recall_at_k(retrieved_doc_ids, expected_doc_ids, k), 4)
        metrics[f"precision_at_{k}"] = round(precision_at_k(retrieved_doc_ids, expected_doc_ids, k), 4)
        metrics[f"ndcg_at_{k}"] = round(ndcg_at_k(retrieved_doc_ids, expected_doc_ids, k), 4)

    return metrics
