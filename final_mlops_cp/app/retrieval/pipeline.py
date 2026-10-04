"""
app/retrieval/pipeline.py
=========================
Multi-stage retrieval pipeline:
1. Candidate Retrieval
2. Ranking / Scoring
3. Relevance Filtering (threshold)
4. Final Context Selection
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import settings
from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve_and_filter_candidates(
    query: str,
    retriever: BaseRetriever,
    candidate_k: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    final_context_k: Optional[int] = None,
    where: Optional[dict[str, Any]] = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    Execute 4-stage retrieval:
    1. Candidate retrieval (retrieve top `candidate_k` candidates)
    2. Ranking / score ordering
    3. Relevance filtering (drop candidates below `relevance_threshold`)
    4. Final context selection (select top `final_context_k` filtered chunks)

    Returns:
        (final_context_chunks, all_candidates)
    """
    top_k = candidate_k if candidate_k is not None else settings.top_k
    thresh = relevance_threshold if relevance_threshold is not None else settings.relevance_threshold
    final_k = final_context_k if final_context_k is not None else settings.final_context_k

    if not query or not query.strip():
        return [], []

    # Stage 1: Candidate retrieval
    candidates = retriever.retrieve(query=query.strip(), top_k=top_k, where=where)
    logger.debug(f"Stage 1 (Candidate Retrieval): retrieved {len(candidates)} chunks.")

    # Stage 2: Ranking
    ranked = sorted(candidates, key=lambda c: float(c.score), reverse=True)

    # Stage 3: Relevance filtering
    filtered = [c for c in ranked if float(c.score) >= float(thresh)]
    logger.debug(f"Stage 3 (Relevance Filtering): {len(filtered)}/{len(ranked)} passed threshold {thresh}.")

    # Stage 4: Final context selection
    final_context = filtered[:final_k]
    logger.debug(f"Stage 4 (Final Context): selected {len(final_context)} chunks.")

    return final_context, ranked
