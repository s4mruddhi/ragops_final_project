"""
app/retrieval/hybrid_retriever.py
=================================
Hybrid retriever combining lexical (BM25) and semantic retrieval with score fusion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _load_hybrid_config(config_path: str = "configs/retrieval.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("methods", {}).get("hybrid", {}).get("params", {})
    return {}


class HybridRetriever(BaseRetriever):
    """
    Combines BM25 and Semantic retrieval using either:
      - 'rrf': Reciprocal Rank Fusion
      - 'weighted_sum': Linear combination of normalized scores
    """

    def __init__(
        self,
        lexical_retriever: Optional[BaseRetriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[Literal["rrf", "weighted_sum"]] = None,
        rrf_k: Optional[int] = None,
        config_path: str = "configs/retrieval.yaml",
    ):
        self._name = "hybrid"
        cfg = _load_hybrid_config(config_path)

        self.alpha = alpha if alpha is not None else float(cfg.get("alpha", 0.5))
        self.fusion_method = fusion_method if fusion_method is not None else str(cfg.get("fusion_method", "rrf"))
        self.rrf_k = rrf_k if rrf_k is not None else int(cfg.get("rrf_k", 60))

        self.lexical = lexical_retriever or BM25Retriever()
        self.semantic = semantic_retriever or SemanticRetriever()

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Run both retrievers, fuse results, and return the top_k chunks.
        """
        if not query or not query.strip():
            return []

        # Retrieve larger candidate pools for fusion
        candidate_k = max(top_k * 3, 20)
        lexical_hits = self.lexical.retrieve(query=query, top_k=candidate_k, where=where)
        semantic_hits = self.semantic.retrieve(query=query, top_k=candidate_k, where=where)

        if not lexical_hits and not semantic_hits:
            return []
        if not lexical_hits:
            return semantic_hits[:top_k]
        if not semantic_hits:
            return lexical_hits[:top_k]

        if self.fusion_method == "weighted_sum":
            return self._fuse_weighted_sum(lexical_hits, semantic_hits, top_k)
        return self._fuse_rrf(lexical_hits, semantic_hits, top_k)

    def _fuse_rrf(
        self,
        lexical_hits: list[RetrievedChunk],
        semantic_hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion."""
        chunk_map: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        # Lexical rank scoring (1-indexed)
        lex_weight = 1.0 - self.alpha
        for rank, chunk in enumerate(lexical_hits, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            score = lex_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        # Semantic rank scoring
        sem_weight = self.alpha
        for rank, chunk in enumerate(semantic_hits, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            score = sem_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results: list[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            original = chunk_map[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=round(rrf_scores[cid], 5),
                    metadata=original.metadata,
                    document_id=original.document_id,
                    retriever=self.name,
                )
            )

        return results

    def _fuse_weighted_sum(
        self,
        lexical_hits: list[RetrievedChunk],
        semantic_hits: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Linear weighted sum of normalized scores."""
        chunk_map: dict[str, RetrievedChunk] = {}
        weighted_scores: dict[str, float] = {}

        def _normalize(chunks: list[RetrievedChunk]) -> dict[str, float]:
            if not chunks:
                return {}
            scores = [float(c.score) for c in chunks]
            max_s, min_s = max(scores), min(scores)
            if max_s > min_s:
                return {c.chunk_id: (float(c.score) - min_s) / (max_s - min_s) for c in chunks}
            elif max_s > 0:
                return {c.chunk_id: float(c.score) / max_s for c in chunks}
            return {c.chunk_id: 1.0 for c in chunks}

        lex_norm = _normalize(lexical_hits)
        sem_norm = _normalize(semantic_hits)

        lex_weight = 1.0 - self.alpha
        for chunk in lexical_hits:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            norm_score = lex_norm.get(cid, 0.0)
            weighted_scores[cid] = weighted_scores.get(cid, 0.0) + (lex_weight * norm_score)

        sem_weight = self.alpha
        for chunk in semantic_hits:
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            norm_score = sem_norm.get(cid, 0.0)
            weighted_scores[cid] = weighted_scores.get(cid, 0.0) + (sem_weight * norm_score)

        sorted_cids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)

        results: list[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            original = chunk_map[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=round(weighted_scores[cid], 4),
                    metadata=original.metadata,
                    document_id=original.document_id,
                    retriever=self.name,
                )
            )

        return results
