"""
app/retrieval/bm25_retriever.py
===============================
BM25 lexical retriever using rank-bm25.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Optional

import jsonlines
import numpy as np
from rank_bm25 import BM25Okapi
import yaml

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _load_bm25_config(config_path: str = "configs/retrieval.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("methods", {}).get("bm25", {}).get("params", {})
    return {}


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric word tokeniser with lowercasing."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever implementing the Okapi BM25 scoring algorithm.
    """

    def __init__(
        self,
        chunks: Optional[list[dict[str, Any]]] = None,
        chunks_path: Optional[str] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        config_path: str = "configs/retrieval.yaml",
    ):
        self._name = "bm25"
        cfg = _load_bm25_config(config_path)

        self.k1 = k1 if k1 is not None else float(cfg.get("k1", 1.5))
        self.b = b if b is not None else float(cfg.get("b", 0.75))

        self.chunks: list[dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks is not None:
            self.index_chunks(chunks)
        elif chunks_path:
            self.load_and_index(chunks_path)
        else:
            default_p = Path("data/processed/chunks.jsonl")
            if default_p.exists():
                self.load_and_index(str(default_p))

    @property
    def name(self) -> str:
        return self._name

    def load_and_index(self, chunks_path: str) -> None:
        """Load chunks from a JSONL file and index them."""
        p = Path(chunks_path)
        if not p.exists():
            logger.warning(f"Chunks file not found: {chunks_path}")
            return
        loaded = []
        with jsonlines.open(p) as reader:
            for item in reader:
                loaded.append(item)
        self.index_chunks(loaded)

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Tokenize chunk texts and build the BM25 index."""
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("No chunks provided to BM25Retriever.")
            self.bm25 = None
            return

        corpus = [tokenize(c.get("text", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25Retriever indexed {len(self.chunks)} chunks (k1={self.k1}, b={self.b}).")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top_k chunks matching query using BM25 scoring.
        """
        if not query or not query.strip() or self.bm25 is None or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        max_score = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0

        indices = np.argsort(raw_scores)[::-1]

        results: list[RetrievedChunk] = []
        for idx in indices:
            raw_s = float(raw_scores[idx])
            if raw_s <= 0.0 and len(results) >= 1:
                # Stop if no relevance
                break

            chunk = self.chunks[idx]
            meta = chunk.get("metadata", {})

            # Filter by where if provided
            if where:
                match = True
                for k, v in where.items():
                    if meta.get(k) != v and chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalize score to [0, 1] relative to highest hit score
            norm_score = (raw_s / max_score) if max_score > 0 else 0.0

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                    text=chunk.get("text", ""),
                    score=round(norm_score, 4),
                    metadata=meta,
                    document_id=chunk.get("document_id") or meta.get("document_id"),
                    retriever=self.name,
                )
            )

            if len(results) >= top_k:
                break

        return results
