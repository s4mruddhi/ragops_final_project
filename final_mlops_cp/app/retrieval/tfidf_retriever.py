"""
app/retrieval/tfidf_retriever.py
================================
TF-IDF baseline retriever using scikit-learn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import jsonlines
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _load_tfidf_config(config_path: str = "configs/retrieval.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("methods", {}).get("tfidf", {}).get("params", {})
    return {}


class TFIDFRetriever(BaseRetriever):
    """
    Lexical baseline retriever based on TF-IDF representation and cosine similarity.
    """

    def __init__(
        self,
        chunks: Optional[list[dict[str, Any]]] = None,
        chunks_path: Optional[str] = None,
        max_features: Optional[int] = None,
        ngram_range: Optional[tuple[int, int]] = None,
        sublinear_tf: Optional[bool] = None,
        config_path: str = "configs/retrieval.yaml",
    ):
        self._name = "tfidf"
        cfg = _load_tfidf_config(config_path)

        self.max_features = max_features if max_features is not None else cfg.get("max_features", 10000)
        raw_ngram = ngram_range if ngram_range is not None else cfg.get("ngram_range", [1, 2])
        self.ngram_range = tuple(raw_ngram) if isinstance(raw_ngram, (list, tuple)) else (1, 2)
        self.sublinear_tf = sublinear_tf if sublinear_tf is not None else cfg.get("sublinear_tf", True)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
        )

        self.chunks: list[dict[str, Any]] = []
        self.matrix = None

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
        """Fit TF-IDF on chunk texts and cache the representation matrix."""
        self.chunks = list(chunks)
        if not self.chunks:
            logger.warning("No chunks provided to TFIDFRetriever.")
            self.matrix = None
            return

        texts = [c.get("text", "") for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TFIDFRetriever indexed {len(self.chunks)} chunks (vocab: {len(self.vectorizer.vocabulary_)}).")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top_k chunks matching query.
        """
        if not query or not query.strip() or self.matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query.strip()])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        # Candidate indices
        indices = np.argsort(scores)[::-1]

        results: list[RetrievedChunk] = []
        for idx in indices:
            score = float(scores[idx])
            if score <= 0.0 and len(results) >= 1:
                # Break once scores hit zero unless we haven't found anything
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

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                    text=chunk.get("text", ""),
                    score=round(score, 4),
                    metadata=meta,
                    document_id=chunk.get("document_id") or meta.get("document_id"),
                    retriever=self.name,
                )
            )

            if len(results) >= top_k:
                break

        return results
