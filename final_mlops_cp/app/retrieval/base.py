"""
app/retrieval/base.py
=====================
Abstract base class for all retrieval methods in RAGOps.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.schemas.retrieval import RetrievalResult, RetrievedChunk


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval models.
    Every retriever must implement `retrieve` and have a `name` property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name of the retriever (e.g. 'tfidf', 'bm25', 'semantic', 'hybrid')."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            query: The user query string.
            top_k: Maximum number of chunks to return.
            where: Optional metadata filter dict (e.g. {"department": "HR"}).

        Returns:
            List of RetrievedChunk objects ordered by descending relevance.
        """
        pass

    def retrieve_with_metrics(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Execute retrieval and record latency.

        Returns:
            RetrievalResult containing retrieved chunks and latency_ms.
        """
        start = time.perf_counter()
        chunks = self.retrieve(query=query, top_k=top_k, where=where)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return RetrievalResult(
            query=query,
            chunks=chunks,
            retriever=self.name,
            latency_ms=round(latency_ms, 2),
            top_k=top_k,
        )
