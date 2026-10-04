"""
app/retrieval/semantic_retriever.py
===================================
Dense semantic retriever backed by ChromaDB and sentence-transformers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.retrieval.base import BaseRetriever
from app.schemas.retrieval import RetrievedChunk
from app.vectorstore.chromadb_store import ChromaDBStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    """
    Dense semantic retriever querying ChromaDB with sentence embeddings.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[ChromaDBStore] = None,
        params_path: str = "params.yaml",
    ):
        self._name = "semantic"
        self.embedder = embedder or Embedder(params_path=params_path)
        self.store = store or ChromaDBStore(params_path=params_path)

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
        Embed the query and retrieve top_k nearest neighbors from ChromaDB.
        """
        if not query or not query.strip():
            return []

        # Errors propagate on purpose: a silent [] hid an empty index and produced MRR=0.0.
        query_embedding = self.embedder.embed_query(query.strip())
        hits = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        results: list[RetrievedChunk] = []
        for hit in hits:
            meta = hit.get("metadata", {})
            results.append(
                RetrievedChunk(
                    chunk_id=hit.get("chunk_id", ""),
                    text=hit.get("text", ""),
                    score=float(hit.get("score", 0.0)),
                    metadata=meta,
                    document_id=meta.get("document_id"),
                    retriever=self.name,
                )
            )

        return results
