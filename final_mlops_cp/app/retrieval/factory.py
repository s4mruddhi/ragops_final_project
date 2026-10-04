"""
app/retrieval/factory.py
========================
Retriever factory for creating configured retriever instances.
"""

from typing import Any, Optional

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.tfidf_retriever import TFIDFRetriever


def get_retriever(
    method: str = "semantic",
    **kwargs: Any,
) -> BaseRetriever:
    """
    Factory function returning an instantiated retriever.

    Args:
        method: One of 'tfidf', 'bm25', 'semantic', 'hybrid'.
        kwargs: Overrides passed to the retriever constructor.

    Returns:
        Instantiated BaseRetriever.
    """
    m = method.lower().strip()
    if m in ("tfidf", "tf-idf"):
        return TFIDFRetriever(**kwargs)
    elif m == "bm25":
        return BM25Retriever(**kwargs)
    elif m == "semantic":
        return SemanticRetriever(**kwargs)
    elif m == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(
            f"Unknown retriever method: '{method}'. Supported methods: 'tfidf', 'bm25', 'semantic', 'hybrid'."
        )
