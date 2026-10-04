"""
app/retrieval
=============
Retrieval layer containing lexical (TF-IDF, BM25), dense (Semantic), and Hybrid retrievers.
"""

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.factory import get_retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.tfidf_retriever import TFIDFRetriever

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "SemanticRetriever",
    "HybridRetriever",
    "get_retriever",
]
