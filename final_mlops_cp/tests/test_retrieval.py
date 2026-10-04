"""
tests/test_retrieval.py
=======================
Unit tests for all retrieval components: TF-IDF, BM25, Semantic, and Hybrid.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25_retriever import BM25Retriever, tokenize
from app.retrieval.factory import get_retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.tfidf_retriever import TFIDFRetriever
from app.schemas.retrieval import RetrievedChunk


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "HR-LEAVE-001_chunk_001",
            "document_id": "HR-LEAVE-001",
            "text": "Full-time employees are entitled to 24 working days of paid annual leave per year.",
            "metadata": {"department": "HR", "title": "Leave Policy", "document_id": "HR-LEAVE-001"},
        },
        {
            "chunk_id": "HR-LEAVE-001_chunk_002",
            "document_id": "HR-LEAVE-001",
            "text": "Employees may carry forward a maximum of 5 unused leave days into the next year.",
            "metadata": {"department": "HR", "title": "Leave Policy", "document_id": "HR-LEAVE-001"},
        },
        {
            "chunk_id": "IT-PASS-001_chunk_001",
            "document_id": "IT-PASS-001",
            "text": "Passwords must be at least 12 characters and include uppercase, numbers, and symbols.",
            "metadata": {"department": "IT", "title": "Password Policy", "document_id": "IT-PASS-001"},
        },
        {
            "chunk_id": "TRAVEL-POL-001_chunk_001",
            "document_id": "TRAVEL-POL-001",
            "text": "The daily hotel rate limit for metro cities is INR 6000 per night for accommodation.",
            "metadata": {"department": "Travel", "title": "Travel Policy", "document_id": "TRAVEL-POL-001"},
        },
    ]


# ===========================================================================
# 1. TF-IDF Retriever Tests
# ===========================================================================

class TestTFIDFRetriever:
    def test_returns_top_k(self, sample_chunks):
        retriever = TFIDFRetriever(chunks=sample_chunks)
        results = retriever.retrieve("annual leave days", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].retriever == "tfidf"

    def test_ranks_relevant_chunk_highest(self, sample_chunks):
        retriever = TFIDFRetriever(chunks=sample_chunks)
        results = retriever.retrieve("password uppercase numbers characters", top_k=3)
        assert results[0].chunk_id == "IT-PASS-001_chunk_001"
        assert results[0].score > 0

    def test_empty_query_returns_empty_list(self, sample_chunks):
        retriever = TFIDFRetriever(chunks=sample_chunks)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_where_filter(self, sample_chunks):
        retriever = TFIDFRetriever(chunks=sample_chunks)
        # Search for year (exists in HR and general context), but filter to Travel
        results = retriever.retrieve("year", top_k=5, where={"department": "Travel"})
        for r in results:
            assert r.metadata.get("department") == "Travel"

    def test_unindexed_retriever_returns_empty(self):
        retriever = TFIDFRetriever(chunks=[])
        assert retriever.retrieve("hello") == []


# ===========================================================================
# 2. BM25 Retriever Tests
# ===========================================================================

class TestBM25Retriever:
    def test_tokenize(self):
        tokens = tokenize("Annual leave, 24 days!")
        assert tokens == ["annual", "leave", "24", "days"]
        assert tokenize("") == []

    def test_returns_top_k(self, sample_chunks):
        retriever = BM25Retriever(chunks=sample_chunks)
        results = retriever.retrieve("hotel rate metro", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0
        assert results[0].chunk_id == "TRAVEL-POL-001_chunk_001"
        assert results[0].retriever == "bm25"
        assert 0.0 <= results[0].score <= 1.0

    def test_empty_query_returns_empty_list(self, sample_chunks):
        retriever = BM25Retriever(chunks=sample_chunks)
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_where_filter(self, sample_chunks):
        retriever = BM25Retriever(chunks=sample_chunks)
        results = retriever.retrieve("policy", top_k=5, where={"department": "IT"})
        for r in results:
            assert r.metadata.get("department") == "IT"


# ===========================================================================
# 3. Semantic Retriever Tests
# ===========================================================================

class TestSemanticRetriever:
    def test_semantic_retriever_with_mocks(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "chunk_id": "MOCK-001_chunk_001",
                "text": "This is mock text",
                "metadata": {"department": "HR", "document_id": "MOCK-001"},
                "score": 0.88,
                "distance": 0.24,
            }
        ]

        retriever = SemanticRetriever(embedder=mock_embedder, store=mock_store)
        results = retriever.retrieve("test question", top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "MOCK-001_chunk_001"
        assert results[0].score == 0.88
        assert results[0].retriever == "semantic"
        mock_embedder.embed_query.assert_called_once_with("test question")

    def test_empty_query_returns_empty_list(self):
        retriever = SemanticRetriever(embedder=MagicMock(), store=MagicMock())
        assert retriever.retrieve("") == []
        assert retriever.retrieve("  ") == []


# ===========================================================================
# 4. Hybrid Retriever Tests
# ===========================================================================

class TestHybridRetriever:
    def test_rrf_fusion(self):
        mock_lexical = MagicMock(spec=BaseRetriever)
        mock_lexical.name = "bm25"
        mock_lexical.retrieve.return_value = [
            RetrievedChunk(chunk_id="chunk_A", text="A", score=0.9, retriever="bm25"),
            RetrievedChunk(chunk_id="chunk_B", text="B", score=0.7, retriever="bm25"),
        ]

        mock_semantic = MagicMock(spec=SemanticRetriever)
        mock_semantic.name = "semantic"
        mock_semantic.retrieve.return_value = [
            RetrievedChunk(chunk_id="chunk_B", text="B", score=0.95, retriever="semantic"),
            RetrievedChunk(chunk_id="chunk_C", text="C", score=0.80, retriever="semantic"),
        ]

        hybrid = HybridRetriever(
            lexical_retriever=mock_lexical,
            semantic_retriever=mock_semantic,
            fusion_method="rrf",
            alpha=0.5,
            rrf_k=60,
        )

        results = hybrid.retrieve("query", top_k=3)
        assert len(results) == 3
        # chunk_B appears in both lists, so its RRF score should be highest!
        assert results[0].chunk_id == "chunk_B"
        assert results[0].retriever == "hybrid"

    def test_weighted_sum_fusion(self):
        mock_lexical = MagicMock(spec=BaseRetriever)
        mock_lexical.name = "bm25"
        mock_lexical.retrieve.return_value = [
            RetrievedChunk(chunk_id="chunk_A", text="A", score=1.0, retriever="bm25"),
        ]

        mock_semantic = MagicMock(spec=SemanticRetriever)
        mock_semantic.name = "semantic"
        mock_semantic.retrieve.return_value = [
            RetrievedChunk(chunk_id="chunk_B", text="B", score=1.0, retriever="semantic"),
        ]

        # alpha=0.8 favors semantic (chunk_B = 0.8, chunk_A = 0.2)
        hybrid = HybridRetriever(
            lexical_retriever=mock_lexical,
            semantic_retriever=mock_semantic,
            fusion_method="weighted_sum",
            alpha=0.8,
        )

        results = hybrid.retrieve("query", top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_B"
        assert results[1].chunk_id == "chunk_A"

    def test_empty_query(self):
        hybrid = HybridRetriever(
            lexical_retriever=MagicMock(),
            semantic_retriever=MagicMock(),
        )
        assert hybrid.retrieve("") == []


# ===========================================================================
# 5. BaseRetriever Metrics & Factory Tests
# ===========================================================================

class TestFactoryAndBase:
    def test_retrieve_with_metrics(self, sample_chunks):
        retriever = TFIDFRetriever(chunks=sample_chunks)
        result = retriever.retrieve_with_metrics("leave policy", top_k=2)
        assert result.query == "leave policy"
        assert result.retriever == "tfidf"
        assert result.latency_ms >= 0.0
        assert len(result.chunks) <= 2

    def test_factory_returns_expected_types(self, sample_chunks):
        assert isinstance(get_retriever("tfidf", chunks=sample_chunks), TFIDFRetriever)
        assert isinstance(get_retriever("bm25", chunks=sample_chunks), BM25Retriever)

    def test_factory_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever method"):
            get_retriever("invalid_method")
