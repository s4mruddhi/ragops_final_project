"""
tests/test_embedding.py
=========================
Unit tests for the Embedder class.
Uses mock sentence-transformers to avoid downloading model weights during CI.

Run: pytest tests/test_embedding.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures: mock SentenceTransformer
# ---------------------------------------------------------------------------

def _make_mock_model(dim: int = 384):
    """Create a mock SentenceTransformer that returns random normalised vectors."""
    mock = MagicMock()

    def fake_encode(texts, **kwargs):
        n = len(texts)
        raw = np.random.randn(n, dim).astype(np.float32)
        # L2-normalise each row (mimic normalize_embeddings=True)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    mock.encode.side_effect = fake_encode
    mock.get_sentence_embedding_dimension.return_value = dim
    return mock


@pytest.fixture
def mock_embedder():
    """Return an Embedder with a mocked SentenceTransformer."""
    with patch("app.vectorstore.embedder.SentenceTransformer", return_value=None) as _:
        from app.vectorstore.embedder import Embedder
        embedder = Embedder(model_name="mock-model", batch_size=16, device="cpu")
        embedder._model = _make_mock_model(dim=384)
        yield embedder


# ---------------------------------------------------------------------------
# Tests: embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def test_returns_numpy_array(self, mock_embedder):
        texts = ["Annual leave policy", "Remote work guidelines"]
        result = mock_embedder.embed_texts(texts)
        assert isinstance(result, np.ndarray)

    def test_shape_is_correct(self, mock_embedder):
        texts = ["text1", "text2", "text3"]
        result = mock_embedder.embed_texts(texts)
        assert result.shape == (3, 384)

    def test_dtype_is_float32(self, mock_embedder):
        result = mock_embedder.embed_texts(["some text"])
        assert result.dtype == np.float32

    def test_empty_input_returns_empty_array(self, mock_embedder):
        result = mock_embedder.embed_texts([])
        assert result.size == 0

    def test_single_text_has_correct_shape(self, mock_embedder):
        result = mock_embedder.embed_texts(["single document"])
        assert result.shape == (1, 384)

    def test_vectors_are_l2_normalised(self, mock_embedder):
        """Vectors should have unit norm (L2 = 1.0) when normalize_embeddings=True."""
        result = mock_embedder.embed_texts(["a", "b", "c"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_different_texts_produce_different_vectors(self, mock_embedder):
        """Two different texts should not produce identical vectors."""
        result = mock_embedder.embed_texts(["policy about leave", "VPN setup guide"])
        assert not np.allclose(result[0], result[1])

    def test_large_batch(self, mock_embedder):
        texts = [f"Document number {i}" for i in range(200)]
        result = mock_embedder.embed_texts(texts)
        assert result.shape[0] == 200


# ---------------------------------------------------------------------------
# Tests: embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_1d_array(self, mock_embedder):
        result = mock_embedder.embed_query("What is the leave policy?")
        assert result.ndim == 1

    def test_correct_dimension(self, mock_embedder):
        result = mock_embedder.embed_query("test query")
        assert result.shape == (384,)

    def test_is_float32(self, mock_embedder):
        result = mock_embedder.embed_query("test query")
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def _make_chunks(self, n: int) -> list[dict]:
        return [
            {
                "chunk_id": f"HR-WFH-001_chunk_{i:03d}",
                "document_id": "HR-WFH-001",
                "chunk_index": i,
                "total_chunks": n,
                "text": f"Work from home policy section {i}. Employees need approval.",
                "char_count": 55,
                "metadata": {
                    "document_id": "HR-WFH-001",
                    "department": "HR",
                    "title": "WFH Policy",
                    "version": "2.1",
                    "effective_date": "2026-01-01",
                    "source": "internal_policy",
                    "access_level": "employee",
                },
            }
            for i in range(n)
        ]

    def test_embedding_added_to_each_chunk(self, mock_embedder):
        chunks = self._make_chunks(5)
        result = mock_embedder.embed_chunks(chunks)
        for c in result:
            assert "embedding" in c
            assert isinstance(c["embedding"], list)
            assert len(c["embedding"]) == 384

    def test_original_fields_preserved(self, mock_embedder):
        chunks = self._make_chunks(3)
        result = mock_embedder.embed_chunks(chunks)
        for r, original in zip(result, chunks):
            assert r["chunk_id"] == original["chunk_id"]
            assert r["text"] == original["text"]
            assert r["metadata"]["department"] == "HR"
            assert r["chunk_index"] == original["chunk_index"]

    def test_output_count_matches_input(self, mock_embedder):
        chunks = self._make_chunks(10)
        result = mock_embedder.embed_chunks(chunks)
        assert len(result) == 10

    def test_embedding_is_list_of_floats(self, mock_embedder):
        chunks = self._make_chunks(2)
        result = mock_embedder.embed_chunks(chunks)
        emb = result[0]["embedding"]
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)


# ---------------------------------------------------------------------------
# Tests: embedding_dim property
# ---------------------------------------------------------------------------

class TestEmbeddingDim:
    def test_returns_correct_dimension(self, mock_embedder):
        assert mock_embedder.embedding_dim == 384


# ---------------------------------------------------------------------------
# Tests: params.yaml config loading
# ---------------------------------------------------------------------------

class TestParamsLoading:
    def test_reads_model_from_params(self):
        """Embedder should pick up model_name from params.yaml."""
        with patch("app.vectorstore.embedder.SentenceTransformer"):
            from app.vectorstore.embedder import Embedder
            e = Embedder(params_path="params.yaml")
            assert "MiniLM" in e.model_name or "all-" in e.model_name.lower()

    def test_reads_batch_size_from_params(self):
        with patch("app.vectorstore.embedder.SentenceTransformer"):
            from app.vectorstore.embedder import Embedder
            e = Embedder(params_path="params.yaml")
            assert e.batch_size == 32

    def test_reads_device_from_params(self):
        with patch("app.vectorstore.embedder.SentenceTransformer"):
            from app.vectorstore.embedder import Embedder
            e = Embedder(params_path="params.yaml")
            assert e.device == "cpu"
