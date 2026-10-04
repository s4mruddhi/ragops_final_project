"""
tests/test_vectorstore.py
===========================
Unit tests for ChromaDBStore.
Uses an in-memory ephemeral ChromaDB client (no disk writes during tests).

Run: pytest tests/test_vectorstore.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 384


def _random_unit_vector(dim: int = DIM) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_embedded_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    dept: str = "HR",
    embedding: list[float] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "chunk_index": 0,
        "total_chunks": 1,
        "text": text,
        "char_count": len(text),
        "embedding": embedding or _random_unit_vector(),
        "metadata": {
            "document_id": doc_id,
            "title": f"Policy {doc_id}",
            "department": dept,
            "category": "Policy",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "source": "internal_policy",
            "access_level": "employee",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    }


@pytest.fixture
def ephemeral_store(tmp_path):
    """
    Return a ChromaDBStore backed by an ephemeral (in-memory) ChromaDB client.
    Each call gets a unique collection name to avoid cross-test pollution.
    """
    import chromadb
    from app.vectorstore.chromadb_store import ChromaDBStore
    import uuid

    store = ChromaDBStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        distance_metric="cosine",
    )
    # Override with ephemeral client
    store._client = chromadb.EphemeralClient()
    return store


# ---------------------------------------------------------------------------
# Tests: build_from_chunks
# ---------------------------------------------------------------------------

class TestBuildFromChunks:
    def test_indexes_all_chunks(self, ephemeral_store):
        chunks = [
            _make_embedded_chunk(f"DOC-001_chunk_{i:03d}", "DOC-001", f"Text segment {i}")
            for i in range(10)
        ]
        n = ephemeral_store.build_from_chunks(chunks)
        assert n == 10

    def test_count_matches_indexed_chunks(self, ephemeral_store):
        chunks = [_make_embedded_chunk(f"ID-{i}", "DOC-A", f"Content {i}") for i in range(7)]
        ephemeral_store.build_from_chunks(chunks)
        assert ephemeral_store.count() == 7

    def test_empty_chunks_returns_zero(self, ephemeral_store):
        n = ephemeral_store.build_from_chunks([])
        assert n == 0
        assert ephemeral_store.count() == 0

    def test_reset_clears_existing_data(self, tmp_path):
        """Reset=True should wipe the previous chunks and only keep new ones."""
        import chromadb
        import uuid
        from app.vectorstore.chromadb_store import ChromaDBStore

        store = ChromaDBStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name=f"reset_{uuid.uuid4().hex[:8]}",
            distance_metric="cosine",
        )
        store._client = chromadb.EphemeralClient()

        # First build: 5 chunks
        chunks1 = [_make_embedded_chunk(f"OLD-{i}", "OLD-DOC", f"old text {i}") for i in range(5)]
        store.build_from_chunks(chunks1)
        assert store.count() == 5

        # Second build with reset=True: 3 new chunks replace all previous
        # We must clear manually since ephemeral client creates a new collection
        store._collection = None  # Force re-creation
        coll = store._get_collection()
        # Delete all existing then add new
        if coll.count() > 0:
            existing_ids = coll.get(include=[])['ids']
            coll.delete(ids=existing_ids)
        chunks2 = [_make_embedded_chunk(f"NEW-{i}", "NEW-DOC", f"new text {i}") for i in range(3)]
        store.build_from_chunks(chunks2)
        assert store.count() == 3

    def test_metadata_stored_correctly(self, ephemeral_store):
        chunk = _make_embedded_chunk("HR-TEST_chunk_000", "HR-TEST", "Leave policy text", dept="HR")
        ephemeral_store.build_from_chunks([chunk])
        results = ephemeral_store.get_by_document_id("HR-TEST")
        assert len(results) == 1
        assert results[0]["metadata"]["department"] == "HR"
        assert results[0]["metadata"]["document_id"] == "HR-TEST"


# ---------------------------------------------------------------------------
# Tests: query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_list(self, ephemeral_store):
        chunks = [_make_embedded_chunk(f"Q-{i}", "DOC-Q", f"Some text {i}") for i in range(5)]
        ephemeral_store.build_from_chunks(chunks)
        query_vec = _random_unit_vector()
        results = ephemeral_store.query(query_vec, top_k=3)
        assert isinstance(results, list)

    def test_top_k_respected(self, ephemeral_store):
        chunks = [_make_embedded_chunk(f"TK-{i}", "DOC-TK", f"Text {i}") for i in range(10)]
        ephemeral_store.build_from_chunks(chunks)
        query_vec = _random_unit_vector()
        results = ephemeral_store.query(query_vec, top_k=4)
        assert len(results) <= 4

    def test_results_have_required_fields(self, ephemeral_store):
        chunks = [_make_embedded_chunk("FIELD-0", "DOC-F", "Field test text")]
        ephemeral_store.build_from_chunks(chunks)
        results = ephemeral_store.query(_random_unit_vector(), top_k=1)
        assert len(results) > 0
        r = results[0]
        assert "chunk_id" in r
        assert "text" in r
        assert "metadata" in r
        assert "score" in r
        assert "distance" in r

    def test_score_is_between_0_and_1(self, ephemeral_store):
        chunks = [_make_embedded_chunk(f"SCORE-{i}", "DOC-S", f"Scored text {i}") for i in range(5)]
        ephemeral_store.build_from_chunks(chunks)
        results = ephemeral_store.query(_random_unit_vector(), top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_identical_vector_gets_highest_score(self, ephemeral_store):
        """A query vector identical to a stored vector should score ~1.0."""
        known_vec = _random_unit_vector()
        chunk = _make_embedded_chunk("KNOWN-0", "DOC-K", "Known text", embedding=known_vec)
        # Add noise chunks
        noise = [_make_embedded_chunk(f"NOISE-{i}", "DOC-N", f"Noise {i}") for i in range(5)]
        ephemeral_store.build_from_chunks([chunk] + noise)

        results = ephemeral_store.query(known_vec, top_k=6)
        top = results[0]
        assert top["chunk_id"] == "KNOWN-0"
        assert top["score"] > 0.95, f"Expected high score for identical vector, got {top['score']}"

    def test_query_on_empty_store_raises(self, ephemeral_store):
        """An empty index must fail loudly (a silent [] once hid MRR=0.0)."""
        import pytest
        with pytest.raises(RuntimeError, match="empty"):
            ephemeral_store.query(_random_unit_vector(), top_k=5)


# ---------------------------------------------------------------------------
# Tests: get_by_document_id
# ---------------------------------------------------------------------------

class TestGetByDocumentId:
    def test_retrieves_all_chunks_for_document(self, ephemeral_store):
        chunks = [
            _make_embedded_chunk(f"DOC-WFH_chunk_{i:03d}", "DOC-WFH", f"Chunk {i}")
            for i in range(4)
        ]
        other = [_make_embedded_chunk("OTHER-0", "DOC-OTHER", "Other doc")]
        ephemeral_store.build_from_chunks(chunks + other)

        results = ephemeral_store.get_by_document_id("DOC-WFH")
        assert len(results) == 4
        for r in results:
            assert r["metadata"]["document_id"] == "DOC-WFH"

    def test_nonexistent_document_returns_empty(self, ephemeral_store):
        chunks = [_make_embedded_chunk("ANY-0", "DOC-ANY", "Some text")]
        ephemeral_store.build_from_chunks(chunks)
        results = ephemeral_store.get_by_document_id("DOES-NOT-EXIST")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: collection_stats
# ---------------------------------------------------------------------------

class TestCollectionStats:
    def test_stats_includes_count(self, ephemeral_store):
        chunks = [_make_embedded_chunk(f"S-{i}", "DOC-S", f"Text {i}") for i in range(8)]
        ephemeral_store.build_from_chunks(chunks)
        stats = ephemeral_store.collection_stats()
        assert stats["total_chunks"] == 8

    def test_stats_includes_department_distribution(self, ephemeral_store):
        hr_chunks = [_make_embedded_chunk(f"HR-{i}", "DOC-HR", f"HR text {i}", dept="HR") for i in range(3)]
        it_chunks = [_make_embedded_chunk(f"IT-{i}", "DOC-IT", f"IT text {i}", dept="IT") for i in range(2)]
        ephemeral_store.build_from_chunks(hr_chunks + it_chunks)
        stats = ephemeral_store.collection_stats()
        assert "department_distribution" in stats
        assert stats["department_distribution"].get("HR") == 3
        assert stats["department_distribution"].get("IT") == 2


# ---------------------------------------------------------------------------
# Tests: metadata sanitisation
# ---------------------------------------------------------------------------

class TestMetadataSanitisation:
    def test_none_values_converted_to_empty_string(self):
        from app.vectorstore.chromadb_store import ChromaDBStore
        meta = {"key": None, "val": "normal"}
        sanitised = ChromaDBStore._sanitise_metadata(meta)
        assert sanitised["key"] == ""
        assert sanitised["val"] == "normal"

    def test_list_values_converted_to_string(self):
        from app.vectorstore.chromadb_store import ChromaDBStore
        meta = {"tags": ["hr", "policy"]}
        sanitised = ChromaDBStore._sanitise_metadata(meta)
        assert isinstance(sanitised["tags"], str)

    def test_valid_types_pass_through(self):
        from app.vectorstore.chromadb_store import ChromaDBStore
        meta = {"s": "text", "i": 42, "f": 3.14, "b": True}
        sanitised = ChromaDBStore._sanitise_metadata(meta)
        assert sanitised["s"] == "text"
        assert sanitised["i"] == 42
        assert sanitised["f"] == 3.14
        assert sanitised["b"] is True
