"""
tests/test_generation.py
========================
Unit tests for RAG generation, groundedness scoring, and citation verification.
"""

from unittest.mock import MagicMock
import pytest

from app.generation.citations import extract_cited_document_ids, verify_citations
from app.generation.generator import AnswerGenerator
from app.generation.groundedness import (
    compute_groundedness,
    extract_claims,
    tokenize_content_words,
)
from app.schemas.retrieval import RetrievedChunk


@pytest.fixture
def sample_context():
    return [
        RetrievedChunk(
            chunk_id="HR-LEAVE-001_chunk_001",
            document_id="HR-LEAVE-001",
            text="All full-time employees are entitled to 24 working days of paid annual leave per calendar year.",
            metadata={"title": "Annual Leave Policy", "document_id": "HR-LEAVE-001", "version": "3.0"},
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="HR-LEAVE-001_chunk_002",
            document_id="HR-LEAVE-001",
            text="Employees may carry forward a maximum of 5 unused leave days into the following calendar year.",
            metadata={"title": "Annual Leave Policy", "document_id": "HR-LEAVE-001", "version": "3.0"},
            score=0.88,
        ),
    ]


# ===========================================================================
# 1. Groundedness Tests
# ===========================================================================

class TestGroundedness:
    def test_extract_claims(self):
        answer = (
            "Employees receive 24 days of annual leave. Up to 5 days can be carried forward.\n\n"
            "Sources:\n- [HR-LEAVE-001] — Annual Leave Policy"
        )
        claims = extract_claims(answer)
        assert len(claims) == 2
        assert "Employees receive 24 days" in claims[0]
        assert "Sources" not in claims[0]
        assert "Sources" not in claims[1]

    def test_high_groundedness(self, sample_context):
        answer = (
            "Full-time employees are entitled to 24 working days of paid annual leave per calendar year."
        )
        res = compute_groundedness(answer, sample_context, threshold=0.75)
        assert res["score"] >= 0.75
        assert res["is_grounded"] is True

    def test_hallucination_detected(self, sample_context):
        answer = (
            "Employees are allowed 60 days of sabbatical vacation with full stock compensation each quarter."
        )
        res = compute_groundedness(answer, sample_context, threshold=0.75)
        assert res["score"] < 0.50
        assert res["is_grounded"] is False

    def test_empty_answer_returns_zero(self, sample_context):
        res = compute_groundedness("", sample_context)
        assert res["score"] == 0.0
        assert res["is_grounded"] is False


# ===========================================================================
# 2. Citation Verification Tests
# ===========================================================================

class TestCitations:
    def test_extract_cited_document_ids(self):
        answer = "Annual leave is 24 days.\n\nSources:\n- [HR-LEAVE-001] — Annual Leave Policy\n- [IT-PASS-001]"
        cited = extract_cited_document_ids(answer)
        assert "HR-LEAVE-001" in cited
        assert "IT-PASS-001" in cited

    def test_verify_valid_citations(self, sample_context):
        answer = "Annual leave is 24 days.\n\nSources:\n- [HR-LEAVE-001] — Annual Leave Policy"
        res = verify_citations(answer, sample_context, threshold=0.80)
        assert res["citation_correctness"] == 1.0
        assert res["is_valid"] is True
        assert "HR-LEAVE-001" in res["valid_document_ids"]

    def test_verify_invalid_citation(self, sample_context):
        answer = "Annual leave is 24 days.\n\nSources:\n- [FAKE-DOC-999] — Fake Policy"
        res = verify_citations(answer, sample_context, threshold=0.80)
        assert res["citation_correctness"] == 0.0
        assert res["is_valid"] is False
        assert "FAKE-DOC-999" in res["invalid_document_ids"]


# ===========================================================================
# 3. Answer Generator Tests
# ===========================================================================

class TestAnswerGenerator:
    def test_format_context(self, sample_context):
        generator = AnswerGenerator()
        formatted = generator.format_context(sample_context)
        assert "Document: HR-LEAVE-001" in formatted
        assert "Annual Leave Policy" in formatted

    def test_empty_context_triggers_fallback(self):
        generator = AnswerGenerator()
        res = generator.generate_answer("How many days of leave?", [])
        assert res["is_fallback"] is True
        assert "unable to find sufficient information" in res["answer"].lower()

    def test_grounded_answer_generation(self, sample_context):
        generator = AnswerGenerator()
        # Mock raw LLM response with grounded text and valid source
        generator.generate_raw_llm_response = MagicMock(
            return_value=(
                "All full-time employees are entitled to 24 working days of paid annual leave.\n\n"
                "Sources:\n- [HR-LEAVE-001] — Annual Leave Policy"
            )
        )
        res = generator.generate_answer("How many leave days?", sample_context)
        assert res["is_fallback"] is False
        assert res["groundedness_score"] >= 0.70
        assert res["citation_correctness"] == 1.0
        assert "24 working days" in res["answer"]

    def test_unsupported_answer_triggers_fallback(self, sample_context):
        generator = AnswerGenerator(groundedness_threshold=0.75)
        # Mock LLM generating unsupported hallucination
        generator.generate_raw_llm_response = MagicMock(
            return_value="Employees receive unlimited vacation and free flights to Mars."
        )
        res = generator.generate_answer("Vacation policy?", sample_context)
        assert res["is_fallback"] is True
        assert "unable to find sufficient information" in res["answer"].lower()
