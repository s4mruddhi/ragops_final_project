"""
app/generation/citations.py
===========================
Citation extraction and verification for RAG generated responses.
Ensures that all cited document sources exist in the retrieved context.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Pattern for document IDs like HR-LEAVE-001, IT-PASS-001, TRAVEL-POL-001
DOC_ID_PATTERN = re.compile(r"\b([A-Z]{2,10}-[A-Z0-9]{2,10}-\d{3})\b")


def extract_cited_document_ids(answer: str) -> list[str]:
    """
    Extract all document IDs cited in the answer text,
    especially focusing on the 'Sources:' section.
    """
    if not answer:
        return []

    cited: list[str] = []

    # Look specifically in the Sources/References section first
    sources_match = re.search(r"(?:Sources|References):\s*(.*)", answer, flags=re.DOTALL | re.IGNORECASE)
    if sources_match:
        sources_text = sources_match.group(1)
        found = DOC_ID_PATTERN.findall(sources_text)
        for doc_id in found:
            if doc_id not in cited:
                cited.append(doc_id)

    # Also search the entire answer if none found in Sources section
    if not cited:
        found = DOC_ID_PATTERN.findall(answer)
        for doc_id in found:
            if doc_id not in cited:
                cited.append(doc_id)

    return cited


def verify_citations(
    answer: str,
    context_chunks: Sequence[Any],
    threshold: float = 0.80,
) -> dict[str, Any]:
    """
    Verify whether cited documents in the answer are actually present in the retrieved context.

    Args:
        answer: Generated answer text.
        context_chunks: Retrieved chunks provided to the LLM.
        threshold: Minimum citation correctness required.

    Returns:
        Dict with keys:
          - 'citation_correctness': float in [0.0, 1.0]
          - 'is_valid': bool (score >= threshold)
          - 'cited_document_ids': list[str]
          - 'valid_document_ids': list[str]
          - 'invalid_document_ids': list[str]
    """
    cited_ids = extract_cited_document_ids(answer)

    # Collect all available document IDs in context
    context_doc_ids: set[str] = set()
    for c in context_chunks:
        doc_id = getattr(c, "document_id", None)
        if not doc_id and hasattr(c, "metadata") and isinstance(c.metadata, dict):
            doc_id = c.metadata.get("document_id")
        if not doc_id and isinstance(c, dict):
            doc_id = c.get("document_id") or c.get("metadata", {}).get("document_id")
        if not doc_id and hasattr(c, "chunk_id"):
            doc_id = c.chunk_id.split("_chunk_")[0]
        if doc_id:
            context_doc_ids.add(doc_id)

    if not cited_ids:
        # If no citation was extracted:
        # If context was provided, missing citation is penalized
        return {
            "citation_correctness": 0.0 if context_doc_ids else 1.0,
            "is_valid": False if context_doc_ids else True,
            "cited_document_ids": [],
            "valid_document_ids": [],
            "invalid_document_ids": [],
            "available_context_doc_ids": list(context_doc_ids),
        }

    valid_ids = [d for d in cited_ids if d in context_doc_ids]
    invalid_ids = [d for d in cited_ids if d not in context_doc_ids]

    correctness = len(valid_ids) / len(cited_ids) if cited_ids else 0.0
    is_valid = correctness >= threshold

    return {
        "citation_correctness": round(correctness, 4),
        "is_valid": is_valid,
        "cited_document_ids": cited_ids,
        "valid_document_ids": valid_ids,
        "invalid_document_ids": invalid_ids,
        "available_context_doc_ids": list(context_doc_ids),
    }
