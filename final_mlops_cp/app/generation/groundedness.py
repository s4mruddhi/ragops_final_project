"""
app/generation/groundedness.py
==============================
Groundedness verification for RAG generated responses.
Measures the extent to which claims in the answer are supported by retrieved context.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Sequence

import yaml

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "won't", "would", "wouldn't", "you", "your", "yours",
    "yourself", "yourselves",
}


def _load_groundedness_params(params_path: str = "params.yaml") -> dict:
    p = Path(params_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("generation", {})
    return {}


def tokenize_content_words(text: str) -> set[str]:
    """Extract lowercase content words, filtering out stopwords and numbers-only punctuation."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def extract_claims(answer: str, min_length: int = 10) -> list[str]:
    """
    Split the answer text into individual sentences/claims,
    ignoring headers and the 'Sources:' section.
    """
    # Remove 'Sources:' section and everything after
    body = re.split(r"\n\s*(?:Sources|References):", answer, flags=re.IGNORECASE)[0]

    # Split by common sentence delimiters
    raw_sentences = re.split(r"[.\n!?]+", body)
    claims = [s.strip() for s in raw_sentences if len(s.strip()) >= min_length]
    return claims


def compute_groundedness(
    answer: str,
    context_chunks: Sequence[Any],
    threshold: float = 0.75,
    min_claim_length: int = 10,
) -> dict[str, Any]:
    """
    Calculate groundedness score for an answer given the retrieved context.

    Args:
        answer: Generated answer text.
        context_chunks: Sequence of RetrievedChunk or dicts containing 'text'.
        threshold: Minimum score to be considered grounded.
        min_claim_length: Minimum character length for a claim.

    Returns:
        Dict with keys:
          - 'score': float in [0.0, 1.0]
          - 'is_grounded': bool (score >= threshold)
          - 'claims_evaluated': int
          - 'grounded_claims': int
          - 'claim_details': list of dicts with claim, overlap, and is_supported
    """
    if not answer or not answer.strip():
        return {
            "score": 0.0,
            "is_grounded": False,
            "claims_evaluated": 0,
            "grounded_claims": 0,
            "claim_details": [],
        }

    # Aggregate context text
    context_texts = []
    for c in context_chunks:
        if hasattr(c, "text"):
            context_texts.append(c.text)
        elif isinstance(c, dict):
            context_texts.append(c.get("text", ""))
        else:
            context_texts.append(str(c))
    full_context = " ".join(context_texts)
    context_words = tokenize_content_words(full_context)

    claims = extract_claims(answer, min_length=min_claim_length)
    if not claims:
        # Fallback if no full sentence detected (e.g. single short statement)
        claims = [answer.strip()]

    claim_details = []
    scores = []

    for claim in claims:
        claim_words = tokenize_content_words(claim)
        if not claim_words:
            continue

        overlap = len(claim_words.intersection(context_words)) / len(claim_words)
        is_supported = overlap >= 0.50  # Over half the substantive content words found
        scores.append(overlap)

        claim_details.append({
            "claim": claim,
            "overlap": round(overlap, 3),
            "is_supported": is_supported,
        })

    if not scores:
        final_score = 1.0 if not context_words else 0.0
    else:
        final_score = float(sum(scores) / len(scores))

    final_score = max(0.0, min(1.0, final_score))
    is_grounded = final_score >= threshold

    grounded_count = sum(1 for c in claim_details if c["is_supported"])

    return {
        "score": round(final_score, 4),
        "is_grounded": is_grounded,
        "claims_evaluated": len(claim_details),
        "grounded_claims": grounded_count,
        "claim_details": claim_details,
    }
