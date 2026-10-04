"""
app/validation/chunker.py
==========================
Splits cleaned document text into overlapping chunks with full metadata.

Design decisions:
  - Character-based splitting (not token-based) for simplicity and reproducibility
    without requiring a tokenizer at index time.
  - Overlap is a hard trailing overlap — the last `overlap` chars of chunk N
    appear at the start of chunk N+1. This ensures context continuity.
  - Chunk boundaries respect sentence boundaries where possible (soft split).
  - Every chunk carries the full document metadata so retrieval results are
    self-contained — no join back to a metadata store is needed.

Configuration (read from params.yaml or passed directly):
  chunk_size:    target chunk size in characters   (default: 512)
  chunk_overlap: overlap in characters             (default: 64)
  min_chunk_size: discard chunks smaller than this (default: 50)
  max_chunk_size: warn on chunks larger than this  (default: 1024)

Each output chunk:
    {
        "chunk_id":       "HR-WFH-001_chunk_007",
        "document_id":    "HR-WFH-001",
        "chunk_index":    7,
        "total_chunks":   12,
        "text":           "...",
        "char_count":     487,
        "metadata": {
            "document_id":    "HR-WFH-001",
            "title":          "Work From Home Policy",
            "department":     "HR",
            "category":       "Remote Work",
            "version":        "2.1",
            "effective_date": "2026-01-01",
            "source":         "internal_policy",
            "access_level":   "employee",
            "chunk_index":    7,
            "total_chunks":   12,
        }
    }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Sentence boundary pattern for soft splitting
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _load_params(params_path: str = "params.yaml") -> dict:
    p = Path(params_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    max_chunk_size: int = 1024

    @classmethod
    def from_params(cls, params_path: str = "params.yaml") -> "ChunkConfig":
        params = _load_params(params_path)
        cfg = params.get("chunking", {})
        return cls(
            chunk_size=cfg.get("chunk_size", 512),
            chunk_overlap=cfg.get("chunk_overlap", 64),
            min_chunk_size=cfg.get("min_chunk_size", 50),
            max_chunk_size=cfg.get("max_chunk_size", 1024),
        )


class DocumentChunker:
    """
    Splits a cleaned document into overlapping text chunks,
    attaching full metadata to every chunk.

    Usage:
        chunker = DocumentChunker()              # reads params.yaml
        chunks  = chunker.chunk(extracted_doc)  # one doc at a time
        all     = chunker.chunk_batch(docs)     # list of docs
    """

    def __init__(self, config: Optional[ChunkConfig] = None, params_path: str = "params.yaml"):
        self.config = config or ChunkConfig.from_params(params_path)
        logger.debug(
            f"Chunker initialised: size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def chunk(self, extracted_doc: dict) -> list[dict]:
        """
        Chunk a single extracted document.

        Args:
            extracted_doc: dict from TextExtractor/TextCleaner with keys:
                           document_id, text, metadata, ...

        Returns:
            List of chunk dicts with chunk_id, text, metadata, etc.
        """
        text = extracted_doc.get("text", "")
        doc_id = extracted_doc.get("document_id", "unknown")
        metadata = extracted_doc.get("metadata", {})

        if not text.strip():
            logger.warning(f"Empty text for document: {doc_id}")
            return []

        raw_chunks = self._split_text(text)

        # Filter out too-small chunks
        valid_chunks = [c for c in raw_chunks if len(c) >= self.config.min_chunk_size]

        # Warn on oversized chunks (don't discard — just flag)
        for i, c in enumerate(valid_chunks):
            if len(c) > self.config.max_chunk_size:
                logger.warning(
                    f"{doc_id} chunk {i}: size {len(c)} > max {self.config.max_chunk_size}"
                )

        total = len(valid_chunks)
        chunks_out = []

        for i, chunk_text in enumerate(valid_chunks):
            chunk_id = f"{doc_id}_chunk_{i:03d}"

            # Build per-chunk metadata — full doc metadata + chunk-specific fields
            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = total
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["document_id"] = doc_id

            chunks_out.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "total_chunks": total,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "metadata": chunk_meta,
            })

        logger.debug(f"{doc_id}: {total} chunks (size={self.config.chunk_size})")
        return chunks_out

    def chunk_batch(self, extracted_docs: list[dict]) -> list[dict]:
        """Chunk a list of extracted documents. Returns a flat list of all chunks."""
        all_chunks = []
        for doc in extracted_docs:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)
        logger.info(
            f"Chunked {len(extracted_docs)} documents -> {len(all_chunks)} total chunks"
        )
        return all_chunks

    # -----------------------------------------------------------------------
    # Splitting logic
    # -----------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into overlapping character-based chunks.

        Strategy:
          1. Try to split at sentence boundaries near the chunk_size target.
          2. If no sentence boundary found, hard-split at chunk_size.
          3. Each new chunk starts `chunk_overlap` chars before the previous end.
        """
        chunks = []
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        start = 0

        while start < len(text):
            end = min(start + size, len(text))

            # If not at the end of text, try to find a sentence boundary
            if end < len(text):
                # Look for a sentence boundary in the last 20% of the chunk
                search_start = max(start, end - size // 5)
                sub = text[search_start:end]
                matches = list(_SENTENCE_BOUNDARY.finditer(sub))
                if matches:
                    # Use the last sentence boundary in the window
                    last_match = matches[-1]
                    end = search_start + last_match.end()

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Next chunk starts with overlap behind the current end
            next_start = end - overlap
            if next_start <= start:
                # Safety: always advance to avoid infinite loop
                next_start = start + 1
            start = next_start

        return chunks

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @property
    def config_dict(self) -> dict:
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
            "max_chunk_size": self.config.max_chunk_size,
        }
