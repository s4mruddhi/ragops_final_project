"""
app/vectorstore/embedder.py
============================
Generates dense vector embeddings for text chunks using sentence-transformers.

Model: all-MiniLM-L6-v2 (default)
  - 384 dimensions
  - Very fast on CPU (~14k sentences/sec)
  - Good quality for semantic similarity

Design:
  - Singleton pattern — model loaded once, reused across calls
  - Batch processing for efficiency
  - Reads model name / batch size / device from params.yaml
  - Returns numpy arrays (float32) compatible with ChromaDB

Usage:
    embedder = Embedder()
    vectors = embedder.embed_texts(["text 1", "text 2"])
    # vectors.shape == (2, 384)
"""

from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


def _load_embedding_params(params_path: str = "params.yaml") -> dict:
    p = Path(params_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("embedding", {})
    return {}


class Embedder:
    """
    Wraps sentence-transformers for batch text embedding.

    Args:
        model_name: HuggingFace model identifier.
        batch_size:  Number of texts per GPU/CPU batch.
        device:      'cpu' or 'cuda'.
        params_path: Path to params.yaml (used if model_name not provided).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        params_path: str = "params.yaml",
    ):
        params = _load_embedding_params(params_path)

        self.model_name = model_name or params.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.batch_size = batch_size or params.get("batch_size", 32)
        self.device = device or params.get("device", "cpu")
        self._model = None  # lazy load
        self.backend = "sentence-transformers"
        self._fallback_dim = 384

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim), float32.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        model = self._get_model()
        if model is None:
            return self._hash_embed(texts)
        logger.info(
            f"Embedding {len(texts)} texts with {self.model_name} "
            f"(batch_size={self.batch_size}, device={self.device})"
        )

        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
        )
        logger.info(f"Embedding complete: shape={vectors.shape}, dtype={vectors.dtype}")
        return vectors.astype(np.float32)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of chunk dicts (from chunker.py).
        Adds 'embedding' key (list[float]) to each chunk dict.

        Args:
            chunks: List of chunk dicts with at least a 'text' key.

        Returns:
            Same list with 'embedding' added.
        """
        texts = [c["text"] for c in chunks]
        vectors = self.embed_texts(texts)

        enriched = []
        for chunk, vec in zip(chunks, vectors):
            out = dict(chunk)
            out["embedding"] = vec.tolist()   # ChromaDB expects list[float]
            enriched.append(out)
        return enriched

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        Returns 1-D numpy array of shape (embedding_dim,).
        """
        vectors = self.embed_texts([query])
        return vectors[0]

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        model = self._get_model()
        if model is None:
            return self._fallback_dim
        return model.get_sentence_embedding_dimension()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None and self.backend == "sentence-transformers":
            strict = os.environ.get("EMBEDDER_STRICT", "0") == "1"
            try:
                if SentenceTransformer is None:
                    raise RuntimeError("sentence-transformers not installed")
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                if strict:
                    raise
                # Offline / no-HF-access fallback so the pipeline never silently returns nothing.
                logger.warning(
                    f"Could not load '{self.model_name}' ({exc}). "
                    "Falling back to deterministic hashing embeddings (backend='hashing'). "
                    "Set EMBEDDER_STRICT=1 to make this an error."
                )
                self.backend = "hashing"
        return self._model

    def _hash_embed(self, texts: list[str]) -> np.ndarray:
        """Deterministic word + bigram hashing embeddings (L2-normalised, 384-d)."""
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(
            n_features=self._fallback_dim,
            ngram_range=(1, 2),
            alternate_sign=True,
            norm="l2",
            stop_words="english",
        )
        return vec.transform(texts).toarray().astype(np.float32)
