"""
app/vectorstore/chromadb_store.py
===================================
ChromaDB vector store wrapper for RAGOps.

Responsibilities:
  - Build (or rebuild) the collection from embedded chunks
  - Semantic search: query -> top-k chunks with scores
  - Persist to disk (data/vectorstore/) so it survives restarts
  - Collection statistics for monitoring

ChromaDB collection schema:
    - id:        chunk_id (e.g. "HR-WFH-001_chunk_007")
    - embedding: list[float] (384 dims, L2-normalised)
    - document:  chunk text
    - metadata:  department, title, version, effective_date, chunk_index, total_chunks, ...

Usage:
    store = ChromaDBStore()
    store.build_from_chunks(embedded_chunks)
    results = store.query("How many days of annual leave?", top_k=5)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logger = logging.getLogger(__name__)


def _load_vectorstore_params(params_path: str = "params.yaml") -> dict:
    p = Path(params_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("vectorstore", {})
    return {}


class ChromaDBStore:
    """
    Persistent ChromaDB vector store.

    Args:
        persist_dir:      Directory to persist the ChromaDB database.
        collection_name:  Name of the ChromaDB collection.
        distance_metric:  'cosine' | 'l2' | 'ip'.
        params_path:      Path to params.yaml.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
        params_path: str = "params.yaml",
    ):
        params = _load_vectorstore_params(params_path)

        self.persist_dir = persist_dir or params.get("persist_dir", "data/vectorstore")
        self.collection_name = collection_name or params.get(
            "collection_name", "ragops_v1"
        )
        self.distance_metric = distance_metric or params.get(
            "distance_metric", "cosine"
        )
        self._client = None
        self._collection = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build_from_chunks(
        self,
        embedded_chunks: list[dict],
        reset: bool = False,
    ) -> int:
        """
        Build (or rebuild) the collection from a list of embedded chunks.

        Args:
            embedded_chunks: List of chunk dicts with 'embedding', 'text',
                             'chunk_id', 'metadata' keys.
            reset:           If True, drop and recreate the collection.

        Returns:
            Number of chunks indexed.
        """
        collection = self._get_collection(reset=reset)

        if not embedded_chunks:
            logger.warning("No chunks to index.")
            return 0

        # Prepare batches: ChromaDB 0.5+ enforces a strict max batch size (e.g. 166).
        # We batch to 100 to guarantee safe ingestion without BatchSizeExceededError.
        BATCH = 100
        total_indexed = 0

        for i in range(0, len(embedded_chunks), BATCH):
            batch = embedded_chunks[i : i + BATCH]

            ids = [c["chunk_id"] for c in batch]
            embeddings = [c["embedding"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = [self._sanitise_metadata(c["metadata"]) for c in batch]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_indexed += len(batch)
            logger.info(
                f"Indexed batch {i // BATCH + 1}: "
                f"{len(batch)} chunks (total so far: {total_indexed})"
            )


        logger.info(
            f"Index build complete. Collection '{self.collection_name}' "
            f"has {total_indexed} chunks."
        )
        return total_indexed

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search: find the top_k most similar chunks.

        Args:
            query_embedding: Query vector (list[float] or np.ndarray).
            top_k:           Number of results to return.
            where:           Optional metadata filter dict (ChromaDB 'where' format).
                             e.g. {"department": "HR"}

        Returns:
            List of result dicts, each containing:
                chunk_id, text, metadata, score (cosine similarity 0-1).
            Sorted by score descending (most relevant first).
        """
        collection = self._get_collection()

        # Guard: if collection is empty, ChromaDB rejects n_results > 0
        current_count = collection.count()
        if current_count == 0:
            raise RuntimeError(
                f"Vector index '{self.collection_name}' is empty. "
                "Run `dvc repro build_index` (or scripts/evaluation/build_index.py) first."
            )

        if hasattr(query_embedding, "tolist"):
            query_vec = query_embedding.tolist()
        else:
            query_vec = [float(x) for x in query_embedding]

        actual_k = min(top_k, current_count)

        kwargs = dict(
            query_embeddings=[query_vec],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        hits = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            score = 1.0 - (dist / 2.0) if self.distance_metric == "cosine" else -dist
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "score": round(float(score), 4),
                "distance": round(float(dist), 4),
            })

        # Already sorted by ChromaDB (ascending distance = descending similarity)
        return hits

    def get_by_document_id(self, document_id: str) -> list[dict]:
        """Retrieve all chunks for a specific document_id."""
        collection = self._get_collection()
        results = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        chunks = []
        for cid, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            chunks.append({"chunk_id": cid, "text": doc, "metadata": meta})
        return chunks

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        collection = self._get_collection()
        return collection.count()

    def collection_stats(self) -> dict:
        """Return statistics about the collection (for monitoring)."""
        collection = self._get_collection()
        n = collection.count()

        if n == 0:
            return {"collection": self.collection_name, "count": 0}

        # Sample some metadata to compute dept distribution
        sample = collection.get(
            limit=min(n, 1000),
            include=["metadatas"],
        )
        depts: dict[str, int] = {}
        for meta in sample["metadatas"]:
            dept = meta.get("department", "unknown")
            depts[dept] = depts.get(dept, 0) + 1

        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "distance_metric": self.distance_metric,
            "total_chunks": n,
            "department_distribution": depts,
        }

    def delete_collection(self) -> None:
        """Drop the entire collection (use with care)."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Collection '{self.collection_name}' deleted.")

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _migrate_sqlite_schema(self, persist_path: Path) -> None:
        """Fix legacy ChromaDB sqlite schema where config_json_str lacks _type."""
        db_file = persist_path / "chroma.sqlite3"
        if not db_file.exists():
            return
        try:
            import json
            import sqlite3
            con = sqlite3.connect(str(db_file))
            cur = con.cursor()
            rows = cur.execute("SELECT id, config_json_str FROM collections").fetchall()
            updated = False
            for cid, cjson in rows:
                if cjson:
                    data = json.loads(cjson)
                    if "_type" not in data:
                        data["_type"] = "CollectionConfigurationInternal"
                        cur.execute(
                            "UPDATE collections SET config_json_str = ? WHERE id = ?",
                            (json.dumps(data), cid),
                        )
                        updated = True
            if updated:
                con.commit()
                logger.info("Auto-migrated ChromaDB collections schema with '_type'.")
            con.close()
        except Exception as exc:
            logger.debug(f"SQLite migration check skipped: {exc}")

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise RuntimeError(
                    "chromadb not installed. Run: pip install chromadb"
                )
            persist_path = Path(self.persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._migrate_sqlite_schema(persist_path)
            self._client = chromadb.PersistentClient(path=str(persist_path))
            logger.info(f"ChromaDB client initialised at: {persist_path}")
        return self._client


    def _get_collection(self, reset: bool = False):
        if reset and self._collection is not None:
            self.delete_collection()

        if self._collection is None:
            client = self._get_client()
            import chromadb

            metadata = {"hnsw:space": self.distance_metric}
            try:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata=metadata,
                )
            except Exception as exc:
                if reset or isinstance(exc, KeyError) or "_type" in str(exc):
                    logger.warning(
                        f"ChromaDB schema version mismatch detected ({exc}). Re-initializing persistence directory."
                    )
                    self._client = None
                    persist_path = Path(self.persist_dir)
                    import shutil
                    for item in persist_path.iterdir():
                        if item.name == "index_stats.json":
                            continue
                        if item.is_file():
                            try:
                                item.unlink()
                            except Exception:
                                pass
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    client = self._get_client()
                    self._collection = client.get_or_create_collection(
                        name=self.collection_name,
                        metadata=metadata,
                    )
                else:
                    raise

            if reset:
                # If reset, clear all existing entries in batches <= 100
                existing = self._collection.count()
                if existing > 0:
                    all_ids = self._collection.get(include=[])["ids"]
                    for i in range(0, len(all_ids), 100):
                        batch_ids = all_ids[i : i + 100]
                        self._collection.delete(ids=batch_ids)
                    logger.info(f"Cleared {existing} existing chunks from collection.")
        return self._collection



    @staticmethod
    def _sanitise_metadata(meta: dict) -> dict:
        """
        ChromaDB metadata values must be str, int, float, or bool.
        Convert any other types to strings.
        """
        sanitised = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                sanitised[k] = v
            elif v is None:
                sanitised[k] = ""
            else:
                sanitised[k] = str(v)
        return sanitised
