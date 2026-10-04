"""
scripts/evaluation/build_index.py
====================================
DVC pipeline Stage 4: Generate embeddings and build ChromaDB index.

Input:  data/processed/chunks.jsonl
Output: data/vectorstore/  (ChromaDB persistent store)

Run:
    python scripts/evaluation/build_index.py
    # or via DVC:
    dvc repro build_index
"""

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jsonlines

from app.vectorstore.embedder import Embedder
from app.vectorstore.chromadb_store import ChromaDBStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
    persist_dir = ROOT / "data" / "vectorstore"
    stats_path = ROOT / "data" / "vectorstore" / "index_stats.json"

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Run chunk_documents.py first.")
        sys.exit(1)

    # Load chunks
    chunks = []
    with jsonlines.open(chunks_path) as reader:
        for chunk in reader:
            chunks.append(chunk)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # -----------------------------------------------------------------------
    # Step 1: Embed all chunks
    # -----------------------------------------------------------------------
    embedder = Embedder(params_path=str(ROOT / "params.yaml"))
    logger.info(f"Embedding model: {embedder.model_name}")
    logger.info(f"Embedding dim: {embedder.embedding_dim}")

    t0 = time.time()
    embedded_chunks = embedder.embed_chunks(chunks)
    embed_time = time.time() - t0
    logger.info(f"Embedded {len(embedded_chunks)} chunks in {embed_time:.1f}s")

    # -----------------------------------------------------------------------
    # Step 2: Build ChromaDB index
    # -----------------------------------------------------------------------
    store = ChromaDBStore(
        persist_dir=str(persist_dir),
        params_path=str(ROOT / "params.yaml"),
    )

    t1 = time.time()
    n_indexed = store.build_from_chunks(embedded_chunks, reset=True)
    index_time = time.time() - t1
    logger.info(f"Indexed {n_indexed} chunks in {index_time:.1f}s")

    # -----------------------------------------------------------------------
    # Step 3: Save index statistics
    # -----------------------------------------------------------------------
    stats = store.collection_stats()
    stats["embedding_model"] = embedder.model_name
    stats["embedding_dim"] = embedder.embedding_dim
    stats["embedding_backend"] = embedder.backend
    stats["persist_dir"] = "data/vectorstore"  # relative: no machine-specific paths
    stats["total_chunks"] = n_indexed
    stats["embed_time_seconds"] = round(embed_time, 2)
    stats["index_time_seconds"] = round(index_time, 2)

    persist_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("INDEX BUILD SUMMARY")
    print("=" * 60)
    print(f"Chunks embedded       : {len(embedded_chunks)}")
    print(f"Chunks indexed        : {n_indexed}")
    print(f"Embedding model       : {embedder.model_name}")
    print(f"Embedding dimensions  : {embedder.embedding_dim}")
    print(f"Embed time            : {embed_time:.1f}s")
    print(f"Index time            : {index_time:.1f}s")
    print(f"Collection            : {store.collection_name}")
    print(f"Persist dir           : {persist_dir}")
    print(f"Department breakdown  : {stats.get('department_distribution', {})}")
    print(f"Stats saved to        : {stats_path}")

    if n_indexed == 0:
        logger.error("No chunks were indexed. Something went wrong.")
        sys.exit(1)

    # Quick smoke-test: query the index with a known question
    print()
    print("--- Smoke test: querying index ---")
    query_vec = embedder.embed_query("How many days of annual leave?")
    results = store.query(query_vec, top_k=3)
    if results:
        print(f"Top result: [{results[0]['score']:.3f}] {results[0]['metadata'].get('document_id')} | {results[0]['text'][:80]}...")
    else:
        logger.warning("Smoke test returned no results.")

    sys.exit(0)


if __name__ == "__main__":
    main()
