"""
scripts/evaluation/chunk_documents.py
=======================================
DVC pipeline Stage 3: Chunk cleaned documents.

Input:  data/processed/extracted_texts.jsonl
Output: data/processed/chunks.jsonl

Run:
    python scripts/evaluation/chunk_documents.py
    # or via DVC:
    dvc repro chunk_documents
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jsonlines

from app.validation.chunker import DocumentChunker, ChunkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    input_path = ROOT / "data" / "processed" / "extracted_texts.jsonl"
    output_path = ROOT / "data" / "processed" / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run extract_and_clean.py first.")
        sys.exit(1)

    # Load extracted texts
    docs = []
    with jsonlines.open(input_path) as reader:
        for doc in reader:
            docs.append(doc)
    logger.info(f"Loaded {len(docs)} documents from {input_path}")

    # Chunk using config from params.yaml
    chunker = DocumentChunker(params_path=str(ROOT / "params.yaml"))
    logger.info(f"Chunker config: {chunker.config_dict}")

    all_chunks = chunker.chunk_batch(docs)

    # Write chunks to JSONL
    with jsonlines.open(output_path, mode="w") as writer:
        for chunk in all_chunks:
            writer.write(chunk)

    # Summary statistics
    sizes = [c["char_count"] for c in all_chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    docs_chunked = len(set(c["document_id"] for c in all_chunks))

    print()
    print("=" * 55)
    print("CHUNKING SUMMARY")
    print("=" * 55)
    print(f"Documents processed  : {docs_chunked}")
    print(f"Total chunks created : {len(all_chunks)}")
    print(f"Avg chunk size       : {avg_size:.0f} chars")
    print(f"Min chunk size       : {min(sizes) if sizes else 0} chars")
    print(f"Max chunk size       : {max(sizes) if sizes else 0} chars")
    print(f"Chunk size config    : {chunker.config.chunk_size} chars")
    print(f"Overlap config       : {chunker.config.chunk_overlap} chars")
    print(f"Output               : {output_path}")

    # Check for issues
    oversized = [c for c in all_chunks if c["char_count"] > chunker.config.max_chunk_size]
    if oversized:
        logger.warning(f"{len(oversized)} oversized chunks (> {chunker.config.max_chunk_size} chars)")

    undersized = [c for c in all_chunks if c["char_count"] < chunker.config.min_chunk_size]
    if undersized:
        logger.warning(f"{len(undersized)} chunks below minimum size — these were NOT written if filtered")

    if len(all_chunks) == 0:
        logger.error("No chunks produced. Check extraction step.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
