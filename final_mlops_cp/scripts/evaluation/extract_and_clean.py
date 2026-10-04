"""
scripts/evaluation/extract_and_clean.py
=========================================
DVC pipeline Stage 2: Extract text from validated documents, clean it.

Input:  data/raw/ + data/processed/validation_report.json
Output: data/processed/extracted_texts.jsonl

Run:
    python scripts/evaluation/extract_and_clean.py
    # or via DVC:
    dvc repro extract_and_clean
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jsonlines

from app.validation.text_extractor import TextExtractor
from app.validation.text_cleaner import TextCleaner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    raw_dir = ROOT / "data" / "raw"
    metadata_dir = raw_dir / "metadata"
    validation_report_path = ROOT / "data" / "processed" / "validation_report.json"
    output_path = ROOT / "data" / "processed" / "extracted_texts.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load validation report to get list of valid files only
    if not validation_report_path.exists():
        logger.error(f"Validation report not found: {validation_report_path}")
        logger.error("Run validate_documents.py first.")
        sys.exit(1)

    with open(validation_report_path, encoding="utf-8") as f:
        report = json.load(f)

    valid_files = [
        Path(r["file_path"])
        for r in report["results"]
        if r["is_valid"]
    ]
    logger.info(f"Extracting text from {len(valid_files)} valid documents")

    extractor = TextExtractor()
    cleaner = TextCleaner()

    extracted = extractor.extract_batch(valid_files, metadata_dir=metadata_dir)
    logger.info(f"Extracted text from {len(extracted)} documents")

    cleaned = cleaner.clean_batch(extracted)
    logger.info(f"Cleaned text: {len(cleaned)} documents")

    # Write to JSONL
    written = 0
    with jsonlines.open(output_path, mode="w") as writer:
        for doc in cleaned:
            writer.write(doc)
            written += 1

    print()
    print("=" * 55)
    print("EXTRACTION & CLEANING SUMMARY")
    print("=" * 55)
    print(f"Valid files from report : {len(valid_files)}")
    print(f"Successfully extracted  : {len(extracted)}")
    print(f"Successfully cleaned    : {len(cleaned)}")
    print(f"Written to JSONL        : {written}")
    total_chars = sum(d.get("char_count", 0) for d in cleaned)
    print(f"Total characters        : {total_chars:,}")
    print(f"Output: {output_path}")

    if written == 0:
        logger.error("No documents extracted — check validation report for errors.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
