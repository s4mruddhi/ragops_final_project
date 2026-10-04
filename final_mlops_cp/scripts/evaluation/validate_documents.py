"""
scripts/evaluation/validate_documents.py
==========================================
DVC pipeline Stage 1: Validate raw documents.

Run:
    python scripts/evaluation/validate_documents.py
    # or via DVC:
    dvc repro validate_documents
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.validation.document_validator import DocumentValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    raw_dir = ROOT / "data" / "raw"
    metadata_dir = raw_dir / "metadata"
    output_path = ROOT / "data" / "processed" / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Validating documents in: {raw_dir}")
    logger.info(f"Metadata directory:       {metadata_dir}")

    validator = DocumentValidator(
        raw_dir=str(raw_dir),
        metadata_dir=str(metadata_dir),
    )
    report = validator.validate_all()
    validator.save_report(report, str(output_path))

    # Print summary
    print()
    print("=" * 55)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 55)
    print(f"Total documents scanned : {report.total}")
    print(f"Valid                   : {report.valid}")
    print(f"Invalid                 : {report.invalid}")
    print()

    if report.invalid > 0:
        print("INVALID DOCUMENTS:")
        for r in report.results:
            if not r.is_valid:
                print(f"  {r.file_path}")
                for issue in r.issues:
                    print(f"    [{issue.severity.upper()}] {issue.code}: {issue.message}")

    print()
    print(f"Report saved: {output_path}")

    # Exit 0 even if some docs are invalid — pipeline continues with valid subset
    # (fail_on_invalid is False in pipeline.yaml)
    sys.exit(0)


if __name__ == "__main__":
    main()
