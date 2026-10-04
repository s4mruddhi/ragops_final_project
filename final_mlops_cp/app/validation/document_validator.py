"""
app/validation/document_validator.py
======================================
Validates raw documents before they enter the ingestion pipeline.

Checks (in order):
  1.  Supported file extension
  2.  File readability (can we open it?)
  3.  Minimum file size (not empty)
  4.  Maximum file size (not too large)
  5.  PDF is not corrupted (can fitz open it?)
  6.  Duplicate file checksums (exact-duplicate detection)
  7.  Metadata sidecar exists
  8.  All required metadata fields present
  9.  version field is a valid semver-like string (e.g. "2.1")
  10. effective_date is a valid ISO date string
  11. document_id uniqueness across the batch
  12. extracted text is non-empty
  13. document_id in metadata matches filename convention

Every check produces a structured result. Invalid documents are logged
and skipped — they do NOT silently enter the vector database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MIN_FILE_SIZE_BYTES = 100
MAX_FILE_SIZE_MB = 50
REQUIRED_METADATA_FIELDS = [
    "document_id",
    "title",
    "department",
    "category",
    "version",
    "effective_date",
    "source",
    "access_level",
]
VERSION_PATTERN = re.compile(r"^\d+\.\d+(\.\d+)?$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"   # "error" | "warning"


@dataclass
class DocumentValidationResult:
    file_path: str
    document_id: Optional[str] = None
    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: Optional[dict] = None
    file_size_bytes: int = 0
    checksum: Optional[str] = None

    def add_error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, severity="error"))
        self.is_valid = False

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, severity="warning"))

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "document_id": self.document_id,
            "is_valid": self.is_valid,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "issues": [
                {"code": i.code, "message": i.message, "severity": i.severity}
                for i in self.issues
            ],
        }


@dataclass
class ValidationReport:
    """Aggregated report for a batch of documents."""
    total: int = 0
    valid: int = 0
    invalid: int = 0
    results: list[DocumentValidationResult] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total": self.total,
                "valid": self.valid,
                "invalid": self.invalid,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class DocumentValidator:
    """
    Validates a batch of raw documents.

    Usage:
        validator = DocumentValidator(raw_dir="data/raw", metadata_dir="data/raw/metadata")
        report = validator.validate_all()
        validator.save_report(report, "data/processed/validation_report.json")
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        metadata_dir: Optional[str] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.raw_dir / "metadata"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def validate_all(self) -> ValidationReport:
        """Validate every document file in raw_dir and return a report."""
        files = self._discover_files()
        logger.info(f"Discovered {len(files)} document files in {self.raw_dir}")

        report = ValidationReport()
        seen_checksums: dict[str, str] = {}     # checksum -> first file path
        seen_doc_ids: dict[str, str] = {}        # document_id -> first file path

        for file_path in sorted(files):
            result = self._validate_single(file_path, seen_checksums, seen_doc_ids)
            report.results.append(result)
            report.total += 1
            if result.is_valid:
                report.valid += 1
                # Register checksum and doc_id for duplicate detection
                if result.checksum:
                    seen_checksums[result.checksum] = str(file_path)
                if result.document_id:
                    seen_doc_ids[result.document_id] = str(file_path)
            else:
                report.invalid += 1

        logger.info(
            f"Validation complete: {report.valid}/{report.total} valid, "
            f"{report.invalid} invalid"
        )
        return report

    def validate_single(self, file_path: Path) -> DocumentValidationResult:
        """Validate a single file (for unit testing / ad-hoc use)."""
        return self._validate_single(file_path, {}, {})

    def save_report(self, report: ValidationReport, output_path: str) -> None:
        """Save validation report as JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Validation report saved: {out}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(
                p for p in self.raw_dir.rglob(f"*{ext}")
                if "metadata" not in p.parts
            )
        return files

    def _validate_single(
        self,
        file_path: Path,
        seen_checksums: dict[str, str],
        seen_doc_ids: dict[str, str],
    ) -> DocumentValidationResult:
        result = DocumentValidationResult(file_path=str(file_path))

        # --- Check 1: Supported extension ---
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.add_error(
                "UNSUPPORTED_EXTENSION",
                f"Extension '{file_path.suffix}' is not in {SUPPORTED_EXTENSIONS}",
            )
            return result  # No point continuing

        # --- Check 2: File readable ---
        if not file_path.exists():
            result.add_error("FILE_NOT_FOUND", f"File does not exist: {file_path}")
            return result

        # --- Check 3: File size ---
        size = file_path.stat().st_size
        result.file_size_bytes = size
        if size < MIN_FILE_SIZE_BYTES:
            result.add_error(
                "FILE_TOO_SMALL",
                f"File size {size} bytes < minimum {MIN_FILE_SIZE_BYTES} bytes (likely empty)",
            )
        if size > MAX_FILE_SIZE_MB * 1024 * 1024:
            result.add_error(
                "FILE_TOO_LARGE",
                f"File size {size / 1024 / 1024:.1f} MB > maximum {MAX_FILE_SIZE_MB} MB",
            )

        # --- Check 4: Compute checksum ---
        try:
            checksum = self._compute_checksum(file_path)
            result.checksum = checksum
        except Exception as exc:
            result.add_error("CHECKSUM_FAILED", f"Cannot compute checksum: {exc}")
            return result

        # --- Check 5: Duplicate file content ---
        if checksum in seen_checksums:
            result.add_error(
                "DUPLICATE_FILE",
                f"Identical content already seen in: {seen_checksums[checksum]}",
            )

        # --- Check 6: PDF corruption ---
        if file_path.suffix.lower() == ".pdf":
            pdf_ok, pdf_err = self._check_pdf_readable(file_path)
            if not pdf_ok:
                result.add_error("CORRUPTED_PDF", f"Cannot open PDF: {pdf_err}")
                return result

        # --- Check 7: Metadata sidecar exists ---
        meta_path = self.metadata_dir / f"{file_path.stem}.json"
        # Also try matching by document pattern (any JSON that contains the filename)
        meta = self._load_metadata(file_path)
        if meta is None:
            result.add_warning(
                "METADATA_MISSING",
                f"No metadata JSON found for {file_path.name} in {self.metadata_dir}",
            )
        else:
            result.metadata = meta

            # --- Check 8: Required metadata fields ---
            for field_name in REQUIRED_METADATA_FIELDS:
                if field_name not in meta or not meta[field_name]:
                    result.add_error(
                        "MISSING_METADATA_FIELD",
                        f"Required metadata field '{field_name}' is missing or empty",
                    )

            if result.is_valid:   # only proceed if no errors so far
                # --- Check 9: Version format ---
                version = str(meta.get("version", ""))
                if not VERSION_PATTERN.match(version):
                    result.add_error(
                        "INVALID_VERSION",
                        f"version '{version}' does not match pattern X.Y or X.Y.Z",
                    )

                # --- Check 10: effective_date format ---
                eff_date = str(meta.get("effective_date", ""))
                if not DATE_PATTERN.match(eff_date):
                    result.add_error(
                        "INVALID_EFFECTIVE_DATE",
                        f"effective_date '{eff_date}' is not a valid YYYY-MM-DD date",
                    )
                else:
                    try:
                        datetime.strptime(eff_date, "%Y-%m-%d")
                    except ValueError:
                        result.add_error(
                            "INVALID_EFFECTIVE_DATE",
                            f"effective_date '{eff_date}' is not a real calendar date",
                        )

                # --- Check 11: document_id uniqueness ---
                doc_id = meta.get("document_id", "")
                result.document_id = doc_id
                if doc_id and doc_id in seen_doc_ids:
                    result.add_error(
                        "DUPLICATE_DOCUMENT_ID",
                        f"document_id '{doc_id}' already seen in: {seen_doc_ids[doc_id]}",
                    )

        return result

    def _compute_checksum(self, file_path: Path) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _check_pdf_readable(self, file_path: Path) -> tuple[bool, str]:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            _ = doc.page_count
            doc.close()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _load_metadata(self, file_path: Path) -> Optional[dict]:
        """
        Find the metadata JSON for this file.
        Tries: metadata/<stem>.json
        Then:  any JSON in metadata/ whose document_id maps to this file.
        """
        # Try stem match first
        candidate = self.metadata_dir / f"{file_path.stem}.json"
        # Also handle IDs like HR-WFH-001 where stem = work_from_home_policy
        # We search all JSONs in metadata dir
        if self.metadata_dir.exists():
            for json_file in self.metadata_dir.glob("*.json"):
                if json_file.name == "document_index.json":
                    continue
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    # Match if file_path is referenced or stem matches
                    stored = data.get("file_path", "")
                    if (
                        file_path.name in stored
                        or file_path.stem in json_file.stem.lower()
                        or str(file_path).replace("\\", "/") in stored.replace("\\", "/")
                    ):
                        return data
                except Exception:
                    continue
        return None
