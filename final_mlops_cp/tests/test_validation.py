"""
tests/test_validation.py
==========================
Unit tests for document validation logic.
Run: pytest tests/test_validation.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.validation.document_validator import (
    DocumentValidator,
    DocumentValidationResult,
    REQUIRED_METADATA_FIELDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_doc_dir(tmp_path):
    """Create a temporary directory structure with valid docs and metadata."""
    raw_dir = tmp_path / "raw"
    meta_dir = raw_dir / "metadata"
    hr_dir = raw_dir / "hr"
    raw_dir.mkdir()
    meta_dir.mkdir()
    hr_dir.mkdir()
    return tmp_path, raw_dir, meta_dir, hr_dir


def _write_txt(path: Path, content: str = "Sample policy content. " * 20) -> None:
    path.write_text(content, encoding="utf-8")


def _write_meta(meta_dir: Path, doc_id: str, file_path: str, overrides: dict = None) -> None:
    meta = {
        "document_id": doc_id,
        "title": f"Test Document {doc_id}",
        "department": "HR",
        "category": "Policy",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "source": "internal_policy",
        "access_level": "employee",
        "file_path": file_path,
    }
    if overrides:
        meta.update(overrides)
    (meta_dir / f"{doc_id}.json").write_text(json.dumps(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: File-level checks
# ---------------------------------------------------------------------------

class TestFileChecks:
    def test_valid_txt_document_passes(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc_file = hr_dir / "test_policy.txt"
        _write_txt(doc_file)
        _write_meta(meta_dir, "HR-TEST-001", str(doc_file))

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc_file)
        assert result.is_valid, f"Expected valid, got issues: {result.issues}"

    def test_unsupported_extension_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        bad_file = hr_dir / "presentation.pptx"
        bad_file.write_text("content")

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(bad_file)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "UNSUPPORTED_EXTENSION" in codes

    def test_empty_file_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        empty_file = hr_dir / "empty.txt"
        empty_file.write_text("")

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(empty_file)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "FILE_TOO_SMALL" in codes

    def test_tiny_file_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        tiny = hr_dir / "tiny.txt"
        tiny.write_text("Hi")   # 2 bytes < 100 minimum

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(tiny)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "FILE_TOO_SMALL" in codes

    def test_nonexistent_file_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, _ = tmp_doc_dir
        missing = raw_dir / "does_not_exist.txt"

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(missing)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "FILE_NOT_FOUND" in codes


# ---------------------------------------------------------------------------
# Tests: Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_duplicate_checksum_detected(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        content = "Identical content. " * 30

        file1 = hr_dir / "policy_a.txt"
        file2 = hr_dir / "policy_b.txt"
        file1.write_text(content, encoding="utf-8")
        file2.write_text(content, encoding="utf-8")

        _write_meta(meta_dir, "HR-A-001", str(file1))
        _write_meta(meta_dir, "HR-B-001", str(file2))

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        report = validator.validate_all()

        codes_all = [i.code for r in report.results for i in r.issues]
        assert "DUPLICATE_FILE" in codes_all

    def test_duplicate_document_id_detected(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir

        file1 = hr_dir / "policy_x.txt"
        file2 = hr_dir / "policy_y.txt"
        _write_txt(file1, "Content of policy x. " * 20)
        _write_txt(file2, "Content of policy y. Different. " * 20)

        # Both have the SAME document_id
        _write_meta(meta_dir, "HR-DUPE-001", str(file1))
        (meta_dir / "HR-DUPE-001-B.json").write_text(json.dumps({
            "document_id": "HR-DUPE-001",  # same ID!
            "title": "Duplicate ID Test",
            "department": "HR",
            "category": "Test",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "source": "internal_policy",
            "access_level": "employee",
            "file_path": str(file2),
        }), encoding="utf-8")

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        report = validator.validate_all()

        codes_all = [i.code for r in report.results for i in r.issues]
        assert "DUPLICATE_DOCUMENT_ID" in codes_all


# ---------------------------------------------------------------------------
# Tests: Metadata validation
# ---------------------------------------------------------------------------

class TestMetadataValidation:
    def test_missing_metadata_gives_warning(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / "no_meta.txt"
        _write_txt(doc)
        # No metadata JSON written

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc)
        codes = [i.code for i in result.issues]
        # Warning but document may still be "valid" (warning != error)
        assert "METADATA_MISSING" in codes

    @pytest.mark.parametrize("missing_field", REQUIRED_METADATA_FIELDS)
    def test_each_required_field_is_enforced(self, tmp_doc_dir, missing_field):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / f"test_{missing_field}.txt"
        _write_txt(doc)

        # Write meta with one field missing
        meta = {
            "document_id": "HR-FIELD-001",
            "title": "Test",
            "department": "HR",
            "category": "Test",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "source": "internal_policy",
            "access_level": "employee",
            "file_path": str(doc),
        }
        del meta[missing_field]
        (meta_dir / "HR-FIELD-001.json").write_text(json.dumps(meta), encoding="utf-8")

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "MISSING_METADATA_FIELD" in codes

    def test_invalid_version_format_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / "bad_version.txt"
        _write_txt(doc)
        _write_meta(meta_dir, "HR-VER-001", str(doc), overrides={"version": "v1.0-beta"})

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "INVALID_VERSION" in codes

    def test_valid_version_formats(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        for ver in ["1.0", "2.1", "10.3", "1.0.1"]:
            doc = hr_dir / f"ver_{ver.replace('.','_')}.txt"
            _write_txt(doc)
            uid = f"HR-VER-{ver.replace('.','')}"
            _write_meta(meta_dir, uid, str(doc), overrides={"version": ver})
            validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
            result = validator.validate_single(doc)
            ver_errors = [i for i in result.issues if i.code == "INVALID_VERSION"]
            assert len(ver_errors) == 0, f"Version '{ver}' should be valid"

    def test_invalid_date_format_fails(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / "bad_date.txt"
        _write_txt(doc)
        _write_meta(meta_dir, "HR-DATE-001", str(doc), overrides={"effective_date": "01-01-2026"})

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc)
        assert not result.is_valid
        codes = [i.code for i in result.issues]
        assert "INVALID_EFFECTIVE_DATE" in codes

    def test_valid_date_passes(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / "good_date.txt"
        _write_txt(doc)
        _write_meta(meta_dir, "HR-DATE-OK", str(doc), overrides={"effective_date": "2026-06-15"})

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        result = validator.validate_single(doc)
        date_errors = [i for i in result.issues if i.code == "INVALID_EFFECTIVE_DATE"]
        assert len(date_errors) == 0


# ---------------------------------------------------------------------------
# Tests: Batch validation and report
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_report_counts_match(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir

        # 2 valid docs — distinct content to avoid duplicate-checksum detection
        for i in range(2):
            f = hr_dir / f"valid_{i}.txt"
            _write_txt(f, content=f"Unique policy content for document {i}. " * 20)
            _write_meta(meta_dir, f"HR-VALID-{i:03d}", str(f))

        # 1 invalid doc (too small)
        bad = hr_dir / "bad.txt"
        bad.write_text("tiny", encoding="utf-8")

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        report = validator.validate_all()

        assert report.total == 3
        assert report.valid == 2
        assert report.invalid == 1

    def test_report_serialises_to_json(self, tmp_doc_dir):
        _, raw_dir, meta_dir, hr_dir = tmp_doc_dir
        doc = hr_dir / "test.txt"
        _write_txt(doc)
        _write_meta(meta_dir, "HR-JSON-001", str(doc))

        validator = DocumentValidator(raw_dir=str(raw_dir), metadata_dir=str(meta_dir))
        report = validator.validate_all()

        data = report.to_dict()
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total"] == 1

        # Must be JSON-serialisable
        import json
        serialised = json.dumps(data)
        assert len(serialised) > 0
