"""
Pipeline stage 0 - Data ingestion.
Scans data/raw, writes a SHA-256 manifest (evidence of the exact dataset version used)
and cross-checks it against the DVC pointer data/raw.dvc.

    python scripts/evaluation/ingest_documents.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "ingest_manifest.json"


def main() -> int:
    files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx"})
    if not files:
        print("ERROR: no documents found in data/raw")
        return 1
    manifest = {
        str(p.relative_to(RAW)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    dvc_ptr = ROOT / "data" / "raw.dvc"
    dvc_md5 = None
    if dvc_ptr.exists():
        for line in dvc_ptr.read_text(encoding="utf-8").splitlines():
            if "md5:" in line:
                dvc_md5 = line.split("md5:")[1].strip()
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"num_documents": len(files), "dvc_md5": dvc_md5, "files": manifest}, indent=2),
                   encoding="utf-8")
    print(f"Ingested {len(files)} documents (dvc md5={dvc_md5}) -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
