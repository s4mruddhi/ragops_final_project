# Data Card

**Name:** RAGOps synthetic enterprise policy corpus (v1)  **Generated:** 2026-10-04

## Composition
* **22 documents** (plain text) in `data/raw/` across HR, IT, Product, Travel, plus one *adversarial* prompt-injection
  test document (`IT-ADVERSARIAL-TEST`). Each has a JSON metadata sidecar (id, title, department, version, effective date, access level).
* **441 chunks** after cleaning + chunking (size 512 chars, overlap 64).
  Department split: {'HR': 143, 'IT': 177, 'Product': 60, 'Travel': 61}.
* **Gold evaluation set:** 32 questions - {'single_doc': 11, 'semantic': 8, 'multi_doc': 6, 'ambiguous': 4, 'unanswerable': 3} - each with expected document ids and a reference answer.

## Provenance and versioning
* **Synthetic**: written by `scripts/generate_synthetic_docs.py`; contains **no real personal or company data**. Policy numbers are fictional.
* Versioned with **DVC** (`data/raw.dvc`, md5 `c6570e82aadd37282aa2e196b3406557.dir`); `scripts/evaluation/ingest_documents.py` writes a SHA-256 manifest per run.
  `scripts/generate_synthetic_docs_v2.py` adds documents to demonstrate a v1 -> v2 dataset change.

## Known limitations / biases
* Synthetic text is cleaner and more consistent than real policies; real-world performance will be lower.
* Single language (English); documents are short, so chunk-level retrieval is easier than on long PDFs.
* Evaluation questions were written by the same author as the documents (lexical overlap is optimistic).
* One adversarial document is deliberately included to test prompt-injection resistance; it must never be treated as an instruction.

## Privacy
No PII. If real HR documents are ever used: restrict by `access_level`, redact personal data before indexing, and log queries without user identifiers.
