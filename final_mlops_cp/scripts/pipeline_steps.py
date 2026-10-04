"""Single source of truth for the pipeline task order (used by the Airflow DAG and the local runner)."""
# (task_id, script path, description, dvc/airflow stage)
STEPS = [
    ("ingest_documents",     "scripts/evaluation/ingest_documents.py",        "Data ingestion + SHA-256 manifest vs DVC pointer"),
    ("validate_documents",   "scripts/evaluation/validate_documents.py",      "Schema / size / duplicate validation"),
    ("preprocess_text",      "scripts/evaluation/extract_and_clean.py",       "Text extraction + cleaning"),
    ("feature_engineering",  "scripts/evaluation/chunk_documents.py",         "Chunking (+ metadata features)"),
    ("build_index",          "scripts/evaluation/build_index.py",             "Embeddings + ChromaDB index"),
    ("train_retrieval_models", "scripts/evaluation/run_retrieval_experiments.py", "Baseline + 3 candidates logged to MLflow"),
    ("evaluate_rag",         "scripts/evaluation/run_rag_evaluation.py",      "End-to-end eval + quality gate (non-zero exit on fail)"),
    ("register_model",       "scripts/evaluation/register_retriever_model.py", "MLflow registry: version + 'champion' alias (only if gate passed)"),
    ("drift_check",          "scripts/monitoring/simulate_drift.py",          "PSI/KS drift report"),
    ("generate_reports",     "scripts/generate_reports.py",                   "Regenerate performance/drift/model/data cards from real results"),
]
