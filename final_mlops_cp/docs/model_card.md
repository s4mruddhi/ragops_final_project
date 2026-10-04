# Model Card - RAGOps retrieval + answer pipeline

**Registered model:** `ragops-retrieval-config` v3 (alias `champion`)  |  **Date:** 2026-10-04

## What the "model" is
RAG is a pipeline, not one network. The MLflow-registered artifact is the **champion configuration**:
retriever = **hybrid**, top_k = 5, embedding model = `sentence-transformers/all-MiniLM-L6-v2`,
chunking = 512/64, prompt = `answer_v1`.
Generation is extractive by default; an OpenAI/Ollama LLM can be enabled via `.env`.

## Intended use
Answer employee questions **only from verified organizational documents**, with citations; abstain when the context does not support an answer.
Decision support for staff - **not** an authority on legal/HR decisions.

## Out of scope
Medical/legal advice, questions outside the document set, any use on confidential data without access controls.

## Experiments (retrieval, 32 questions)
| Retriever | Recall@5 | MRR |
|---|---|---|
| tfidf (baseline) | 0.8906 | 0.8438 |
| bm25 | 0.8802 | 0.7943 |
| semantic | 0.8385 | 0.8125 |
| hybrid | 0.8906 | 0.875 |

## Quality gate (blocks promotion)
PASSED - recall@5, precision@5, MRR, groundedness, citation correctness, latency, abstention (thresholds in `params.yaml`).

## Limitations
Small gold set; lexical groundedness proxy; extractive answers are terse; weak refusal on out-of-scope questions (see performance_report.md section 4);
semantic quality depends on `all-MiniLM-L6-v2`. Prompt injection inside documents is tested with `IT-ADVERSARIAL-TEST` and must be handled as data.

## Monitoring & retraining
Prometheus/Grafana (latency, errors, fallback share, groundedness, PSI drift); `/feedback` ratings are the label signal. PSI > 0.2
raises an alert -> re-run the Airflow DAG.

## Reproducibility
Seed 42; DVC-versioned data; params in `params.yaml`; MLflow run ids for every experiment.
