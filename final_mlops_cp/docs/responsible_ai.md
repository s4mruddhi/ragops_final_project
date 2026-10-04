# Responsible AI Statement

## Intended use
RAGOps is an **employee self-service assistant** that answers policy questions (HR, IT, Product, Travel) *only* from verified
organizational documents, always with citations. Answers are **decision support**; for anything binding (pay, discipline, legal)
the employee must confirm with HR / IT.

## Not intended for
Automated HR decisions, performance evaluation, legal/medical advice, or any use on confidential data without access controls.

## Explainability (SHAP/LIME equivalent for RAG)
Classical SHAP/LIME explain feature contributions of a tabular/text classifier. A RAG system's explanation is **provenance**:
every response returns the retrieved chunks (`sources`) with scores, the cited document ids, a **groundedness score** and a
**citation-correctness score**. `app/generation/groundedness.py` shows which answer claims are (not) supported by context, and
low-groundedness answers are replaced by a safe fallback. (If your instructor insists on SHAP/LIME, apply LIME to the TF-IDF
retriever: perturb query terms and measure change in top-k score - TF-IDF weights are already directly interpretable.)

## Risks and mitigations
| Risk | Mitigation in this repo |
|---|---|
| Hallucination | Groundedness threshold -> fallback; abstention metric in the quality gate |
| Prompt injection inside documents | Context is treated as data; adversarial doc `IT-ADVERSARIAL-TEST` kept as a regression case |
| Stale / wrong policy | Versioned documents (DVC), metadata `version` + `effective_date` shown in citations |
| Unauthorized access | `access_level` metadata field; department filter in the API (full RBAC is future work) |
| Silent quality decay | Prometheus + Grafana, PSI/KS drift, user ratings via `/feedback`, retraining DAG |
| Over-reliance | Fallback text sends users to HR/IT; UI/API label answers as assistive |

## Bias / fairness
No demographic attributes are used or inferred, so group-fairness metrics do not apply. The relevant fairness risk is *coverage
bias*: departments with more/better documents get better answers. Check per-department recall (`department_distribution` in the
index stats) before adding new departments.

## Privacy
Synthetic corpus, no PII. For real data: redact PII before indexing, minimise and anonymise query logs
(`data/monitoring/query_log.jsonl` stores only numeric features, not query text), set retention limits, never commit `.env`/keys.

## Ethical limitations (be explicit)
Small evaluation set; lexical groundedness proxy; extractive generator; single language; no formal red-teaming beyond one injection test.
