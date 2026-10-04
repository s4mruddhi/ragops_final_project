# Performance Report
> Generated 2026-10-04 by `scripts/generate_reports.py` from `data/evaluation/reports/*.json` and `docs/benchmark_results.json`.

## 1. Experiment comparison (32 gold questions, retrieval on document-level labels)

| Retriever | Recall@1 | Recall@5 | Precision@5 | MRR | NDCG@5 | Mean latency ms | p95 latency ms |
|---|---|---|---|---|---|---|---|
| tfidf (baseline) | 0.7500 | 0.8906 | 0.5938 | 0.8438 | 0.8146 | 2.29 | 3.62 |
| bm25 | 0.7083 | 0.8802 | 0.4813 | 0.7943 | 0.7698 | 1.31 | 2.39 |
| semantic | 0.7344 | 0.8385 | 0.6250 | 0.8125 | 0.7649 | 211.27 | 82.98 |
| **hybrid - champion** | 0.7969 | 0.8906 | 0.6188 | 0.8750 | 0.8269 | 148.09 | 58.75 |

Baseline = `tfidf`; candidates = the other three. Champion is chosen by (Recall@5, MRR), registered in MLflow as `ragops-retrieval-config` **v3** with alias `champion`.

## 2. Business + model metrics (required by the project brief)

| Metric | Value | Meaning |
|---|---|---|
| **Model metric** - Recall@5 / MRR | 0.8906 / 0.875 | Correct policy document is in the top 5 chunks |
| **Business metric** - answerable questions answered (1 - false-abstention) | 89.66% | Employee gets a usable cited answer instead of being sent to HR |
| **Business metric** - unanswerable questions correctly refused | 33.33% | Hallucination avoided (n=3 questions - small sample) |
| Groundedness / citation correctness | 0.8789 / 0.875 | Lexical-overlap groundedness (see limitations) |
| p95 latency / throughput / error rate | *run `make load-test`* | `scripts/benchmark/load_test.py` writes docs/benchmark_results.json |
| **Drift metric** - max PSI on simulated shift | 6.3603 (threshold 0.2) | See drift_report.md |

## 3. Quality gate (promotion threshold)

| Check | Value | Threshold | Result |
|---|---|---|---|
| recall_at_5 | 0.8906 | >= 0.7 | PASS |
| precision_at_5 | 0.6188 | >= 0.55 | PASS |
| mrr | 0.875 | >= 0.65 | PASS |
| groundedness | 0.8789 | >= 0.75 | PASS |
| citation_correctness | 0.875 | >= 0.8 | PASS |
| retrieval_latency_ms | 425.73 | <= 500 | PASS |
| total_latency_ms | 426.5 | <= 5000 | PASS |
| abstention_rate | 0.3333 | >= 0.3 | PASS |

**Overall: PASSED - model promoted to alias `champion`**

## 4. Honest limitations (say these in the viva before the examiner does)

* **Gold set is small** (32 questions, 3 unanswerable) - differences of a few points between retrievers are within noise.
* **Default generator is extractive** (no LLM key): it returns the best-matching sentence, so groundedness is naturally high.
  Groundedness is a *lexical-overlap* heuristic, not an NLI/LLM judge.
* **Abstention** on unanswerable questions is the weakest point; the relevance rule trades false-abstention against refusal.
  Plugging in an LLM (`OPENAI_API_KEY`) is the intended fix.
* **Benchmark** numbers depend on the machine; rerun `make load-test` on the demo machine and regenerate this report.
