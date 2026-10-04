# Drift Report

> Generated 2026-10-04 from `data/monitoring/drift_report.json` (`scripts/monitoring/simulate_drift.py`).

**Setup.** Reference = 32 gold questions through the `hybrid` retriever. Window A = 60 resampled normal queries. Window B = 60 *shifted* queries (1/3 off-topic, 1/3 very long, 1/3 typo-heavy). Method: PSI (threshold 0.2) and two-sample KS test (alpha 0.05).

| Feature | PSI normal | PSI shifted | KS p (shifted) | Drift flagged (shifted) |
|---|---|---|---|---|
| query_length | 0.0088 | 0.2507 | 0.014349 | YES |
| top_retrieval_score | 0.0447 | 0.8076 | 0.001754 | YES |
| groundedness_score | 0.0001 | 6.3603 | 0.0 | YES |
| fallback | 0.0001 | 4.8921 | 0.0 | YES |

**Result:** normal window flagged = **False** (no false alarm); shifted window flagged = **True**.

**Action policy.** Prometheus alert `InputDriftDetected` (PSI > 0.2) -> review new queries -> add/refresh documents in
`data/raw` (DVC version bump) -> re-run the Airflow DAG (`ragops_pipeline`) -> quality gate -> new MLflow model version.
Live view: `GET /api/v1/drift`, Grafana panels *Feature PSI* and *Drift detected*.

**Caveat.** PSI is biased upward on small samples; bins are therefore adaptive (>=10 reference points per bin) and the
detector needs >= 30 samples per window.
