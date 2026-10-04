# 5-Minute Live Demo Script

| Time | Show | Say |
|---|---|---|
| 0:00 | README architecture diagram | Problem: employees need verified answers; 18 -> RAGOps. Pipeline: data -> DAG -> MLflow -> FastAPI -> monitoring -> retrain. |
| 0:30 | `git log --oneline`, `data/raw.dvc`, `dvc.yaml` | Reproducible: Git + DVC data version, seed 42, params.yaml, lock file. |
| 1:00 | Airflow UI: `ragops_pipeline` green run | 9 tasks: ingest -> validate -> preprocess -> features -> index -> train -> evaluate -> register -> drift. Gate failure blocks registration. |
| 1:45 | MLflow UI: `ragops-retrieval` runs + Models tab | Baseline TF-IDF + 3 candidates; champion registered with version, alias `champion`, metadata tags. |
| 2:30 | `docker compose up`; Swagger `/docs`; POST `/query` (answerable, then an unanswerable) | Validation (422), health, cited sources, groundedness, fallback. |
| 3:15 | Grafana dashboard while `make load-test` runs | Request rate, p95 latency, error rate, fallback share, groundedness. |
| 4:00 | `make drift` -> PSI panel turns red, alert rule | Simulated off-topic/long/typo queries -> PSI > 0.2 -> retrain DAG. |
| 4:30 | GitHub Actions green run; S3/CloudWatch screenshot | CI: tests, gate, Docker build+API test. Cloud: S3, SageMaker, CloudWatch - then torn down. |
| 5:00 | Limitations slide | Small gold set, extractive generator, weak refusal - and how I would fix them. |

Pre-demo checklist: `docker compose up -d`, `make pipeline`, open Airflow/MLflow/Grafana tabs, run `make load-test` once, keep AWS screenshots ready.
