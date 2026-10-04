.PHONY: setup data pipeline test serve up down drift load-test lock
PY ?= python

setup:            ## install deps
	$(PY) -m pip install -r requirements.txt
data:             ## pull versioned data
	dvc pull data/raw.dvc
pipeline:         ## run the full DAG locally (ingest -> ... -> register -> drift)
	$(PY) scripts/run_dag_pipeline.py
test:
	pytest tests -q
serve:
	uvicorn app.main:app --reload --port 8000
up:               ## full stack: API + MLflow + Prometheus + Grafana
	docker compose up --build -d
down:
	docker compose down
drift:
	$(PY) scripts/monitoring/simulate_drift.py --api http://localhost:8000
load-test:
	$(PY) scripts/benchmark/load_test.py --url http://localhost:8000 --requests 300 --concurrency 8

lock:             ## pin exact versions (run once on Python 3.10, commit requirements.lock)
	pip install pip-tools && pip-compile requirements.txt -o requirements.lock --strip-extras
reports:
	$(PY) scripts/generate_reports.py
