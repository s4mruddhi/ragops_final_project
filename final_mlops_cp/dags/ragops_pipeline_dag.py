"""
RAGOps training / evaluation / registration DAG (Apache Airflow 2.x and 3.x).

ingest -> validate -> preprocess -> feature_engineering -> build_index -> train(retrieval experiments)
       -> evaluate (quality gate) -> register (MLflow) -> drift_check

Every task runs the SAME scripts that `dvc repro` and CI run, so there is one implementation.
A failed quality gate exits non-zero, fails `evaluate_rag`, and blocks `register_model`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(os.environ.get("RAGOPS_HOME", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.pipeline_steps import STEPS  # noqa: E402

PYTHON = os.environ.get("RAGOPS_PYTHON", "python")

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    dag_id="ragops_pipeline",
    default_args=default_args,
    description="RAGOps: ingest -> validate -> preprocess -> features -> train -> evaluate -> register",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["ragops", "mlflow", "dvc"],
) as dag:
    previous = None
    for task_id, script, doc in STEPS:
        task = BashOperator(
            task_id=task_id,
            bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON} {script}',
            env={"PYTHONPATH": str(PROJECT_ROOT), "MLFLOW_TRACKING_URI":
                 os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db"),
                 "PATH": os.environ.get("PATH", "")},
            append_env=True,
            doc_md=doc,
        )
        if previous is not None:
            previous >> task
        previous = task
