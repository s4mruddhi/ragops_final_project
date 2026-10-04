"""MLflow helpers: one place to configure the tracking URI and seeds."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int | None = None) -> int:
    seed = seed if seed is not None else int(load_params().get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def configure_mlflow():
    """Point MLflow at a SQLite backend (needed for the Model Registry)."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    exp = load_params().get("mlflow", {}).get("experiment_name", "ragops-retrieval")
    mlflow.set_experiment(exp)
    return mlflow
