"""
Register the champion retriever configuration in the MLflow Model Registry.

Only runs if data/evaluation/reports/quality_gate.json says passed=true.
Creates a new model version, sets alias `champion`, and attaches metadata tags.

    python scripts/evaluation/register_retriever_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.pyfunc

from app.core.mlflow_utils import configure_mlflow, load_params, set_seed

REPORTS = ROOT / "data" / "evaluation" / "reports"


class RetrieverConfigModel(mlflow.pyfunc.PythonModel):
    """Tiny pyfunc wrapper so the retrieval config is a versioned, loadable MLflow model."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def predict(self, context, model_input, params=None):
        return [self.config for _ in range(len(model_input))]


def main() -> int:
    params = load_params()
    cfg = params.get("mlflow", {})
    name = cfg.get("registered_model_name", "ragops-retrieval-config")
    alias = cfg.get("model_alias", "champion")

    gate = json.loads((REPORTS / "quality_gate.json").read_text(encoding="utf-8"))
    if not gate.get("passed"):
        print("Quality gate did not pass -> NOT registering a new version.")
        return 2

    ret = json.loads((REPORTS / "retrieval_metrics.json").read_text(encoding="utf-8"))
    ret = {k: v for k, v in ret.items() if not k.startswith("_")}
    champion = max(ret, key=lambda m: (ret[m]["mean_recall_at_5"], ret[m]["mean_mrr"]))
    m = ret[champion]

    set_seed()
    configure_mlflow()
    config = {
        "retriever": champion,
        "top_k": params["retrieval"]["top_k"],
        "embedding_model": params["embedding"]["model"],
        "chunking": params["chunking"],
        "prompt_version": params["generation"]["prompt_version"],
        "dataset_version": params["dataset"]["current_version"],
    }

    with mlflow.start_run(run_name=f"register_{champion}") as run:
        mlflow.set_tag("stage", "registration")
        mlflow.log_params({"champion_retriever": champion, "seed": params.get("seed", 42)})
        mlflow.log_metrics({
            "recall_at_5": m["mean_recall_at_5"],
            "precision_at_5": m["mean_precision_at_5"],
            "mrr": m["mean_mrr"],
            "p95_latency_ms": m["p95_latency_ms"],
        })
        mlflow.log_artifact(str(REPORTS / "quality_gate.json"))
        info = mlflow.pyfunc.log_model(
            artifact_path=cfg.get("artifact_path", "retrieval_config"),
            python_model=RetrieverConfigModel(config),
        )
        mv = mlflow.register_model(info.model_uri, name)

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, alias, mv.version)
    client.update_registered_model(
        name, description="Champion retrieval configuration for the RAGOps knowledge assistant."
    )
    for k, v in {
        "retriever": champion,
        "recall_at_5": str(m["mean_recall_at_5"]),
        "mrr": str(m["mean_mrr"]),
        "quality_gate": "passed",
        "dataset_version": params["dataset"]["current_version"],
        "prompt_version": params["generation"]["prompt_version"],
    }.items():
        client.set_model_version_tag(name, mv.version, k, v)

    # Hand the decision to the serving layer without needing MLflow at runtime.
    (REPORTS / "champion.json").write_text(
        json.dumps({"name": name, "version": mv.version, "alias": alias, **config}, indent=2),
        encoding="utf-8",
    )
    print(f"Registered {name} v{mv.version} -> alias '{alias}' (retriever={champion})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
