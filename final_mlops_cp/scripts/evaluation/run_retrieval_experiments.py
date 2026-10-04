"""
scripts/evaluation/run_retrieval_experiments.py
================================================
DVC Pipeline Stage 5: Run retrieval experiments across all candidate retrievers
(TF-IDF, BM25, Semantic, Hybrid) on data/evaluation/rag_questions.jsonl,
log runs to MLflow, and produce data/evaluation/reports/retrieval_metrics.json.

Run:
    python scripts/evaluation/run_retrieval_experiments.py
    # or via DVC:
    dvc repro retrieval_experiments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jsonlines
import mlflow
import numpy as np
import yaml

from app.core.mlflow_utils import configure_mlflow, set_seed
from app.evaluation.retrieval_metrics import evaluate_query_retrieval
from app.retrieval.factory import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    params_file = ROOT / "params.yaml"
    if params_file.exists():
        with open(params_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_dataset(dataset_path: Path) -> list[dict]:
    dataset = []
    with jsonlines.open(dataset_path) as reader:
        for item in reader:
            dataset.append(item)
    return dataset


def evaluate_method(
    method_name: str,
    retriever: Any,
    dataset: list[dict],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, Any]:
    """Evaluate a single retriever across the dataset."""
    logger.info(f"Evaluating retriever: '{method_name}' on {len(dataset)} questions...")
    max_k = max(k_values)
    per_question_results = []
    latencies = []

    for item in dataset:
        q = item["question"]
        expected = item.get("expected_document_ids", [])

        start = time.perf_counter()
        retrieved_chunks = retriever.retrieve(q, top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        retrieved_doc_ids = []
        for c in retrieved_chunks:
            doc_id = getattr(c, "document_id", None) or c.metadata.get("document_id")
            if not doc_id and "_" in c.chunk_id:
                doc_id = c.chunk_id.split("_chunk_")[0]
            if doc_id:
                retrieved_doc_ids.append(doc_id)

        metrics = evaluate_query_retrieval(retrieved_doc_ids, expected, k_values=k_values)
        metrics["latency_ms"] = round(latency_ms, 2)
        metrics["question"] = q
        metrics["expected_document_ids"] = expected
        metrics["retrieved_document_ids"] = retrieved_doc_ids[:5]
        per_question_results.append(metrics)

    # Compute aggregates
    aggregated: dict[str, Any] = {
        "method": method_name,
        "num_questions": len(dataset),
        "mean_mrr": round(float(np.mean([r["mrr"] for r in per_question_results])), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    }

    for k in k_values:
        aggregated[f"mean_recall_at_{k}"] = round(
            float(np.mean([r[f"recall_at_{k}"] for r in per_question_results])), 4
        )
        aggregated[f"mean_precision_at_{k}"] = round(
            float(np.mean([r[f"precision_at_{k}"] for r in per_question_results])), 4
        )
        aggregated[f"mean_ndcg_at_{k}"] = round(
            float(np.mean([r[f"ndcg_at_{k}"] for r in per_question_results])), 4
        )

    return {
        "summary": aggregated,
        "details": per_question_results,
    }


def main():
    params = load_params()
    eval_cfg = params.get("evaluation", {})
    mlflow_cfg = params.get("mlflow", {})
    k_values = eval_cfg.get("k_values", [1, 3, 5, 10])

    dataset_path = ROOT / eval_cfg.get("dataset_path", "data/evaluation/rag_questions.jsonl")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation questions.")

    report_dir = ROOT / "data" / "evaluation" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    seed = set_seed()
    configure_mlflow()
    baseline = mlflow_cfg.get("baseline_method", "tfidf")

    methods_to_test = ["tfidf", "bm25", "semantic", "hybrid"]
    all_results: dict[str, Any] = {}
    failed: list[str] = []

    for method in methods_to_test:
        try:
            retriever = get_retriever(method)
            eval_output = evaluate_method(method, retriever, dataset, k_values=k_values)
            summary = eval_output["summary"]
            all_results[method] = summary

            # Log to MLflow
            with mlflow.start_run(run_name=f"retrieval_{method}"):
                mlflow.set_tag("role", "baseline" if method == baseline else "candidate")
                mlflow.set_tag("stage", "retrieval_experiment")
                mlflow.log_param("seed", seed)
                mlflow.log_param("retriever_method", method)
                mlflow.log_param("num_questions", len(dataset))
                mlflow.log_param("k_values", str(k_values))

                for metric_name, val in summary.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(metric_name, val)

                # Save detailed artifact
                detail_file = report_dir / f"{method}_details.json"
                with open(detail_file, "w", encoding="utf-8") as f:
                    json.dump(eval_output["details"], f, indent=2)
                mlflow.log_artifact(str(detail_file))

        except Exception as exc:
            logger.error(f"Error evaluating retriever '{method}': {exc}", exc_info=True)
            failed.append(method)

    if failed:
        logger.error(f"Retrievers FAILED and are excluded from the report: {failed}")
        all_results["_failed"] = failed
    if len([m for m in all_results if not m.startswith("_")]) < 3:
        logger.error("Need baseline + >=2 candidates for a valid experiment.")
        sys.exit(1)

    # Output consolidated metrics
    output_metrics_file = report_dir / "retrieval_metrics.json"
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Consolidated metrics saved to {output_metrics_file}")

    # Determine best method based on Recall@5
    best_method = max(
        [m for m in all_results if not m.startswith("_")],
        key=lambda m: (all_results[m].get("mean_recall_at_5", 0), all_results[m].get("mean_mrr", 0)),
    )
    logger.info(f"Best performing retriever: '{best_method}' (Recall@5: {all_results[best_method]['mean_recall_at_5']})")

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Method':<12} | {'Recall@5':<10} | {'Prec@5':<10} | {'MRR':<8} | {'NDCG@5':<8} | {'Latency (ms)':<12}")
    print("-" * 78)
    for m, s in [(k, v) for k, v in all_results.items() if not k.startswith("_")]:
        print(
            f"{m:<12} | {s.get('mean_recall_at_5', 0):<10.4f} | {s.get('mean_precision_at_5', 0):<10.4f} | "
            f"{s.get('mean_mrr', 0):<8.4f} | {s.get('mean_ndcg_at_5', 0):<8.4f} | {s.get('avg_latency_ms', 0):<12.2f}"
        )
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
