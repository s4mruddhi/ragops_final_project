"""
scripts/evaluation/run_rag_evaluation.py
========================================
DVC Pipeline Stage 6: End-to-end RAG Evaluation & Quality Gate.
Evaluates end-to-end generation, groundedness, citations, and enforces quality gate thresholds.

Output:
    data/evaluation/reports/rag_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
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
from app.generation.generator import AnswerGenerator
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


def load_champion_retriever_method() -> str:
    """Read the best retriever from retrieval_metrics.json if available."""
    metrics_file = ROOT / "data" / "evaluation" / "reports" / "retrieval_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, encoding="utf-8") as f:
                data = json.load(f)
            best_method = max(
                [m for m in data if not m.startswith("_")],
                key=lambda m: (data[m].get("mean_recall_at_5", 0), data[m].get("mean_mrr", 0)),
            )
            return best_method
        except Exception:
            pass
    return "semantic"


def evaluate_rag_pipeline(dataset: list[dict], retriever_method: str) -> dict[str, Any]:
    from app.retrieval.pipeline import retrieve_and_filter_candidates

    retriever = get_retriever(retriever_method)
    has_real_llm = bool(
        os.environ.get("OPENAI_API_KEY")
        and not os.environ.get("OPENAI_API_KEY", "").startswith("sk-your")
    )
    generator = AnswerGenerator(allow_test_mock=not has_real_llm)
    if not has_real_llm:
        logger.warning(
            "OPENAI_API_KEY not configured. Running offline evaluation with test mock responses. "
            "Set OPENAI_API_KEY in .env for production LLM generation."
        )

    results = []
    groundedness_scores = []
    citation_scores = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    fallback_count = 0
    abstain_hits, unanswerable_total = 0, 0
    answerable_fallbacks, answerable_total = 0, 0

    thresh = float(load_params().get("retrieval", {}).get("score_threshold", 0.0))

    for item in dataset:
        q = item["question"]
        t0 = time.perf_counter()

        # Step 1: Retrieval + Relevance Filtering
        t_ret_start = time.perf_counter()
        final_context, all_candidates = retrieve_and_filter_candidates(
            query=q,
            retriever=retriever,
            candidate_k=5,
            relevance_threshold=thresh,
        )
        ret_latency = (time.perf_counter() - t_ret_start) * 1000.0

        # Step 2: Generation & Verification
        gen_output = generator.generate_answer(q, final_context)
        gen_latency = gen_output["generation_latency_ms"]
        tot_latency = (time.perf_counter() - t0) * 1000.0

        g_score = gen_output["groundedness_score"]
        c_score = gen_output["citation_correctness"]
        is_fallback = gen_output["is_fallback"]

        groundedness_scores.append(g_score)
        citation_scores.append(c_score)
        retrieval_latencies.append(ret_latency)
        generation_latencies.append(gen_latency)
        total_latencies.append(tot_latency)

        if is_fallback:
            fallback_count += 1
        if item.get("answerable", True):
            answerable_total += 1
            answerable_fallbacks += int(is_fallback)
        else:
            unanswerable_total += 1
            abstain_hits += int(is_fallback)

        results.append({
            "question": q,
            "answer": gen_output["answer"],
            "is_fallback": is_fallback,
            "groundedness_score": g_score,
            "citation_correctness": c_score,
            "retrieval_latency_ms": round(ret_latency, 2),
            "generation_latency_ms": round(gen_latency, 2),
            "total_latency_ms": round(tot_latency, 2),
            "retrieved_chunk_ids": [c.chunk_id for c in final_context],
        })

    summary = {
        "retriever_used": retriever_method,
        "num_questions": len(dataset),
        "abstention_rate_unanswerable": round(abstain_hits / unanswerable_total, 4) if unanswerable_total else 1.0,
        "false_abstention_rate_answerable": round(answerable_fallbacks / answerable_total, 4) if answerable_total else 0.0,
        "mean_groundedness": round(float(np.mean(groundedness_scores)), 4),
        "mean_citation_correctness": round(float(np.mean(citation_scores)), 4),
        "fallback_rate": round(float(fallback_count / len(dataset)), 4),
        "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2),
        "avg_generation_latency_ms": round(float(np.mean(generation_latencies)), 2),
        "avg_total_latency_ms": round(float(np.mean(total_latencies)), 2),
        "p95_total_latency_ms": round(float(np.percentile(total_latencies, 95)), 2),
    }

    return {"summary": summary, "details": results}


def check_quality_gate(summary: dict[str, Any], retrieval_summary: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    passed = True

    # 1. Recall@5
    recall_at_5 = retrieval_summary.get("mean_recall_at_5", 0.0)
    min_recall = gate_cfg.get("min_recall_at_5", 0.70)
    checks["recall_at_5"] = {"value": recall_at_5, "threshold": min_recall, "pass": recall_at_5 >= min_recall}
    if not checks["recall_at_5"]["pass"]:
        passed = False

    # 2. Precision@5
    precision_at_5 = retrieval_summary.get("mean_precision_at_5", 0.0)
    min_precision = gate_cfg.get("min_precision_at_5", 0.60)
    checks["precision_at_5"] = {"value": precision_at_5, "threshold": min_precision, "pass": precision_at_5 >= min_precision}
    if not checks["precision_at_5"]["pass"]:
        passed = False

    # 3. MRR
    mrr = retrieval_summary.get("mean_mrr", 0.0)
    min_mrr = gate_cfg.get("min_mrr", 0.65)
    checks["mrr"] = {"value": mrr, "threshold": min_mrr, "pass": mrr >= min_mrr}
    if not checks["mrr"]["pass"]:
        passed = False

    # 4. Groundedness
    groundedness = summary.get("mean_groundedness", 0.0)
    min_groundedness = gate_cfg.get("min_groundedness", 0.75)
    checks["groundedness"] = {"value": groundedness, "threshold": min_groundedness, "pass": groundedness >= min_groundedness}
    if not checks["groundedness"]["pass"]:
        passed = False

    # 5. Citation Correctness
    citation = summary.get("mean_citation_correctness", 0.0)
    min_citation = gate_cfg.get("min_citation_correctness", 0.80)
    checks["citation_correctness"] = {"value": citation, "threshold": min_citation, "pass": citation >= min_citation}
    if not checks["citation_correctness"]["pass"]:
        passed = False

    # 6. Latency checks
    ret_lat = summary.get("avg_retrieval_latency_ms", 0.0)
    max_ret_lat = gate_cfg.get("max_retrieval_latency_ms", 500)
    checks["retrieval_latency_ms"] = {"value": ret_lat, "threshold": max_ret_lat, "pass": ret_lat <= max_ret_lat}
    if not checks["retrieval_latency_ms"]["pass"]:
        passed = False

    tot_lat = summary.get("avg_total_latency_ms", 0.0)
    max_tot_lat = gate_cfg.get("max_total_latency_ms", 5000)
    checks["total_latency_ms"] = {"value": tot_lat, "threshold": max_tot_lat, "pass": tot_lat <= max_tot_lat}
    if not checks["total_latency_ms"]["pass"]:
        passed = False

    # 7. Abstention on unanswerable questions (hallucination guard)
    ab = summary.get("abstention_rate_unanswerable", 1.0)
    min_ab = gate_cfg.get("min_abstention_rate", 0.5)
    checks["abstention_rate"] = {"value": ab, "threshold": min_ab, "pass": ab >= min_ab}
    if not checks["abstention_rate"]["pass"]:
        passed = False

    return {"passed": passed, "checks": checks}


def main():
    params = load_params()
    eval_cfg = params.get("evaluation", {})
    gate_cfg = params.get("quality_gate", {})
    mlflow_cfg = params.get("mlflow", {})

    dataset_path = ROOT / eval_cfg.get("dataset_path", "data/evaluation/rag_questions.jsonl")
    dataset = []
    with jsonlines.open(dataset_path) as reader:
        for item in reader:
            dataset.append(item)

    champion_method = load_champion_retriever_method()
    logger.info(f"Running RAG evaluation using champion retriever: '{champion_method}'")

    eval_result = evaluate_rag_pipeline(dataset, champion_method)
    summary = eval_result["summary"]

    # Load retrieval metrics for the champion method
    retrieval_metrics_file = ROOT / "data" / "evaluation" / "reports" / "retrieval_metrics.json"
    retrieval_data = {}
    if retrieval_metrics_file.exists():
        with open(retrieval_metrics_file, encoding="utf-8") as f:
            retrieval_data = json.load(f).get(champion_method, {})

    gate_result = check_quality_gate(summary, retrieval_data, gate_cfg)
    summary["quality_gate"] = gate_result

    # Log to MLflow
    set_seed()
    configure_mlflow()

    with mlflow.start_run(run_name="rag_end_to_end_evaluation"):
        mlflow.set_tag("stage", "rag_end_to_end")
        mlflow.log_artifact(str(ROOT / "params.yaml"))
        mlflow.log_param("champion_retriever", champion_method)
        mlflow.log_param("num_questions", len(dataset))
        mlflow.log_param("quality_gate_passed", gate_result["passed"])

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        for check_name, check_data in gate_result["checks"].items():
            mlflow.log_metric(f"gate_{check_name}_val", check_data["value"])

    # Save to report file
    report_file = ROOT / "data" / "evaluation" / "reports" / "rag_metrics.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "quality_gate": gate_result, "details": eval_result["details"]}, f, indent=2)

    quality_gate_file = ROOT / "data" / "evaluation" / "reports" / "quality_gate.json"
    with open(quality_gate_file, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)

    logger.info(f"RAG evaluation metrics saved to {report_file} and {quality_gate_file}")

    print("\n" + "=" * 70)
    print("RAG QUALITY GATE VERIFICATION REPORT")
    print("=" * 70)
    status_str = "PASSED [GREEN]" if gate_result["passed"] else "FAILED [RED]"
    print(f"Overall Status: {status_str}")
    print("-" * 70)
    for metric_name, c in gate_result["checks"].items():
        pass_symbol = "PASS" if c["pass"] else "FAIL"
        comp = "<=" if "latency" in metric_name else ">="
        print(f"{metric_name:<26}: {c['value']:>8.4f} (req {comp} {c['threshold']:<6.4f})  [{pass_symbol}]")
    print("=" * 70 + "\n")

    if not gate_result["passed"] and gate_cfg.get("require_all_tests_pass", True):
        logger.error("Quality gate FAILED - exiting non-zero so CI/Airflow blocks promotion.")
        sys.exit(2)


if __name__ == "__main__":
    main()
