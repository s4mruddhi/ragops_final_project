"""Quality-gate logic (promotion threshold) - pure unit tests."""
from scripts.evaluation.run_rag_evaluation import check_quality_gate

GOOD_SUMMARY = {"mean_groundedness": 0.9, "mean_citation_correctness": 0.9, "avg_retrieval_latency_ms": 5,
                "avg_total_latency_ms": 10, "abstention_rate_unanswerable": 0.5}
GOOD_RET = {"mean_recall_at_5": 0.9, "mean_precision_at_5": 0.7, "mean_mrr": 0.8}
CFG = {"min_recall_at_5": 0.7, "min_precision_at_5": 0.55, "min_mrr": 0.65, "min_groundedness": 0.75,
       "min_citation_correctness": 0.8, "max_retrieval_latency_ms": 500, "max_total_latency_ms": 5000,
       "min_abstention_rate": 0.3}


def test_gate_passes_when_all_thresholds_met():
    assert check_quality_gate(GOOD_SUMMARY, GOOD_RET, CFG)["passed"] is True


def test_gate_blocks_on_low_recall():
    res = check_quality_gate(GOOD_SUMMARY, {**GOOD_RET, "mean_recall_at_5": 0.4}, CFG)
    assert res["passed"] is False and res["checks"]["recall_at_5"]["pass"] is False


def test_gate_blocks_on_hallucination_risk():
    res = check_quality_gate({**GOOD_SUMMARY, "mean_groundedness": 0.5}, GOOD_RET, CFG)
    assert res["passed"] is False


def test_gate_blocks_when_system_never_abstains():
    res = check_quality_gate({**GOOD_SUMMARY, "abstention_rate_unanswerable": 0.0}, GOOD_RET, CFG)
    assert res["passed"] is False
