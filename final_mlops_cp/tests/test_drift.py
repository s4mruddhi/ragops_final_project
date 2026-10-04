import numpy as np

from app.monitoring.drift import detect_drift, ks_pvalue, psi


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    assert psi(a, b) < 0.05


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 2000), rng.normal(2, 1, 2000)) > 0.2


def test_psi_handles_binary_flag():
    assert psi([0.0] * 100, [1.0] * 50 + [0.0] * 50) > 0.2
    assert psi([0.0] * 100, [0.0] * 100) == 0.0


def test_ks_detects_shift():
    rng = np.random.default_rng(1)
    assert ks_pvalue(rng.normal(0, 1, 500), rng.normal(1, 1, 500)) < 0.05


def test_detect_drift_flags_and_respects_min_samples():
    rng = np.random.default_rng(2)
    ref = {f: list(rng.normal(0, 1, 100)) for f in ("query_length", "top_retrieval_score", "groundedness_score", "fallback")}
    cur = {f: list(rng.normal(3, 1, 100)) for f in ref}
    assert detect_drift(ref, cur)["drift_detected"] is True
    small = {f: v[:5] for f, v in cur.items()}
    rep = detect_drift(ref, small, min_samples=30)
    assert rep["drift_detected"] is False
    assert all(v["status"] == "insufficient_samples" for v in rep["features"].values())
