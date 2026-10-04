"""
Drift detection for RAGOps.

Monitored features (per query): query_length, top_retrieval_score, groundedness_score,
fallback flag. Methods: Population Stability Index (PSI) and two-sample KS test.
Thresholds come from params.yaml -> drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
QUERY_LOG = ROOT / "data" / "monitoring" / "query_log.jsonl"
REFERENCE = ROOT / "data" / "monitoring" / "reference_features.json"
FEATURES = ["query_length", "top_retrieval_score", "groundedness_score", "fallback"]


def psi(reference: Sequence[float], current: Sequence[float], bins: int | None = None) -> float:
    """Population Stability Index. >0.2 = significant drift.

    Low-cardinality features (e.g. the 0/1 fallback flag) use one bin per distinct value;
    continuous ones use reference-quantile bins.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    # PSI is biased upward on small samples (~ (bins-1)*(1/n_ref+1/n_cur)), so keep >=10 reference
    # points per bin: 32 samples -> 3 bins, >=100 samples -> 10 bins.
    bins = bins or int(min(10, max(3, len(ref) // 10)))
    uniq = np.unique(np.concatenate([ref, cur]))
    if len(np.unique(ref)) <= 5:
        r = np.array([(ref == u).mean() for u in uniq])
        c = np.array([(cur == u).mean() for u in uniq])
    else:
        inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
        r = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1) / len(ref)
        c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1) / len(cur)
    eps = 1e-3
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_pvalue(reference: Sequence[float], current: Sequence[float]) -> float:
    if len(reference) < 2 or len(current) < 2:
        return 1.0
    return float(stats.ks_2samp(reference, current).pvalue)


def detect_drift(reference: dict[str, list[float]], current: dict[str, list[float]],
                 psi_threshold: float = 0.2, ks_alpha: float = 0.05,
                 min_samples: int = 30) -> dict:
    out = {"features": {}, "drift_detected": False, "n_reference": 0, "n_current": 0}
    for f in FEATURES:
        r, c = reference.get(f, []), current.get(f, [])
        out["n_reference"], out["n_current"] = len(r), len(c)
        if len(c) < min_samples or len(r) < min_samples:
            out["features"][f] = {"status": "insufficient_samples", "psi": None, "ks_p": None, "drift": False}
            continue
        p, k = psi(r, c), ks_pvalue(r, c)
        drift = bool(p > psi_threshold or k < ks_alpha)
        out["features"][f] = {"psi": round(p, 4), "ks_p": round(k, 6), "drift": drift,
                              "ref_mean": round(float(np.mean(r)), 4), "cur_mean": round(float(np.mean(c)), 4)}
        out["drift_detected"] |= drift
    return out


def record_query(features: dict) -> None:
    """Append one inference's features to the query log (used by the API)."""
    QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(features) + "\n")


def load_features(path: Path, last_n: int | None = None) -> dict[str, list[float]]:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    if last_n:
        rows = rows[-last_n:]
    return {k: [float(r[k]) for r in rows if k in r] for k in FEATURES}


def load_reference() -> dict[str, list[float]]:
    if REFERENCE.exists():
        return json.loads(REFERENCE.read_text(encoding="utf-8"))
    return {k: [] for k in FEATURES}
