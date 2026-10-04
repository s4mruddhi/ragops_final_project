"""
Drift simulation (evidence for the report + Grafana).

1. Reference  = champion retriever + generator on the gold evaluation questions.
2. Drifted    = realistic production shift: off-topic, very long, and typo-heavy queries.
3. Computes PSI / KS per feature -> data/monitoring/drift_report.json
4. Optional: --api http://localhost:8000 replays both phases against the live API so the
   Prometheus/Grafana dashboard shows the shift, then calls /api/v1/drift.

    python scripts/monitoring/simulate_drift.py
    python scripts/monitoring/simulate_drift.py --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jsonlines  # noqa: E402

from app.core.mlflow_utils import load_params, set_seed  # noqa: E402
from app.generation.generator import AnswerGenerator  # noqa: E402
from app.monitoring import drift  # noqa: E402
from app.retrieval.factory import get_retriever  # noqa: E402

OFF_TOPIC = [
    "What is the best pizza topping for a team lunch celebration this Friday evening",
    "Can you recommend a good science fiction novel to read over the long weekend",
    "Who won the cricket world cup final and what was the score of the match",
    "How do I train for a marathon in six months with a full time desk job",
    "Explain the plot of the latest superhero movie and its post credit scenes",
    "What is the weather forecast for the coming week in the northern hemisphere",
]
VERBOSE = (
    "I have a rather long and detailed question about how everything works at the company, "
    "including every possible exception, special case, regional variation and historical change, "
    "and I would like a complete and exhaustive explanation regarding "
)


def drifted_queries(base: list[str], n: int, rng: random.Random) -> list[str]:
    out = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            out.append(rng.choice(OFF_TOPIC))
        elif kind == 1:
            out.append(VERBOSE + rng.choice(base).lower())
        else:  # typo-heavy
            q = list(rng.choice(base))
            for _ in range(max(3, len(q) // 8)):
                j = rng.randrange(len(q))
                q[j] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            out.append("".join(q))
    return out


def features_for(queries: list[str], retriever, generator) -> dict[str, list[float]]:
    feats = {k: [] for k in drift.FEATURES}
    for q in queries:
        chunks = retriever.retrieve(q, top_k=5)
        out = generator.generate_answer(q, chunks)
        feats["query_length"].append(len(q.split()))
        feats["top_retrieval_score"].append(float(chunks[0].score) if chunks else 0.0)
        feats["groundedness_score"].append(float(out["groundedness_score"]))
        feats["fallback"].append(1.0 if out["is_fallback"] else 0.0)
    return feats


def replay_to_api(api: str, queries: list[str], label: str) -> None:
    import httpx

    ok = 0
    for q in queries:
        try:
            ok += httpx.post(f"{api}/api/v1/query", json={"question": q}, timeout=30).status_code == 200
        except Exception as exc:
            print("  request failed:", exc)
            break
    print(f"  [{label}] {ok}/{len(queries)} requests OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    params = load_params()
    rng = random.Random(set_seed())
    with jsonlines.open(ROOT / params["evaluation"]["dataset_path"]) as r:
        base = [x["question"] for x in r]
    method = "tfidf"
    try:
        method = json.loads((ROOT / "data/evaluation/reports/champion.json").read_text())["retriever"]
    except Exception:
        pass
    import os
    has_real_llm = bool(
        os.environ.get("OPENAI_API_KEY")
        and not os.environ.get("OPENAI_API_KEY", "").startswith("sk-your")
    )
    retriever = get_retriever(method)
    generator = AnswerGenerator(allow_test_mock=not has_real_llm)

    normal = [rng.choice(base) for _ in range(args.n)]
    shifted = drifted_queries(base, args.n, rng)
    ref = features_for(base, retriever, generator)
    # reference = gold questions; "normal" traffic = resampled gold questions; "drifted" = shifted
    cur_normal = features_for(normal, retriever, generator)
    cur_drift = features_for(shifted, retriever, generator)

    d_cfg = params["drift"]
    kw = dict(psi_threshold=d_cfg["psi_threshold"], ks_alpha=d_cfg["ks_p_value_threshold"],
              min_samples=d_cfg["min_samples"])
    rep_normal = drift.detect_drift(ref, cur_normal, **kw)
    rep_drift = drift.detect_drift(ref, cur_drift, **kw)

    drift.REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    drift.REFERENCE.write_text(json.dumps(ref), encoding="utf-8")
    report = {"retriever": method, "reference_queries": len(base), "window_size": args.n,
              "thresholds": kw, "no_drift_window": rep_normal, "drifted_window": rep_drift}
    out = ROOT / "data" / "monitoring" / "drift_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'feature':<22}{'PSI normal':>12}{'PSI drifted':>14}{'KS p drifted':>14}  drift?")
    for f in drift.FEATURES:
        a, b = rep_normal["features"][f], rep_drift["features"][f]
        print(f"{f:<22}{a['psi']:>12}{b['psi']:>14}{b['ks_p']:>14}  {b['drift']}")
    print("no-drift window flagged:", rep_normal["drift_detected"], "| drifted window flagged:", rep_drift["drift_detected"])
    print("report ->", out.relative_to(ROOT))

    if args.api:
        print("Replaying traffic against", args.api)
        replay_to_api(args.api, normal, "normal")
        replay_to_api(args.api, shifted, "drifted")
        import httpx
        print(json.dumps(httpx.get(f"{args.api}/api/v1/drift").json(), indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
