"""
Load test -> p50/p95/p99 latency, throughput (req/s), error rate. Writes docs/benchmark_results.json.

    python scripts/benchmark/load_test.py --url http://localhost:8000 --requests 300 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retriever", default=None)
    a = ap.parse_args()

    qs = [json.loads(l)["question"] for l in open(ROOT / "data/evaluation/rag_questions.jsonl", encoding="utf-8")]
    rng = random.Random(42)
    payloads = [{"question": rng.choice(qs), **({"retriever": a.retriever} if a.retriever else {})}
                for _ in range(a.requests)]

    client = httpx.Client(base_url=a.url, timeout=60, limits=httpx.Limits(max_connections=a.concurrency))
    client.post("/api/v1/query", json=payloads[0])  # warm-up (model / index load)

    def one(p):
        t = time.perf_counter()
        try:
            ok = client.post("/api/v1/query", json=p).status_code == 200
        except Exception:
            ok = False
        return (time.perf_counter() - t) * 1000, ok

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(one, payloads))
    wall = time.perf_counter() - t0
    lat = np.array([r[0] for r in res])
    errs = sum(1 for r in res if not r[1])
    out = {
        "requests": a.requests, "concurrency": a.concurrency, "retriever": a.retriever or "champion",
        "throughput_rps": round(a.requests / wall, 2),
        "latency_ms": {"mean": round(float(lat.mean()), 2), "p50": round(float(np.percentile(lat, 50)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2), "p99": round(float(np.percentile(lat, 99)), 2)},
        "error_rate": round(errs / a.requests, 4), "errors": errs,
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmark_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
