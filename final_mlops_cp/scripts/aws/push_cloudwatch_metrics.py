"""Push the latest evaluation + benchmark + drift results to CloudWatch (custom namespace RAGOps).

    ENABLE_CLOUDWATCH=1 python scripts/aws/push_cloudwatch_metrics.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.monitoring.cloudwatch import put_metrics  # noqa: E402

rag = json.loads((ROOT / "data/evaluation/reports/rag_metrics.json").read_text())["summary"]
m = {"Groundedness": (rag["mean_groundedness"], "None"), "FallbackRate": (rag["fallback_rate"], "None"),
     "P95TotalLatency": (rag["p95_total_latency_ms"], "Milliseconds")}
bench = ROOT / "docs/benchmark_results.json"
if bench.exists():
    b = json.loads(bench.read_text())
    m.update({"ApiP95Latency": (b["latency_ms"]["p95"], "Milliseconds"),
              "ApiThroughput": (b["throughput_rps"], "Count/Second"), "ApiErrorRate": (b["error_rate"], "None")})
drift = ROOT / "data/monitoring/drift_report.json"
if drift.exists():
    d = json.loads(drift.read_text())["drifted_window"]
    m["DriftDetected"] = (1.0 if d["drift_detected"] else 0.0, "None")
import os
os.environ.setdefault("ENABLE_CLOUDWATCH", "1")
print("sent" if put_metrics(m) else "NOT sent (check credentials)", list(m))
