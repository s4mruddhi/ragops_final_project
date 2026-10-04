"""Run the exact DAG task order locally without an Airflow daemon (also used by CI).

    python scripts/run_dag_pipeline.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.pipeline_steps import STEPS  # noqa: E402

env = {**os.environ, "PYTHONPATH": str(ROOT), "ANONYMIZED_TELEMETRY": "False"}
t_all = time.time()
for i, (task_id, script, doc) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {task_id}: {doc}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, script], cwd=ROOT, env=env).returncode
    if rc != 0:
        print(f"[FAIL] {task_id} exit={rc}")
        sys.exit(rc)
    print(f"[PASS] {task_id} ({time.time() - t0:.1f}s)")
print(f"\n[SUCCESS] pipeline finished in {time.time() - t_all:.1f}s")
