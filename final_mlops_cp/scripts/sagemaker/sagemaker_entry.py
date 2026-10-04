"""SageMaker training entry point: runs the same pipeline, writes reports to /opt/ml/model."""
import os
import shutil
import subprocess
import sys

root = os.environ.get("SM_SOURCE_DIR", os.getcwd())
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=root)
subprocess.check_call([sys.executable, "scripts/run_dag_pipeline.py"], cwd=root)  # logs stream to CloudWatch
out = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
shutil.copytree(os.path.join(root, "data", "evaluation", "reports"), os.path.join(out, "reports"), dirs_exist_ok=True)
