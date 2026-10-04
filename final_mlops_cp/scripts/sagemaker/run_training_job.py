"""
Run the retrieval-experiment + quality-gate pipeline as an Amazon SageMaker *training job*
(managed, ephemeral compute). Source = this repo tarred to S3; entry point = sagemaker_entry.py.

    export AWS_S3_BUCKET=...  SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<role>
    python scripts/sagemaker/run_training_job.py
    # cleanup afterwards (cost control):  python scripts/aws/cleanup.py --bucket $AWS_S3_BUCKET
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = (".git", ".venv", "mlruns", "mlflow.db", "__pycache__", ".dvc/cache", "airflow", ".airflow", "data/vectorstore")


def package_source() -> Path:
    tmp = Path(tempfile.mkdtemp()) / "sourcedir.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not any(rel.startswith(e) or f"/{e}/" in f"/{rel}" for e in EXCLUDE):
                tar.add(p, arcname=rel)
    return tmp


def main() -> int:
    bucket, role = os.environ["AWS_S3_BUCKET"], os.environ["SAGEMAKER_ROLE_ARN"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    s3, sm = boto3.client("s3", region_name=region), boto3.client("sagemaker", region_name=region)
    job = f"ragops-eval-{int(time.time())}"
    key = f"ragops/sagemaker/{job}/sourcedir.tar.gz"
    s3.upload_file(str(package_source()), bucket, key)
    image = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"  # us-east-1 account id
    sm.create_training_job(
        TrainingJobName=job, RoleArn=role,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        HyperParameters={"sagemaker_program": '"scripts/sagemaker/sagemaker_entry.py"',
                         "sagemaker_submit_directory": f'"s3://{bucket}/{key}"',
                         "sagemaker_region": f'"{region}"'},
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/ragops/sagemaker/output"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )
    print("Started", job, "- watch logs in CloudWatch: /aws/sagemaker/TrainingJobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
