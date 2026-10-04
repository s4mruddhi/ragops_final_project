"""
Upload / download the versioned data + artifacts to AWS S3.

    export AWS_S3_BUCKET=ragops-demo-<you>-2026
    python scripts/aws/s3_sync.py upload        # data/raw, evaluation set, reports, mlflow.db, params.yaml
    python scripts/aws/s3_sync.py download
    python scripts/aws/s3_sync.py create-bucket
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "ragops/"
UPLOAD_PATHS = ["data/raw", "data/evaluation", "data/processed", "params.yaml", "mlflow.db"]


def client(region=None):
    return boto3.client("s3", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))


def create_bucket(bucket: str, region: str = "us-east-1") -> None:
    s3 = client(region)
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})


def upload(bucket: str, root: Path = ROOT, paths=UPLOAD_PATHS, s3=None) -> int:
    s3 = s3 or client()
    n = 0
    for rel in paths:
        p = root / rel
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        for f in files:
            s3.upload_file(str(f), bucket, PREFIX + f.relative_to(root).as_posix())
            n += 1
    return n


def download(bucket: str, dest: Path = ROOT, s3=None) -> int:
    s3 = s3 or client()
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            target = dest / obj["Key"][len(PREFIX):]
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create-bucket", "upload", "download"])
    ap.add_argument("--bucket", default=os.environ.get("AWS_S3_BUCKET"))
    a = ap.parse_args()
    if not a.bucket:
        sys.exit("Set AWS_S3_BUCKET or pass --bucket")
    if a.action == "create-bucket":
        create_bucket(a.bucket, os.environ.get("AWS_REGION", "us-east-1"))
        print("created", a.bucket)
    elif a.action == "upload":
        print("uploaded", upload(a.bucket), "files to s3://%s/%s" % (a.bucket, PREFIX))
    else:
        print("downloaded", download(a.bucket), "files")
