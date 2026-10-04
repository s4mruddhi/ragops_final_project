"""
app/core/storage.py
===================
Storage adapter — abstracts local filesystem vs AWS S3.

In local mode  (APP_ENV=local):  reads/writes to local paths.
In AWS mode    (APP_ENV=aws):    reads/writes to S3 bucket.

Usage:
    from app.core.storage import storage
    storage.upload("data/processed/chunks.jsonl", "processed/chunks.jsonl")
    storage.download("processed/chunks.jsonl", "data/processed/chunks.jsonl")
    exists = storage.exists("processed/chunks.jsonl")
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalStorageAdapter:
    """
    Local filesystem adapter.
    Remote keys are treated as relative paths under a local base directory.
    Used during development and CI.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: str, remote_key: str) -> None:
        """Copy local_path → base_dir/remote_key."""
        src = Path(local_path)
        dst = self.base_dir / remote_key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"[LOCAL] Uploaded {src} → {dst}")

    def download(self, remote_key: str, local_path: str) -> None:
        """Copy base_dir/remote_key → local_path."""
        src = self.base_dir / remote_key
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"[LOCAL] Downloaded {src} → {dst}")

    def exists(self, remote_key: str) -> bool:
        return (self.base_dir / remote_key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys under prefix."""
        target = self.base_dir / prefix
        if not target.exists():
            return []
        return [
            str(p.relative_to(self.base_dir))
            for p in target.rglob("*")
            if p.is_file()
        ]


class S3StorageAdapter:
    """
    AWS S3 adapter.
    Wraps boto3 — only imported when in AWS mode to avoid unnecessary dependency.
    Credentials are read from environment / IAM role — never hardcoded.
    """

    def __init__(self, bucket: str, region: str):
        try:
            import boto3
            self._s3 = boto3.client("s3", region_name=region)
            self._bucket = bucket
            logger.info(f"[S3] Adapter initialised: s3://{bucket} in {region}")
        except ImportError:
            raise RuntimeError(
                "boto3 is not installed. Run: pip install boto3"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise S3 client: {exc}") from exc

    def upload(self, local_path: str, remote_key: str) -> None:
        """Upload local file to S3."""
        self._s3.upload_file(local_path, self._bucket, remote_key)
        logger.info(f"[S3] Uploaded {local_path} → s3://{self._bucket}/{remote_key}")

    def download(self, remote_key: str, local_path: str) -> None:
        """Download S3 object to local file."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self._bucket, remote_key, local_path)
        logger.info(f"[S3] Downloaded s3://{self._bucket}/{remote_key} → {local_path}")

    def exists(self, remote_key: str) -> bool:
        """Check if an S3 key exists without downloading."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=remote_key)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys under prefix."""
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


# ---------------------------------------------------------------------------
# Factory — returns the right adapter based on APP_ENV
# ---------------------------------------------------------------------------

def _build_storage_adapter():
    """
    Called once at import time. Returns the correct adapter.
    Reads APP_ENV, AWS_S3_BUCKET, AWS_REGION from environment.
    """
    import os
    env = os.getenv("APP_ENV", "local").lower()
    if env == "aws":
        bucket = os.getenv("AWS_S3_BUCKET", "")
        region = os.getenv("AWS_REGION", "us-east-1")
        if not bucket:
            logger.warning(
                "APP_ENV=aws but AWS_S3_BUCKET is not set. "
                "Falling back to local storage adapter."
            )
            return LocalStorageAdapter()
        return S3StorageAdapter(bucket=bucket, region=region)
    return LocalStorageAdapter()


# Singleton — import this everywhere
storage = _build_storage_adapter()
