#!/usr/bin/env python3
"""
scripts/setup_dvc.py
====================
Initialises DVC and configures remotes for the RAGOps project.

Run this ONCE after cloning the repository:
    python scripts/setup_dvc.py

What it does:
    1. Initialises DVC (dvc init)
    2. Adds a local fallback remote (for offline development)
    3. Adds the S3 remote (if AWS_S3_BUCKET is set)
    4. Sets the default remote
    5. Tracks data directories with dvc add
    6. Creates .gitignore entries DVC expects

Environment variables required for S3 remote:
    AWS_S3_BUCKET  — your S3 bucket name
    AWS_REGION     — e.g. us-east-1

Usage:
    # Local only (no S3):
    python scripts/setup_dvc.py --mode local

    # With S3 remote:
    AWS_S3_BUCKET=my-ragops-bucket AWS_REGION=us-east-1 \\
        python scripts/setup_dvc.py --mode s3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and print it."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  [ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Set up DVC for RAGOps")
    parser.add_argument(
        "--mode",
        choices=["local", "s3"],
        default="local",
        help="Storage backend: 'local' (default) or 's3' (requires AWS_S3_BUCKET)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAGOps DVC Setup")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. DVC init (idempotent — safe to run again)
    # -----------------------------------------------------------------------
    print("\n[1] Initialising DVC...")
    if (ROOT / ".dvc").exists():
        print("  .dvc/ already exists — skipping dvc init")
    else:
        run(["dvc", "init"])
        print("  DVC initialised.")

    # -----------------------------------------------------------------------
    # 2. Configure local fallback remote
    # -----------------------------------------------------------------------
    print("\n[2] Configuring local DVC remote (fallback)...")
    local_cache = ROOT / ".dvc_local_remote"
    local_cache.mkdir(exist_ok=True)
    run(["dvc", "remote", "add", "-f", "localremote", str(local_cache)])
    print(f"  Local remote: {local_cache}")

    # -----------------------------------------------------------------------
    # 3. Configure S3 remote (if requested)
    # -----------------------------------------------------------------------
    if args.mode == "s3":
        bucket = os.environ.get("AWS_S3_BUCKET", "")
        region = os.environ.get("AWS_REGION", "us-east-1")
        if not bucket:
            print("\n[ERROR] AWS_S3_BUCKET environment variable is not set.")
            print("  Set it and retry: AWS_S3_BUCKET=my-bucket python scripts/setup_dvc.py --mode s3")
            sys.exit(1)

        s3_url = f"s3://{bucket}/dvc"
        print(f"\n[3] Configuring S3 DVC remote: {s3_url}")
        run(["dvc", "remote", "add", "-f", "s3remote", s3_url])
        run(["dvc", "remote", "modify", "s3remote", "region", region])
        run(["dvc", "remote", "default", "s3remote"])
        print(f"  S3 remote set as default: {s3_url} (region: {region})")
    else:
        print("\n[3] Setting local remote as default (local mode)...")
        run(["dvc", "remote", "default", "localremote"])
        print("  Local remote is default.")

    # -----------------------------------------------------------------------
    # 4. Track data directories
    # -----------------------------------------------------------------------
    print("\n[4] Tracking data directories with DVC...")

    data_dirs = [
        "data/raw",
        "data/processed",
        "data/evaluation",
    ]

    for d in data_dirs:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        result = run(["dvc", "add", d], check=False)
        if result.returncode == 0:
            print(f"  Tracked: {d}")
        else:
            print(f"  Already tracked or empty: {d} (exit={result.returncode})")

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    print("\n[5] Verifying DVC status...")
    run(["dvc", "status"], check=False)

    print()
    print("=" * 60)
    print("DVC setup complete.")
    print()
    print("Next steps:")
    print("  git add .dvc/ .dvcignore data/*.dvc")
    print("  git commit -m 'feat: initialise DVC tracking for data directories'")
    if args.mode == "s3":
        print("  dvc push    # push data to S3")
    else:
        print("  dvc push    # push data to local remote")
        print()
        print("  When ready for S3:")
        print("  AWS_S3_BUCKET=<bucket> python scripts/setup_dvc.py --mode s3")
        print("  dvc push")
    print("=" * 60)


if __name__ == "__main__":
    main()
