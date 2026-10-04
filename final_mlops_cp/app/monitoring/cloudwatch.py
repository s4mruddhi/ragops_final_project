"""
CloudWatch monitoring (cloud extension). Pushes RAG quality/latency metrics and drift flags as
custom metrics in the `RAGOps` namespace. No-op unless ENABLE_CLOUDWATCH=1, so local runs and CI
never need AWS credentials.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
NAMESPACE = "RAGOps"


def enabled() -> bool:
    return os.environ.get("ENABLE_CLOUDWATCH", "0") == "1"


def put_metrics(metrics: dict[str, tuple[float, str]], dimensions: Optional[dict[str, str]] = None,
                client=None) -> bool:
    """metrics = {name: (value, unit)}; returns True if sent."""
    if client is None:
        if not enabled():
            return False
        import boto3
        client = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    dims = [{"Name": k, "Value": v} for k, v in (dimensions or {"Service": "ragops-api"}).items()]
    try:
        client.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{"MetricName": n, "Value": float(v), "Unit": u, "Dimensions": dims}
                        for n, (v, u) in metrics.items()],
        )
        return True
    except Exception as exc:  # monitoring must never break inference
        logger.warning(f"CloudWatch put_metric_data failed: {exc}")
        return False
