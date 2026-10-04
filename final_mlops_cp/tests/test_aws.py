"""AWS extension tests against moto (no real credentials / cost)."""
import os

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@mock_aws
def test_s3_upload_download_roundtrip(tmp_path):
    from scripts.aws import s3_sync

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello")
    (tmp_path / "params.yaml").write_text("seed: 1")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3_sync.create_bucket("test-bucket")
    assert s3_sync.upload("test-bucket", root=tmp_path, paths=["data", "params.yaml"], s3=s3) == 2
    out = tmp_path / "restore"
    assert s3_sync.download("test-bucket", dest=out, s3=s3) == 2
    assert (out / "data" / "a.txt").read_text() == "hello"


@mock_aws
def test_cloudwatch_put_metrics():
    from app.monitoring.cloudwatch import NAMESPACE, put_metrics

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    assert put_metrics({"Groundedness": (0.9, "None")}, client=cw) is True
    names = [m["MetricName"] for m in cw.list_metrics(Namespace=NAMESPACE)["Metrics"]]
    assert "Groundedness" in names


def test_cloudwatch_disabled_by_default(monkeypatch):
    from app.monitoring.cloudwatch import put_metrics

    monkeypatch.delenv("ENABLE_CLOUDWATCH", raising=False)
    assert put_metrics({"X": (1, "None")}) is False
