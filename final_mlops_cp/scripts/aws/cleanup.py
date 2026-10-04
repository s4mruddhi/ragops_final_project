"""Delete the demo S3 bucket (all objects) and stop any running SageMaker jobs - run after the demo.

    python scripts/aws/cleanup.py --bucket ragops-demo-<you>-2026
"""
import argparse
import os

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
region = os.environ.get("AWS_REGION", "us-east-1")
if not a.yes and input(f"Delete bucket {a.bucket} and ALL objects? type 'yes': ") != "yes":
    raise SystemExit("aborted")
s3 = boto3.resource("s3", region_name=region)
b = s3.Bucket(a.bucket)
b.objects.all().delete()
b.object_versions.all().delete()
b.delete()
print("deleted bucket", a.bucket)
sm = boto3.client("sagemaker", region_name=region)
for j in sm.list_training_jobs(StatusEquals="InProgress")["TrainingJobSummaries"]:
    sm.stop_training_job(TrainingJobName=j["TrainingJobName"])
    print("stopped", j["TrainingJobName"])
for ep in sm.list_endpoints()["Endpoints"]:
    if ep["EndpointName"].startswith("ragops"):
        sm.delete_endpoint(EndpointName=ep["EndpointName"])
        print("deleted endpoint", ep["EndpointName"])
print("Also check: CloudWatch log groups /aws/sagemaker/*  and the IAM role if you created one.")
