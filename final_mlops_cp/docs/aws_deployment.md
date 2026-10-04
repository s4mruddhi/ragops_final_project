# AWS Cloud Extension (S3 + SageMaker + CloudWatch)

Everything is **optional** for local runs, and **must be torn down after the demo** (cost control). Expected cost for one demo < $5.

## 0. Prereqs
```bash
aws configure                                    # IAM user with S3, SageMaker, CloudWatch access
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=ragops-demo-<yourname>-2026 # globally unique
```

## 1. S3 for data / artifacts
```bash
python scripts/aws/s3_sync.py create-bucket
python scripts/aws/s3_sync.py upload             # data/raw, eval set, processed, params.yaml, mlflow.db
# DVC remote on S3 (version evidence in the cloud):
dvc remote modify s3remote url s3://$AWS_S3_BUCKET/dvc
dvc push -r s3remote
```
Screenshot: S3 console listing `ragops/` and `dvc/`.

## 2. SageMaker training job
Create an execution role (`AmazonSageMakerFullAccess` + S3 access) and export its ARN:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role>
python scripts/sagemaker/run_training_job.py
```
The job runs `scripts/run_dag_pipeline.py` on `ml.m5.large` and writes reports to `s3://$AWS_S3_BUCKET/ragops/sagemaker/output`.
> The AWS-owned scikit-learn image account id in the script is for `us-east-1`; for another region look up the id in the
> SageMaker docs ("AWS Deep Learning / built-in algorithm image URIs") and edit `image` in `run_training_job.py`.
> This script could not be executed against real AWS by the author of the scaffold - test it once before the demo.

## 3. CloudWatch
* SageMaker job logs appear automatically in `/aws/sagemaker/TrainingJobs`.
* App metrics: `ENABLE_CLOUDWATCH=1 uvicorn app.main:app` pushes `TotalLatency`, `Groundedness`, `Fallback` to namespace `RAGOps`.
* Batch push of eval/benchmark/drift results: `ENABLE_CLOUDWATCH=1 python scripts/aws/push_cloudwatch_metrics.py`.
Screenshot: CloudWatch -> Metrics -> RAGOps, and the training-job log stream.

## 4. TEAR DOWN (mandatory)
```bash
python scripts/aws/cleanup.py --bucket $AWS_S3_BUCKET     # deletes bucket, stops jobs/endpoints
```
Then confirm in console: no S3 bucket, no running SageMaker jobs/endpoints, optionally delete the log groups and IAM role.
Screenshot the empty console as evidence.

## Tested locally with moto
`tests/test_aws.py` mocks S3 + CloudWatch, so CI exercises the code paths without credentials.
