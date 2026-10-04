# AWS Setup Guide — RAGOps Project

> **For academic demonstration only.**  
> Estimated AWS cost for a single demo run: **< $5 USD** (if cleaned up promptly).

---

## Prerequisites

```bash
# Install AWS CLI v2 (https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
aws --version   # must be >= 2.x

# Configure your credentials (use a dedicated IAM user with limited permissions)
aws configure
# AWS Access Key ID: <your key>
# AWS Secret Access Key: <your secret>
# Default region: us-east-1
# Default output format: json
```

---

## Step 1 — Create S3 Bucket

```bash
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=ragops-demo-$(whoami)-2026   # must be globally unique

# Create the bucket
aws s3 mb s3://${AWS_S3_BUCKET} --region ${AWS_REGION}

# Create the expected prefix structure
for prefix in raw-documents processed-documents dvc mlflow embeddings \
              vector-index evaluation models reports; do
    aws s3api put-object --bucket ${AWS_S3_BUCKET} --key ${prefix}/
    echo "Created prefix: ${prefix}/"
done

# Verify
aws s3 ls s3://${AWS_S3_BUCKET}
```

---

## Step 2 — Configure DVC S3 Remote

```bash
# Set bucket name in DVC config (no hardcoded credentials)
dvc remote modify s3remote url s3://${AWS_S3_BUCKET}/dvc
dvc remote modify s3remote region ${AWS_REGION}
dvc remote default s3remote

# Push data to S3
dvc push

# Verify
aws s3 ls s3://${AWS_S3_BUCKET}/dvc/ --recursive | head -20
```

---

## Step 3 — Configure MLflow with S3 Artifact Store

```bash
# Update .env (do NOT commit)
echo "MLFLOW_ARTIFACT_ROOT=s3://${AWS_S3_BUCKET}/mlflow" >> .env
echo "MLFLOW_TRACKING_URI=http://localhost:5000" >> .env   # or remote MLflow server

# Start MLflow with S3 backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://${AWS_S3_BUCKET}/mlflow \
    --host 0.0.0.0 \
    --port 5000
```

---

## Step 4 — Create IAM Role for SageMaker

```bash
# Create trust policy
cat > /tmp/sagemaker-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
    --role-name ragops-sagemaker-role \
    --assume-role-policy-document file:///tmp/sagemaker-trust.json

# Attach needed policies
aws iam attach-role-policy \
    --role-name ragops-sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name ragops-sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get the role ARN for use in .env
aws iam get-role --role-name ragops-sagemaker-role \
    --query 'Role.Arn' --output text
```

Set the ARN in your `.env`:
```
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/ragops-sagemaker-role
```

---

## Step 5 — Set Up CloudWatch Log Groups

```bash
aws logs create-log-group \
    --log-group-name /ragops/api \
    --region ${AWS_REGION}

aws logs create-log-group \
    --log-group-name /ragops/pipeline \
    --region ${AWS_REGION}

# Set retention to 7 days (cost control)
aws logs put-retention-policy \
    --log-group-name /ragops/api \
    --retention-in-days 7

aws logs put-retention-policy \
    --log-group-name /ragops/pipeline \
    --retention-in-days 7

echo "CloudWatch log groups created"
```

---

## Step 6 — Deploy FastAPI on EC2 (Optional — for full AWS demo)

```bash
# Launch a t3.small instance with Amazon Linux 2023
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.small \
    --key-name <your-key-pair> \
    --security-group-ids <your-sg-id> \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ragops-api}]' \
    --count 1

# SSH in and run:
# sudo dnf install docker -y
# sudo systemctl start docker
# docker run -d -p 8000:8000 \
#   -e APP_ENV=aws \
#   -e AWS_S3_BUCKET=${AWS_S3_BUCKET} \
#   -e AWS_REGION=${AWS_REGION} \
#   ragops-api:latest
```

---

## Step 7 — Switch DVC to S3 for the Demo

```bash
# Confirm data is in S3
dvc push   # pushes data/raw to s3://${AWS_S3_BUCKET}/dvc

# Test round-trip (simulate a fresh clone)
dvc pull   # pulls from S3
```

---

## Estimated Costs (Demo Run Only)

| Service | Usage | Estimated Cost |
|---|---|---|
| S3 | ~100MB storage + transfers | < $0.01 |
| SageMaker | 1× ml.t3.medium job, ~30 min | ~$0.50 |
| EC2 | 1× t3.small, ~2 hours | ~$0.05 |
| CloudWatch | Logs ingestion ~1MB | < $0.01 |
| **Total** | | **< $1 USD** |

> Always run `scripts/aws/cleanup.md` after demo to avoid ongoing charges.
