# AWS Cleanup Guide — RAGOps Project

> **Run this after every demo to avoid ongoing AWS charges.**  
> Follow these steps in order to remove all resources created by the setup guide.

---

## Variables (set these first)

```bash
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=ragops-demo-<your-username>-2026   # your actual bucket name
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

---

## Step 1 — Stop and Remove EC2 Instances

```bash
# Find the RAGOps instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ragops-api" \
              "Name=instance-state-name,Values=running,stopped" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text)

echo "Instance to terminate: ${INSTANCE_ID}"

# Terminate (this permanently deletes the instance)
if [ -n "${INSTANCE_ID}" ]; then
    aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}
    echo "Termination initiated for ${INSTANCE_ID}"
fi
```

---

## Step 2 — Delete SageMaker Jobs and Endpoints

```bash
# List recent training jobs
aws sagemaker list-training-jobs \
    --name-contains ragops \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 10

# SageMaker training jobs incur cost only while RUNNING.
# Once Complete/Failed, they do NOT incur charges.
# However, clean up any leftover endpoints:

# List endpoints
aws sagemaker list-endpoints --name-contains ragops

# Delete any active endpoints (none expected for this project — we use training jobs)
# aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

---

## Step 3 — Empty and Delete S3 Bucket

> **WARNING:** This permanently deletes all data in the bucket including DVC artifacts, MLflow runs, and evaluation reports. Download anything you need first.

```bash
# First download any evaluation reports you want to keep
aws s3 cp s3://${AWS_S3_BUCKET}/evaluation/ ./local_backup/ --recursive

# Remove all objects (required before bucket deletion)
aws s3 rm s3://${AWS_S3_BUCKET} --recursive
echo "All objects removed from ${AWS_S3_BUCKET}"

# Delete the bucket
aws s3 rb s3://${AWS_S3_BUCKET}
echo "Bucket deleted: ${AWS_S3_BUCKET}"

# Verify it's gone
aws s3 ls | grep ragops || echo "Bucket successfully deleted"
```

---

## Step 4 — Remove CloudWatch Log Groups

```bash
# Delete log groups (this deletes all logs — download first if needed)
aws logs delete-log-group \
    --log-group-name /ragops/api \
    --region ${AWS_REGION}

aws logs delete-log-group \
    --log-group-name /ragops/pipeline \
    --region ${AWS_REGION}

echo "CloudWatch log groups deleted"
```

---

## Step 5 — Remove IAM Role

```bash
# Detach policies first (required before deletion)
aws iam detach-role-policy \
    --role-name ragops-sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam detach-role-policy \
    --role-name ragops-sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Delete the role
aws iam delete-role --role-name ragops-sagemaker-role
echo "IAM role deleted"
```

---

## Step 6 — Revert DVC to Local Remote

```bash
# After deleting S3, switch DVC back to local remote
dvc remote default localremote
git add .dvc/config
git commit -m "chore: revert DVC remote to local after demo cleanup"
```

---

## Step 7 — Verify No Resources Remain

```bash
# Check for any running EC2 instances
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ragops*" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name]' \
    --output table

# Check for active SageMaker training jobs
aws sagemaker list-training-jobs \
    --status-equals InProgress \
    --name-contains ragops

# Check S3 bucket is gone
aws s3 ls | grep ragops || echo "No RAGOps S3 buckets found"

echo "Cleanup complete."
```

---

## Cost Safety Checklist

```
[ ] EC2 instance terminated
[ ] SageMaker training jobs — Completed or Stopped (no in-progress jobs)
[ ] S3 bucket emptied and deleted (or confirm 0 objects)
[ ] CloudWatch log groups deleted
[ ] IAM role deleted
[ ] No ongoing charges in AWS Billing console
```

> After cleanup, verify in the **AWS Billing → Cost Explorer** that no unexpected charges are accruing.
