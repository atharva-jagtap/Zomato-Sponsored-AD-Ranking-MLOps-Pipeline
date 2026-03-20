# Zomato Restaurant Ads Ranking - End-to-End MLOps Pipeline

This project is meant to be completed in four connected phases:

1. Install local tools and credentials
2. Provision required AWS services
3. Start local MLOps services
4. Run the full data, training, serving, and monitoring pipeline

If you want the full project exactly as designed, AWS services and Terraform are
not optional. The main mismatch on your machine is the shell and OS commands,
not the need for cloud infrastructure.

## How To Read The Guide On Your System

You are using Windows with Git Bash. That means:

- AWS CLI is still required
- Terraform is still required
- Docker is still required
- The AWS setup phases are still required
- Some Linux or macOS commands from the guide need Windows Git Bash equivalents

So the correct interpretation is:

- Keep all AWS and Terraform steps
- Translate the shell commands for Windows Git Bash where needed
- Treat local services like MLflow, Redis, Grafana, and Prometheus as part of the full workflow, not as a replacement for AWS

## Full Architecture

```text
Raw Data (S3 + DVC)
  -> Validation (Great Expectations)
  -> Feature Engineering
  -> Feature Store (Feast offline + Redis online)
  -> Training (XGBoost LambdaMART)
  -> Experiment Tracking + Registry (MLflow)
  -> CI/CD (GitHub Actions + ECR + SageMaker)
  -> Serving (FastAPI / BentoML on EC2 behind ALB)
  -> Monitoring (Prometheus + Grafana + Loki)
  -> Drift Detection (Evidently)
  -> Retraining Trigger
```

## Repo Layout

```text
src/ingestion/            Kaggle download + session simulation
src/validation/           Data quality checks
src/features/             Batch features + Feast integration
src/training/             Train, evaluate, and register models
src/serving/              FastAPI and BentoML serving
src/monitoring/           Drift detection + alert rules
src/pipeline/             SageMaker pipeline definition
docker/                   Training and serving Dockerfiles
infra/main.tf             AWS infrastructure
docker-compose.yml        Local MLflow / Redis / Prometheus / Grafana / Loki stack
tests/test_all.py         Unit tests
```

## Phase 1: Local Setup

### Required tools

- Python 3.11
- Docker Desktop
- AWS CLI v2
- Terraform 1.7+
- Kaggle API credentials

### Windows Git Bash setup

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

### AWS CLI on Windows Git Bash

Install AWS CLI with Windows tools such as `winget`, then reopen Git Bash:

```bash
aws --version
aws configure
aws sts get-caller-identity
```

### Terraform on Windows Git Bash

Install Terraform with Windows tools such as `winget`, then reopen Git Bash:

```bash
terraform version
```

### Kaggle credentials on Git Bash

```bash
mkdir -p ~/.kaggle
cp /c/Users/chess/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Phase 2: AWS Services Setup

These steps are required for the complete project:

1. Create the S3 bucket
2. Create the ECR repository
3. Create IAM roles
4. Create the EC2 key pair
5. Update the Terraform backend
6. Run Terraform
7. Register the SageMaker pipeline

### Required environment values

Fill these into `.env` as you complete the AWS setup:

```bash
AWS_REGION=ap-south-1
S3_BUCKET=
FEAST_S3_BUCKET=
DRIFT_REPORTS_BUCKET=
SAGEMAKER_ROLE_ARN=
ECR_URI=
ECR_IMAGE_URI_TRAINING=
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=zomato-ads-ranker
MODEL_STAGE=Production
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Terraform on this repo

Terraform is driven from [main.tf](/c:/Users/chess/Desktop/zomato-ads-mlops/infra/main.tf).

Before running it, update the backend bucket in that file to your real S3 bucket.

Then from Git Bash:

```bash
cd infra
terraform init
terraform plan -var="key_pair_name=zomato-mlops"
terraform apply -var="key_pair_name=zomato-mlops" -auto-approve
cd ..
```

## Local MLOps Services

Service URLs:

- MLflow: `http://localhost:5000`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Loki: `http://localhost:3100`

## CI/CD And SageMaker

The full guide also includes:

- pushing training and serving images to ECR
- using [ci.yml](/c:/Users/chess/Desktop/zomato-ads-mlops/.github/workflows/ci.yml) for tests and image build
- registering and running [sagemaker_pipeline.py](/c:/Users/chess/Desktop/zomato-ads-mlops/src/pipeline/sagemaker_pipeline.py)
- deploying through EC2 and ALB from [main.tf](/c:/Users/chess/Desktop/zomato-ads-mlops/infra/main.tf)
