"""
SageMaker Pipelines — Full Training DAG
-----------------------------------------
Orchestrates the complete ML training workflow as a reproducible, versioned pipeline.

Steps:
  preprocess → validate → train → evaluate → conditional_register

Why SageMaker Pipelines (not Airflow/Prefect):
  - Native integration with S3, ECR, MLflow on AWS
  - Each step runs in its own Docker container (full reproducibility)
  - Pipeline execution is logged and versioned in AWS Console
  - Conditional step: only register if NDCG@5 improves (using ConditionStep)
  - CI/CD trigger: GitHub Actions starts this pipeline on every push to main

Usage:
    # First time: create/update the pipeline definition in AWS
    python src/pipeline/sagemaker_pipeline.py --action upsert

    # Trigger a run manually
    python src/pipeline/sagemaker_pipeline.py --action run

    # Describe latest execution status
    python src/pipeline/sagemaker_pipeline.py --action status

Environment variables required:
    AWS_REGION              ap-south-1
    SAGEMAKER_ROLE_ARN      arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
    ECR_IMAGE_URI_TRAINING  ACCOUNT.dkr.ecr.REGION.amazonaws.com/zomato-ads-mlops:training-latest
    S3_BUCKET               your-mlops-bucket
    MLFLOW_TRACKING_URI     http://mlflow.internal:5000
"""

import argparse
import json
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep

# ─── Config ───────────────────────────────────────────────────────────────────

AWS_REGION          = os.environ.get("AWS_REGION", "ap-south-1")
ROLE_ARN            = os.environ.get("SAGEMAKER_ROLE_ARN", "")
ECR_TRAINING_IMAGE  = os.environ.get("ECR_IMAGE_URI_TRAINING", "")
S3_BUCKET           = os.environ.get("S3_BUCKET", "your-mlops-bucket")
MLFLOW_URI          = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
PIPELINE_NAME       = "zomato-ads-training-pipeline"

S3_DATA_PREFIX      = f"s3://{S3_BUCKET}/data"
S3_FEATURES_PREFIX  = f"s3://{S3_BUCKET}/features"
S3_ARTIFACTS_PREFIX = f"s3://{S3_BUCKET}/artifacts"


def get_pipeline_session() -> PipelineSession:
    boto_session = boto3.Session(region_name=AWS_REGION)
    return PipelineSession(boto_session=boto_session)


# ─── Pipeline parameters (overridable per execution) ─────────────────────────

param_n_estimators = ParameterString(
    name="NEstimators",
    default_value="300",
)
param_max_depth = ParameterString(
    name="MaxDepth",
    default_value="6",
)
param_learning_rate = ParameterString(
    name="LearningRate",
    default_value="0.1",
)
param_ndcg_threshold = ParameterFloat(
    name="NDCGThreshold",
    default_value=0.0,  # 0.0 = always register first model
)
param_commit_hash = ParameterString(
    name="CommitHash",
    default_value="manual",
)


# ─── Step 1: Preprocess ───────────────────────────────────────────────────────

def make_preprocess_step(pipeline_session: PipelineSession) -> ProcessingStep:
    """
    Runs simulate_sessions.py + batch_features.py inside the training container.
    Input:  raw Zomato CSV from S3
    Output: ltr_train.parquet, ltr_test.parquet → S3
    """
    processor = ScriptProcessor(
        image_uri=ECR_TRAINING_IMAGE,
        command=["python"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session,
        env={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    )

    return ProcessingStep(
        name="Preprocess",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"{S3_DATA_PREFIX}/raw/zomato/",
                destination="/opt/ml/processing/input/raw",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output/features",
                destination=f"{S3_FEATURES_PREFIX}/latest",
            ),
            ProcessingOutput(
                output_name="validation_report",
                source="/opt/ml/processing/output/validation",
                destination=f"{S3_ARTIFACTS_PREFIX}/validation",
            ),
        ],
        code="src/pipeline/scripts/preprocess.py",
    )


# ─── Step 2: Train ────────────────────────────────────────────────────────────

def make_train_step(
    pipeline_session: PipelineSession,
    preprocess_step: ProcessingStep,
) -> ProcessingStep:
    """
    Runs LambdaMART training + MLflow logging inside the training container.
    Input:  feature parquets from S3
    Output: evaluation_metrics.json (used by ConditionStep), model artifacts
    """
    processor = ScriptProcessor(
        image_uri=ECR_TRAINING_IMAGE,
        command=["python"],
        instance_type="ml.m5.2xlarge",  # more memory for XGBoost
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session,
        env={
            "MLFLOW_TRACKING_URI": MLFLOW_URI,
            "N_ESTIMATORS":        param_n_estimators,
            "MAX_DEPTH":           param_max_depth,
            "LEARNING_RATE":       param_learning_rate,
            "COMMIT_HASH":         param_commit_hash,
        },
    )

    # PropertyFile lets ConditionStep read metrics from this step's output
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_metrics.json",
    )

    return ProcessingStep(
        name="Train",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig
                       .Outputs["features"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/features",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=f"{S3_ARTIFACTS_PREFIX}/evaluation",
            ),
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/output/model",
                destination=f"{S3_ARTIFACTS_PREFIX}/model",
            ),
        ],
        code="src/pipeline/scripts/train_step.py",
        property_files=[evaluation_report],
    ), evaluation_report


# ─── Step 3: Conditional Register ─────────────────────────────────────────────

def make_register_step(
    pipeline_session: PipelineSession,
    train_step: ProcessingStep,
    evaluation_report: PropertyFile,
) -> ConditionStep:
    """
    Only registers the model if NDCG@5 exceeds the threshold.
    This prevents regressions from being promoted to production.

    The ConditionStep is the key MLOps pattern here:
      if ndcg_at_5 > param_ndcg_threshold:
          run register_step
      else:
          skip (pipeline succeeds but model is not registered)
    """
    processor = ScriptProcessor(
        image_uri=ECR_TRAINING_IMAGE,
        command=["python"],
        instance_type="ml.t3.medium",
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session,
        env={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    )

    register_step = ProcessingStep(
        name="RegisterModel",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig
                       .Outputs["model"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        code="src/pipeline/scripts/register_step.py",
        depends_on=[train_step],
    )

    # Read NDCG@5 from the evaluation output JSON
    ndcg_value = JsonGet(
        step_name=train_step.name,
        property_file=evaluation_report,
        json_path="ndcg_at_5",
    )

    condition = ConditionGreaterThan(
        left=ndcg_value,
        right=param_ndcg_threshold,
    )

    return ConditionStep(
        name="CheckNDCGAndRegister",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[],  # skip registration — pipeline still succeeds
    )


# ─── Build Pipeline ───────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    pipeline_session = get_pipeline_session()

    preprocess_step = make_preprocess_step(pipeline_session)
    train_step, evaluation_report = make_train_step(pipeline_session, preprocess_step)
    condition_step = make_register_step(pipeline_session, train_step, evaluation_report)

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            param_n_estimators,
            param_max_depth,
            param_learning_rate,
            param_ndcg_threshold,
            param_commit_hash,
        ],
        steps=[preprocess_step, train_step, condition_step],
        sagemaker_session=pipeline_session,
    )


# ─── Actions ──────────────────────────────────────────────────────────────────

def upsert_pipeline() -> None:
    """Create or update the pipeline definition in AWS."""
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f"Pipeline '{PIPELINE_NAME}' upserted successfully.")
    definition = json.loads(pipeline.definition())
    print(f"Steps: {[s['Name'] for s in definition['Steps']]}")


def run_pipeline(commit_hash: str = "manual") -> None:
    """Trigger a new pipeline execution."""
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    response = sm_client.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=f"run-{commit_hash[:7]}",
        PipelineParameters=[
            {"Name": "CommitHash", "Value": commit_hash},
        ],
    )
    arn = response["PipelineExecutionArn"]
    print(f"Pipeline execution started: {arn}")
    print(f"Track at: https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home#/pipelines")


def describe_latest() -> None:
    """Print the status of the most recent pipeline execution."""
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    executions = sm_client.list_pipeline_executions(
        PipelineName=PIPELINE_NAME,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["PipelineExecutionSummaries"]

    if not executions:
        print("No executions found.")
        return

    latest = executions[0]
    arn = latest["PipelineExecutionArn"]
    status = latest["PipelineExecutionStatus"]
    created = latest["CreationTime"]
    print(f"Latest execution: {arn}")
    print(f"Status:  {status}")
    print(f"Created: {created}")

    # Show step statuses
    steps = sm_client.list_pipeline_execution_steps(
        PipelineExecutionArn=arn
    )["PipelineExecutionSteps"]

    print("\nStep statuses:")
    for step in steps:
        print(f"  {step['StepName']:30s} {step['StepStatus']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        default="upsert",
        choices=["upsert", "run", "status"],
    )
    parser.add_argument("--commit-hash", default="manual")
    args = parser.parse_args()

    if args.action == "upsert":
        upsert_pipeline()
    elif args.action == "run":
        run_pipeline(args.commit_hash)
    elif args.action == "status":
        describe_latest()


if __name__ == "__main__":
    main()
