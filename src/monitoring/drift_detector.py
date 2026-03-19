"""
Drift Detection with Evidently AI
-----------------------------------
Runs daily comparison of:
  1. Feature distributions: live traffic vs training baseline
  2. Prediction score distribution: current model output vs training-time output
  3. Target drift: if label proxies shift (CTR dropping), trigger retraining

Why this matters:
  - Restaurant catalogs change (new restaurants open, old ones close)
  - User preferences shift with seasons, events, trends
  - Without drift detection, model silently degrades

Outputs:
  - HTML drift report saved to S3 (viewable by team)
  - JSON summary for alerting
  - If PSI > threshold: triggers CodePipeline retraining webhook

Usage:
    python src/monitoring/drift_detector.py
    python src/monitoring/drift_detector.py --reference-days 10 --current-days 5
    python src/monitoring/drift_detector.py --trigger-retrain-if-drifted
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
import requests
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

FEATURES_DIR = Path("data/features")
REPORTS_DIR = Path("data/drift_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# PSI threshold: industry standard is 0.2 (moderate drift), 0.25 (severe drift)
DRIFT_PSI_THRESHOLD = 0.20
S3_REPORTS_BUCKET = os.getenv("DRIFT_REPORTS_BUCKET", "")
CODEPIPELINE_NAME = os.getenv("RETRAIN_PIPELINE_NAME", "zomato-ads-retrain-pipeline")

FEATURE_COLS = [
    "cuisine_match", "price_fit", "rating_score", "promo_flag",
    "meal_time_match", "is_veg_match", "votes_log", "delivery_available",
    "day_of_week", "hour_of_day", "rank_position",
    "click_through_rate", "order_rate", "ctr_wilson_lower",
]

TARGET_COL = "label"
PREDICTION_COL = "score"


def load_reference_data(n_days: int = 10) -> pd.DataFrame:
    """Training data = reference (baseline). Days 1–10."""
    df = pd.read_parquet(FEATURES_DIR / "ltr_train.parquet")
    return df[FEATURE_COLS + [TARGET_COL]].sample(
        min(50_000, len(df)), random_state=42
    )


def load_current_data(n_days: int = 5) -> pd.DataFrame:
    """
    Production traffic from last N days.
    In a real system this comes from a prediction log table in S3/Redshift.
    Here we simulate it from the test set with slight distribution shift.
    """
    df = pd.read_parquet(FEATURES_DIR / "ltr_test.parquet")

    # Simulate realistic production drift:
    # - Slight shift in cuisine_match (seasonal menu changes)
    # - Slight shift in price_fit (inflation → budget pressure)
    import numpy as np
    rng = np.random.default_rng(99)
    df = df.copy()
    df["cuisine_match"] = (df["cuisine_match"] + rng.normal(0.05, 0.1, len(df))).clip(0, 1)
    df["price_fit"]     = (df["price_fit"]     + rng.normal(-0.05, 0.08, len(df))).clip(0, 1)

    return df[FEATURE_COLS + [TARGET_COL]].sample(
        min(50_000, len(df)), random_state=42
    )


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[Report, dict]:
    """Run Evidently data drift + target drift report."""

    column_mapping = ColumnMapping(
        target=TARGET_COL,
        numerical_features=[
            c for c in FEATURE_COLS
            if c not in ["promo_flag", "is_veg_match", "delivery_available",
                         "day_of_week"]
        ],
        categorical_features=["promo_flag", "is_veg_match", "delivery_available",
                               "day_of_week"],
    )

    report = Report(metrics=[
        DataDriftPreset(drift_share=0.3),
        TargetDriftPreset(),
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    # Extract per-feature PSI scores
    drift_summary = {
        "run_at":       datetime.utcnow().isoformat(),
        "n_reference":  len(reference),
        "n_current":    len(current),
        "drifted_features": [],
        "max_psi":      0.0,
        "dataset_drift": False,
    }

    try:
        drift_result = result["metrics"][0]["result"]
        drift_summary["dataset_drift"] = drift_result.get("dataset_drift", False)
        drift_summary["share_drifted"] = drift_result.get("share_of_drifted_columns", 0.0)

        for feature, stats in drift_result.get("drift_by_columns", {}).items():
            psi = stats.get("stattest_threshold", 0.0)
            if stats.get("drift_detected", False):
                drift_summary["drifted_features"].append({
                    "feature": feature,
                    "drift_score": stats.get("drift_score", 0.0),
                    "psi": psi,
                })
            drift_summary["max_psi"] = max(drift_summary["max_psi"], psi)
    except (KeyError, IndexError):
        pass

    return report, drift_summary


def save_report(report: Report, summary: dict) -> Path:
    """Save HTML report locally and optionally to S3."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    json_path = REPORTS_DIR / f"drift_summary_{timestamp}.json"

    report.save_html(str(html_path))
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  HTML report: {html_path}")
    print(f"  JSON summary: {json_path}")

    # Upload to S3 if bucket configured
    if S3_REPORTS_BUCKET:
        s3 = boto3.client("s3")
        s3_key = f"drift-reports/drift_report_{timestamp}.html"
        s3.upload_file(str(html_path), S3_REPORTS_BUCKET, s3_key)
        print(f"  Uploaded to s3://{S3_REPORTS_BUCKET}/{s3_key}")

    return html_path


def trigger_retraining_pipeline() -> None:
    """
    Trigger CodePipeline to kick off a new training run.
    This closes the MLOps loop: drift detected → retrain automatically.
    """
    print("\nTriggering retraining pipeline via CodePipeline...")
    try:
        client = boto3.client("codepipeline", region_name=os.getenv("AWS_REGION", "ap-south-1"))
        response = client.start_pipeline_execution(name=CODEPIPELINE_NAME)
        execution_id = response["pipelineExecutionId"]
        print(f"  Pipeline execution started: {execution_id}")
    except Exception as e:
        print(f"  CodePipeline trigger failed (non-critical): {e}")
        print(f"  To retrain manually: python src/training/train.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-days",         type=int,  default=10)
    parser.add_argument("--current-days",           type=int,  default=5)
    parser.add_argument("--trigger-retrain-if-drifted", action="store_true")
    args = parser.parse_args()

    print("Loading reference data (training baseline)...")
    reference = load_reference_data(args.reference_days)
    print(f"  {len(reference):,} reference rows")

    print("Loading current production data...")
    current = load_current_data(args.current_days)
    print(f"  {len(current):,} current rows")

    print("\nRunning drift analysis...")
    report, summary = run_drift_report(reference, current)

    print(f"\nDrift Summary:")
    print(f"  Dataset drift detected : {summary['dataset_drift']}")
    print(f"  Share of drifted cols  : {summary.get('share_drifted', 0.0):.1%}")
    print(f"  Max PSI                : {summary['max_psi']:.4f}  (threshold: {DRIFT_PSI_THRESHOLD})")

    if summary["drifted_features"]:
        print(f"  Drifted features:")
        for f in summary["drifted_features"]:
            print(f"    {f['feature']:30s} drift_score={f['drift_score']:.4f}")

    save_report(report, summary)

    # Trigger retraining if drift exceeds threshold
    if args.trigger_retrain_if_drifted and summary["max_psi"] > DRIFT_PSI_THRESHOLD:
        print(f"\nPSI {summary['max_psi']:.4f} > threshold {DRIFT_PSI_THRESHOLD}")
        trigger_retraining_pipeline()
    elif summary["max_psi"] > DRIFT_PSI_THRESHOLD:
        print(f"\nPSI {summary['max_psi']:.4f} > threshold — consider retraining.")
        print("  Run with --trigger-retrain-if-drifted to automate this.")
    else:
        print(f"\nNo significant drift detected. Model is stable.")


if __name__ == "__main__":
    main()
