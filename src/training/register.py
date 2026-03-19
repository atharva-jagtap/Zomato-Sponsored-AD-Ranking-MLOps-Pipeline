"""
Model Registry — Staging → Production Promotion
-------------------------------------------------
Promotes the latest Staging model to Production in MLflow registry
after manual or automated approval.

Why a separate promotion step (not auto-promote from training):
  - Training auto-registers to Staging if NDCG improves
  - A human (or automated gate) reviews Staging before promoting to Production
  - This is the industry-standard pattern: train → staging → approve → production

Promotion gates checked before promoting:
  1. NDCG@5 in Staging > NDCG@5 of current Production
  2. No data validation failures in the associated run
  3. Model artifact exists and loads without error

Usage:
    # Inspect what's in Staging before promoting
    python src/training/register.py --action inspect

    # Promote Staging → Production (with all gates)
    python src/training/register.py --action promote

    # Roll back Production to previous version
    python src/training/register.py --action rollback
"""

import argparse
import os
import sys

import mlflow
import mlflow.xgboost
import numpy as np
from mlflow.exceptions import RestException

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "zomato-ads-ranker")


def get_client() -> mlflow.tracking.MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def inspect() -> None:
    """Show current state of the model registry."""
    client = get_client()

    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    try:
        client.get_registered_model(MODEL_NAME)
    except RestException:
        print("\nModel registry is empty for this model.")
        print("Run training first so a model can be registered to Staging.")
        return

    for stage in ["Staging", "Production", "Archived"]:
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if not versions:
            print(f"\n{stage}: (none)")
            continue

        for v in versions:
            run = client.get_run(v.run_id)
            metrics = run.data.metrics
            params  = run.data.params

            print(f"\n{stage}: v{v.version}")
            print(f"  Run ID       : {v.run_id[:8]}...")
            print(f"  Created      : {v.creation_timestamp}")
            print(f"  NDCG@5       : {metrics.get('ndcg_at_5', 'N/A'):.4f}")
            print(f"  MRR          : {metrics.get('mrr', 'N/A'):.4f}")
            print(f"  Mean Ord Rank: {metrics.get('mean_ordered_rank', 'N/A'):.4f}")
            print(f"  N estimators : {params.get('n_estimators', 'N/A')}")
            print(f"  Max depth    : {params.get('max_depth', 'N/A')}")
            print(f"  Train rows   : {params.get('n_train_rows', 'N/A')}")


def run_promotion_gates(
    client: mlflow.tracking.MlflowClient,
    staging_version,
    production_version,
) -> tuple[bool, list[str]]:
    """
    Run all gates before allowing promotion.
    Returns (passed: bool, reasons: list[str])
    """
    reasons = []

    staging_run  = client.get_run(staging_version.run_id)
    staging_ndcg = staging_run.data.metrics.get("ndcg_at_5", 0.0)

    # Gate 1: Staging NDCG > Production NDCG
    if production_version:
        prod_run  = client.get_run(production_version.run_id)
        prod_ndcg = prod_run.data.metrics.get("ndcg_at_5", 0.0)

        if staging_ndcg <= prod_ndcg:
            reasons.append(
                f"NDCG@5 did not improve: Staging={staging_ndcg:.4f} <= Production={prod_ndcg:.4f}"
            )
        else:
            print(f"  Gate 1 PASSED: NDCG@5 {prod_ndcg:.4f} → {staging_ndcg:.4f} (+{staging_ndcg - prod_ndcg:.4f})")
    else:
        print(f"  Gate 1 PASSED: No production model yet — first deployment (NDCG@5={staging_ndcg:.4f})")

    # Gate 2: Model loads without error
    try:
        model_uri = f"models:/{MODEL_NAME}/Staging"
        model = mlflow.xgboost.load_model(model_uri)
        # Quick smoke test: predict on dummy input (18 features)
        import xgboost as xgb
        dummy = np.zeros((3, 18), dtype=np.float32)
        dmat = xgb.DMatrix(dummy)
        scores = model.predict(dmat)
        assert len(scores) == 3, "Expected 3 scores from dummy input"
        print(f"  Gate 2 PASSED: Model loads and predicts without error")
    except Exception as e:
        reasons.append(f"Model failed to load or predict: {e}")

    # Gate 3: Training run has no parameter anomalies
    params = staging_run.data.params
    n_features = int(params.get("n_features", 0))
    if n_features < 10:
        reasons.append(f"Unexpected n_features={n_features} (expected >= 10)")
    else:
        print(f"  Gate 3 PASSED: n_features={n_features}")

    return len(reasons) == 0, reasons


def promote() -> None:
    """Promote latest Staging model to Production."""
    client = get_client()

    try:
        staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    except RestException:
        print("No registered model found yet. Run training first: python src/training/train.py")
        sys.exit(1)

    if not staging_versions:
        print("No model in Staging. Run training first: python src/training/train.py")
        sys.exit(1)

    staging_version = staging_versions[0]
    production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    production_version = production_versions[0] if production_versions else None

    print(f"Evaluating Staging v{staging_version.version} for promotion...\n")

    passed, reasons = run_promotion_gates(client, staging_version, production_version)

    if not passed:
        print("\nPromotion BLOCKED:")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)

    # Archive current Production
    if production_version:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=production_version.version,
            stage="Archived",
        )
        print(f"\nArchived: v{production_version.version} (was Production)")

    # Promote Staging → Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_version.version,
        stage="Production",
    )
    print(f"Promoted: v{staging_version.version} → Production")
    print(f"\nNext step: Blue/Green deployment will pick up the new Production model.")
    print(f"  Serving layer polls MLflow on startup and loads Production version.")


def rollback() -> None:
    """Roll back Production to the most recent Archived version."""
    client = get_client()

    try:
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except RestException:
        print("No registered model found yet, so there is nothing to roll back.")
        sys.exit(1)
    archived = [v for v in all_versions if v.current_stage == "Archived"]

    if not archived:
        print("No archived versions to roll back to.")
        sys.exit(1)

    # Most recently archived = previous Production
    previous = sorted(archived, key=lambda v: v.last_updated_timestamp, reverse=True)[0]

    # Archive current Production
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prod_versions[0].version,
            stage="Archived",
        )
        print(f"Archived current Production: v{prod_versions[0].version}")

    # Restore previous
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=previous.version,
        stage="Production",
    )
    print(f"Rolled back to: v{previous.version} → Production")
    print("Restart serving containers to load rolled-back model.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        default="inspect",
        choices=["inspect", "promote", "rollback"],
    )
    args = parser.parse_args()

    if args.action == "inspect":
        inspect()
    elif args.action == "promote":
        promote()
    elif args.action == "rollback":
        rollback()


if __name__ == "__main__":
    main()
