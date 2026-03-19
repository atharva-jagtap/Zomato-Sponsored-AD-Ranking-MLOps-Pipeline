"""
LambdaMART Training — XGBoost + MLflow
----------------------------------------
Trains a Learning to Rank model using XGBoost's lambdarank objective.
This is the same algorithm Zomato uses in production.

Key design decisions:
  - Out-of-time train/test split (days 1–10 train, 11–15 test) — same as Zomato
  - Tracks all params, metrics, and artifacts in MLflow
  - Conditional registration: only promotes to registry if NDCG@5 improves
  - Feature importance plot saved as artifact (useful for interview demo)

Usage:
    python src/training/train.py
    python src/training/train.py --n-estimators 500 --max-depth 8
    python src/training/train.py --run-name "experiment_v2_more_trees"

Metrics logged to MLflow:
  - ndcg_at_5        (primary metric — same as Zomato's Accuracy@N equivalent)
  - mrr              (mean reciprocal rank)
  - mean_ordered_rank
  - ndcg_at_1, ndcg_at_10
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score

from evaluate import compute_mrr, compute_mean_ordered_rank, compute_ndcg

FEATURES_DIR = Path("data/features")
ARTIFACTS_DIR = Path("data/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "zomato-restaurant-ads-ranking"

FEATURE_COLS = [
    "cuisine_match", "price_fit", "rating_score", "promo_flag",
    "meal_time_match", "is_veg_match", "votes_log", "delivery_available",
    "day_of_week", "hour_of_day", "rank_position",
    "click_through_rate", "order_rate", "ctr_wilson_lower", "avg_rank_position",
    "avg_label", "user_order_rate", "user_promo_click_rate",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(FEATURES_DIR / "ltr_train.parquet")
    test  = pd.read_parquet(FEATURES_DIR / "ltr_test.parquet")
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def prepare_dmatrix(df: pd.DataFrame) -> xgb.DMatrix:
    """
    XGBoost LTR requires group sizes (how many items per query/session).
    Each session_id is one query group.
    """
    X = df[FEATURE_COLS].values
    y = df["label"].values

    # Group sizes: number of candidate restaurants per session
    groups = df.groupby("session_id").size().values

    dmat = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
    dmat.set_group(groups)
    return dmat


def plot_feature_importance(model: xgb.Booster, run_id: str) -> Path:
    """Generate and save feature importance plot as MLflow artifact."""
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#534AB7")
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (LambdaMART — Gain)")
    plt.tight_layout()

    path = ARTIFACTS_DIR / f"feature_importance_{run_id[:8]}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def get_production_ndcg() -> float:
    """
    Fetch the NDCG@5 of the current production model from MLflow registry.
    Returns 0.0 if no production model exists yet.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(
            "zomato-ads-ranker", stages=["Production"]
        )
        if not versions:
            return 0.0
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get("ndcg_at_5", 0.0))
    except Exception:
        return 0.0


def train(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    run_name: str | None = None,
) -> None:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, test_df = load_data()
    dtrain = prepare_dmatrix(train_df)
    dtest  = prepare_dmatrix(test_df)

    params = {
        "objective":        "rank:ndcg",   # LambdaMART
        "eval_metric":      ["ndcg@5", "ndcg@1", "ndcg@10"],
        "tree_method":      "hist",
        "max_depth":        max_depth,
        "learning_rate":    learning_rate,
        "min_child_weight": min_child_weight,
        "subsample":        subsample,
        "colsample_bytree": colsample_bytree,
        "seed":             42,
    }

    with mlflow.start_run(run_name=run_name or f"lambdamart_d{max_depth}_n{n_estimators}") as run:
        run_id = run.info.run_id
        print(f"\nMLflow run: {run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "n_estimators":       n_estimators,
            "max_depth":          max_depth,
            "learning_rate":      learning_rate,
            "min_child_weight":   min_child_weight,
            "subsample":          subsample,
            "colsample_bytree":   colsample_bytree,
            "algorithm":          "lambdamart",
            "train_days":         "1-10",
            "test_days":          "11-15",
            "n_train_rows":       len(train_df),
            "n_test_rows":        len(test_df),
            "n_features":         len(FEATURE_COLS),
        })

        # Train with early stopping
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "eval")],
            early_stopping_rounds=30,
            evals_result=evals_result,
            verbose_eval=50,
        )

        # Compute evaluation metrics
        test_scores = model.predict(dtest)
        test_df = test_df.copy()
        test_df["score"] = test_scores

        ndcg_5  = compute_ndcg(test_df, k=5)
        ndcg_1  = compute_ndcg(test_df, k=1)
        ndcg_10 = compute_ndcg(test_df, k=10)
        mrr     = compute_mrr(test_df)
        mor     = compute_mean_ordered_rank(test_df)

        metrics = {
            "ndcg_at_5":        ndcg_5,
            "ndcg_at_1":        ndcg_1,
            "ndcg_at_10":       ndcg_10,
            "mrr":              mrr,
            "mean_ordered_rank": mor,
            "best_iteration":   model.best_iteration,
        }
        mlflow.log_metrics(metrics)

        print(f"\nEvaluation results (out-of-time test, days 11–15):")
        print(f"  NDCG@5  : {ndcg_5:.4f}")
        print(f"  NDCG@1  : {ndcg_1:.4f}")
        print(f"  NDCG@10 : {ndcg_10:.4f}")
        print(f"  MRR     : {mrr:.4f}")
        print(f"  Mean Ordered Rank: {mor:.4f}")

        # Log feature importance plot
        fi_path = plot_feature_importance(model, run_id)
        mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=None,  # register conditionally below
            input_example=train_df[FEATURE_COLS].head(1),
        )

        # Conditional registration — only promote if NDCG@5 improves
        prod_ndcg = get_production_ndcg()
        print(f"\nProduction NDCG@5: {prod_ndcg:.4f}  |  This run: {ndcg_5:.4f}")

        if ndcg_5 > prod_ndcg:
            print(f"  Improvement: +{ndcg_5 - prod_ndcg:.4f} — registering model")
            client = mlflow.tracking.MlflowClient()
            mv = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="zomato-ads-ranker",
            )
            client.transition_model_version_stage(
                name="zomato-ads-ranker",
                version=mv.version,
                stage="Staging",
            )
            print(f"  Registered: zomato-ads-ranker v{mv.version} -> Staging")
            print("  Run: python src/training/register.py --action promote to promote to Production")
        else:
            print(f"  No improvement — model NOT registered")

        print(f"\nMLflow UI: {MLFLOW_TRACKING_URI}/#/experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",    type=int,   default=300)
    parser.add_argument("--max-depth",       type=int,   default=6)
    parser.add_argument("--learning-rate",   type=float, default=0.1)
    parser.add_argument("--min-child-weight",type=int,   default=5)
    parser.add_argument("--subsample",       type=float, default=0.8)
    parser.add_argument("--colsample-bytree",type=float, default=0.8)
    parser.add_argument("--run-name",        type=str,   default=None)
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        run_name=args.run_name,
    )
