"""
BentoML Service — Model Packaging for Production Deployment
-------------------------------------------------------------
BentoML wraps the MLflow XGBoost model into a standardized serving container
that handles: model loading, input validation, batching, and serialization.

Why BentoML on top of FastAPI:
  - FastAPI handles HTTP routing and Prometheus metrics (operational layer)
  - BentoML handles model lifecycle: loading, versioning, adaptive batching
  - Separation of concerns: ML serving logic vs API logic
  - BentoML generates a Docker image with correct model artifact bundled in

In this project:
  - BentoML loads the Production model from MLflow on startup
  - FastAPI (api.py) calls the BentoML runner for inference
  - This means model swaps (new Production version) only require container restart

Build and serve:
    # Save current Production model to BentoML store
    python src/serving/bento_service.py --action save

    # Serve locally for testing
    bentoml serve src/serving/bento_service.py:RankingService --port 8080

    # Build Docker image (pushed to ECR for blue/green deployment)
    bentoml build
    bentoml containerize zomato-ads-ranker:latest \
        --image-tag ACCOUNT.dkr.ecr.REGION.amazonaws.com/zomato-ads-mlops:serving-latest
"""

import argparse
import os

import bentoml
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "zomato-ads-ranker"

FEATURE_COLS = [
    "cuisine_match", "price_fit", "rating_score", "promo_flag",
    "meal_time_match", "is_veg_match", "votes_log", "delivery_available",
    "day_of_week", "hour_of_day", "rank_position",
    "click_through_rate", "order_rate", "ctr_wilson_lower", "avg_rank_position",
    "avg_label", "user_order_rate", "user_promo_click_rate",
]


# ─── Input/Output schemas ─────────────────────────────────────────────────────

class RankInput(BaseModel):
    """Single candidate restaurant feature vector."""
    features: list[float]  # ordered as FEATURE_COLS


class RankBatchInput(BaseModel):
    session_id: str
    candidates: list[RankInput]


class RankBatchOutput(BaseModel):
    session_id: str
    scores: list[float]
    ranked_indices: list[int]  # indices into input candidates, sorted by score desc


# ─── Runner ──────────────────────────────────────────────────────────────────

class XGBoostRankingRunner(bentoml.Runnable):
    """
    BentoML Runnable wraps the XGBoost model.
    Handles batching: multiple /rank requests are batched together for efficiency.
    """
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/Production"
        self._model: xgb.Booster = mlflow.xgboost.load_model(model_uri)
        print(f"BentoML runner loaded: {MODEL_NAME} (Production)")

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for a batch of candidate feature vectors.
        batchable=True: BentoML automatically batches concurrent requests.
        batch_dim=0: stack along first axis.
        """
        dmat = xgb.DMatrix(features, feature_names=FEATURE_COLS)
        return self._model.predict(dmat)


# ─── Service definition ───────────────────────────────────────────────────────

ranking_runner = bentoml.Runner(
    XGBoostRankingRunner,
    name="xgboost_ranking_runner",
    max_batch_size=64,          # batch up to 64 concurrent candidate lists
    max_latency_ms=100,         # wait at most 100ms to fill a batch
)

svc = bentoml.Service("zomato-ads-ranker", runners=[ranking_runner])


@svc.api(input=JSON(pydantic_model=RankBatchInput), output=JSON(pydantic_model=RankBatchOutput))
async def rank(request: RankBatchInput) -> RankBatchOutput:
    """
    Rank a list of candidate restaurants by predicted relevance score.
    """
    feature_matrix = np.array(
        [c.features for c in request.candidates], dtype=np.float32
    )

    scores: np.ndarray = await ranking_runner.predict.async_run(feature_matrix)

    # Sort by score descending
    ranked_indices = np.argsort(scores)[::-1].tolist()

    return RankBatchOutput(
        session_id=request.session_id,
        scores=scores.tolist(),
        ranked_indices=ranked_indices,
    )


# ─── Save model to BentoML store ─────────────────────────────────────────────

def save_model_to_bento_store() -> None:
    """
    Pull the Production model from MLflow and save it into BentoML's model store.
    This is run once after each model promotion before building the serving image.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"

    print(f"Loading Production model from MLflow: {model_uri}")
    xgb_model = mlflow.xgboost.load_model(model_uri)

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    version = versions[0].version if versions else "unknown"

    saved = bentoml.xgboost.save_model(
        "zomato-ads-ranker",
        xgb_model,
        labels={"mlflow_version": str(version), "stage": "production"},
        metadata={"feature_columns": FEATURE_COLS},
    )
    print(f"Saved to BentoML store: {saved}")
    print(f"  To serve: bentoml serve src/serving/bento_service.py:svc --port 8080")
    print(f"  To build image: bentoml build && bentoml containerize zomato-ads-ranker:latest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="save", choices=["save"])
    args = parser.parse_args()

    if args.action == "save":
        save_model_to_bento_store()


if __name__ == "__main__":
    main()
