"""
FastAPI Ranking Inference API
------------------------------
Serves the trained LambdaMART model as a REST endpoint.
Deployed inside an EC2 VPC behind an ALB.

Endpoints:
  POST /rank          — rank a list of candidate restaurants for a user session
  GET  /health        — liveness check for ALB target group
  GET  /ready         — readiness check (model loaded)
  GET  /metrics       — Prometheus metrics scrape endpoint

Request/Response example:
  POST /rank
  {
    "session_id": "S00012345",
    "user_context": {
      "preferred_cuisine": "north_indian",
      "budget_segment": "mid",
      "is_veg_preference": 0,
      "promo_sensitive": 1,
      "meal_time": "dinner",
      "hour_of_day": 20,
      "day_of_week": 5
    },
    "candidates": [
      {
        "restaurant_id": "000123",
        "cuisine_match": 0.9,
        "price_fit": 0.8,
        "rating_score": 0.75,
        ...
      },
      ...
    ]
  }

  Response:
  {
    "session_id": "S00012345",
    "ranked": [
      {"restaurant_id": "000456", "score": 0.823, "rank": 1},
      ...
    ],
    "latency_ms": 4.2
  }
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

# ─── Config ───────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "zomato-ads-ranker")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

FEATURE_COLS = [
    "cuisine_match", "price_fit", "rating_score", "promo_flag",
    "meal_time_match", "is_veg_match", "votes_log", "delivery_available",
    "day_of_week", "hour_of_day", "rank_position",
    "click_through_rate", "order_rate", "ctr_wilson_lower", "avg_rank_position",
    "avg_label", "user_order_rate", "user_promo_click_rate",
]

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "ranking_requests_total",
    "Total ranking requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "ranking_request_latency_ms",
    "Ranking request latency in milliseconds",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500],
)
CANDIDATES_PER_REQUEST = Histogram(
    "ranking_candidates_per_request",
    "Number of candidate restaurants per ranking request",
    buckets=[5, 10, 15, 20, 30, 50],
)
MODEL_SCORE_DISTRIBUTION = Histogram(
    "ranking_model_score",
    "Distribution of model output scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
MODEL_VERSION_GAUGE = Gauge(
    "ranking_model_version",
    "Currently loaded model version",
)


# ─── Model loader ─────────────────────────────────────────────────────────────

class ModelState:
    model: xgb.Booster | None = None
    model_version: str = "unknown"
    ready: bool = False


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading model: {model_uri}")
        state.model = mlflow.xgboost.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            state.model_version = versions[0].version
            MODEL_VERSION_GAUGE.set(float(state.model_version))

        state.ready = True
        print(f"Model loaded: {MODEL_NAME} v{state.model_version} ({MODEL_STAGE})")
    except Exception as e:
        print(f"Failed to load model: {e}")
        state.ready = False

    yield

    state.ready = False
    print("Shutting down ranking service.")


app = FastAPI(
    title="Zomato Restaurant Ads Ranking API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class UserContext(BaseModel):
    preferred_cuisine: str
    budget_segment: str
    is_veg_preference: int
    promo_sensitive: int
    meal_time: str
    hour_of_day: int
    day_of_week: int


class CandidateRestaurant(BaseModel):
    restaurant_id: str
    cuisine_match: float
    price_fit: float
    rating_score: float
    promo_flag: int
    meal_time_match: float
    is_veg_match: int
    votes_log: float
    delivery_available: int
    rank_position: int
    # Learned features (from feature store)
    click_through_rate: float = 0.0
    order_rate: float = 0.0
    ctr_wilson_lower: float = 0.0
    avg_rank_position: float = 10.0
    avg_label: float = 0.0
    user_order_rate: float = 0.0
    user_promo_click_rate: float = 0.0


class RankRequest(BaseModel):
    session_id: str
    user_context: UserContext
    candidates: list[CandidateRestaurant]


class RankedRestaurant(BaseModel):
    restaurant_id: str
    score: float
    rank: int


class RankResponse(BaseModel):
    session_id: str
    ranked: list[RankedRestaurant]
    latency_ms: float
    model_version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """ALB liveness check — always returns 200 if process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """ALB readiness check — returns 503 if model not loaded yet."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": state.model_version}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/rank", response_model=RankResponse)
async def rank(request: RankRequest) -> RankResponse:
    if not state.ready:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not ready")

    start = time.perf_counter()

    try:
        # Build feature matrix — one row per candidate
        feature_rows = []
        for c in request.candidates:
            row_data = c.model_dump()
            row_data["day_of_week"] = request.user_context.day_of_week
            row_data["hour_of_day"] = request.user_context.hour_of_day
            row = [row_data[col] for col in FEATURE_COLS]
            feature_rows.append(row)

        X = np.array(feature_rows, dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_COLS)

        # Predict relevance scores
        scores = state.model.predict(dmat)

        # Build ranked output
        ranked = sorted(
            zip([c.restaurant_id for c in request.candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked_response = [
            RankedRestaurant(restaurant_id=rid, score=float(score), rank=i + 1)
            for i, (rid, score) in enumerate(ranked)
        ]

        latency_ms = (time.perf_counter() - start) * 1000

        # Update Prometheus metrics
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency_ms)
        CANDIDATES_PER_REQUEST.observe(len(request.candidates))
        for _, score in ranked:
            MODEL_SCORE_DISTRIBUTION.observe(float(score))

        return RankResponse(
            session_id=request.session_id,
            ranked=ranked_response,
            latency_ms=round(latency_ms, 2),
            model_version=state.model_version,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
