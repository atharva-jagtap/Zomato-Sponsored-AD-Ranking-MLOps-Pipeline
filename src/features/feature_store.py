"""
Feast Feature Store
--------------------
Registers feature views for offline (S3/Parquet) and online (Redis) serving.

This mirrors Zomato's architecture:
  - Offline store: S3 — batch features for training and backfill
  - Online store:  Redis — low-latency feature retrieval during serving (<5ms)

Two feature views:
  1. restaurant_stats_fv — per-restaurant engagement aggregates
  2. user_stats_fv       — per-user behavioral preferences

Usage:
    # Register feature definitions
    python src/features/feature_store.py --action apply

    # Materialize batch features into Redis (online store)
    python src/features/feature_store.py --action materialize

    # Fetch features for a batch of (user, restaurant) pairs
    python src/features/feature_store.py --action fetch-sample

Config:
    Set FEAST_S3_BUCKET and REDIS_HOST in environment or .env
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, Int32

FEATURES_DIR = Path("data/features")
FEAST_REPO = Path("feature_repo")
FEAST_REPO.mkdir(parents=True, exist_ok=True)


# ─── Feature store config ─────────────────────────────────────────────────────

def write_feature_store_config() -> None:
    """
    Write feature_store.yaml for Feast.
    In production this points to S3 offline store + Redis online store.
    Locally uses file-based offline store.
    """
    s3_bucket = os.getenv("FEAST_S3_BUCKET", "")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")

    if s3_bucket:
        offline_config = f"""
  type: s3
  bucket: {s3_bucket}
  path: feast/offline"""
    else:
        offline_config = """
  type: file"""

    config = f"""project: zomato_ads_ranking
provider: local
registry: {FEAST_REPO}/registry.db

offline_store:{offline_config}

online_store:
  type: redis
  connection_string: {redis_host}:{redis_port}

entity_key_serialization_version: 2
"""
    (FEAST_REPO / "feature_store.yaml").write_text(config)
    print(f"Feature store config written: {FEAST_REPO}/feature_store.yaml")


# ─── Entities ─────────────────────────────────────────────────────────────────

restaurant_entity = Entity(
    name="restaurant_id",
    description="Zomato restaurant identifier",
)

user_entity = Entity(
    name="user_id",
    description="Simulated user identifier",
)


# ─── Data sources ─────────────────────────────────────────────────────────────

restaurant_source = FileSource(
    path=str(FEATURES_DIR / "restaurant_stats.parquet"),
    timestamp_field="event_timestamp",
)

user_source = FileSource(
    path=str(FEATURES_DIR / "user_stats.parquet"),
    timestamp_field="event_timestamp",
)


# ─── Feature views ────────────────────────────────────────────────────────────

restaurant_stats_fv = FeatureView(
    name="restaurant_stats",
    entities=[restaurant_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="click_through_rate",      dtype=Float32),
        Field(name="order_rate",               dtype=Float32),
        Field(name="ctr_wilson_lower",         dtype=Float32),
        Field(name="avg_rank_position",        dtype=Float32),
        Field(name="total_impressions",        dtype=Int32),
        Field(name="total_clicks",             dtype=Int32),
        Field(name="total_orders",             dtype=Int32),
        Field(name="view_rate",                dtype=Float32),
        Field(name="view_rate_norm",           dtype=Float32),
        Field(name="click_through_rate_norm",  dtype=Float32),
        Field(name="order_rate_norm",          dtype=Float32),
    ],
    source=restaurant_source,
    description="Per-restaurant engagement aggregates. Updated daily.",
)

user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="total_sessions",          dtype=Int32),
        Field(name="avg_label",               dtype=Float32),
        Field(name="order_rate",              dtype=Float32),
        Field(name="click_rate",              dtype=Float32),
        Field(name="promo_click_rate",        dtype=Float32),
    ],
    source=user_source,
    description="Per-user historical behavior preferences. Updated weekly.",
)


# ─── Actions ─────────────────────────────────────────────────────────────────

def apply_feature_store() -> None:
    """Register all feature views with the Feast registry."""
    write_feature_store_config()
    store = FeatureStore(repo_path=str(FEAST_REPO))
    store.apply([restaurant_entity, user_entity, restaurant_stats_fv, user_stats_fv])
    print("Feature store applied. Registry updated.")
    print(f"  Feature views: {[fv.name for fv in store.list_feature_views()]}")


def materialize_to_online_store() -> None:
    """Push latest batch features into Redis for low-latency online serving."""
    store = FeatureStore(repo_path=str(FEAST_REPO))
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    print(f"Materializing features to online store ({start_date.date()} → {end_date.date()})...")
    store.materialize(start_date=start_date, end_date=end_date)
    print("Materialization complete. Redis now has latest features.")


def fetch_sample_features() -> None:
    """Demo: retrieve features for a batch of (user, restaurant) pairs."""
    store = FeatureStore(repo_path=str(FEAST_REPO))

    # Sample entity keys
    restaurant_ids = pd.read_parquet(FEATURES_DIR / "restaurant_stats.parquet")[
        "restaurant_id"
    ].head(5).tolist()

    entity_rows = [{"restaurant_id": rid} for rid in restaurant_ids]

    features = store.get_online_features(
        features=[
            "restaurant_stats:click_through_rate",
            "restaurant_stats:order_rate",
            "restaurant_stats:ctr_wilson_lower",
        ],
        entity_rows=entity_rows,
    ).to_df()

    print("\nSample online feature fetch:")
    print(features.to_string())


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        default="apply",
        choices=["apply", "materialize", "fetch-sample"],
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "apply":
        apply_feature_store()
        print("\nNext step: python src/features/feature_store.py --action materialize")
    elif args.action == "materialize":
        materialize_to_online_store()
        print("\nNext step: python src/training/train.py")
    elif args.action == "fetch-sample":
        fetch_sample_features()


if __name__ == "__main__":
    main()
