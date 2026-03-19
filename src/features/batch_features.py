"""
Batch Feature Engineering
--------------------------
Computes offline features from the processed sessions table.
These are stored in S3 and registered in the Feast feature store.

Feature groups:
  1. restaurant_stats     — per-restaurant aggregated signals (rating, CTR, order rate)
  2. user_preferences     — per-user historical preference signals
  3. session_features     — per (session, restaurant) interaction features
  4. ltr_training_dataset — final joined table ready for LambdaMART training

Why batch features vs raw columns:
  - Raw columns (rating_score, cuisine_match) are static catalog attributes
  - Batch features are DERIVED from interaction history:
    e.g. restaurant_ctr_7d = clicks / impressions over last 7 days
    These time-windowed aggregates are what a feature store is actually for

Usage:
    python src/features/batch_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ─── Restaurant-level features ────────────────────────────────────────────────

def compute_restaurant_stats(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate restaurant performance signals from interaction history.
    These are the 'learned' features that improve over time with more data.
    """
    grp = sessions.groupby("restaurant_id")

    stats = pd.DataFrame({
        "restaurant_id":          grp["restaurant_id"].first(),
        # Impression and engagement counts
        "total_impressions":      grp["label"].count(),
        "total_views":            grp["label"].apply(lambda x: (x >= 1).sum()),
        "total_clicks":           grp["label"].apply(lambda x: (x >= 2).sum()),
        "total_orders":           grp["label"].apply(lambda x: (x == 3).sum()),
        # Rate signals (key ranking features)
        "view_rate":              grp["label"].apply(lambda x: (x >= 1).mean()),
        "click_through_rate":     grp["label"].apply(lambda x: (x >= 2).mean()),
        "order_rate":             grp["label"].apply(lambda x: (x == 3).mean()),
        # Position-adjusted CTR (Wilson score lower bound for robustness)
        "avg_rank_position":      grp["rank_position"].mean(),
        # Recency signal (last seen day index)
        "last_seen_day":          grp["day_index"].max(),
        "first_seen_day":         grp["day_index"].min(),
    }).reset_index(drop=True)

    # Wilson score lower bound for CTR (more reliable for low-impression restaurants)
    n = stats["total_impressions"]
    p = stats["click_through_rate"]
    z = 1.96  # 95% confidence
    stats["ctr_wilson_lower"] = (
        (p + z**2 / (2 * n) - z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n))
        / (1 + z**2 / n)
    ).fillna(0.0)

    # Normalize rates to [0, 1] for model consumption
    for col in ["view_rate", "click_through_rate", "order_rate", "ctr_wilson_lower"]:
        stats[f"{col}_norm"] = (
            (stats[col] - stats[col].min()) /
            (stats[col].max() - stats[col].min() + 1e-8)
        )

    # Timestamp for Feast feature store point-in-time correctness
    stats["event_timestamp"] = pd.Timestamp.now()

    return stats


# ─── User-level features ──────────────────────────────────────────────────────

def compute_user_stats(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate user behavior signals.
    These inform personalization: some users click fast food, others click continental.
    """
    grp = sessions.groupby("user_id")

    stats = pd.DataFrame({
        "user_id":                   grp["user_id"].first(),
        "total_sessions":            grp["session_id"].nunique(),
        "avg_label":                 grp["label"].mean(),
        "order_rate":                grp["label"].apply(lambda x: (x == 3).mean()),
        "click_rate":                grp["label"].apply(lambda x: (x >= 2).mean()),
        # Preference signals (what this user tends to engage with)
        "avg_cuisine_match_on_click":
            sessions[sessions["label"] >= 2].groupby("user_id")["cuisine_match"].mean(),
        "avg_price_fit_on_click":
            sessions[sessions["label"] >= 2].groupby("user_id")["price_fit"].mean(),
        "promo_click_rate":
            sessions[sessions["promo_flag"] == 1].groupby("user_id")["label"]
            .apply(lambda x: (x >= 2).mean()),
    }).reset_index(drop=True)

    stats = stats.fillna(0.0)
    stats["event_timestamp"] = pd.Timestamp.now()

    return stats


# ─── Final LTR dataset ────────────────────────────────────────────────────────

def build_ltr_dataset(
    sessions: pd.DataFrame,
    restaurant_stats: pd.DataFrame,
    user_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all features into the final LTR training table.
    One row = one (session, candidate_restaurant) pair with label + all features.
    """
    df = sessions.copy()

    # Join restaurant stats
    df = df.merge(
        restaurant_stats[[
            "restaurant_id", "click_through_rate", "order_rate",
            "ctr_wilson_lower", "avg_rank_position",
        ]],
        on="restaurant_id",
        how="left",
    )

    # Join user stats
    df = df.merge(
        user_stats[[
            "user_id", "avg_label", "order_rate", "promo_click_rate",
        ]].rename(columns={
            "order_rate":       "user_order_rate",
            "promo_click_rate": "user_promo_click_rate",
        }),
        on="user_id",
        how="left",
    )

    df = df.fillna(0.0)

    # Final feature columns in the order the model expects
    FEATURE_COLS = [
        # Static catalog features
        "cuisine_match",
        "price_fit",
        "rating_score",
        "promo_flag",
        "meal_time_match",
        "is_veg_match",
        "votes_log",
        "delivery_available",
        # Context features
        "day_of_week",
        "hour_of_day",
        "rank_position",
        # Learned restaurant features (batch aggregates)
        "click_through_rate",
        "order_rate",
        "ctr_wilson_lower",
        "avg_rank_position",
        # Learned user features
        "avg_label",
        "user_order_rate",
        "user_promo_click_rate",
    ]

    return df[["session_id", "restaurant_id", "label", "day_index"] + FEATURE_COLS]


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading processed sessions...")
    sessions = pd.read_parquet(PROCESSED_DIR / "sessions_ltr.parquet")
    print(f"  {len(sessions):,} rows, {sessions['session_id'].nunique():,} sessions")

    print("Computing restaurant stats...")
    restaurant_stats = compute_restaurant_stats(sessions)
    restaurant_stats.to_parquet(FEATURES_DIR / "restaurant_stats.parquet", index=False)
    print(f"  {len(restaurant_stats):,} restaurants")

    print("Computing user stats...")
    user_stats = compute_user_stats(sessions)
    user_stats.to_parquet(FEATURES_DIR / "user_stats.parquet", index=False)
    print(f"  {len(user_stats):,} users")

    print("Building final LTR dataset...")
    ltr = build_ltr_dataset(sessions, restaurant_stats, user_stats)

    # Out-of-time train/test split (Zomato's exact approach)
    train = ltr[ltr["day_index"] <= 10]
    test  = ltr[ltr["day_index"] > 10]

    train.to_parquet(FEATURES_DIR / "ltr_train.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "ltr_test.parquet",  index=False)

    print(f"\nSplit (out-of-time, same as Zomato):")
    print(f"  Train (days  1–10): {len(train):,} rows")
    print(f"  Test  (days 11–15): {len(test):,} rows")

    print(f"\nFeature columns: {ltr.columns.tolist()}")
    print("\nNext step: python src/features/feature_store.py")


if __name__ == "__main__":
    main()
