"""
Evaluation Metrics for Learning to Rank
-----------------------------------------
Implements the exact metrics Zomato uses:
  - NDCG@K  — normalized discounted cumulative gain
  - MRR     — mean reciprocal rank
  - Mean Ordered Rank — Zomato's custom metric for top-slot accuracy

All metrics are computed per-session then averaged — this is the correct
approach for LTR evaluation (not flattening all rows into one computation).
"""

import numpy as np
import pandas as pd


def compute_ndcg(df: pd.DataFrame, k: int = 5) -> float:
    """
    Per-session NDCG@K averaged across all test sessions.

    NDCG measures whether highly relevant items appear at the top of the ranking.
    K=5 is the primary metric (5 sponsored slots shown to user).
    """
    scores = []
    for _, session in df.groupby("session_id"):
        session = session.sort_values("score", ascending=False)
        labels = session["label"].values[:k]
        ideal_labels = np.sort(labels)[::-1]

        if ideal_labels.sum() == 0:
            continue  # skip sessions with no positive labels

        dcg  = sum(l / np.log2(i + 2) for i, l in enumerate(labels))
        idcg = sum(l / np.log2(i + 2) for i, l in enumerate(ideal_labels))
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def compute_mrr(df: pd.DataFrame) -> float:
    """
    Mean Reciprocal Rank — what rank did the first relevant item appear at?
    Relevant = label >= 2 (clicked or ordered).
    """
    scores = []
    for _, session in df.groupby("session_id"):
        session = session.sort_values("score", ascending=False).reset_index(drop=True)
        relevant_positions = session[session["label"] >= 2].index.tolist()
        if not relevant_positions:
            continue
        first_relevant_rank = relevant_positions[0] + 1  # 1-indexed
        scores.append(1.0 / first_relevant_rank)
    return float(np.mean(scores)) if scores else 0.0


def compute_mean_ordered_rank(df: pd.DataFrame) -> float:
    """
    Zomato's custom metric: for each session, what is the average rank
    of ordered restaurants in the predicted ranking?

    Lower is better (ordered restaurants should appear near the top).
    """
    scores = []
    for _, session in df.groupby("session_id"):
        session = session.sort_values("score", ascending=False).reset_index(drop=True)
        ordered = session[session["label"] == 3]
        if ordered.empty:
            continue
        avg_rank = float(ordered.index.to_numpy().mean()) + 1  # 1-indexed
        scores.append(avg_rank)
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_model(df: pd.DataFrame) -> dict:
    """Run all metrics and return as a dict for MLflow logging."""
    return {
        "ndcg_at_1":         compute_ndcg(df, k=1),
        "ndcg_at_5":         compute_ndcg(df, k=5),
        "ndcg_at_10":        compute_ndcg(df, k=10),
        "mrr":               compute_mrr(df),
        "mean_ordered_rank": compute_mean_ordered_rank(df),
    }
