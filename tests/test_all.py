"""
Unit Tests
-----------
Tests for the three most critical layers:
  1. Feature engineering — correct computation, no leakage
  2. Model evaluation metrics — NDCG, MRR correctness
  3. API schema — correct input/output validation

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short  # CI mode
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Feature tests ────────────────────────────────────────────────────────────

class TestBatchFeatures:

    def _make_sessions(self, n: int = 100) -> pd.DataFrame:
        """Minimal sessions DataFrame for testing."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "session_id":     [f"S{i:04d}" for i in range(n)],
            "user_id":        [f"U{i % 10:04d}" for i in range(n)],
            "restaurant_id":  [f"R{i % 20:04d}" for i in range(n)],
            "label":          rng.integers(0, 4, n),
            "rank_position":  rng.integers(1, 21, n),
            "day_index":      rng.integers(1, 16, n),
            "cuisine_match":  rng.uniform(0, 1, n),
            "price_fit":      rng.uniform(0, 1, n),
        })

    def test_restaurant_stats_shape(self):
        from features.batch_features import compute_restaurant_stats
        sessions = self._make_sessions(200)
        stats = compute_restaurant_stats(sessions)
        assert len(stats) == sessions["restaurant_id"].nunique()
        assert "click_through_rate" in stats.columns
        assert "order_rate" in stats.columns
        assert "ctr_wilson_lower" in stats.columns

    def test_ctr_wilson_lower_bounded(self):
        """Wilson score lower bound must always be <= raw CTR."""
        from features.batch_features import compute_restaurant_stats
        sessions = self._make_sessions(500)
        stats = compute_restaurant_stats(sessions)
        assert (stats["ctr_wilson_lower"] <= stats["click_through_rate"] + 1e-6).all(), \
            "Wilson lower bound must not exceed raw CTR"

    def test_no_data_leakage_in_train_test_split(self):
        """
        Critical: test sessions must not contain any session_ids from train.
        Out-of-time split means day_index 1-10 = train, 11-15 = test.
        """
        from features.batch_features import compute_restaurant_stats, compute_user_stats, build_ltr_dataset
        sessions = self._make_sessions(500)

        restaurant_stats = compute_restaurant_stats(sessions)
        user_stats = compute_user_stats(sessions)
        ltr = build_ltr_dataset(sessions, restaurant_stats, user_stats)

        train = ltr[ltr["day_index"] <= 10]
        test  = ltr[ltr["day_index"] > 10]

        # Session IDs must not overlap between train and test
        # (a session happens on one day only)
        train_sessions = set(train["session_id"])
        test_sessions  = set(test["session_id"])
        overlap = train_sessions & test_sessions
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} sessions appear in both train and test"

    def test_feature_ranges(self):
        """All bounded features must stay within [0, 1]."""
        from features.batch_features import compute_restaurant_stats, compute_user_stats, build_ltr_dataset
        sessions = self._make_sessions(300)
        restaurant_stats = compute_restaurant_stats(sessions)
        user_stats = compute_user_stats(sessions)
        ltr = build_ltr_dataset(sessions, restaurant_stats, user_stats)

        for col in ["cuisine_match", "price_fit", "rating_score"]:
            if col in ltr.columns:
                assert ltr[col].between(0.0, 1.0).all(), f"{col} out of [0, 1] range"


# ─── Evaluation metric tests ──────────────────────────────────────────────────

class TestEvaluationMetrics:

    def _make_predictions(self) -> pd.DataFrame:
        """
        Synthetic session with known perfect and worst rankings.
        Session A: model correctly ranks most relevant item first → high NDCG
        Session B: model ranks worst item first → low NDCG
        """
        return pd.DataFrame({
            "session_id": ["A", "A", "A", "B", "B", "B"],
            "label":      [3,    1,   0,   3,   1,   0],
            "score":      [0.9,  0.5, 0.1, 0.1, 0.5, 0.9],  # A: correct, B: reversed
        })

    def test_ndcg_perfect_ranking(self):
        from training.evaluate import compute_ndcg
        # Session A only: model scores match relevance order → NDCG should be 1.0
        df = pd.DataFrame({
            "session_id": ["A", "A", "A"],
            "label":      [3,    1,   0],
            "score":      [0.9,  0.5, 0.1],
        })
        ndcg = compute_ndcg(df, k=3)
        assert ndcg == pytest.approx(1.0, abs=0.01), f"Perfect ranking should give NDCG≈1.0, got {ndcg}"

    def test_ndcg_worst_ranking(self):
        from training.evaluate import compute_ndcg
        # Model ranks least relevant item first → NDCG should be well below 1.0
        df = pd.DataFrame({
            "session_id": ["B", "B", "B"],
            "label":      [3,    1,   0],
            "score":      [0.1,  0.5, 0.9],  # worst item scored highest
        })
        ndcg = compute_ndcg(df, k=3)
        assert ndcg < 0.8, f"Worst ranking should give NDCG < 0.8, got {ndcg}"

    def test_mrr_first_click(self):
        from training.evaluate import compute_mrr
        # Clicked item is ranked 1st → MRR = 1.0
        df = pd.DataFrame({
            "session_id": ["A", "A", "A"],
            "label":      [2,    0,   0],
            "score":      [0.9,  0.5, 0.1],
        })
        mrr = compute_mrr(df)
        assert mrr == pytest.approx(1.0, abs=0.01)

    def test_mrr_second_click(self):
        from training.evaluate import compute_mrr
        # Clicked item is ranked 2nd → MRR = 0.5
        df = pd.DataFrame({
            "session_id": ["A", "A", "A"],
            "label":      [0,    2,   0],
            "score":      [0.9,  0.5, 0.1],
        })
        mrr = compute_mrr(df)
        assert mrr == pytest.approx(0.5, abs=0.01)

    def test_ndcg_at_k_ignores_beyond_k(self):
        from training.evaluate import compute_ndcg
        # Only top-1 matters for NDCG@1
        df = pd.DataFrame({
            "session_id": ["A", "A", "A"],
            "label":      [3,    0,   3],
            "score":      [0.9,  0.5, 0.1],
        })
        ndcg1 = compute_ndcg(df, k=1)
        assert ndcg1 == pytest.approx(1.0, abs=0.01), "Top-1 is the best item → NDCG@1 should be 1.0"


# ─── API schema tests ─────────────────────────────────────────────────────────

class TestAPISchemas:

    def test_rank_request_valid(self):
        from serving.api import RankRequest, UserContext, CandidateRestaurant
        req = RankRequest(
            session_id="S001",
            user_context=UserContext(
                preferred_cuisine="north_indian",
                budget_segment="mid",
                is_veg_preference=0,
                promo_sensitive=1,
                meal_time="dinner",
                hour_of_day=20,
                day_of_week=5,
            ),
            candidates=[
                CandidateRestaurant(
                    restaurant_id="R001",
                    cuisine_match=0.9,
                    price_fit=0.8,
                    rating_score=0.75,
                    promo_flag=0,
                    meal_time_match=0.9,
                    is_veg_match=1,
                    votes_log=5.2,
                    delivery_available=1,
                    rank_position=1,
                )
            ],
        )
        assert req.session_id == "S001"
        assert len(req.candidates) == 1
        assert req.candidates[0].restaurant_id == "R001"

    def test_candidate_default_learned_features(self):
        """Learned features from feature store should have sensible defaults."""
        from serving.api import CandidateRestaurant
        c = CandidateRestaurant(
            restaurant_id="R001",
            cuisine_match=0.5,
            price_fit=0.5,
            rating_score=0.5,
            promo_flag=0,
            meal_time_match=0.5,
            is_veg_match=0,
            votes_log=3.0,
            delivery_available=1,
            rank_position=5,
        )
        # Defaults should not cause errors
        assert c.click_through_rate == 0.0
        assert c.order_rate == 0.0
        assert c.avg_rank_position == 10.0


# ─── Session simulator tests ──────────────────────────────────────────────────

class TestSessionSimulator:

    def test_relevance_score_bounded(self):
        from ingestion.simulate_sessions import compute_relevance_score
        session_ctx = {
            "preferred_cuisine": "north_indian",
            "budget_segment": "mid",
            "is_veg_preference": 0,
            "promo_sensitive": 1,
            "meal_time": "dinner",
        }
        restaurant = pd.Series({
            "cuisine_group": "north_indian",
            "budget_segment": "mid",
            "rating_score": 0.8,
            "promo_flag": 1,
            "votes_log": 6.0,
            "is_veg": 0,
        })
        score = compute_relevance_score(session_ctx, restaurant)
        assert 0.0 <= score <= 1.0, f"Relevance score out of bounds: {score}"

    def test_perfect_match_higher_than_mismatch(self):
        from ingestion.simulate_sessions import compute_relevance_score
        base_ctx = {
            "preferred_cuisine": "north_indian",
            "budget_segment": "mid",
            "is_veg_preference": 0,
            "promo_sensitive": 0,
            "meal_time": "dinner",
        }
        matching = pd.Series({
            "cuisine_group": "north_indian",
            "budget_segment": "mid",
            "rating_score": 0.8,
            "promo_flag": 0,
            "votes_log": 5.0,
            "is_veg": 0,
        })
        mismatching = pd.Series({
            "cuisine_group": "desserts",
            "budget_segment": "luxury",
            "rating_score": 0.3,
            "promo_flag": 0,
            "votes_log": 1.0,
            "is_veg": 1,
        })
        assert compute_relevance_score(base_ctx, matching) > \
               compute_relevance_score(base_ctx, mismatching), \
               "Matching restaurant should always score higher than mismatch"
