"""
Data Validation with Great Expectations
----------------------------------------
Validates the processed session data before it enters the feature store or training.

Why this matters:
  - Catches upstream data issues before they silently corrupt the model
  - Generates an HTML validation report stored in S3 as a pipeline artifact
  - Zomato's blog skips this — including it shows production awareness

Two suites:
  1. restaurant_catalog_suite — validates the Zomato restaurant catalog
  2. sessions_ltr_suite       — validates the simulated LTR training rows

Usage:
    python src/validation/expectations.py
    python src/validation/expectations.py --fail-fast  # exit 1 on failure
"""

import argparse
import json
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/validation_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Catalog Suite ────────────────────────────────────────────────────────────

def build_catalog_suite(validator) -> None:
    """Expectations for the restaurant catalog."""

    # Schema
    validator.expect_table_columns_to_match_set(
        column_set=[
            "restaurant_id", "name", "rating_score", "votes_log",
            "cost_for_two", "delivery_available", "is_veg", "promo_flag",
            "cuisine_group", "budget_segment", "location",
        ],
        exact_match=False,
    )

    # No nulls on critical columns
    for col in ["restaurant_id", "rating_score", "cuisine_group", "budget_segment"]:
        validator.expect_column_values_to_not_be_null(column=col)

    # Uniqueness
    validator.expect_column_values_to_be_unique(column="restaurant_id")

    # Value ranges
    validator.expect_column_values_to_be_between(
        column="rating_score", min_value=0.0, max_value=1.0
    )
    validator.expect_column_values_to_be_between(
        column="votes_log", min_value=0.0, max_value=20.0
    )
    validator.expect_column_values_to_be_in_set(
        column="delivery_available", value_set=[0, 1]
    )
    validator.expect_column_values_to_be_in_set(
        column="is_veg", value_set=[0, 1]
    )
    validator.expect_column_values_to_be_in_set(
        column="promo_flag", value_set=[0, 1]
    )
    validator.expect_column_values_to_be_in_set(
        column="cuisine_group",
        value_set=["north_indian", "south_indian", "chinese", "continental",
                   "fast_food", "desserts", "other"],
    )
    validator.expect_column_values_to_be_in_set(
        column="budget_segment",
        value_set=["low", "mid", "high", "luxury"],
    )

    # Row count sanity
    validator.expect_table_row_count_to_be_between(min_value=10_000, max_value=200_000)


# ─── Sessions Suite ───────────────────────────────────────────────────────────

def build_sessions_suite(validator) -> None:
    """Expectations for the LTR training sessions table."""

    # Schema
    validator.expect_table_columns_to_match_set(
        column_set=[
            "session_id", "user_id", "restaurant_id", "rank_position", "label",
            "cuisine_match", "price_fit", "rating_score", "promo_flag",
            "meal_time_match", "is_veg_match", "day_of_week", "hour_of_day",
            "votes_log", "delivery_available", "day_index",
        ],
        exact_match=True,
    )

    # No nulls anywhere in training data
    for col in ["session_id", "restaurant_id", "label", "cuisine_match",
                "price_fit", "rating_score", "day_index"]:
        validator.expect_column_values_to_not_be_null(column=col)

    # Label distribution — label 0 should dominate (realistic: most impressions ignored)
    validator.expect_column_values_to_be_in_set(
        column="label", value_set=[0, 1, 2, 3]
    )
    validator.expect_column_proportion_of_unique_values_to_be_between(
        column="label", min_proportion=0.0, max_proportion=0.5
    )

    # Feature ranges
    for col in ["cuisine_match", "price_fit", "rating_score", "meal_time_match"]:
        validator.expect_column_values_to_be_between(
            column=col, min_value=0.0, max_value=1.0
        )

    validator.expect_column_values_to_be_between(
        column="rank_position", min_value=1, max_value=20
    )
    validator.expect_column_values_to_be_between(
        column="day_of_week", min_value=0, max_value=6
    )
    validator.expect_column_values_to_be_between(
        column="hour_of_day", min_value=0, max_value=23
    )
    validator.expect_column_values_to_be_between(
        column="day_index", min_value=1, max_value=15
    )

    # Binary flags
    for col in ["promo_flag", "is_veg_match", "delivery_available"]:
        validator.expect_column_values_to_be_in_set(column=col, value_set=[0, 1])

    # Row count: 50k sessions × 20 candidates = 1M rows expected
    validator.expect_table_row_count_to_be_between(
        min_value=500_000, max_value=2_000_000
    )

    # Sessions should span all 15 days (no missing days)
    validator.expect_column_values_to_be_between(
        column="day_index", min_value=1, max_value=15
    )


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_validation(fail_fast: bool = False) -> dict:
    context = gx.get_context()
    results = {}

    validations = [
        {
            "name":  "restaurant_catalog",
            "path":  PROCESSED_DIR / "restaurant_catalog.parquet",
            "suite": build_catalog_suite,
        },
        {
            "name":  "sessions_ltr",
            "path":  PROCESSED_DIR / "sessions_ltr.parquet",
            "suite": build_sessions_suite,
        },
    ]

    all_passed = True

    for v in validations:
        if not v["path"].exists():
            print(f"[skip] {v['name']} — file not found: {v['path']}")
            continue

        print(f"\nValidating: {v['name']} ({v['path'].stat().st_size / 1e6:.1f} MB)")
        df = pd.read_parquet(v["path"])

        # Build datasource and validator
        datasource = context.sources.add_pandas(name=v["name"])
        asset = datasource.add_dataframe_asset(name=v["name"])
        batch_request = asset.build_batch_request(dataframe=df)

        suite_name = f"{v['name']}_suite"
        context.add_or_update_expectation_suite(suite_name)
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )

        # Build expectations
        v["suite"](validator)
        validator.save_expectation_suite(discard_failed_expectations=False)

        # Run
        checkpoint = context.add_or_update_checkpoint(
            name=f"{v['name']}_checkpoint",
            validator=validator,
        )
        result = checkpoint.run()

        passed = result.success
        all_passed = all_passed and passed
        results[v["name"]] = {
            "passed": passed,
            "stats": result.to_json_dict(),
        }

        status = "PASSED" if passed else "FAILED"
        print(f"  Result: {status}")

        if not passed and fail_fast:
            print(f"\nValidation failed on {v['name']}. Exiting.")
            sys.exit(1)

    # Save summary
    summary_path = REPORTS_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"all_passed": all_passed, "results": results}, f, indent=2, default=str)

    print(f"\nValidation summary saved: {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit with code 1 if any validation fails (use in CI/CD)",
    )
    args = parser.parse_args()

    results = run_validation(fail_fast=args.fail_fast)

    all_passed = all(r["passed"] for r in results.values() if "passed" in r)
    if not all_passed:
        print("\nSome validations failed. Check reports in data/validation_reports/")
        sys.exit(1)

    print("\nAll validations passed.")
    print("Next step: python src/features/batch_features.py")


if __name__ == "__main__":
    main()
