"""
tests/test_feature_engineering.py — Unit Tests for Feature Engineering Module
=============================================================================
Tests all 7 engineered features, encoding, and feature matrix preparation.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    BINARY_COLUMNS,
    ONEHOT_COLUMNS,
    TARGET_COL,
    build_preprocessor,
    derive_renewal_feature,
    derive_revenue_features,
    derive_tenure_features,
    derive_usage_features,
    encode_and_scale,
    prepare_feature_matrix,
    run_feature_engineering,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_abt():
    """Create a small ABT mimicking abt_cleaned.csv structure."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "account_id": [f"A-{i:04d}" for i in range(n)],
            "account_name": [f"Company_{i}" for i in range(n)],
            "industry": np.random.choice(["FinTech", "EdTech", "HealthTech"], n),
            "country": np.random.choice(["US", "UK", "DE", "IN"], n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="7D").astype(str),
            "referral_source": np.random.choice(["organic", "paid", "partner"], n),
            "plan_tier": np.random.choice(["Basic", "Pro", "Enterprise"], n),
            "seats": np.random.randint(1, 50, n).astype(float),
            "is_trial": np.random.choice([True, False], n),
            "churn_flag": np.random.choice([0, 1], n, p=[0.78, 0.22]),
            "subscription_id": [f"S-{i:04d}" for i in range(n)],
            "start_date": pd.date_range("2023-06-01", periods=n, freq="5D").astype(str),
            "sub_plan_tier": np.random.choice(["Basic", "Pro"], n),
            "sub_seats": np.random.randint(1, 50, n).astype(float),
            "mrr_amount": np.random.uniform(100, 2000, n).round(2),
            "arr_amount": np.random.uniform(1200, 24000, n).round(2),
            "sub_is_trial": np.random.choice([True, False], n),
            "upgrade_flag": np.random.choice([True, False], n),
            "downgrade_flag": np.random.choice([True, False], n),
            "sub_churn_flag": np.random.choice([True, False], n),
            "billing_frequency": np.random.choice(["monthly", "annual"], n),
            "auto_renew_flag": np.random.choice([True, False], n),
            "total_events": np.random.randint(50, 500, n).astype(float),
            "avg_daily_usage": np.random.uniform(1, 20, n).round(2),
            "total_duration_secs": np.random.randint(1000, 200000, n).astype(float),
            "avg_duration_secs": np.random.uniform(100, 5000, n).round(2),
            "total_errors": np.random.randint(0, 50, n).astype(float),
            "avg_errors": np.random.uniform(0, 2, n).round(4),
            "active_days": np.random.randint(5, 60, n).astype(float),
            "last_activity_date": pd.date_range("2024-11-01", periods=n, freq="1D").astype(str),
            "first_activity_date": pd.date_range("2023-06-01", periods=n, freq="5D").astype(str),
            "unique_features_used": np.random.randint(1, 30, n),
            "beta_feature_usage": np.random.randint(0, 10, n),
            "ticket_count": np.random.randint(0, 10, n).astype(float),
            "avg_satisfaction": np.random.uniform(1, 5, n).round(1),
            "min_satisfaction": np.random.uniform(1, 3, n).round(1),
            "avg_resolution_hours": np.random.uniform(5, 72, n).round(1),
            "avg_first_response_min": np.random.uniform(10, 120, n).round(1),
            "escalation_count": np.random.randint(0, 5, n).astype(float),
            "last_ticket_date": pd.date_range("2024-10-01", periods=n, freq="1D").astype(str),
        }
    )


# ---------------------------------------------------------------------------
# Test: Tenure Features
# ---------------------------------------------------------------------------
class TestTenureFeatures:
    def test_tenure_months_created(self, sample_abt):
        result = derive_tenure_features(sample_abt)
        assert "tenure_months" in result.columns
        assert result["tenure_months"].dtype == float
        assert (result["tenure_months"] >= 0).all()

    def test_tenure_bucket_has_4_categories(self, sample_abt):
        result = derive_tenure_features(sample_abt)
        assert "tenure_bucket" in result.columns
        # Bins: 0-3m, 3-6m, 6-12m, 12+m
        expected_cats = {"0-3m", "3-6m", "6-12m", "12+m"}
        actual_cats = set(result["tenure_bucket"].cat.categories)
        assert actual_cats == expected_cats


# ---------------------------------------------------------------------------
# Test: Usage Features
# ---------------------------------------------------------------------------
class TestUsageFeatures:
    def test_usage_intensity_created(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        result = derive_usage_features(df)
        assert "usage_intensity" in result.columns
        assert result["usage_intensity"].dtype == float

    def test_active_day_ratio_bounded(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        result = derive_usage_features(df)
        assert "active_day_ratio" in result.columns
        assert (result["active_day_ratio"] >= 0).all()
        assert (result["active_day_ratio"] <= 1).all()

    def test_days_since_last_activity_created(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        result = derive_usage_features(df)
        assert "days_since_last_activity" in result.columns
        assert (result["days_since_last_activity"] >= 0).all()


# ---------------------------------------------------------------------------
# Test: Revenue Features
# ---------------------------------------------------------------------------
class TestRevenueFeatures:
    def test_mrr_segment_has_3_categories(self, sample_abt):
        result = derive_revenue_features(sample_abt)
        assert "mrr_segment" in result.columns
        assert set(result["mrr_segment"].dropna().unique()) == {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Test: Renewal Feature
# ---------------------------------------------------------------------------
class TestRenewalFeature:
    def test_near_renewal_is_binary(self, sample_abt):
        result = derive_renewal_feature(sample_abt)
        assert "near_renewal" in result.columns
        assert set(result["near_renewal"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test: Feature Matrix
# ---------------------------------------------------------------------------
class TestFeatureMatrix:
    def test_excludes_id_columns(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, y = prepare_feature_matrix(df)
        assert "account_id" not in X.columns
        assert "account_name" not in X.columns
        assert "subscription_id" not in X.columns

    def test_excludes_date_columns(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, y = prepare_feature_matrix(df)
        date_cols = [c for c in X.columns if "date" in c.lower()]
        assert len(date_cols) == 0

    def test_excludes_leakage_columns(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, y = prepare_feature_matrix(df)
        assert "sub_churn_flag" not in X.columns
        assert "sub_is_trial" not in X.columns

    def test_target_is_correct(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, y = prepare_feature_matrix(df)
        assert y.name == "churn_flag" or isinstance(y, pd.Series)
        assert set(y.unique()).issubset({0, 1})
        assert len(y) == len(X)

    def test_drops_arr_amount(self, sample_abt):
        """arr_amount is perfectly correlated with mrr_amount; must be dropped."""
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, y = prepare_feature_matrix(df)
        assert "arr_amount" not in X.columns


# ---------------------------------------------------------------------------
# Test: Preprocessor (ColumnTransformer)
# ---------------------------------------------------------------------------
class TestPreprocessor:
    def test_build_preprocessor_returns_tuple(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, _ = prepare_feature_matrix(df)
        preprocessor, num_cols, ohe_cols, bin_cols = build_preprocessor(X)
        assert len(num_cols) > 0
        assert len(ohe_cols) > 0
        assert len(bin_cols) > 0

    def test_transform_produces_all_numeric(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, _ = prepare_feature_matrix(df)

        # Ensure types for preprocessor
        for col in BINARY_COLUMNS:
            if col in X.columns:
                X[col] = X[col].astype(int)
        for col in ONEHOT_COLUMNS:
            if col in X.columns:
                X[col] = X[col].astype(str)

        preprocessor, _, _, _ = build_preprocessor(X)
        X_transformed = encode_and_scale(X, preprocessor, fit=True)
        assert X_transformed.select_dtypes(exclude=["number"]).shape[1] == 0
        assert X_transformed.isnull().sum().sum() == 0

    def test_feature_count_after_transform(self, sample_abt):
        df = derive_tenure_features(sample_abt)
        df = derive_usage_features(df)
        df = derive_revenue_features(df)
        df = derive_renewal_feature(df)
        X, _ = prepare_feature_matrix(df)

        for col in BINARY_COLUMNS:
            if col in X.columns:
                X[col] = X[col].astype(int)
        for col in ONEHOT_COLUMNS:
            if col in X.columns:
                X[col] = X[col].astype(str)

        preprocessor, _, _, _ = build_preprocessor(X)
        X_transformed = encode_and_scale(X, preprocessor, fit=True)
        # Should have more columns than input (due to one-hot expansion)
        assert X_transformed.shape[1] > X.select_dtypes(include=["number"]).shape[1]


# ---------------------------------------------------------------------------
# Test: Full Pipeline
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_run_returns_correct_types(self, sample_abt):
        X, y, preprocessor = run_feature_engineering(sample_abt, output_path=None)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert preprocessor is not None

    def test_run_preserves_row_count(self, sample_abt):
        X, y, _ = run_feature_engineering(sample_abt, output_path=None)
        assert len(X) == len(sample_abt)
        assert len(y) == len(sample_abt)

    def test_all_engineered_features_present(self, sample_abt):
        X, _, _ = run_feature_engineering(sample_abt, output_path=None)
        expected = [
            "tenure_months",
            "tenure_bucket",
            "usage_intensity",
            "active_day_ratio",
            "days_since_last_activity",
            "mrr_segment",
            "near_renewal",
        ]
        for feat in expected:
            assert feat in X.columns, f"Missing engineered feature: {feat}"

    def test_saves_csv(self, sample_abt, tmp_path):
        output_path = str(tmp_path / "test_features.csv")
        X, y, _ = run_feature_engineering(sample_abt, output_path=output_path)
        loaded = pd.read_csv(output_path)
        assert TARGET_COL in loaded.columns
        assert len(loaded) == len(sample_abt)
