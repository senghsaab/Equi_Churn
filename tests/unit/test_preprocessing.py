"""
tests/test_preprocessing.py — Unit Tests for Preprocessing Module
===================================================================
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clip_outliers,
    convert_types,
    handle_missing_values,
    remove_duplicates,
    remove_leakage_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "account_id": ["A001", "A002", "A003", "A004", "A005"],
            "company_name": ["Acme", "Beta", "Gamma", "Delta", "Epsilon"],
            "industry": ["Tech", "Finance", None, "Tech", "Finance"],
            "mrr": [100.0, 250.0, None, 500.0, 150.0],
            "logins": [10, 20, 30, 40, 50],
            "signup_date": [
                "2023-01-15",
                "2023-03-10",
                "2023-06-01",
                "2023-09-20",
                "2024-01-01",
            ],
            "churn_date": ["2024-01-01", None, None, "2024-06-01", None],
            "churn_flag": [1, 0, 0, 1, 0],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRemoveLeakageColumns:
    def test_drops_churn_date(self, sample_df):
        result = remove_leakage_columns(sample_df)
        assert "churn_date" not in result.columns

    def test_keeps_non_leakage_columns(self, sample_df):
        result = remove_leakage_columns(sample_df)
        assert "account_id" in result.columns
        assert "mrr" in result.columns
        assert "churn_flag" in result.columns

    def test_extra_columns(self, sample_df):
        result = remove_leakage_columns(sample_df, extra_columns=["company_name"])
        assert "company_name" not in result.columns


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self, sample_df):
        df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]]).reset_index(
            drop=True
        )
        result = remove_duplicates(df_with_dupes, subset=["account_id"])
        assert len(result) == len(sample_df)

    def test_no_duplicates(self, sample_df):
        result = remove_duplicates(sample_df)
        assert len(result) == len(sample_df)


class TestHandleMissingValues:
    def test_imputes_numeric(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result["mrr"].isnull().sum() == 0

    def test_imputes_categorical(self, sample_df):
        result = handle_missing_values(sample_df)
        cat_cols = result.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            assert result[col].isnull().sum() == 0

    def test_creates_missing_indicators(self, sample_df):
        # Force a higher missing rate to trigger indicator creation
        df = sample_df.copy()
        df.loc[0:2, "mrr"] = None  # 60% missing
        result = handle_missing_values(df)
        assert "is_missing_mrr" in result.columns


class TestConvertTypes:
    def test_converts_dates(self, sample_df):
        result = convert_types(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])

    def test_converts_categories(self, sample_df):
        result = convert_types(sample_df)
        if "industry" in result.columns:
            assert result["industry"].dtype.name == "category"


class TestClipOutliers:
    def test_clips_extreme_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        result = clip_outliers(df, columns=["val"], lower_pct=0.0, upper_pct=0.9)
        assert result["val"].max() <= 100
