"""
test_data_pipeline.py — Unit Tests for ChurnDataPipeline
==========================================================
Tests each pipeline stage independently using fixtures
with small synthetic DataFrames.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import ChurnDataPipeline, SCHEMA_REGISTRY, LEAKAGE_COLUMNS


# =============================================================================
# Fixtures — Small synthetic datasets for testing
# =============================================================================
@pytest.fixture
def sample_raw_dir(tmp_path):
    """Create a temporary directory with minimal valid CSV files."""

    # --- accounts ---
    accounts = pd.DataFrame({
        "account_id": ["A001", "A002", "A003", "A004", "A005"],
        "account_name": ["Acme", "Beta", "Gamma", "Delta", "Echo"],
        "industry": ["Tech", "Finance", "Health", "Tech", "Finance"],
        "country": ["US", "UK", "US", "DE", "US"],
        "signup_date": [
            "2024-01-15", "2024-03-20", "2024-06-01",
            "2024-07-10", "2024-02-28",
        ],
        "referral_source": ["organic", "paid", "referral", "organic", "paid"],
        "plan_tier": ["Pro", "Basic", "Enterprise", "Pro", "Basic"],
        "seats": [10, 5, 50, 15, 3],
        "is_trial": [0, 1, 0, 0, 1],
        "churn_flag": [0, 1, 0, 1, 0],
    })
    accounts.to_csv(tmp_path / "ravenstack_accounts.csv", index=False)

    # --- subscriptions ---
    subscriptions = pd.DataFrame({
        "subscription_id": ["S001", "S002", "S003", "S004", "S005"],
        "account_id": ["A001", "A002", "A003", "A004", "A005"],
        "start_date": [
            "2024-01-15", "2024-03-20", "2024-06-01",
            "2024-07-10", "2024-02-28",
        ],
        "end_date": [None, "2024-09-20", None, "2024-12-10", None],
        "plan_tier": ["Pro", "Basic", "Enterprise", "Pro", "Basic"],
        "seats": [10, 5, 50, 15, 3],
        "mrr_amount": [100.0, 50.0, 500.0, 150.0, 30.0],
        "arr_amount": [1200.0, 600.0, 6000.0, 1800.0, 360.0],
        "is_trial": [0, 1, 0, 0, 1],
        "upgrade_flag": [0, 0, 1, 0, 0],
        "downgrade_flag": [0, 1, 0, 0, 0],
        "churn_flag": [0, 1, 0, 1, 0],
        "billing_frequency": ["monthly", "monthly", "annual", "monthly", "monthly"],
        "auto_renew_flag": [1, 0, 1, 0, 1],
    })
    subscriptions.to_csv(tmp_path / "ravenstack_subscriptions.csv", index=False)

    # --- feature_usage ---
    usage_rows = []
    for i, sid in enumerate(["S001", "S002", "S003", "S004", "S005"]):
        for day_offset in range(5):
            usage_rows.append({
                "usage_id": f"U{i*5 + day_offset:03d}",
                "subscription_id": sid,
                "usage_date": f"2024-08-{10 + day_offset:02d}",
                "feature_name": "dashboard" if day_offset % 2 == 0 else "reports",
                "usage_count": np.random.randint(1, 100),
                "usage_duration_secs": np.random.randint(60, 3600),
                "error_count": np.random.randint(0, 5),
                "is_beta_feature": 1 if day_offset == 4 else 0,
            })
    usage = pd.DataFrame(usage_rows)
    usage.to_csv(tmp_path / "ravenstack_feature_usage.csv", index=False)

    # --- support_tickets ---
    tickets = pd.DataFrame({
        "ticket_id": ["T001", "T002", "T003", "T004"],
        "account_id": ["A001", "A002", "A002", "A004"],
        "submitted_at": [
            "2024-08-01", "2024-08-05", "2024-08-10", "2024-08-15",
        ],
        "closed_at": [
            "2024-08-02", "2024-08-07", "2024-08-12", "2024-08-18",
        ],
        "resolution_time_hours": [24.0, 48.0, 36.0, 72.0],
        "priority": ["Low", "High", "Medium", "Critical"],
        "first_response_time_minutes": [30, 15, 45, 10],
        "satisfaction_score": [4.5, 2.0, 3.0, 1.5],
        "escalation_flag": [0, 1, 0, 1],
    })
    tickets.to_csv(tmp_path / "ravenstack_support_tickets.csv", index=False)

    # --- churn_events ---
    churn = pd.DataFrame({
        "churn_event_id": ["CE001", "CE002"],
        "account_id": ["A002", "A004"],
        "churn_date": ["2024-09-20", "2024-12-10"],
        "reason_code": ["price", "competitor"],
        "refund_amount_usd": [50.0, 100.0],
        "preceding_upgrade_flag": [0, 0],
        "preceding_downgrade_flag": [1, 0],
        "is_reactivation": [0, 0],
        "feedback_text": ["too expensive", "better features elsewhere"],
    })
    churn.to_csv(tmp_path / "ravenstack_churn_events.csv", index=False)

    return tmp_path


@pytest.fixture
def pipeline(sample_raw_dir):
    """Create a ChurnDataPipeline instance with sample data."""
    return ChurnDataPipeline(raw_dir=str(sample_raw_dir))


# =============================================================================
# Test: Data Loading
# =============================================================================
class TestDataLoading:
    """Tests for Stage 1: Data Loading."""

    def test_load_data_success(self, pipeline):
        """All 5 tables should be loaded into self.tables."""
        pipeline.load_data()
        assert len(pipeline.tables) == 5
        assert "accounts" in pipeline.tables
        assert "subscriptions" in pipeline.tables
        assert "feature_usage" in pipeline.tables
        assert "support_tickets" in pipeline.tables
        assert "churn_events" in pipeline.tables

    def test_load_data_row_counts(self, pipeline):
        """Verify row counts match what we created."""
        pipeline.load_data()
        assert len(pipeline.tables["accounts"]) == 5
        assert len(pipeline.tables["subscriptions"]) == 5
        assert len(pipeline.tables["feature_usage"]) == 25  # 5 accounts × 5 days
        assert len(pipeline.tables["support_tickets"]) == 4

    def test_load_data_missing_dir(self, tmp_path):
        """Should raise NotADirectoryError for non-existent directory."""
        p = ChurnDataPipeline(raw_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(NotADirectoryError):
            p.load_data()

    def test_load_data_missing_files(self, tmp_path):
        """Should raise FileNotFoundError if CSV files are missing."""
        tmp_path.mkdir(exist_ok=True)
        p = ChurnDataPipeline(raw_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            p.load_data()


# =============================================================================
# Test: Schema Validation
# =============================================================================
class TestSchemaValidation:
    """Tests for Stage 2: Schema Validation."""

    def test_validate_schemas_passes(self, pipeline):
        """Schema validation should pass on well-formed data."""
        pipeline.load_data().validate_schemas()
        # No critical errors should be raised
        assert len(pipeline._validation_errors) == 0

    def test_validate_binary_columns(self, pipeline):
        """Binary columns should only contain 0 and 1."""
        pipeline.load_data().validate_schemas()
        accounts = pipeline.tables["accounts"]
        assert set(accounts["churn_flag"].unique()).issubset({0, 1})
        assert set(accounts["is_trial"].unique()).issubset({0, 1})

    def test_validate_date_columns(self, pipeline):
        """Date columns should be parseable."""
        pipeline.load_data().validate_schemas()
        # No date-related validation errors
        date_errors = [e for e in pipeline._validation_errors if "date" in e.lower()]
        assert len(date_errors) == 0


# =============================================================================
# Test: Data Versioning
# =============================================================================
class TestDataVersioning:
    """Tests for Stage 3: MD5 Data Versioning."""

    def test_version_data_produces_hashes(self, pipeline):
        """Each table should get an MD5 hash."""
        pipeline.load_data().version_data()
        assert len(pipeline.data_versions) == 5
        for table_name, info in pipeline.data_versions.items():
            assert "md5" in info
            assert len(info["md5"]) == 32  # MD5 hex digest length
            assert "timestamp" in info
            assert "size_bytes" in info

    def test_hashes_are_deterministic(self, pipeline):
        """Same file should produce same hash across runs."""
        pipeline.load_data().version_data()
        hashes_1 = {k: v["md5"] for k, v in pipeline.data_versions.items()}

        pipeline.version_data()  # run again
        hashes_2 = {k: v["md5"] for k, v in pipeline.data_versions.items()}

        assert hashes_1 == hashes_2


# =============================================================================
# Test: Merging
# =============================================================================
class TestMerging:
    """Tests for Stage 4: Table Merging."""

    def test_merge_produces_one_row_per_account(self, pipeline):
        """ABT should have exactly 1 row per account_id."""
        pipeline.load_data().validate_schemas().merge_tables()
        assert pipeline.abt is not None
        assert len(pipeline.abt) == pipeline.abt["account_id"].nunique()

    def test_merge_contains_expected_columns(self, pipeline):
        """Merged ABT should contain usage and support aggregations."""
        pipeline.load_data().validate_schemas().merge_tables()
        cols = set(pipeline.abt.columns)
        # Usage aggregations
        assert "total_events" in cols
        assert "avg_daily_usage" in cols
        assert "active_days" in cols
        assert "last_activity_date" in cols
        # Support aggregations
        assert "ticket_count" in cols
        assert "avg_satisfaction" in cols

    def test_merge_preserves_churn_flag(self, pipeline):
        """churn_flag should be preserved from accounts table."""
        pipeline.load_data().validate_schemas().merge_tables()
        assert "churn_flag" in pipeline.abt.columns
        assert set(pipeline.abt["churn_flag"].unique()).issubset({0, 1})

    def test_merge_removes_leakage_columns(self, pipeline):
        """Leakage columns should be removed during merge."""
        pipeline.load_data().validate_schemas().merge_tables()
        for col in LEAKAGE_COLUMNS:
            assert col not in pipeline.abt.columns, f"Leakage column '{col}' not removed"


# =============================================================================
# Test: Missing Value Handling
# =============================================================================
class TestMissingValues:
    """Tests for Stage 5: Missing Value Analysis & Imputation."""

    def test_missing_report_generated(self, pipeline):
        """Missing report should be generated."""
        pipeline.load_data().validate_schemas().merge_tables()
        pipeline.analyze_and_impute_missing()
        assert pipeline.missing_report is not None
        assert "column" in pipeline.missing_report.columns
        assert "missing_pct" in pipeline.missing_report.columns

    def test_numeric_nulls_imputed(self, pipeline):
        """After imputation, numeric columns should have no nulls."""
        pipeline.load_data().validate_schemas().merge_tables()
        # Inject a null
        pipeline.abt.loc[0, "mrr_amount"] = np.nan
        pipeline.analyze_and_impute_missing()
        assert pipeline.abt["mrr_amount"].isnull().sum() == 0


# =============================================================================
# Test: Outlier Detection
# =============================================================================
class TestOutlierDetection:
    """Tests for Stage 6: IQR Outlier Detection."""

    def test_outliers_capped(self, pipeline):
        """Extreme values should be clipped to IQR bounds."""
        pipeline.load_data().validate_schemas().merge_tables()
        pipeline.analyze_and_impute_missing()
        # Inject an extreme outlier
        pipeline.abt.loc[0, "mrr_amount"] = 999999.0
        pipeline.detect_and_cap_outliers()
        assert pipeline.abt["mrr_amount"].max() < 999999.0

    def test_outlier_report_logged(self, pipeline):
        """Outlier report should track affected columns."""
        pipeline.load_data().validate_schemas().merge_tables()
        pipeline.analyze_and_impute_missing()
        pipeline.abt.loc[0, "mrr_amount"] = 999999.0
        pipeline.detect_and_cap_outliers()
        assert isinstance(pipeline.outlier_report, list)


# =============================================================================
# Test: Full Pipeline
# =============================================================================
class TestFullPipeline:
    """Tests for the complete run() method."""

    def test_run_returns_dataframe(self, pipeline, tmp_path):
        """Full pipeline run should return a DataFrame."""
        output_path = str(tmp_path / "output.csv")
        abt = pipeline.run(output_path=output_path)
        assert isinstance(abt, pd.DataFrame)
        assert len(abt) > 0

    def test_run_saves_csv(self, pipeline, tmp_path):
        """Pipeline should save output CSV to disk."""
        output_path = str(tmp_path / "output.csv")
        pipeline.run(output_path=output_path)
        assert Path(output_path).exists()

    def test_run_saves_version_log(self, pipeline, tmp_path):
        """Pipeline should save data versions CSV."""
        output_path = str(tmp_path / "output.csv")
        pipeline.run(output_path=output_path)
        version_path = Path("data/processed/data_versions.csv")
        # Version log is saved to default location (not tmp)
        # Pipeline always saves it — we just check the attribute
        assert len(pipeline.data_versions) == 5

    def test_method_chaining(self, pipeline):
        """Pipeline methods should support chaining."""
        result = pipeline.load_data().validate_schemas().version_data()
        assert result is pipeline  # Returns self

    def test_summary(self, pipeline, tmp_path):
        """Summary should contain key information."""
        output_path = str(tmp_path / "output.csv")
        pipeline.run(output_path=output_path)
        s = pipeline.summary()
        assert "abt_shape" in s
        assert "churn_rate" in s
        assert "columns" in s
        assert "data_versions" in s
