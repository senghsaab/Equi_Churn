"""
data_pipeline.py — Class-Based Data Ingestion & Preprocessing Pipeline
========================================================================
Modular, reusable, and testable pipeline for the RavenStack SaaS Churn
Analytics Dataset. Handles the full journey from raw CSVs to a clean,
model-ready Analytical Base Table (ABT).

Pipeline stages:
    1. Data Loading        — pd.read_csv with validation
    2. Schema Validation   — column existence, dtype checks
    3. Data Versioning     — MD5 hash of raw files for reproducibility
    4. Merging             — accounts + subscriptions + usage + support → ABT
    5. Missing Value Analysis & Imputation
    6. Outlier Detection   — IQR method with winsorization
    7. Export              — cleaned ABT to data/processed/

Usage:
    pipeline = ChurnDataPipeline(raw_dir="data/raw")
    abt = pipeline.run()                     # full pipeline
    pipeline.save(abt, "data/processed/abt_pipeline.csv")

CLI:
    python -m src.data_pipeline --raw-dir data/raw --output data/processed/abt_pipeline.csv
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

# =============================================================================
# Logging Configuration
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# =============================================================================
# Schema Definitions — Expected columns and dtypes for each table
# =============================================================================
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # ravenstack_accounts.csv — WHO are our customers?
    # -------------------------------------------------------------------------
    "accounts": {
        "file": "ravenstack_accounts.csv",
        "expected_columns": [
            "account_id",       # str  — primary key
            "account_name",     # str  — company name
            "industry",         # str  — customer's industry
            "country",          # str  — customer's country
            "signup_date",      # str  → datetime — when they signed up
            "referral_source",  # str  — how they found us
            "plan_tier",        # str  — Basic / Pro / Enterprise
            "seats",            # int  — number of licensed seats
            "is_trial",         # int  — 1 if on trial, 0 if paid
            "churn_flag",       # int  — ⭐ TARGET: 0=active, 1=churned
        ],
        "dtype_checks": {
            "seats": "numeric",
            "is_trial": "binary_int",
            "churn_flag": "binary_int",
            "signup_date": "date_parseable",
        },
    },
    # -------------------------------------------------------------------------
    # ravenstack_subscriptions.csv — WHAT are they paying?
    # -------------------------------------------------------------------------
    "subscriptions": {
        "file": "ravenstack_subscriptions.csv",
        "expected_columns": [
            "subscription_id",    # str  — unique subscription ID
            "account_id",         # str  — FK → accounts
            "start_date",         # str  → datetime
            "end_date",           # str  → datetime ⚠️ LEAKAGE
            "plan_tier",          # str  — plan level
            "seats",              # int  — seats for this subscription
            "mrr_amount",         # float — Monthly Recurring Revenue ($)
            "arr_amount",         # float — Annual Recurring Revenue ($)
            "is_trial",           # int  — trial subscription?
            "upgrade_flag",       # int  — upgraded?
            "downgrade_flag",     # int  — downgraded?
            "churn_flag",         # int  — churned during this subscription?
            "billing_frequency",  # str  — monthly / annual
            "auto_renew_flag",    # int  — auto-renewal enabled?
        ],
        "dtype_checks": {
            "mrr_amount": "numeric",
            "arr_amount": "numeric",
            "seats": "numeric",
            "is_trial": "binary_int",
            "churn_flag": "binary_int",
            "start_date": "date_parseable",
        },
    },
    # -------------------------------------------------------------------------
    # ravenstack_feature_usage.csv — HOW are they using the product?
    # -------------------------------------------------------------------------
    "feature_usage": {
        "file": "ravenstack_feature_usage.csv",
        "expected_columns": [
            "usage_id",            # str  — unique usage record
            "subscription_id",     # str  — FK → subscriptions
            "usage_date",          # str  → datetime
            "feature_name",        # str  — which feature was used
            "usage_count",         # int  — times used that day
            "usage_duration_secs", # float — seconds of usage
            "error_count",         # int  — errors encountered
            "is_beta_feature",     # int  — beta feature?
        ],
        "dtype_checks": {
            "usage_count": "numeric",
            "usage_duration_secs": "numeric",
            "error_count": "numeric",
            "usage_date": "date_parseable",
        },
    },
    # -------------------------------------------------------------------------
    # ravenstack_support_tickets.csv — ARE they having problems?
    # -------------------------------------------------------------------------
    "support_tickets": {
        "file": "ravenstack_support_tickets.csv",
        "expected_columns": [
            "ticket_id",                   # str  — unique ticket ID
            "account_id",                  # str  — FK → accounts
            "submitted_at",                # str  → datetime
            "closed_at",                   # str  → datetime
            "resolution_time_hours",       # float — hours to resolve
            "priority",                    # str  — Low / Medium / High / Critical
            "first_response_time_minutes", # float — minutes to first reply
            "satisfaction_score",          # float — CSAT (1–5)
            "escalation_flag",             # int  — escalated?
        ],
        "dtype_checks": {
            "resolution_time_hours": "numeric",
            "first_response_time_minutes": "numeric",
            "satisfaction_score": "numeric",
            "submitted_at": "date_parseable",
        },
    },
    # -------------------------------------------------------------------------
    # ravenstack_churn_events.csv — WHY did they leave?
    # ⚠️ ALL fields here (except account_id) are POST-OUTCOME.
    #    We use this ONLY to derive the churn_flag target variable,
    #    then DROP every other column to prevent leakage.
    # -------------------------------------------------------------------------
    "churn_events": {
        "file": "ravenstack_churn_events.csv",
        "expected_columns": [
            "churn_event_id",             # str
            "account_id",                 # str  — FK → accounts
            "churn_date",                 # str  ⚠️ LEAKAGE
            "reason_code",                # str  ⚠️ LEAKAGE
            "refund_amount_usd",          # float ⚠️ LEAKAGE
            "preceding_upgrade_flag",     # int  ⚠️ LEAKAGE
            "preceding_downgrade_flag",   # int  ⚠️ LEAKAGE
            "is_reactivation",            # int
            "feedback_text",              # str  ⚠️ LEAKAGE
        ],
        "dtype_checks": {
            "refund_amount_usd": "numeric",
            "churn_date": "date_parseable",
        },
    },
}

# Columns that MUST be dropped before training — they contain information
# that is only available AFTER the churn outcome has occurred.
LEAKAGE_COLUMNS = [
    "end_date",
    "churn_date",
    "reason_code",
    "refund_amount_usd",
    "preceding_upgrade_flag",
    "preceding_downgrade_flag",
    "feedback_text",
    "churn_event_id",
    "closed_at",              # resolution time implies ticket is closed
]


# =============================================================================
# ChurnDataPipeline — The Main Class
# =============================================================================
class ChurnDataPipeline:
    """
    End-to-end data ingestion and preprocessing pipeline for the
    RavenStack SaaS Churn Analytics Dataset.

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing the raw CSV files.
    config : dict or None
        Optional configuration overrides (loaded from YAML or passed directly).

    Attributes
    ----------
    tables : dict[str, pd.DataFrame]
        Loaded raw tables keyed by table name.
    abt : pd.DataFrame or None
        The merged Analytical Base Table (set after run()).
    data_versions : dict
        MD5 hashes and timestamps of raw CSVs.
    missing_report : pd.DataFrame or None
        Missing value analysis report.
    outlier_report : list[dict]
        Log of all outlier winsorization actions.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        config: Optional[Dict] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.config = config or {}
        self.tables: Dict[str, pd.DataFrame] = {}
        self.abt: Optional[pd.DataFrame] = None
        self.data_versions: Dict[str, Dict] = {}
        self.missing_report: Optional[pd.DataFrame] = None
        self.outlier_report: List[Dict] = []
        self._validation_errors: List[str] = []

        logger.info("ChurnDataPipeline initialized | raw_dir=%s", self.raw_dir)

    # =========================================================================
    # STAGE 1 — Data Loading
    # =========================================================================
    def load_data(self) -> "ChurnDataPipeline":
        """
        Load all five CSV files from raw_dir into self.tables.
        Validates that all expected files exist before loading.

        Returns self for method chaining.
        """
        logger.info("=" * 70)
        logger.info("STAGE 1: DATA LOADING")
        logger.info("=" * 70)

        if not self.raw_dir.is_dir():
            raise NotADirectoryError(
                f"Raw data directory not found: {self.raw_dir}. "
                f"Download dataset with: kaggle datasets download "
                f"rivalytics/saas-subscription-and-churn-analytics-dataset "
                f"-p {self.raw_dir} --unzip"
            )

        # Check all files exist before loading any
        missing_files = []
        for table_name, schema in SCHEMA_REGISTRY.items():
            filepath = self.raw_dir / str(schema["file"])
            if not filepath.exists():
                missing_files.append(schema["file"])

        if missing_files:
            raise FileNotFoundError(
                f"Missing CSV files in {self.raw_dir}: {missing_files}"
            )

        # Load each table
        for table_name, schema in SCHEMA_REGISTRY.items():
            filepath = self.raw_dir / str(schema["file"])
            df = pd.read_csv(filepath)

            self.tables[table_name] = df
            logger.info(
                "  Loaded %-22s | rows=%5d | cols=%2d | memory=%.2f MB",
                schema["file"],
                len(df),
                len(df.columns),
                df.memory_usage(deep=True).sum() / 1e6,
            )

        total_rows = sum(len(df) for df in self.tables.values())
        logger.info(
            "  ✓ All %d tables loaded | total_rows=%d",
            len(self.tables),
            total_rows,
        )
        return self

    # =========================================================================
    # STAGE 2 — Schema Validation
    # =========================================================================
    def validate_schemas(self) -> "ChurnDataPipeline":
        """
        Validate that each loaded table has the expected columns and
        that key columns have the correct dtypes.

        Logs warnings for issues but does not raise (lenient mode).
        Raises on critical failures (missing join keys).
        """
        logger.info("=" * 70)
        logger.info("STAGE 2: SCHEMA VALIDATION")
        logger.info("=" * 70)
        self._validation_errors = []

        for table_name, schema in SCHEMA_REGISTRY.items():
            if table_name not in self.tables:
                self._validation_errors.append(f"Table '{table_name}' not loaded")
                continue

            df = self.tables[table_name]
            expected = set(schema["expected_columns"])
            actual = set(df.columns)

            # --- Column existence check ---
            missing_cols = expected - actual
            extra_cols = actual - expected

            if missing_cols:
                logger.warning(
                    "  [%s] Missing expected columns: %s",
                    table_name,
                    sorted(missing_cols),
                )
                self._validation_errors.append(
                    f"{table_name}: missing columns {sorted(missing_cols)}"
                )

            if extra_cols:
                logger.info(
                    "  [%s] Extra columns found (OK): %s",
                    table_name,
                    sorted(extra_cols),
                )

            # --- Dtype checks ---
            dtype_checks: Dict[str, str] = schema["dtype_checks"]
            for col, expected_type in dtype_checks.items():
                if col not in df.columns:
                    continue

                if expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # Try to coerce
                        try:
                            pd.to_numeric(df[col], errors="raise")
                            logger.info(
                                "  [%s.%s] Can be coerced to numeric ✓",
                                table_name,
                                col,
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                "  [%s.%s] Expected numeric but got %s",
                                table_name,
                                col,
                                df[col].dtype,
                            )
                            self._validation_errors.append(
                                f"{table_name}.{col}: not numeric"
                            )

                elif expected_type == "binary_int":
                    unique_vals = set(df[col].dropna().unique())
                    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                        logger.warning(
                            "  [%s.%s] Expected binary {0,1} but got %s",
                            table_name,
                            col,
                            unique_vals,
                        )
                        self._validation_errors.append(
                            f"{table_name}.{col}: not binary"
                        )
                    else:
                        logger.info(
                            "  [%s.%s] Binary check passed ✓", table_name, col
                        )

                elif expected_type == "date_parseable":
                    sample = df[col].dropna().head(10)
                    try:
                        pd.to_datetime(sample, errors="raise")
                        logger.info(
                            "  [%s.%s] Date-parseable check passed ✓",
                            table_name,
                            col,
                        )
                    except Exception:
                        logger.warning(
                            "  [%s.%s] Cannot parse as date. Sample: %s",
                            table_name,
                            col,
                            sample.tolist()[:3],
                        )
                        self._validation_errors.append(
                            f"{table_name}.{col}: not date-parseable"
                        )

            # --- Critical check: account_id must exist for joins ---
            join_key = "account_id"
            if table_name == "feature_usage":
                join_key = "subscription_id"  # usage joins via subscription

            if join_key not in df.columns:
                msg = f"CRITICAL: '{join_key}' missing from {table_name} — cannot join"
                logger.error("  %s", msg)
                raise ValueError(msg)
            else:
                n_unique = df[join_key].nunique()
                n_null = df[join_key].isnull().sum()
                logger.info(
                    "  [%s.%s] Join key OK | unique=%d | nulls=%d",
                    table_name,
                    join_key,
                    n_unique,
                    n_null,
                )

        if self._validation_errors:
            logger.warning(
                "  ⚠ Schema validation completed with %d warnings: %s",
                len(self._validation_errors),
                self._validation_errors,
            )
        else:
            logger.info("  ✓ Schema validation passed — all checks OK")

        return self

    # =========================================================================
    # STAGE 3 — Data Versioning (MD5 Hashing)
    # =========================================================================
    def version_data(self) -> "ChurnDataPipeline":
        """
        Compute MD5 hash of each raw CSV file for reproducibility tracking.
        Logs hash + timestamp so we can verify data hasn't changed between runs.
        """
        logger.info("=" * 70)
        logger.info("STAGE 3: DATA VERSIONING")
        logger.info("=" * 70)

        timestamp = datetime.now().isoformat()

        for table_name, schema in SCHEMA_REGISTRY.items():
            filepath = self.raw_dir / str(schema["file"])
            if filepath.exists():
                md5_hash = self._compute_md5(filepath)
                file_size = filepath.stat().st_size
                self.data_versions[table_name] = {
                    "file": schema["file"],
                    "md5": md5_hash,
                    "size_bytes": file_size,
                    "timestamp": timestamp,
                }
                md5_prefix = md5_hash[:16]  # type: ignore[index]
                logger.info(
                    "  %-22s | md5=%s | size=%s",
                    str(schema["file"]),
                    md5_prefix + "...",
                    self._format_bytes(file_size),
                )

        logger.info("  ✓ Data versions recorded at %s", timestamp)
        return self

    @staticmethod
    def _compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file efficiently (streaming)."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Format byte count as human-readable string."""
        fsize: float = float(size)
        for unit in ["B", "KB", "MB", "GB"]:
            if fsize < 1024:
                return f"{fsize:.1f} {unit}"
            fsize /= 1024
        return f"{fsize:.1f} TB"

    # =========================================================================
    # STAGE 4 — Merging Strategy
    # =========================================================================
    def merge_tables(self) -> "ChurnDataPipeline":
        """
        Merge all tables into a single Analytical Base Table (ABT).

        Strategy:
            1. accounts (base, 1 row per customer)
            2. LEFT JOIN latest subscription per account
            3. LEFT JOIN aggregated feature_usage
               (total_events, avg_daily_usage, last_activity_date, active_days)
            4. LEFT JOIN aggregated support_tickets
               (ticket_count, avg_satisfaction, avg_resolution_time, escalations)
            5. Derive churn_flag from accounts table (already present)
            6. Remove leakage columns

        Result: one row per customer, one churn label.
        """
        logger.info("=" * 70)
        logger.info("STAGE 4: TABLE MERGING")
        logger.info("=" * 70)

        accounts = self.tables["accounts"].copy()
        subscriptions = self.tables["subscriptions"].copy()
        feature_usage = self.tables["feature_usage"].copy()
        support_tickets = self.tables["support_tickets"].copy()

        # ----- Step 4a: Parse dates in subscriptions -----
        for date_col in ["start_date", "end_date"]:
            if date_col in subscriptions.columns:
                subscriptions[date_col] = pd.to_datetime(
                    subscriptions[date_col], errors="coerce"
                )

        # ----- Step 4b: Latest subscription per account -----
        # Sort by start_date descending, take first per account → latest sub
        latest_sub = (
            subscriptions.sort_values("start_date", ascending=False)
            .groupby("account_id")
            .first()
            .reset_index()
        )
        # Rename to avoid column collisions
        sub_rename = {
            "plan_tier": "sub_plan_tier",
            "seats": "sub_seats",
            "is_trial": "sub_is_trial",
            "churn_flag": "sub_churn_flag",
        }
        latest_sub.rename(
            columns={k: v for k, v in sub_rename.items() if k in latest_sub.columns},
            inplace=True,
        )

        abt = accounts.merge(latest_sub, on="account_id", how="left")
        logger.info(
            "  Step 4b: accounts + latest_subscription → %d rows, %d cols",
            len(abt),
            len(abt.columns),
        )

        # ----- Step 4c: Aggregate feature usage to account level -----
        # feature_usage links via subscription_id, so first map to account_id
        sub_account_map = subscriptions[["subscription_id", "account_id"]].drop_duplicates()
        usage_with_account = feature_usage.merge(
            sub_account_map, on="subscription_id", how="left"
        )

        # Parse usage_date for active_days / last_activity_date
        usage_with_account["usage_date"] = pd.to_datetime(
            usage_with_account["usage_date"], errors="coerce"
        )

        usage_agg = (
            usage_with_account.groupby("account_id")
            .agg(
                total_events=("usage_count", "sum"),
                avg_daily_usage=("usage_count", "mean"),
                total_duration_secs=("usage_duration_secs", "sum"),
                avg_duration_secs=("usage_duration_secs", "mean"),
                total_errors=("error_count", "sum"),
                avg_errors=("error_count", "mean"),
                active_days=("usage_date", "nunique"),
                last_activity_date=("usage_date", "max"),
                first_activity_date=("usage_date", "min"),
                unique_features_used=("feature_name", "nunique"),
                beta_feature_usage=("is_beta_feature", "sum"),
            )
            .reset_index()
        )

        abt = abt.merge(usage_agg, on="account_id", how="left")
        logger.info(
            "  Step 4c: + aggregated_usage → %d rows, %d cols",
            len(abt),
            len(abt.columns),
        )

        # ----- Step 4d: Aggregate support tickets -----
        support_tickets["submitted_at"] = pd.to_datetime(
            support_tickets["submitted_at"], errors="coerce"
        )

        ticket_agg = (
            support_tickets.groupby("account_id")
            .agg(
                ticket_count=("ticket_id", "count"),
                avg_satisfaction=("satisfaction_score", "mean"),
                min_satisfaction=("satisfaction_score", "min"),
                avg_resolution_hours=("resolution_time_hours", "mean"),
                avg_first_response_min=("first_response_time_minutes", "mean"),
                escalation_count=("escalation_flag", "sum"),
                last_ticket_date=("submitted_at", "max"),
            )
            .reset_index()
        )

        abt = abt.merge(ticket_agg, on="account_id", how="left")
        logger.info(
            "  Step 4d: + aggregated_support → %d rows, %d cols",
            len(abt),
            len(abt.columns),
        )

        # ----- Step 4e: Remove leakage columns -----
        leaked = [c for c in LEAKAGE_COLUMNS if c in abt.columns]
        if leaked:
            abt.drop(columns=leaked, inplace=True)
            logger.warning(
                "  LEAKAGE GUARD: Dropped %d post-outcome columns: %s",
                len(leaked),
                leaked,
            )

        # ----- Step 4f: Ensure churn_flag is the target -----
        if "churn_flag" in abt.columns:
            abt["churn_flag"] = abt["churn_flag"].fillna(0).astype(int)
            churn_rate = abt["churn_flag"].mean() * 100
            logger.info(
                "  Target: churn_flag | churned=%d | not_churned=%d | rate=%.2f%%",
                int(abt["churn_flag"].sum()),
                int((abt["churn_flag"] == 0).sum()),  # type: ignore[union-attr]
                churn_rate,
            )

        logger.info(
            "  ✓ Final ABT: %d rows × %d columns (1 row per customer)",
            len(abt),
            len(abt.columns),
        )
        self.abt = abt
        return self

    # =========================================================================
    # STAGE 5 — Missing Value Analysis & Imputation
    # =========================================================================
    def analyze_and_impute_missing(self) -> "ChurnDataPipeline":
        """
        Analyze missing values across the ABT:
            1. Report % missing per column
            2. Log warnings for columns > 5% missing
            3. Impute: numeric → median, categorical → mode or 'Unknown'
        """
        logger.info("=" * 70)
        logger.info("STAGE 5: MISSING VALUE ANALYSIS & IMPUTATION")
        logger.info("=" * 70)

        if self.abt is None:
            raise RuntimeError("Must call merge_tables() before missing analysis")

        df: pd.DataFrame = self.abt  # type: ignore[assignment]

        # ----- Analysis -----
        total_cells = df.shape[0] * df.shape[1]  # type: ignore[index]
        total_missing = df.isnull().sum().sum()  # type: ignore[union-attr]
        missing_by_col = df.isnull().mean() * 100  # type: ignore[union-attr]

        report = pd.DataFrame({
            "column": missing_by_col.index,
            "missing_pct": missing_by_col.values,
            "missing_count": df.isnull().sum().values,  # type: ignore[union-attr]
            "dtype": [str(df[c].dtype) for c in df.columns],  # type: ignore[union-attr, arg-type]
        }).sort_values("missing_pct", ascending=False)

        self.missing_report = report

        logger.info(
            "  Overall: %d / %d cells missing (%.2f%%)",
            total_missing,
            total_cells,
            (total_missing / total_cells) * 100 if total_cells > 0 else 0,
        )

        # Log columns with missing values
        cols_with_missing = report[report["missing_pct"] > 0]
        for _, row in cols_with_missing.iterrows():
            level = logging.WARNING if row["missing_pct"] > 5 else logging.INFO
            logger.log(
                level,
                "  %-35s | %.1f%% missing (%d rows)",
                row["column"],
                row["missing_pct"],
                row["missing_count"],
            )

        # ----- Imputation -----
        imputation_log = []

        # Numeric columns → median
        numeric_cols = df.select_dtypes(include=["number"]).columns  # type: ignore[union-attr]
        for col in numeric_cols:
            n_missing = df[col].isnull().sum()  # type: ignore[union-attr]
            if n_missing > 0:
                median_val = df[col].median()  # type: ignore[union-attr]
                df[col] = df[col].fillna(median_val)  # type: ignore[index]
                imputation_log.append({
                    "column": col,
                    "strategy": "median",
                    "fill_value": round(median_val, 4),
                    "rows_filled": n_missing,
                })
                logger.info(
                    "  Imputed %-30s | strategy=median | value=%.4f | rows=%d",
                    col,
                    median_val,
                    n_missing,
                )

        # Categorical/object columns → mode or 'Unknown'
        cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns  # type: ignore[union-attr]
        for col in cat_cols:
            n_missing = df[col].isnull().sum()  # type: ignore[union-attr]
            if n_missing > 0:
                mode_val = df[col].mode()  # type: ignore[union-attr]
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)  # type: ignore[index]
                imputation_log.append({
                    "column": col,
                    "strategy": "mode",
                    "fill_value": fill_val,
                    "rows_filled": n_missing,
                })
                logger.info(
                    "  Imputed %-30s | strategy=mode | value='%s' | rows=%d",
                    col,
                    fill_val,
                    n_missing,
                )

        # Verify no remaining nulls in critical columns
        remaining = df.isnull().sum().sum()  # type: ignore[union-attr]
        if remaining > 0:
            logger.warning(
                "  ⚠ %d null values remain after imputation (in date columns — expected)",
                remaining,
            )
        else:
            logger.info("  ✓ Zero null values remaining after imputation")

        logger.info("  ✓ Imputation complete | %d columns filled", len(imputation_log))
        self.abt = df
        return self

    # =========================================================================
    # STAGE 6 — Outlier Detection (IQR Method with Winsorization)
    # =========================================================================
    def detect_and_cap_outliers(
        self,
        columns: Optional[List[str]] = None,
        iqr_multiplier: float = 1.5,
    ) -> "ChurnDataPipeline":
        """
        Detect outliers using the IQR method and cap (winsorize) them.

        For each numeric column:
            Q1 = 25th percentile
            Q3 = 75th percentile
            IQR = Q3 - Q1
            Lower bound = Q1 - iqr_multiplier × IQR
            Upper bound = Q3 + iqr_multiplier × IQR

        Values below lower or above upper are clipped to the bound.
        All changes are logged for audit.

        Parameters
        ----------
        columns : list[str] or None
            Specific columns to check. If None, auto-selects:
            MRR, tenure, usage, and support metric columns.
        iqr_multiplier : float
            Multiplier for IQR range (default 1.5 = standard Tukey fence).
        """
        logger.info("=" * 70)
        logger.info("STAGE 6: OUTLIER DETECTION (IQR METHOD)")
        logger.info("=" * 70)

        if self.abt is None:
            raise RuntimeError("Must call merge_tables() before outlier detection")

        df: pd.DataFrame = self.abt  # type: ignore[assignment]

        # Auto-select columns if not provided
        if columns is None:
            columns = [  # type: ignore[assignment]
                str(col) for col in df.select_dtypes(include=["number"]).columns  # type: ignore[union-attr]
                if col not in ["account_id", "churn_flag", "is_trial",
                               "sub_is_trial", "auto_renew_flag", "upgrade_flag",
                               "downgrade_flag", "sub_churn_flag",
                               "escalation_flag", "is_beta_feature",
                               "beta_feature_usage"]
            ]

        self.outlier_report = []

        for col in columns:  # type: ignore[union-attr]
            if col not in df.columns:  # type: ignore[union-attr]
                continue

            q1 = df[col].quantile(0.25)  # type: ignore[union-attr]
            q3 = df[col].quantile(0.75)  # type: ignore[union-attr]
            iqr = q3 - q1

            if iqr == 0:
                logger.info("  %-35s | IQR=0 → skipping (no spread)", col)
                continue

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            # Count outliers before clipping
            n_below = (df[col] < lower_bound).sum()  # type: ignore[union-attr]
            n_above = (df[col] > upper_bound).sum()  # type: ignore[union-attr]
            n_total = n_below + n_above

            if n_total > 0:
                # Winsorize (cap at bounds)
                original_min = df[col].min()  # type: ignore[union-attr]
                original_max = df[col].max()  # type: ignore[union-attr]
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)  # type: ignore[index]

                record = {
                    "column": col,
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "outliers_below": int(n_below),
                    "outliers_above": int(n_above),
                    "total_outliers": int(n_total),
                    "pct_affected": round(n_total / len(df) * 100, 2),  # type: ignore[arg-type]
                    "original_range": f"[{original_min:.2f}, {original_max:.2f}]",
                    "capped_range": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                }
                self.outlier_report.append(record)

                logger.warning(
                    "  %-35s | outliers=%d (%.1f%%) | below=%d above=%d | "
                    "capped to [%.2f, %.2f]",
                    col,
                    n_total,
                    n_total / len(df) * 100,  # type: ignore[arg-type]
                    n_below,
                    n_above,
                    lower_bound,
                    upper_bound,
                )
            else:
                logger.info(
                    "  %-35s | No outliers (IQR bounds: [%.2f, %.2f])",
                    col,
                    lower_bound,
                    upper_bound,
                )

        logger.info(
            "  ✓ Outlier detection complete | %d columns checked, %d had outliers",
            len(columns),  # type: ignore[arg-type]
            len(self.outlier_report),
        )
        self.abt = df
        return self

    # =========================================================================
    # STAGE 7 — Remove Duplicates
    # =========================================================================
    def remove_duplicates(self) -> "ChurnDataPipeline":
        """Remove duplicate rows by account_id, keeping the last occurrence."""
        logger.info("=" * 70)
        logger.info("STAGE 7: DEDUPLICATION")
        logger.info("=" * 70)

        if self.abt is None:
            raise RuntimeError("Must call merge_tables() first")
        assert self.abt is not None

        n_before = len(self.abt)  # type: ignore[arg-type]
        self.abt = self.abt.drop_duplicates(  # type: ignore[union-attr]
            subset=["account_id"], keep="last"
        ).reset_index(drop=True)
        n_removed = n_before - len(self.abt)  # type: ignore[arg-type]

        if n_removed > 0:
            logger.warning(
                "  Removed %d duplicate rows (%.2f%%)",
                n_removed,
                n_removed / n_before * 100,
            )
        else:
            logger.info("  ✓ No duplicate account_ids found")

        return self

    # =========================================================================
    # STAGE 8 — Type Conversions
    # =========================================================================
    def convert_types(self) -> "ChurnDataPipeline":
        """Convert date strings to datetime and categorical strings to category dtype."""
        logger.info("=" * 70)
        logger.info("STAGE 8: TYPE CONVERSIONS")
        logger.info("=" * 70)

        if self.abt is None:
            raise RuntimeError("Must call merge_tables() first")

        df: pd.DataFrame = self.abt  # type: ignore[assignment]

        # Date columns
        date_cols = [c for c in df.columns if any(  # type: ignore[union-attr]
            kw in c for kw in ["_date", "_at"]
        )]
        for col in date_cols:
            if col in df.columns and df[col].dtype == "object":  # type: ignore[union-attr]
                df[col] = pd.to_datetime(df[col], errors="coerce")  # type: ignore[index]
                logger.info("  Converted %-30s → datetime64", col)

        # Categorical columns
        cat_candidates = [
            "industry", "country", "referral_source", "plan_tier",
            "sub_plan_tier", "billing_frequency", "priority",
        ]
        for col in cat_candidates:
            if col in df.columns:  # type: ignore[union-attr]
                df[col] = df[col].astype("category")  # type: ignore[index]
                logger.info(
                    "  Converted %-30s → category (%d unique)",
                    col,
                    df[col].nunique(),  # type: ignore[union-attr]
                )

        # Ensure target is int
        if "churn_flag" in df.columns:  # type: ignore[union-attr]
            df["churn_flag"] = df["churn_flag"].astype(int)  # type: ignore[index]

        self.abt = df
        logger.info("  ✓ Type conversions complete")
        return self

    # =========================================================================
    # Save & Export
    # =========================================================================
    def save(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = "data/processed/abt_pipeline.csv",
    ) -> str:
        """
        Save the ABT (or a provided DataFrame) to CSV.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame to save. If None, uses self.abt.
        output_path : str
            Output file path.

        Returns
        -------
        str
            Path to the saved file.
        """
        df = df if df is not None else self.abt
        if df is None:
            raise RuntimeError("No data to save. Run the pipeline first.")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("  Saved ABT to %s (%d rows × %d cols)", out, len(df), len(df.columns))
        return str(out)

    def save_version_log(
        self, output_path: str = "data/processed/data_versions.csv"
    ) -> None:
        """Save data version hashes to CSV for audit trail."""
        if not self.data_versions:
            logger.warning("No version data to save — run version_data() first")
            return

        records = []
        for table_name, info in self.data_versions.items():
            records.append({"table": table_name, **info})

        version_df = pd.DataFrame(records)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        version_df.to_csv(out, index=False)
        logger.info("  Saved data version log to %s", out)

    # =========================================================================
    # Summary & Reporting
    # =========================================================================
    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the pipeline state."""
        result: Dict[str, Any] = {
            "tables_loaded": list(self.tables.keys()),
            "validation_errors": self._validation_errors,
            "data_versions": self.data_versions,
        }
        if self.abt is not None:
            abt: pd.DataFrame = self.abt  # type: ignore[assignment]
            churn_rate: Optional[float] = (
                float(abt["churn_flag"].mean()) * 100  # type: ignore[union-attr]
                if "churn_flag" in abt.columns  # type: ignore[union-attr]
                else None
            )
            result["abt_shape"] = abt.shape  # type: ignore[union-attr]
            result["churn_rate"] = churn_rate
            result["columns"] = list(abt.columns)  # type: ignore[union-attr]
            result["dtypes"] = {str(k): int(v) for k, v in abt.dtypes.value_counts().items()}  # type: ignore[union-attr]
            result["memory_mb"] = round(
                abt.memory_usage(deep=True).sum() / 1e6, 2  # type: ignore[union-attr]
            )
        if self.outlier_report:
            result["outliers_capped"] = len(self.outlier_report)
        return result

    # =========================================================================
    # FULL PIPELINE — run()
    # =========================================================================
    def run(
        self,
        output_path: str = "data/processed/abt_pipeline.csv",
        save_versions: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the full pipeline end-to-end:
            1. Load data
            2. Validate schemas
            3. Version data (MD5 hashes)
            4. Merge tables into ABT
            5. Remove duplicates
            6. Analyze & impute missing values
            7. Detect & cap outliers (IQR)
            8. Convert types
            9. Save output

        Parameters
        ----------
        output_path : str
            Where to save the final ABT.
        save_versions : bool
            Whether to save the data version log.

        Returns
        -------
        pd.DataFrame
            The cleaned, merged ABT ready for feature engineering.
        """
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  CHURN DATA PIPELINE — STARTING FULL RUN" + " " * 27 + "║")
        logger.info("╚" + "═" * 68 + "╝")

        # Execute pipeline stages in sequence
        (
            self.load_data()
            .validate_schemas()
            .version_data()
            .merge_tables()
            .remove_duplicates()
            .analyze_and_impute_missing()
            .detect_and_cap_outliers()
            .convert_types()
        )

        # Save outputs
        self.save(output_path=output_path)
        if save_versions:
            self.save_version_log()

        # Final summary
        s = self.summary()
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  PIPELINE COMPLETE" + " " * 49 + "║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  ABT Shape     : %-48s ║", f"{s['abt_shape']}")
        logger.info("║  Churn Rate    : %-48s ║", f"{s.get('churn_rate', 'N/A'):.2f}%")
        logger.info("║  Memory        : %-48s ║", f"{s.get('memory_mb', 'N/A')} MB")
        logger.info("║  Outliers Fixed: %-48s ║", f"{s.get('outliers_capped', 0)} columns")
        logger.info("║  Output        : %-48s ║", output_path)
        logger.info("╚" + "═" * 68 + "╝")

        return self.abt


# =============================================================================
# CLI Entrypoint
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ChurnDataPipeline end-to-end"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Path to directory containing raw CSV files",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abt_pipeline.csv",
        help="Output path for the cleaned ABT",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    args = parser.parse_args()

    pipeline = ChurnDataPipeline(raw_dir=args.raw_dir)
    abt = pipeline.run(output_path=args.output)
    print(f"\nDone! ABT saved to {args.output} | shape={abt.shape}")
