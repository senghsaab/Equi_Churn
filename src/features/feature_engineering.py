"""
feature_engineering.py — Derive Predictive Features for SaaS Churn
====================================================================
Transforms the cleaned ABT into a model-ready feature matrix using
sklearn Pipeline + ColumnTransformer for full reproducibility.

Engineered features:
    1. tenure_months            — subscription duration in months
    2. tenure_bucket            — categorical bins [0–3, 3–6, 6–12, 12+]
    3. usage_intensity          — total_events / tenure_months
    4. active_day_ratio         — active_days / subscription_tenure_days
    5. days_since_last_activity — reference_date − last_activity_date
    6. mrr_segment              — low / medium / high (MRR tertiles)
    7. near_renewal             — 1 if days to contract end < 30

Encoding:
    - One-hot  : plan_tier, country, industry, billing_frequency,
                 tenure_bucket, mrr_segment, referral_source
    - Binary   : is_trial, upgrade_flag, downgrade_flag,
                 auto_renew_flag, near_renewal  (already 0/1)

Scaling:
    - StandardScaler on numeric features AFTER train/test split
      (fit on train only to prevent leakage)

No XGBoost, SHAP, or advanced libraries.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants — Reference date for tenure / recency calculations
# ---------------------------------------------------------------------------
# Use a fixed reference date (latest date in the dataset) for reproducibility.
# Using Timestamp.today() would make features non-deterministic.
REFERENCE_DATE = pd.Timestamp("2025-01-01")

# Columns that should NEVER be in the final feature matrix
ID_AND_META_COLUMNS = [
    "account_id",
    "account_name",
    "subscription_id",
    "company_name",
]

# All date columns to be dropped after feature derivation
DATE_COLUMNS = [
    "signup_date",
    "start_date",
    "last_activity_date",
    "first_activity_date",
    "last_ticket_date",
    "end_date",
]

# Leakage columns — post-outcome information
LEAKAGE_COLUMNS = [
    "sub_churn_flag",   # subscription-level churn flag (leaks target)
    "sub_plan_tier",    # redundant with plan_tier
    "sub_seats",        # redundant with seats
    "sub_is_trial",     # redundant with is_trial
]

# Nominal categoricals to one-hot encode
ONEHOT_COLUMNS = [
    "plan_tier",
    "country",
    "industry",
    "billing_frequency",
    "tenure_bucket",
    "mrr_segment",
    "referral_source",
]

# Binary flags — already 0/1, no encoding needed, just ensure int type
BINARY_COLUMNS = [
    "is_trial",
    "upgrade_flag",
    "downgrade_flag",
    "auto_renew_flag",
    "near_renewal",
]

# Target column
TARGET_COL = "churn_flag"


# ===========================================================================
# 1. Feature Derivation Functions
# ===========================================================================
def derive_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tenure-based features:
        - tenure_months      : (reference_date - start_date) / 30
        - tenure_bucket      : categorical [0-3, 3-6, 6-12, 12+]
        - subscription_tenure_days : raw days (used internally)

    Shorter tenure = higher churn risk (new customers haven't seen value).
    """
    df = df.copy()

    start = pd.to_datetime(df["start_date"], errors="coerce")
    df["subscription_tenure_days"] = (REFERENCE_DATE - start).dt.days.clip(lower=0)
    df["tenure_months"] = (df["subscription_tenure_days"] / 30.0).round(2)

    # Categorical bins capturing non-linear tenure effects
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 6, 12, float("inf")],
        labels=["0-3m", "3-6m", "6-12m", "12+m"],
        right=True,
        include_lowest=True,
    )

    logger.info(
        "Tenure features: median=%.1f months | buckets=%s",
        df["tenure_months"].median(),
        df["tenure_bucket"].value_counts().to_dict(),
    )
    return df


def derive_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive usage intensity features:
        - usage_intensity       : total_events / tenure_months
          (low events per month = disengaged customer → churn signal)
        - active_day_ratio      : active_days / subscription_tenure_days
          (captures consistency of usage, not just volume)
        - days_since_last_activity : reference_date - last_activity_date
          (recent inactivity is a leading churn indicator)
    """
    df = df.copy()

    # usage_intensity — guard against division by zero for very new accounts
    tenure_safe = df["tenure_months"].replace(0, np.nan)
    df["usage_intensity"] = (df["total_events"] / tenure_safe).fillna(0).round(2)

    # active_day_ratio — consistency of engagement
    tenure_days_safe = df["subscription_tenure_days"].replace(0, np.nan)
    df["active_day_ratio"] = (df["active_days"] / tenure_days_safe).fillna(0).clip(0, 1).round(4)

    # days_since_last_activity
    last_activity = pd.to_datetime(df["last_activity_date"], errors="coerce")
    df["days_since_last_activity"] = (REFERENCE_DATE - last_activity).dt.days.clip(lower=0)
    # Fill NaN (no activity data) with a high value indicating disengagement
    df["days_since_last_activity"] = df["days_since_last_activity"].fillna(
        df["days_since_last_activity"].max() if df["days_since_last_activity"].notna().any() else 365
    )

    logger.info(
        "Usage features: intensity_median=%.1f | active_ratio_median=%.3f | "
        "days_since_last_median=%.0f",
        df["usage_intensity"].median(),
        df["active_day_ratio"].median(),
        df["days_since_last_activity"].median(),
    )
    return df


def derive_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive MRR segment using tertiles (3 bins: low / medium / high).
    High-MRR accounts need different retention tactics than SMBs.
    """
    df = df.copy()

    try:
        df["mrr_segment"] = pd.qcut(
            df["mrr_amount"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
    except (ValueError, TypeError) as e:
        logger.warning("Could not create MRR tertiles: %s — using quartile fallback", e)
        median_mrr = df["mrr_amount"].median()
        df["mrr_segment"] = pd.cut(
            df["mrr_amount"],
            bins=[-float("inf"), median_mrr * 0.5, median_mrr * 1.5, float("inf")],
            labels=["low", "medium", "high"],
        )

    logger.info(
        "MRR segments: %s", df["mrr_segment"].value_counts().to_dict()
    )
    return df


def derive_renewal_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive near_renewal flag: 1 if days to NEXT billing cycle renewal < 30, else 0.
    Customers near renewal are at a decision point -- high churn risk.

    Uses billing_frequency (monthly=30 days, annual=365 days) to determine
    the next renewal date from start_date. This correctly captures accounts
    approaching a payment / renewal decision point at reference date.
    """
    df = df.copy()

    start = pd.to_datetime(df["start_date"], errors="coerce")
    days_since_start = (REFERENCE_DATE - start).dt.days.clip(lower=0)

    # Determine cycle length per account based on billing_frequency
    if "billing_frequency" in df.columns:
        cycle_days = df["billing_frequency"].map(
            {"monthly": 30, "quarterly": 90, "annual": 365}
        ).fillna(365).astype(float)
    else:
        cycle_days = pd.Series(365.0, index=df.index)
        logger.info("billing_frequency not available -- defaulting to 365-day annual cycle")

    # cycles_elapsed = floor(days_since_start / cycle_days)
    # days_to_next_renewal = (cycles_elapsed + 1) * cycle_days - days_since_start
    cycles_elapsed = (days_since_start / cycle_days).astype(int)
    days_to_next_renewal = (cycles_elapsed + 1) * cycle_days - days_since_start

    df["near_renewal"] = (days_to_next_renewal < 30).astype(int)
    df["near_renewal"] = df["near_renewal"].fillna(0).astype(int)

    logger.info(
        "Near-renewal accounts: %d (%.1f%%)",
        df["near_renewal"].sum(),
        df["near_renewal"].mean() * 100,
    )
    return df


# ===========================================================================
# 2. Feature Matrix Preparation
# ===========================================================================
def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y), dropping non-predictive columns.

    Drops:
        - ID / metadata columns (account_id, company_name, etc.)
        - Raw date columns (already used to derive tenure/recency features)
        - Leakage columns (sub_churn_flag, etc.)
        - Any remaining non-numeric / non-category columns

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    df = df.copy()

    # Extract target
    y = df[target_col].astype(int).copy()

    # Columns to drop
    drop_cols = set(ID_AND_META_COLUMNS + DATE_COLUMNS + LEAKAGE_COLUMNS + [target_col])
    # Also drop subscription_tenure_days (intermediate, used to compute other features)
    drop_cols.add("subscription_tenure_days")
    # Also drop arr_amount (perfectly correlated with mrr_amount: r=1.0)
    drop_cols.add("arr_amount")
    # Drop any is_missing indicators for dropped columns
    drop_cols.update([c for c in df.columns if c.startswith("is_missing_")])

    existing_drops = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drops, errors="ignore")

    # Log what we're keeping vs dropping
    logger.info(
        "Dropped %d columns: %s",
        len(existing_drops),
        sorted(existing_drops),
    )

    return X, y


# ===========================================================================
# 3. Encoding & Scaling via sklearn Pipeline + ColumnTransformer
# ===========================================================================
def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """
    Build a ColumnTransformer that:
        - One-hot encodes nominal categoricals
        - Passes through binary flags as-is (already 0/1)
        - Scales all numeric features with StandardScaler

    Returns
    -------
    tuple[ColumnTransformer, list, list, list]
        Preprocessor, numeric_cols, onehot_cols, binary_cols
    """
    # Identify column groups present in X
    onehot_cols = [c for c in ONEHOT_COLUMNS if c in X.columns]
    binary_cols = [c for c in BINARY_COLUMNS if c in X.columns]
    numeric_cols = [
        c for c in X.select_dtypes(include=["number"]).columns
        if c not in binary_cols
    ]

    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("One-hot features (%d): %s", len(onehot_cols), onehot_cols)
    logger.info("Binary features  (%d): %s", len(binary_cols), binary_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",          # avoid multicollinearity
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                onehot_cols,
            ),
            (
                "bin",
                "passthrough",
                binary_cols,
            ),
        ],
        remainder="drop",  # drops any remaining uncategorized columns
        verbose_feature_names_out=True,
    )

    return preprocessor, numeric_cols, onehot_cols, binary_cols


def encode_and_scale(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Apply the ColumnTransformer to produce a fully numeric feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix with mixed types.
    preprocessor : ColumnTransformer
        The fitted (or to-be-fitted) preprocessor.
    fit : bool
        If True, fit_transform. If False, transform only (for test set).

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix with named columns.
    """
    # Ensure binary columns are numeric before transform
    for col in BINARY_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Ensure categorical columns are string type for OneHotEncoder
    for col in ONEHOT_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    # Get feature names from the transformer
    feature_names = preprocessor.get_feature_names_out()

    result = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    logger.info(
        "Transformed feature matrix: %s (%d features)",
        result.shape,
        len(feature_names),
    )
    return result


# ===========================================================================
# 4. Pipeline Entrypoint
# ===========================================================================
def run_feature_engineering(
    df: pd.DataFrame,
    output_path: Optional[str] = "data/processed/abt_features.csv",
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Full feature engineering pipeline:
        1. Derive tenure features (tenure_months, tenure_bucket)
        2. Derive usage features (usage_intensity, active_day_ratio,
           days_since_last_activity)
        3. Derive revenue features (mrr_segment via tertiles)
        4. Derive near_renewal flag
        5. Prepare feature matrix (drop IDs, dates, leakage)
        6. Build ColumnTransformer (scaler + one-hot + passthrough)
        7. Fit-transform on full dataset (scaling done inside Pipeline
           at train time; here we just save the pre-encoding features)

    NOTE: StandardScaler should be fit on TRAIN ONLY. This function
    returns the raw (pre-scaled) feature matrix and the unfitted
    preprocessor. The train.py module handles the train/test split
    and calls preprocessor.fit_transform(X_train) / .transform(X_test).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, ColumnTransformer]
        X (pre-scaled features), y (target), preprocessor (unfitted).
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # --- Step 1–4: Derive all engineered features ---
    df = derive_tenure_features(df)
    df = derive_usage_features(df)
    df = derive_revenue_features(df)
    df = derive_renewal_feature(df)

    # --- Step 5: Prepare feature matrix ---
    X, y = prepare_feature_matrix(df)

    # --- Step 6: Build preprocessor (unfitted) ---
    preprocessor, numeric_cols, onehot_cols, binary_cols = build_preprocessor(X)

    # --- Step 7: Save pre-encoded feature matrix (for inspection) ---
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_df = X.copy()
        save_df[TARGET_COL] = y.values
        save_df.to_csv(out, index=False)
        logger.info("Saved pre-encoded feature matrix to %s", out)

    # Log summary
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("  Features (X): %s", X.shape)
    logger.info("  Target  (y): %s | churn_rate=%.2f%%", y.shape, y.mean() * 100)
    logger.info(
        "  Engineered: tenure_months, tenure_bucket, usage_intensity, "
        "active_day_ratio, days_since_last_activity, mrr_segment, near_renewal"
    )
    logger.info("  Preprocessor: StandardScaler + OneHotEncoder (UNFITTED)")
    logger.info("=" * 60)

    return X, y, preprocessor


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--input",
        default="data/processed/abt_cleaned.csv",
        help="Path to cleaned ABT",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abt_features.csv",
        help="Output path for feature matrix",
    )
    args = parser.parse_args()
    cleaned_abt = pd.read_csv(args.input)
    X, y, preprocessor = run_feature_engineering(cleaned_abt, output_path=args.output)
    print(f"\nDone! Feature matrix: {X.shape} | Target: {y.shape}")
    print(f"Churn rate: {y.mean()*100:.1f}%")
    print(f"Features: {list(X.columns)}")
