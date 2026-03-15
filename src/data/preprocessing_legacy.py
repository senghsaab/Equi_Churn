"""
preprocessing.py — Data Cleaning & Preprocessing
===================================================
Handles missing values, duplicates, type conversions, and leakage-safe
column filtering for the churn prediction pipeline.

Key principles:
    1. Never leak future information into features.
    2. All transformations are idempotent and logged.
    3. Preserve original data — write to data/processed/.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — Leakage-Prone Columns
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    "churn_date",
    "churn_reason",
    "refund_amount",
    "cancellation_date",
    "subscription_end_date",
    "end_date",
]

DATE_COLUMNS = [
    "signup_date",
    "start_date",
    "created_date",
    "usage_date",
]

CATEGORICAL_COLUMNS = [
    "industry",
    "company_size",
    "country",
    "plan",
    "status",
    "priority",
]


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------
def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Handle missing values with configurable strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_strategy : str
        Strategy for numeric columns: 'median', 'mean', or 'zero'.
    categorical_strategy : str
        Strategy for categorical columns: 'mode' or 'unknown'.
    threshold : float
        Drop columns with more than this fraction of missing values.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()
    initial_shape = df.shape

    # 1. Drop columns above missing threshold
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    if drop_cols:
        logger.warning(
            "Dropping %d columns with >%.0f%% missing: %s",
            len(drop_cols),
            threshold * 100,
            drop_cols,
        )
        df.drop(columns=drop_cols, inplace=True)

    # 2. Create missing-value indicator features (before imputation)
    for col in df.columns:
        if df[col].isnull().any():
            missing_rate = df[col].isnull().mean()
            if missing_rate > 0.01:  # Only if > 1% missing
                df[f"is_missing_{col}"] = df[col].isnull().astype(int)
                logger.info(
                    "Created indicator: is_missing_%s (%.1f%% missing)",
                    col,
                    missing_rate * 100,
                )

    # 3. Impute numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == "median":
                fill_val = df[col].median()
            elif numeric_strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
            logger.info("Imputed numeric '%s' with %s=%.4f", col, numeric_strategy, fill_val)

    # 4. Impute categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            if categorical_strategy == "mode" and not df[col].mode().empty:
                fill_val = df[col].mode()[0]
            else:
                fill_val = "Unknown"
            df[col] = df[col].fillna(fill_val)
            logger.info("Imputed categorical '%s' with '%s'", col, fill_val)

    logger.info(
        "Missing value handling: %s → %s | indicators added=%d",
        initial_shape,
        df.shape,
        len([c for c in df.columns if c.startswith("is_missing_")]),
    )
    return df


# ---------------------------------------------------------------------------
# Duplicate Removal
# ---------------------------------------------------------------------------
def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "last",
) -> pd.DataFrame:
    """Remove duplicate rows, logging the count."""
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d duplicate rows (%.2f%%)", n_removed, n_removed / n_before * 100)
    else:
        logger.info("No duplicates found.")
    return df


# ---------------------------------------------------------------------------
# Type Conversions
# ---------------------------------------------------------------------------
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate dtypes:
        - date strings → datetime64
        - known categorical columns → category
        - boolean-like columns → int
    """
    df = df.copy()

    # Date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info("Converted '%s' to datetime64", col)

    # Categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
            logger.info(
                "Converted '%s' to category (%d unique values)",
                col,
                df[col].nunique(),
            )

    # Ensure churn_flag is integer
    if "churn_flag" in df.columns:
        df["churn_flag"] = df["churn_flag"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Leakage Prevention
# ---------------------------------------------------------------------------
def remove_leakage_columns(
    df: pd.DataFrame,
    extra_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop columns that carry information from after the churn event,
    which would not be available at prediction time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    extra_columns : list[str] or None
        Additional columns to drop beyond the default LEAKAGE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Dataframe with leakage columns removed.
    """
    cols_to_drop = LEAKAGE_COLUMNS.copy()
    if extra_columns:
        cols_to_drop.extend(extra_columns)

    present = [c for c in cols_to_drop if c in df.columns]
    if present:
        df = df.drop(columns=present)
        logger.warning("LEAKAGE GUARD: Dropped %d columns: %s", len(present), present)
    else:
        logger.info("No leakage columns found to remove.")

    return df


def detect_leakage_features(
    df: pd.DataFrame,
    target_col: str = "churn_flag",
    auc_threshold: float = 0.95,
) -> List[str]:
    """
    Flag features with suspiciously high single-feature AUC against target.
    This helps catch subtle leakage that isn't caught by column-name checks.

    Returns list of suspicious feature names.
    """
    from sklearn.metrics import roc_auc_score

    suspicious = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(
        target_col, errors="ignore"
    )

    for col in numeric_cols:
        valid = df[[col, target_col]].dropna()
        if len(valid) < 50 or valid[target_col].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(valid[target_col], valid[col])
            auc = max(auc, 1 - auc)  # Handle inversely-correlated features
            if auc > auc_threshold:
                suspicious.append(col)
                logger.warning(
                    "LEAKAGE ALERT: '%s' has AUC=%.4f against target. Investigate!",
                    col,
                    auc,
                )
        except Exception:
            continue

    if not suspicious:
        logger.info("No suspicious high-AUC features detected.")
    return suspicious


# ---------------------------------------------------------------------------
# Outlier Detection
# ---------------------------------------------------------------------------
def clip_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """Clip numeric outliers to percentile bounds."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    for col in columns:
        if col in df.columns:
            lower = df[col].quantile(lower_pct)
            upper = df[col].quantile(upper_pct)
            clipped = df[col].clip(lower, upper)
            n_clipped = (df[col] != clipped).sum()
            if n_clipped > 0:
                df[col] = clipped
                logger.info(
                    "Clipped %d values in '%s' to [%.2f, %.2f]",
                    n_clipped,
                    col,
                    lower,
                    upper,
                )
    return df


# ---------------------------------------------------------------------------
# Pipeline Entrypoint
# ---------------------------------------------------------------------------
def run_preprocessing(
    df: pd.DataFrame,
    output_path: Optional[str] = "data/processed/abt_cleaned.csv",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
        1. Remove leakage columns
        2. Remove duplicates
        3. Convert types
        4. Handle missing values
        5. Clip outliers
        6. Detect remaining leakage

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged ABT from data_ingestion.
    output_path : str or None
        If provided, saves the cleaned ABT.

    Returns
    -------
    pd.DataFrame
        Cleaned, preprocessed ABT.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df = remove_leakage_columns(df)
    df = remove_duplicates(df, subset=["account_id"])
    df = convert_types(df)
    df = handle_missing_values(df)
    df = clip_outliers(df)

    # Final leakage check
    if "churn_flag" in df.columns:
        suspicious = detect_leakage_features(df)
        if suspicious:
            logger.warning(
                "Review these suspicious features before training: %s", suspicious
            )

    if output_path:
        from pathlib import Path

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved cleaned ABT to %s", out)

    logger.info("PREPROCESSING COMPLETE | shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--input",
        default="data/processed/abt_pipeline.csv",
        help="Path to merged ABT",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abt_cleaned.csv",
        help="Output path for cleaned ABT",
    )
    args = parser.parse_args()
    raw_abt = pd.read_csv(args.input)
    run_preprocessing(raw_abt, output_path=args.output)
