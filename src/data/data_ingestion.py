"""
data_ingestion.py — Load & Merge Raw CSV Tables
=================================================
Loads the five RavenStack CSV files, validates schemas, and produces
a single merged analytical base table (ABT) keyed on account_id.

Tables:
    - accounts.csv         → customer metadata
    - subscriptions.csv    → subscription lifecycle & MRR
    - feature_usage.csv    → daily product interaction logs
    - support_tickets.csv  → support activity & satisfaction
    - churn_events.csv     → churn dates, reasons, refunds
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_FILES = [
    "ravenstack_accounts.csv",
    "ravenstack_subscriptions.csv",
    "ravenstack_feature_usage.csv",
    "ravenstack_support_tickets.csv",
    "ravenstack_churn_events.csv",
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def load_single_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a single CSV with basic validation."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, **kwargs)
    logger.info(
        "Loaded %-25s | rows=%d | cols=%d | mem=%.2f MB",
        path.name,
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def load_all_tables(
    raw_dir: str = "data/raw",
) -> Dict[str, pd.DataFrame]:
    """
    Load all five RavenStack CSV files from the raw directory.

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by table name (without .csv extension).
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {raw_path}")

    # Validate all expected files are present
    missing = [f for f in EXPECTED_FILES if not (raw_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data files in {raw_path}: {missing}. "
            f"Download from Kaggle: rivalytics/saas-subscription-and-churn-analytics-dataset"
        )

    tables = {}
    for filename in EXPECTED_FILES:
        key = filename.replace(".csv", "").replace("ravenstack_", "")
        tables[key] = load_single_csv(raw_path / filename)

    logger.info("All %d tables loaded successfully.", len(tables))
    return tables


def validate_schemas(tables: Dict[str, pd.DataFrame]) -> None:
    """
    Run basic schema checks on loaded tables.
    Logs warnings for unexpected issues but does not raise.
    """
    # Check for account_id in all tables (primary join key)
    for name, df in tables.items():
        if "account_id" not in df.columns:
            logger.warning(
                "Table '%s' does not contain 'account_id' column. "
                "Available columns: %s",
                name,
                list(df.columns),
            )

    # Check for duplicate account_ids in accounts table
    if "accounts" in tables:
        acct = tables["accounts"]
        n_dupes = acct["account_id"].duplicated().sum()
        if n_dupes > 0:
            logger.warning(
                "Found %d duplicate account_ids in accounts table.", n_dupes
            )

    # Log basic stats
    for name, df in tables.items():
        null_pct = (df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100
        logger.info(
            "Schema check %-20s | dtypes=%s | null_pct=%.2f%%",
            name,
            dict(df.dtypes.value_counts()),
            null_pct,
        )


def merge_abt(
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all tables into a single Analytical Base Table (ABT) keyed
    on account_id.

    Merge strategy:
    1. accounts ← LEFT JOIN subscriptions (latest subscription per account)
    2. result   ← LEFT JOIN aggregated feature_usage
    3. result   ← LEFT JOIN aggregated support_tickets
    4. result   ← LEFT JOIN churn_events (to derive target variable)

    Returns
    -------
    pd.DataFrame
        Merged ABT with one row per account.
    """
    accounts = tables["accounts"].copy()
    subscriptions = tables["subscriptions"].copy()
    feature_usage = tables["feature_usage"].copy()
    support_tickets = tables["support_tickets"].copy()
    churn_events = tables["churn_events"].copy()

    # --- 1. Latest subscription per account ---
    if "start_date" in subscriptions.columns:
        subscriptions["start_date"] = pd.to_datetime(
            subscriptions["start_date"], errors="coerce"
        )
        latest_sub = (
            subscriptions.sort_values("start_date")
            .groupby("account_id")
            .last()
            .reset_index()
        )
    else:
        latest_sub = subscriptions.groupby("account_id").last().reset_index()

    abt = accounts.merge(latest_sub, on="account_id", how="left", suffixes=("", "_sub"))
    logger.info("After subscription merge: %d rows", len(abt))

    # --- 2. Aggregated feature usage ---
    # feature_usage may use subscription_id instead of account_id;
    # map through subscriptions to get account_id.
    if "account_id" not in feature_usage.columns and "subscription_id" in feature_usage.columns:
        sub_id_map = subscriptions[["subscription_id", "account_id"]].drop_duplicates()
        feature_usage = feature_usage.merge(sub_id_map, on="subscription_id", how="left")
        logger.info("Mapped feature_usage subscription_id → account_id")

    usage_numeric = feature_usage.select_dtypes(include="number")
    usage_numeric["account_id"] = feature_usage["account_id"]
    # Drop subscription_id from aggregation if present
    agg_cols = [c for c in usage_numeric.columns if c not in ("account_id", "subscription_id")]
    usage_agg = usage_numeric.groupby("account_id")[agg_cols].agg(["mean", "sum", "std"]).reset_index()
    # Flatten multi-level columns
    usage_agg.columns = [
        f"usage_{col[0]}_{col[1]}" if col[1] else col[0]
        for col in usage_agg.columns
    ]

    abt = abt.merge(usage_agg, on="account_id", how="left")
    logger.info("After usage merge: %d rows, %d cols", len(abt), len(abt.columns))

    # --- 3. Aggregated support tickets ---
    ticket_agg = (
        support_tickets.groupby("account_id")
        .agg(
            ticket_count=("account_id", "count"),
            **{
                col: (col, "mean")
                for col in support_tickets.select_dtypes(include="number").columns
                if col != "account_id"
            },
        )
        .reset_index()
    )
    abt = abt.merge(ticket_agg, on="account_id", how="left")
    logger.info("After support merge: %d rows, %d cols", len(abt), len(abt.columns))

    # --- 4. Churn target ---
    churned_accounts = churn_events["account_id"].unique()
    abt["churn_flag"] = abt["account_id"].isin(churned_accounts).astype(int)
    logger.info(
        "Final ABT: %d rows, %d cols | churn_rate=%.2f%%",
        len(abt),
        len(abt.columns),
        abt["churn_flag"].mean() * 100,
    )

    return abt


# ---------------------------------------------------------------------------
# Pipeline Entrypoint
# ---------------------------------------------------------------------------
def run_ingestion(
    raw_dir: str = "data/raw",
    output_path: Optional[str] = "data/processed/abt_merged.csv",
) -> pd.DataFrame:
    """
    Full ingestion pipeline: load → validate → merge → save.

    Parameters
    ----------
    raw_dir : str
        Path to raw CSV directory.
    output_path : str or None
        If provided, saves the merged ABT to this path.

    Returns
    -------
    pd.DataFrame
        The merged Analytical Base Table.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    tables = load_all_tables(raw_dir)
    validate_schemas(tables)
    abt = merge_abt(tables)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        abt.to_csv(out, index=False)
        logger.info("Saved ABT to %s", out)

    logger.info("INGESTION COMPLETE")
    return abt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Path to raw CSV directory"
    )
    parser.add_argument(
        "--output",
        default="data/processed/abt_merged.csv",
        help="Output path for merged ABT",
    )
    args = parser.parse_args()
    run_ingestion(raw_dir=args.raw_dir, output_path=args.output)
