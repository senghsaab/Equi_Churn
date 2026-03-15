"""
inference.py — Batch & Single-Account Inference
==================================================
Loads a saved model artifact and scores new customer data.

Usage:
    # Single account
    python -m src.inference --model models/gradient_boosting.pkl --input new_account.csv

    # Batch
    python -m src.inference --model models/gradient_boosting.pkl --input batch.csv --output scored.csv
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model_artifact(model_path: str) -> Dict[str, Any]:
    """
    Load a saved model artifact (model + scaler + feature names).

    Parameters
    ----------
    model_path : str
        Path to the .pkl file saved by train.py.

    Returns
    -------
    dict
        Keys: 'model', 'model_name', 'scaler', 'feature_names'.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    artifact = joblib.load(path)
    logger.info(
        "Loaded model: %s (features=%d)",
        artifact.get("model_name", "unknown"),
        len(artifact.get("feature_names", [])),
    )
    return artifact


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(
    artifact: Dict[str, Any],
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score accounts using a loaded model artifact.

    Parameters
    ----------
    artifact : dict
        Model artifact from load_model_artifact().
    X : pd.DataFrame
        Feature matrix (same columns as training).
    threshold : float
        Probability threshold for churn classification.

    Returns
    -------
    pd.DataFrame
        Original features plus:
            - churn_probability : float [0, 1]
            - churn_prediction  : int {0, 1}
            - risk_tier         : str {"Low", "Medium", "High", "Critical"}
    """
    model = artifact["model"]
    scaler = artifact.get("scaler")
    expected_features = artifact.get("feature_names", [])

    # Align features
    if expected_features:
        missing = set(expected_features) - set(X.columns)
        extra = set(X.columns) - set(expected_features)
        if missing:
            logger.warning(
                "Missing features (filled with 0): %s", missing
            )
            for col in missing:
                X[col] = 0
        if extra:
            logger.warning("Extra features (dropped): %s", extra)
        X = X[expected_features]

    # Scale
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )
    else:
        X_scaled = X

    # Predict
    results = X.copy()

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)[:, 1]
        results["churn_probability"] = probas
        results["churn_prediction"] = (probas >= threshold).astype(int)
    else:
        results["churn_probability"] = float("nan")
        results["churn_prediction"] = model.predict(X_scaled)

    # Risk tiering
    results["risk_tier"] = pd.cut(
        results["churn_probability"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    )

    logger.info(
        "Scored %d accounts | churn_predicted=%d (%.1f%%) | threshold=%.2f",
        len(results),
        results["churn_prediction"].sum(),
        results["churn_prediction"].mean() * 100,
        threshold,
    )
    return results


def predict_single(
    artifact: Dict[str, Any],
    account_features: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Score a single account (convenience wrapper).

    Parameters
    ----------
    artifact : dict
        Model artifact.
    account_features : dict
        Feature dict for one account.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Prediction results for the single account.
    """
    X = pd.DataFrame([account_features])
    result = predict(artifact, X, threshold=threshold)
    return result.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Risk Summary Report
# ---------------------------------------------------------------------------
def generate_risk_report(
    scored_df: pd.DataFrame,
    account_id_col: str = "account_id",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a CS-ready risk report sorted by churn probability.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of predict() function.
    account_id_col : str
        Column name for account identifier.
    output_path : str or None
        If provided, saves the report to CSV.

    Returns
    -------
    pd.DataFrame
        Risk report with account_id, probability, prediction, tier.
    """
    report_cols = [
        col for col in [account_id_col, "churn_probability", "churn_prediction", "risk_tier"]
        if col in scored_df.columns
    ]
    report = (
        scored_df[report_cols]
        .sort_values("churn_probability", ascending=False)
        .reset_index(drop=True)
    )

    # Summary stats
    tier_counts = report["risk_tier"].value_counts()
    logger.info("\n=== Risk Distribution ===")
    for tier in ["Critical", "High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        logger.info("  %-10s: %d accounts", tier, count)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
        logger.info("Saved risk report to %s", out)

    return report


# ---------------------------------------------------------------------------
# Pipeline Entrypoint
# ---------------------------------------------------------------------------
def run_inference(
    model_path: str = "models/gradient_boosting.pkl",
    input_path: str = "data/processed/new_accounts.csv",
    output_path: str = "reports/risk_report.csv",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Full inference pipeline:
        1. Load model artifact
        2. Load new account data
        3. Score all accounts
        4. Generate risk report

    Returns
    -------
    pd.DataFrame
        Scored accounts with risk tiers.
    """
    logger.info("=" * 60)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("=" * 60)

    artifact = load_model_artifact(model_path)
    new_data = pd.read_csv(input_path)
    logger.info("Loaded %d new accounts from %s", len(new_data), input_path)

    # Separate account_id if present (not a feature)
    account_ids = None
    if "account_id" in new_data.columns:
        account_ids = new_data["account_id"]
        new_data = new_data.drop(columns=["account_id"])

    # Drop target if accidentally included
    if "churn_flag" in new_data.columns:
        new_data = new_data.drop(columns=["churn_flag"])

    scored = predict(artifact, new_data, threshold=threshold)

    # Re-attach account_ids
    if account_ids is not None:
        scored.insert(0, "account_id", account_ids.values)

    report = generate_risk_report(scored, output_path=output_path)

    logger.info("INFERENCE COMPLETE")
    return scored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument(
        "--model",
        default="models/gradient_boosting.pkl",
        help="Path to saved model .pkl",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to new account data CSV",
    )
    parser.add_argument(
        "--output",
        default="reports/risk_report.csv",
        help="Output path for risk report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default 0.5)",
    )
    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
    )
