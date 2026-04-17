"""
run_drift_check.py — Evidently Data Drift Detection for EquiChurn
=================================================================
Compares the training baseline distribution against current-week
prediction logs. Outputs PSI scores per feature and generates an
Evidently HTML report.

Used by: .github/workflows/evidently-monitoring.yml
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DRIFT_THRESHOLD = 0.20  # PSI threshold for alerting
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "monitoring_baseline.csv")
PREDICTIONS_LOG = os.path.join(PROJECT_ROOT, "logs", "prediction_audit.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def load_baseline():
    """Load the training baseline for drift comparison."""
    if os.path.exists(BASELINE_PATH):
        return pd.read_csv(BASELINE_PATH)

    # Fallback: Use processed training data as baseline
    fallback = os.path.join(PROJECT_ROOT, "data", "processed", "abt_merged.csv")
    if os.path.exists(fallback):
        print(f"[WARN] monitoring_baseline.csv not found, using {fallback}")
        df = pd.read_csv(fallback)
        # Save as baseline for future runs
        df.to_csv(BASELINE_PATH, index=False)
        return df

    print("[ERROR] No baseline data found. Create data/processed/monitoring_baseline.csv")
    sys.exit(1)


def load_current_predictions():
    """Load the current week's predictions from the audit log."""
    if not os.path.exists(PREDICTIONS_LOG):
        print("[WARN] No prediction audit log found. Generating synthetic current data.")
        # Return a small sample from baseline with noise (for CI demo)
        baseline = load_baseline()
        numeric_cols = baseline.select_dtypes(include=[np.number]).columns
        sample = baseline[numeric_cols].sample(min(200, len(baseline)), random_state=42).copy()
        # Add slight drift for testing
        for col in sample.columns[:3]:
            sample[col] = sample[col] * np.random.normal(1.02, 0.05, len(sample))
        return sample

    # Parse JSON-lines audit log
    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not records:
        print("[WARN] Audit log is empty.")
        return pd.DataFrame()

    return pd.DataFrame(records)


def compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index between two distributions."""
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def run_drift_report(baseline_df, current_df):
    """Run Evidently drift report and save HTML."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # Align columns
        common_cols = list(set(baseline_df.columns) & set(current_df.columns))
        numeric_cols = baseline_df[common_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            print("[WARN] No common numeric columns for drift comparison.")
            return

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=baseline_df[numeric_cols].dropna(),
            current_data=current_df[numeric_cols].dropna()
        )

        week_num = datetime.now().isocalendar()[1]
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(REPORTS_DIR, f"drift_week_{week_num}.html")
        report.save_html(report_path)
        print(f"[OK] Evidently report saved to {report_path}")

    except ImportError:
        print("[WARN] Evidently not installed. Skipping HTML report generation.")
    except Exception as e:
        print(f"[WARN] Evidently report failed: {e}")


def main():
    print("=" * 60)
    print("  EquiChurn Data Drift Detection")
    print("=" * 60)

    baseline = load_baseline()
    current = load_current_predictions()

    if current.empty:
        print("[SKIP] No current data to compare. Exiting.")
        return

    # Align to common numeric columns
    common_cols = list(set(baseline.columns) & set(current.columns))
    numeric_cols = [c for c in common_cols
                    if pd.api.types.is_numeric_dtype(baseline[c])
                    and pd.api.types.is_numeric_dtype(current[c])]

    if not numeric_cols:
        print("[SKIP] No overlapping numeric features between baseline and current.")
        return

    print(f"\nAnalyzing {len(numeric_cols)} features for drift...")
    print("-" * 50)

    drifted_features = []
    for col in sorted(numeric_cols):
        psi = compute_psi(baseline[col].dropna().values, current[col].dropna().values)
        status = "DRIFT" if psi > DRIFT_THRESHOLD else "OK"
        if psi > DRIFT_THRESHOLD:
            drifted_features.append((col, psi))
        print(f"  {col:40s}  PSI={psi:.4f}  [{status}]")

    print("-" * 50)

    if drifted_features:
        print(f"\nDRIFT_ALERT: {len(drifted_features)} feature(s) exceeded PSI threshold ({DRIFT_THRESHOLD})")
        print("\nDrifted features:")
        for feat, psi in drifted_features:
            print(f"  - {feat}: PSI={psi:.4f}")
        print("\nRecommendation: Trigger manual retrain review.")
    else:
        print(f"\n[OK] No significant drift detected (threshold={DRIFT_THRESHOLD})")

    # Generate Evidently HTML report
    run_drift_report(baseline, current)

    print("\nDrift check complete.")


if __name__ == "__main__":
    main()
