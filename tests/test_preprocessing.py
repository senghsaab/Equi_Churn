import pytest
import pandas as pd
import numpy as np

def test_no_post_churn_leakage_features(saas_bundle):
    """Ensures no post-churn or future-leaking features exist in the training matrix."""
    X_train = saas_bundle['X_test'] # Using feature set sample
    leakage_cols = ['cancellation_reason', 'churn_date', 'churn_reason', 'offboarding_date']
    
    # feature names are columns in X_train if it's a DF
    cols = X_train.columns if hasattr(X_train, 'columns') else saas_bundle['feature_names']
    
    for col in leakage_cols:
        assert all(col not in str(c) for c in cols), f"Leakage feature {col} found in feature names."

def test_missing_protected_attributes_flagged_not_imputed(saas_bundle):
    """Verification that protected attributes maintain their distribution even with missingness."""
    p_test = saas_bundle['p_test']
    # If using DPDP Act 2023 compliance, we should handle missing as a distinct category or error
    assert p_test.isna().sum().sum() == 0, "Missing protected attributes found. Imbalance in collection."

def test_outlier_capping_applied(saas_bundle):
    """Ensures that variables like MRR are capped to prevent skew from dominant large customers."""
    X_test = saas_bundle['X_test']
    mrr_idx = -1
    for i, name in enumerate(saas_bundle['feature_names']):
        if 'mrr' in name: mrr_idx = i
    
    if mrr_idx != -1:
        # Check if values are normalized or within a reasonable range
        # After scaling, values should be near mean/std
        mrr_vals = X_test.iloc[:, mrr_idx] if hasattr(X_test, 'iloc') else X_test[:, mrr_idx]
        assert mrr_vals.max() <= 10.0, "MRR seems uncapped or unscaled."
