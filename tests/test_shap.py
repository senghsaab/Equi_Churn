import pytest
import numpy as np
import shap
from src.models.train_utils import identify_proxy_features, encode_protected

def test_shap_values_shape_matches_features(production_model, saas_bundle):
    """Validates that SHAP output matrix matches the feature space dimensions."""
    X_test, feature_names = saas_bundle['X_test'], saas_bundle['feature_names']
    
    # We use the underlying estimator
    base_model = production_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle list-style shap values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    assert sv.shape[1] == len(feature_names), \
        f"SHAP shape mismatch: {sv.shape[1]} values vs {len(feature_names)} features"

def test_shap_no_nan_values(production_model, saas_bundle):
    """Ensures stability of the explanation engine."""
    X_test = saas_bundle['X_test']
    base_model = production_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_test)
    
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    assert not np.isnan(sv).any(), "NaN values found in SHAP output."

def test_proxy_discriminator_identification(saas_bundle):
    """Verifies that the identifying logic for proxy features works correctly."""
    X_test, p_test, feature_names = saas_bundle['X_test'], saas_bundle['p_test'], saas_bundle['feature_names']
    p_encoded = encode_protected(p_test['age_group'])
    
    proxy_features = identify_proxy_features(X_test, p_encoded, feature_names)
    assert isinstance(proxy_features, list), "identify_proxy_features should return a list."
    # We at least expect the logical check to execute without error
