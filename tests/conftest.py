import pytest
import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Local imports
PROJECT_ROOT = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
sys.path.append(PROJECT_ROOT)

from src.api import app
from src.models.train_utils import get_training_data, encode_protected

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with startup/shutdown events triggered."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def production_model():
    """Load the Production XGBoost Mitigated model."""
    path = os.path.join(PROJECT_ROOT, "models", "production_model.pkl")
    if not os.path.exists(path):
        pytest.skip(f"Production model not found at {path}")
    return joblib.load(path)

@pytest.fixture(scope="session")
def preprocessor():
    """Load the production preprocessing pipeline."""
    path = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.pkl")
    if not os.path.exists(path):
        pytest.skip(f"Preprocessor not found at {path}")
    return joblib.load(path)

@pytest.fixture(scope="session")
def saas_bundle():
    """Returns a sampled test bundle for SaaS dataset."""
    X_train_bal, y_train_bal, X_test, y_test, p_train, p_test, feature_names = get_training_data('saas')
    # Sampling for speed (first 500 rows) and resetting index to avoid mismatch
    return {
        'X_test': X_test[:500].reset_index(drop=True) if hasattr(X_test, 'reset_index') else X_test[:500],
        'y_test': np.array(y_test[:500]), # Ensure numpy array
        'p_test': p_test[:500].reset_index(drop=True),
        'feature_names': feature_names,
        'X_train': X_train_bal[:10]
    }

@pytest.fixture(scope="session")
def bank_bundle():
    """Returns a sampled test bundle for Bank dataset."""
    X_train_bal, y_train_bal, X_test, y_test, p_train, p_test, feature_names = get_training_data('bank')
    return {
        'X_test': X_test[:500].reset_index(drop=True) if hasattr(X_test, 'reset_index') else X_test[:500],
        'y_test': np.array(y_test[:500]),
        'p_test': p_test[:500].reset_index(drop=True),
        'feature_names': feature_names
    }

@pytest.fixture(scope="session")
def valid_payload():
    """A valid customer input payload for API testing."""
    return {
        "tenure_months": 24.5,
        "mrr": 1500.0,
        "seats_purchased": 10,
        "feature_adoption_rate": 0.85,
        "days_since_last_activity": 2.0,
        "support_tickets_count": 1,
        "avg_resolution_days": 1.5,
        "plan_type": "enterprise",
        "customer_segment": "strategic",
        "near_renewal": 0,
        "has_discount": 1,
        "age_group": "25-34",
        "region": "NA",
        "customer_id": "cust123"
    }
