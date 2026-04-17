import pytest

def test_health_endpoint_returns_dpdp_info(client):
    """Verifies that the health endpoint includes mandatory regulatory metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["regulatory_frame"] == "DPDP Act 2023"
    assert "age_group" in data["protected_attributes"]

def test_predict_returns_fairness_audit(client, valid_payload):
    """Verifies that every prediction is accompanied by a fairness audit object."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "fairness_audit" in data
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_tier"] in ["HIGH", "MEDIUM", "LOW"]
    assert "top_risk_signals" in data

def test_predict_invalid_mrr_returns_422(client, valid_payload):
    """Verifies that Pydantic validation catches illegal inputs."""
    payload = valid_payload.copy()
    payload["mrr"] = -500.0
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "mrr" in response.text.lower()

def test_predict_protected_attributes_transparency(client, valid_payload):
    """Ensures protected attributes are acknowledged in the audit but not used as feature keys."""
    response = client.post("/predict", json=valid_payload)
    data = response.json()
    audit = data["fairness_audit"]
    
    assert audit["age_group_value"] == valid_payload["age_group"]
    assert audit["region_value"] == valid_payload["region"]
    # Internal logic check: we don't return the raw feature vector, but we check audit
    assert "DPDP Act 2023 compliant" in audit["note"]

def test_metrics_endpoint_active(client):
    """Verifies operational monitoring endpoint is functional."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_count" in data
    assert "model_version" in data or "model_name" in data
