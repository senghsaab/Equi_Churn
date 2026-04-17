"""
============================================================
  EquiChurn - Live Demo Script
  Run this in VS Code terminal to see the full pipeline demo
============================================================
"""
import requests
import json
import time

API_URL = "http://127.0.0.1:8080"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_json(data, indent=4):
    print(json.dumps(data, indent=indent))

# -------------------------------------------------------
# 1. HEALTH CHECK
# -------------------------------------------------------
print_header("1. API HEALTH CHECK")
try:
    r = requests.get(f"{API_URL}/health")
    health = r.json()
    print(f"\n  Status         : {health['status']}")
    print(f"  Model          : {health['model']}")
    print(f"  Regulation     : {health['regulatory_frame']}")
    print(f"  Protected Attrs: {health['protected_attributes']}")
    print(f"\n  [OK] API is running!")
except Exception as e:
    print(f"\n  [ERROR] API not reachable: {e}")
    print("  Start the server first:")
    print("    python -m uvicorn src.api:app --host 127.0.0.1 --port 8080")
    exit(1)

# -------------------------------------------------------
# 2. PREDICT - LOW RISK CUSTOMER
# -------------------------------------------------------
print_header("2. PREDICT - LOW RISK CUSTOMER")

low_risk = {
    "tenure_months": 24,
    "mrr": 1200.0,
    "seats_purchased": 20,
    "feature_adoption_rate": 0.85,
    "days_since_last_activity": 2,
    "support_tickets_count": 1,
    "avg_resolution_days": 1.5,
    "plan_type": "Enterprise",
    "customer_segment": "Enterprise",
    "near_renewal": 0,
    "has_discount": 0,
    "age_group": "36-50",
    "region": "North America",
    "customer_id": "LOYAL-001"
}

print("\n  Sending LOW risk customer data...")
print(f"  - Tenure: {low_risk['tenure_months']} months")
print(f"  - MRR: ${low_risk['mrr']}")
print(f"  - Feature Adoption: {low_risk['feature_adoption_rate']*100}%")
print(f"  - Days Since Last Activity: {low_risk['days_since_last_activity']}")
print(f"  - Support Tickets: {low_risk['support_tickets_count']}")

r = requests.post(f"{API_URL}/predict", json=low_risk)
result = r.json()

print(f"\n  --- PREDICTION RESULT ---")
print(f"  Customer ID     : {result.get('customer_id', 'N/A')}")
print(f"  Churn Prediction: {'YES - WILL CHURN' if result.get('churn_prediction') == 1 else 'NO - WILL STAY'}")
print(f"  Churn Probability: {result.get('churn_probability', 0)*100:.1f}%")
print(f"  Risk Tier       : {result.get('risk_tier', 'N/A')}")

if 'fairness_audit' in result:
    audit = result['fairness_audit']
    print(f"\n  --- FAIRNESS AUDIT (DPDP Act 2023) ---")
    print(f"  Framework       : {audit.get('regulation', 'N/A')}")
    print(f"  Protected Attrs : {audit.get('protected_attributes_tracked', 'N/A')}")

time.sleep(1)

# -------------------------------------------------------
# 3. PREDICT - HIGH RISK CUSTOMER
# -------------------------------------------------------
print_header("3. PREDICT - HIGH RISK CUSTOMER")

high_risk = {
    "tenure_months": 3,
    "mrr": 50.0,
    "seats_purchased": 2,
    "feature_adoption_rate": 0.15,
    "days_since_last_activity": 45,
    "support_tickets_count": 8,
    "avg_resolution_days": 12.0,
    "plan_type": "Basic",
    "customer_segment": "SMB",
    "near_renewal": 1,
    "has_discount": 0,
    "age_group": "18-25",
    "region": "LATAM",
    "customer_id": "AT-RISK-002"
}

print("\n  Sending HIGH risk customer data...")
print(f"  - Tenure: {high_risk['tenure_months']} months")
print(f"  - MRR: ${high_risk['mrr']}")
print(f"  - Feature Adoption: {high_risk['feature_adoption_rate']*100}%")
print(f"  - Days Since Last Activity: {high_risk['days_since_last_activity']}")
print(f"  - Support Tickets: {high_risk['support_tickets_count']}")

r = requests.post(f"{API_URL}/predict", json=high_risk)
result = r.json()

print(f"\n  --- PREDICTION RESULT ---")
print(f"  Customer ID     : {result.get('customer_id', 'N/A')}")
print(f"  Churn Prediction: {'YES - WILL CHURN' if result.get('churn_prediction') == 1 else 'NO - WILL STAY'}")
print(f"  Churn Probability: {result.get('churn_probability', 0)*100:.1f}%")
print(f"  Risk Tier       : {result.get('risk_tier', 'N/A')}")

if 'fairness_audit' in result:
    audit = result['fairness_audit']
    print(f"\n  --- FAIRNESS AUDIT (DPDP Act 2023) ---")
    print(f"  Framework       : {audit.get('regulation', 'N/A')}")
    print(f"  Protected Attrs : {audit.get('protected_attributes_tracked', 'N/A')}")

if 'top_risk_signals' in result:
    print(f"\n  --- TOP RISK SIGNALS ---")
    for i, signal in enumerate(result['top_risk_signals'][:5], 1):
        print(f"  {i}. {signal.get('feature', 'N/A')} -> {signal.get('direction', 'N/A')} ({signal.get('impact', 'N/A')})")

time.sleep(1)

# -------------------------------------------------------
# 4. MONITORING METRICS
# -------------------------------------------------------
print_header("4. PREDICTION MONITORING METRICS")

try:
    r = requests.get(f"{API_URL}/metrics")
    metrics = r.json()
    print(f"\n  Total Predictions: {metrics.get('total_predictions', 'N/A')}")
    print(f"  Avg Probability  : {metrics.get('avg_churn_probability', 0)*100:.1f}%")
    print(f"  Churn Rate       : {metrics.get('churn_rate', 0)*100:.1f}%")
except:
    print("  Metrics endpoint not available")

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print_header("DEMO COMPLETE")
print("""
  What you just saw:
  ------------------
  1. Health check - API status and compliance info
  2. Low-risk customer  -> predicted as STAY (low churn probability)
  3. High-risk customer -> predicted as CHURN (high churn probability)
  4. Monitoring metrics  -> live prediction stats

  Key Features Demonstrated:
  - Real-time churn prediction
  - Per-prediction fairness auditing (DPDP Act 2023)
  - Protected attribute tracking (age_group, region)
  - Risk tier classification (LOW/MEDIUM/HIGH)
  - SHAP-based explainability (top risk signals)

  Try it yourself:
  - Open http://127.0.0.1:8080/docs in Chrome
  - Modify customer values and see predictions change!
""")
