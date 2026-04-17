import requests

test_cases = [
    {"name": "Loyal Enterprise", "data": {"tenure_months": 36, "mrr": 2000.0, "seats_purchased": 25, "feature_adoption_rate": 0.90, "days_since_last_activity": 1, "support_tickets_count": 0, "avg_resolution_days": 0, "plan_type": "Enterprise", "customer_segment": "Enterprise", "near_renewal": 0, "has_discount": 0, "age_group": "36-50", "region": "North America", "customer_id": "LOYAL-001"}},
    {"name": "Engaged Mid-Mkt", "data": {"tenure_months": 12, "mrr": 800.0, "seats_purchased": 10, "feature_adoption_rate": 0.65, "days_since_last_activity": 3, "support_tickets_count": 2, "avg_resolution_days": 4.0, "plan_type": "Pro", "customer_segment": "Mid-Market", "near_renewal": 0, "has_discount": 0, "age_group": "26-35", "region": "Europe", "customer_id": "MID-002"}},
    {"name": "Cooling Off SMB", "data": {"tenure_months": 6, "mrr": 150.0, "seats_purchased": 3, "feature_adoption_rate": 0.30, "days_since_last_activity": 25, "support_tickets_count": 5, "avg_resolution_days": 8.0, "plan_type": "Basic", "customer_segment": "SMB", "near_renewal": 1, "has_discount": 1, "age_group": "26-35", "region": "LATAM", "customer_id": "COOL-003"}},
    {"name": "High Risk Churn", "data": {"tenure_months": 2, "mrr": 30.0, "seats_purchased": 1, "feature_adoption_rate": 0.10, "days_since_last_activity": 60, "support_tickets_count": 10, "avg_resolution_days": 15.0, "plan_type": "Basic", "customer_segment": "SMB", "near_renewal": 1, "has_discount": 0, "age_group": "18-25", "region": "LATAM", "customer_id": "CHURN-004"}},
]

for tc in test_cases:
    r = requests.post("http://127.0.0.1:8080/predict", json=tc["data"])
    res = r.json()
    pred = res.get("churn_prediction", "?")
    prob = res.get("churn_probability", 0)
    risk = res.get("risk_tier", "?")
    label = "WILL CHURN" if pred == 1 else "WILL STAY"
    print(f"{tc['name']:20s} -> {label:10s}  Prob={prob*100:5.1f}%  Risk={risk}")
    if 'top_risk_signals' in res:
        for s in res['top_risk_signals'][:2]:
            print(f"                       - {s['feature']}: {s['impact']}")
    print()
