import json
import os
from datetime import datetime
import threading

AUDIT_LOG_PATH = "logs/prediction_audit.json"
os.makedirs("logs", exist_ok=True)

# Thread-safe counters
prediction_counter = 0
high_risk_counter = 0
fairness_flag_counter = 0
total_prob_sum = 0.0
lock = threading.Lock()

def log_prediction(customer_id, probability, protected, risk_tier):
    """
    Logs prediction metadata to a local JSON file for auditing.
    """
    global prediction_counter, high_risk_counter, fairness_flag_counter, total_prob_sum
    
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "customer_id": customer_id,
        "probability": probability,
        "protected_attributes": protected,
        "risk_tier": risk_tier
    }
    
    # Update global counters
    with lock:
        prediction_counter += 1
        total_prob_sum += probability
        if risk_tier == "HIGH":
            high_risk_counter += 1
            
    # Append to file
    try:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error logging prediction: {e}")

def get_stats():
    """Returns current API statistics."""
    with lock:
        avg_prob = total_prob_sum / prediction_counter if prediction_counter > 0 else 0.0
        return {
            "prediction_count": prediction_counter,
            "avg_probability": round(avg_prob, 4),
            "high_risk_count": high_risk_counter,
            "fairness_flags_triggered": fairness_flag_counter
        }
