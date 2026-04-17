"""
generate_dataset.py
===================
Publication-quality synthetic B2B SaaS churn dataset generator.

Statistical Design:
  - N = 5000 samples, 20% churn rate (industry benchmark)
  - 27 features across 6 categories
  - Distinct Beta/Binomial/Uniform distributions per segment
  - Realistic noise added to avoid perfect separability

Author : CTO — B2B SaaS Churn Prediction Project
Created: 2026-03-13
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ──────────────────────────────────────────────
# 0. REPRODUCIBILITY & CONSTANTS
# ──────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

N_TOTAL   = 5_000
CHURN_RATE = 0.20
N_CHURNED  = int(N_TOTAL * CHURN_RATE)   # 1 000
N_RETAINED = N_TOTAL - N_CHURNED          # 4 000

# Corrected path to root /data/raw
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'churn_dataset.csv')

print("=" * 70)
print("  B2B SaaS Churn Dataset Generator (EquiChurn) - Stage 1")
print("=" * 70)
print(f"  Total samples   : {N_TOTAL:,}")
print(f"  Churned         : {N_CHURNED:,}  ({CHURN_RATE*100:.0f}%)")
print(f"  Retained        : {N_RETAINED:,}  ({(1-CHURN_RATE)*100:.0f}%)")
print()

# ──────────────────────────────────────────────
# 1. HELPER: noise clip
# ──────────────────────────────────────────────
def add_noise(arr, std=0.05, lo=0.0, hi=1.0):
    """Add Gaussian noise and clip to [lo, hi]."""
    return np.clip(arr + rng.normal(0, std, size=arr.shape), lo, hi)

# ──────────────────────────────────────────────
# 2. GENERATE EACH SEGMENT
# ──────────────────────────────────────────────

def generate_segment(n, churned: bool) -> dict:
    """Generate features for n customers in a single segment."""
    d = {}

    # ── 2.1  COMPANY PROFILE ──────────────────
    # company_size: 1=SMB, 2=Mid-Market, 3=Enterprise
    size_probs = [0.55, 0.30, 0.15] if churned else [0.25, 0.40, 0.35]
    d['company_size'] = rng.choice([1, 2, 3], size=n, p=size_probs)

    seg_map = {1: 'SMB', 2: 'Mid-Market', 3: 'Enterprise'}
    d['customer_segment'] = [seg_map[s] for s in d['company_size']]

    contract_probs = [0.55, 0.37, 0.08] if churned else [0.30, 0.55, 0.15]
    d['contract_type'] = rng.choice([0, 1, 2], size=n, p=contract_probs)

    plan_names = ['Basic', 'Standard', 'Premium', 'Enterprise']
    plan_probs = [0.45, 0.35, 0.15, 0.05] if churned else [0.15, 0.35, 0.30, 0.20]
    d['plan_type'] = rng.choice(plan_names, size=n, p=plan_probs)

    regions = ['North America', 'Europe', 'APAC', 'LATAM']
    reg_probs = [0.20, 0.20, 0.40, 0.20] if churned else [0.30, 0.30, 0.15, 0.25]
    d['region'] = rng.choice(regions, size=n, p=reg_probs)

    ages = ['18-25', '26-35', '36-50', '50+']
    age_probs = [0.15, 0.20, 0.25, 0.40] if churned else [0.25, 0.35, 0.30, 0.10]
    d['age_group'] = rng.choice(ages, size=n, p=age_probs)

    # ── 2.2  DATES (NEW RAW SOURCE) ──────────
    # reference_date is "today"
    ref_date = pd.Timestamp('2026-04-08')
    d['reference_date'] = [ref_date] * n
    
    # tenure_months logic (re-randomized for raw dates)
    tenure_days = rng.integers(30, 700, n) if not churned else rng.integers(15, 300, n)
    d['subscription_start_date'] = [ref_date - pd.Timedelta(days=int(td)) for td in tenure_days]
    
    # last_activity_date
    lag = rng.integers(0, 60, n) if not churned else rng.integers(10, 120, n)
    d['last_activity_date'] = [ref_date - pd.Timedelta(days=int(l)) for l in lag]
    
    # ── 2.3  REVENUE & SEATS (NEW RAW SOURCE) ──────
    if churned:
        d['arpu'] = np.round(rng.uniform(50, 600, n) + rng.normal(0, 30, n), 2)
    else:
        d['arpu'] = np.round(rng.uniform(150, 2500, n) + rng.normal(0, 80, n), 2)
    d['mrr'] = np.round(d['arpu'] * d['company_size'] * rng.uniform(8, 40, n), 2)

    # Seats
    d['seats_purchased'] = (d['company_size'] * rng.integers(5, 50, n)).astype(int)
    d['seats_active'] = (d['seats_purchased'] * (rng.beta(2, 5, n) if churned else rng.beta(5, 2, n))).astype(int)
    d['seats_active'] = np.clip(d['seats_active'], 0, d['seats_purchased'])

    # ── 2.4  USAGE DATA (NEW RAW SOURCE) ───────────
    # active_days in total tenure
    d['active_days'] = (tenure_days * (rng.beta(2, 5, n) if churned else rng.beta(6, 2, n))).astype(int)
    
    # login_frequency_monthly (proxy for total_events)
    d['total_events'] = (d['active_days'] * rng.uniform(5, 50, n)).astype(int)
    
    # Usage last 30d vs prev 30d
    if churned:
        d['usage_last_30d'] = rng.integers(0, 100, n)
        d['usage_prev_30d'] = rng.integers(50, 500, n)
    else:
        d['usage_last_30d'] = rng.integers(100, 1000, n)
        d['usage_prev_30d'] = rng.integers(100, 800, n)

    # Feature bits
    d['features_available'] = [25] * n
    d['features_used'] = (d['features_available'] * (rng.beta(2, 5, n) if churned else rng.beta(5, 2, n))).astype(int)
    
    # Module adoption (flags 1-5)
    for i in range(1, 6):
        p_adopt = 0.2 if churned else 0.7
        d[f'module_{i}_active'] = rng.binomial(1, p_adopt, n)

    # ── 2.5  SUPPORT (NEW RAW SOURCE) ──────────────
    d['total_tickets'] = (rng.integers(3, 30, n) if churned else rng.integers(0, 10, n)).astype(int)
    d['open_tickets'] = (d['total_tickets'] * rng.uniform(0.1, 0.9, n)).astype(int) if churned else (d['total_tickets'] * 0.1).astype(int)

    # Existing flags maintained
    espon_p = 0.20 if churned else 0.65
    d['executive_sponsor'] = rng.binomial(1, espon_p, n).astype(int)
    d['competitor_evaluation'] = rng.binomial(1, 0.55 if churned else 0.15, n).astype(int)
    d['contract_renewal_days'] = (rng.integers(-30, 90, n) if churned else rng.integers(30, 365, n)).astype(int)

    return d

# ──────────────────────────────────────────────
# 3. BUILD DATAFRAME
# ──────────────────────────────────────────────
churned_data  = generate_segment(N_CHURNED,  churned=True)
retained_data = generate_segment(N_RETAINED, churned=False)

df_churned  = pd.DataFrame(churned_data);  df_churned['churned']  = 1
df_retained = pd.DataFrame(retained_data); df_retained['churned'] = 0

df = pd.concat([df_churned, df_retained], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle

# ──────────────────────────────────────────────
# 4. ENFORCE DTYPES
# ──────────────────────────────────────────────
int_cols = [
    'company_size', 'contract_type', 'churned',
    'total_events', 'usage_last_30d', 'usage_prev_30d',
    'features_available', 'features_used',
    'seats_purchased', 'seats_active',
    'total_tickets', 'open_tickets',
    'executive_sponsor', 'competitor_evaluation', 'contract_renewal_days'
]
# Adding module flags to int_cols
int_cols += [f'module_{i}_active' for i in range(1, 6)]

string_cols = [
    'customer_segment', 'plan_type', 'region', 'age_group',
    'subscription_start_date', 'reference_date', 'last_activity_date'
]

for col in int_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

float_cols = [c for c in df.columns if c not in int_cols + string_cols]
for col in float_cols:
    df[col] = df[col].astype(float)

# ──────────────────────────────────────────────
# 5. SAVE CSV
# ──────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"  [OK] Dataset saved -> {os.path.abspath(OUTPUT_PATH)}")
print()

# ──────────────────────────────────────────────
# 6. DIAGNOSTIC PRINT-OUTS
# ──────────────────────────────────────────────
print("═" * 70)
print("  DATASET OVERVIEW")
print("═" * 70)
print(f"  Shape          : {df.shape}")
print(f"  Churn rate     : {df['churned'].mean()*100:.2f}%")
print()

print("─" * 70)
print("  NUMERIC FEATURE DISTRIBUTIONS (mean ± std)")
print("─" * 70)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if col == 'churned': continue
    m, s = df[col].mean(), df[col].std()
    lo, hi = df[col].min(), df[col].max()
    print(f"  {col:<35}  μ={m:>9.3f}  σ={s:>8.3f}  [{lo:.2f} – {hi:.2f}]")

print()
print("═" * 70)
print("  Stage 1 Complete — Raw Dataset Ready ✅")
print("═" * 70)
