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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'churn_dataset.csv')

print("=" * 70)
print("  B2B SaaS Churn Dataset Generator — Stage 1")
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
    # Churned skew toward SMB, Retained toward Enterprise
    size_probs = [0.55, 0.30, 0.15] if churned else [0.25, 0.40, 0.35]
    d['company_size'] = rng.choice([1, 2, 3], size=n, p=size_probs)

    # contract_type: 0=Monthly, 1=Annual, 2=Multi-Year
    contract_probs = [0.55, 0.37, 0.08] if churned else [0.30, 0.55, 0.15]
    d['contract_type'] = rng.choice([0, 1, 2], size=n, p=contract_probs)

    # onboarding_completion_pct: churned Low (Beta(2,6)), retained High (Beta(6,3))
    if churned:
        d['onboarding_completion_pct'] = np.round(
            add_noise(rng.beta(2, 6, n) * 100, std=5, lo=10, hi=95), 1)
    else:
        d['onboarding_completion_pct'] = np.round(
            add_noise(rng.beta(6, 3, n) * 100, std=5, lo=40, hi=100), 1)

    # ── 2.2  REVENUE SIGNALS ───────────────────
    # arpu  (Average Revenue Per User, USD/month)
    if churned:
        d['arpu'] = np.round(rng.uniform(50, 600, n) + rng.normal(0, 30, n), 2)
    else:
        d['arpu'] = np.round(rng.uniform(150, 2500, n) + rng.normal(0, 80, n), 2)
    d['arpu'] = np.clip(d['arpu'], 20, 3000)

    # mrr  (Monthly Recurring Revenue, USD) — correlated with company_size × arpu
    d['mrr'] = np.round(d['arpu'] * d['company_size'] * rng.uniform(8, 40, n), 2)

    # expansion_mrr (USD/month — upsells)
    if churned:
        d['expansion_mrr'] = np.round(rng.uniform(0, 200, n), 2)
    else:
        d['expansion_mrr'] = np.round(rng.uniform(0, 1500, n), 2)

    # payment_failures_6m: Binomial — churned higher failure probability
    p_fail = 0.35 if churned else 0.05
    d['payment_failures_6m'] = rng.binomial(8, p_fail, n).astype(int)

    # ── 2.3  PRODUCT USAGE ────────────────────
    # tenure_months
    if churned:
        d['tenure_months'] = rng.integers(1, 24, size=n)
    else:
        d['tenure_months'] = rng.integers(3, 72, size=n)

    # dau_wau_ratio: Churned Beta(2,8) ~ low engagement
    if churned:
        d['dau_wau_ratio'] = np.round(
            add_noise(rng.beta(2, 8, n), std=0.05, lo=0.01, hi=0.90), 3)
    else:
        d['dau_wau_ratio'] = np.round(
            add_noise(rng.beta(5, 5, n), std=0.05, lo=0.05, hi=0.99), 3)

    # feature_adoption_rate (0–1)
    if churned:
        d['feature_adoption_rate'] = np.round(
            add_noise(rng.beta(2, 5, n), std=0.05, lo=0.0, hi=0.8), 3)
    else:
        d['feature_adoption_rate'] = np.round(
            add_noise(rng.beta(5, 3, n), std=0.05, lo=0.1, hi=1.0), 3)

    # login_frequency_monthly (sessions/month)
    if churned:
        d['login_frequency_monthly'] = np.round(
            rng.uniform(1, 15, n) + rng.normal(0, 2, n), 1)
    else:
        d['login_frequency_monthly'] = np.round(
            rng.uniform(8, 60, n) + rng.normal(0, 5, n), 1)
    d['login_frequency_monthly'] = np.clip(d['login_frequency_monthly'], 0, 90)

    # api_calls_monthly
    if churned:
        d['api_calls_monthly'] = np.round(
            rng.uniform(0, 5_000, n) + rng.normal(0, 200, n))
    else:
        d['api_calls_monthly'] = np.round(
            rng.uniform(1_000, 100_000, n) + rng.normal(0, 2_000, n))
    d['api_calls_monthly'] = np.clip(d['api_calls_monthly'], 0, 150_000)

    # integrations_active
    if churned:
        d['integrations_active'] = rng.integers(0, 4, size=n)
    else:
        d['integrations_active'] = rng.integers(1, 12, size=n)

    # seats_utilized_pct (%)
    if churned:
        d['seats_utilized_pct'] = np.round(
            add_noise(rng.beta(2, 5, n) * 100, std=8, lo=5, hi=85), 1)
    else:
        d['seats_utilized_pct'] = np.round(
            add_noise(rng.beta(5, 2, n) * 100, std=8, lo=30, hi=100), 1)

    # usage_trend_3m  — slope of usage over 3 months
    if churned:
        d['usage_trend_3m'] = np.round(
            rng.uniform(-1.0, 0.2, n) + rng.normal(0, 0.1, n), 3)
    else:
        d['usage_trend_3m'] = np.round(
            rng.uniform(-0.3, 1.0, n) + rng.normal(0, 0.1, n), 3)

    # usage_trend_6m  — slope over 6 months (correlated with 3m)
    d['usage_trend_6m'] = np.round(
        d['usage_trend_3m'] * rng.uniform(0.7, 1.3, n) + rng.normal(0, 0.05, n), 3)

    # ── 2.4  SUPPORT / CX ────────────────────
    # support_tickets_6m
    if churned:
        d['support_tickets_6m'] = rng.integers(3, 21, size=n)
    else:
        d['support_tickets_6m'] = rng.integers(0, 8, size=n)

    # avg_ticket_resolution_days
    if churned:
        d['avg_ticket_resolution_days'] = np.round(
            rng.uniform(3, 15, n) + rng.normal(0, 1.5, n), 1)
    else:
        d['avg_ticket_resolution_days'] = np.round(
            rng.uniform(0.5, 5, n) + rng.normal(0, 0.5, n), 1)
    d['avg_ticket_resolution_days'] = np.clip(d['avg_ticket_resolution_days'], 0.2, 20)

    # nps_score (0–10 Likert)
    if churned:
        d['nps_score'] = rng.integers(0, 8, size=n)    # 0–7
    else:
        d['nps_score'] = rng.integers(5, 11, size=n)   # 5–10

    # csat_score (1–5)
    if churned:
        d['csat_score'] = np.round(
            add_noise(rng.beta(2, 5, n) * 4 + 1, std=0.3, lo=1, hi=5), 1)
    else:
        d['csat_score'] = np.round(
            add_noise(rng.beta(5, 2, n) * 4 + 1, std=0.2, lo=1, hi=5), 1)

    # ── 2.5  RELATIONSHIP HEALTH ──────────────
    # executive_sponsor: 0=No, 1=Yes
    espon_p = 0.20 if churned else 0.65
    d['executive_sponsor'] = rng.binomial(1, espon_p, n).astype(int)

    # decision_maker_change (# of DM changes in last 12m)
    dm_p = 0.45 if churned else 0.15
    d['decision_maker_change'] = rng.binomial(3, dm_p, n).astype(int)

    # qbr_meetings_yearly (Quarterly Business Reviews)
    if churned:
        d['qbr_meetings_yearly'] = rng.integers(0, 3, size=n)
    else:
        d['qbr_meetings_yearly'] = rng.integers(1, 5, size=n)

    # health_score (composite, 0–100)
    if churned:
        d['health_score'] = np.round(
            add_noise(rng.uniform(10, 55, n), std=5, lo=5, hi=60), 1)
    else:
        d['health_score'] = np.round(
            add_noise(rng.uniform(50, 100, n), std=5, lo=40, hi=100), 1)

    # customer_growth_pct (YoY revenue growth of the customer's company, %)
    if churned:
        d['customer_growth_pct'] = np.round(
            rng.uniform(-30, 10, n) + rng.normal(0, 3, n), 2)
    else:
        d['customer_growth_pct'] = np.round(
            rng.uniform(-5, 80, n) + rng.normal(0, 5, n), 2)

    # competitor_evaluation: 0=No, 1=Yes (known competitive review)
    comp_p = 0.55 if churned else 0.15
    d['competitor_evaluation'] = rng.binomial(1, comp_p, n).astype(int)

    # ── 2.6  FINANCIAL / CONTRACT SIGNALS ─────
    # contract_renewal_days (-N = already churned / overdue, +N = days until renewal)
    if churned:
        d['contract_renewal_days'] = rng.integers(-30, 90, size=n)
    else:
        d['contract_renewal_days'] = rng.integers(30, 365, size=n)

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
    'company_size', 'contract_type', 'tenure_months',
    'integrations_active', 'support_tickets_6m', 'nps_score',
    'executive_sponsor', 'decision_maker_change', 'qbr_meetings_yearly',
    'payment_failures_6m', 'competitor_evaluation',
    'contract_renewal_days', 'churned'
]
for col in int_cols:
    df[col] = df[col].astype(int)

float_cols = [c for c in df.columns if c not in int_cols]
for col in float_cols:
    df[col] = df[col].astype(float)

# ──────────────────────────────────────────────
# 5. SAVE CSV
# ──────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"  ✅  Dataset saved → {os.path.abspath(OUTPUT_PATH)}")
print()

# ──────────────────────────────────────────────
# 6. DIAGNOSTIC PRINT-OUTS
# ──────────────────────────────────────────────
print("═" * 70)
print("  DATASET OVERVIEW")
print("═" * 70)
print(f"  Shape          : {df.shape}")
print(f"  Churn rate     : {df['churned'].mean()*100:.2f}%")
print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

print("─" * 70)
print("  FEATURE DISTRIBUTIONS (mean ± std)")
print("─" * 70)
for col in df.columns[:-1]:
    m, s = df[col].mean(), df[col].std()
    lo, hi = df[col].min(), df[col].max()
    print(f"  {col:<35}  μ={m:>9.3f}  σ={s:>8.3f}  [{lo:.2f} – {hi:.2f}]")

print()
print("─" * 70)
print("  PEARSON CORRELATION WITH CHURN LABEL (|r| descending)")
print("─" * 70)
corr_results = []
y = df['churned'].values
for col in df.columns[:-1]:
    r, p = pearsonr(df[col].values, y)
    corr_results.append((col, r, p))

corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
for col, r, p in corr_results:
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    direction = '↑ churn' if r > 0 else '↓ churn'
    print(f"  {col:<35}  r={r:+.4f}  {direction}  {sig}")

print()
print("═" * 70)
print("  Stage 1 Complete — Dataset Ready ✅")
print("═" * 70)
