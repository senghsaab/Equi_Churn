# Data Dictionary — B2B SaaS Churn Prediction Dataset

**Version**: 1.0 | **Stage**: 1 | **Samples**: 5,000 | **Churn Rate**: 20%

---

## Category 1: Company Profile

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `company_size` | int | 1=SMB, 2=Mid-Market, 3=Enterprise | Customer's company tier. SMB customers churn most (~55% on monthly contracts). |
| `contract_type` | int | 0=Monthly, 1=Annual, 2=Multi-Year | Billing cadence. Monthly = highest churn risk (no lock-in). |
| `onboarding_completion_pct` | float | 10–100 | % of onboarding steps completed. Low completion predicts early churn. |

---

## Category 2: Revenue Signals

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `arpu` | float | 20–3000 USD | Average Revenue Per User per month. Low ARPU = lower switching cost. |
| `mrr` | float | 0–∞ USD | Monthly Recurring Revenue. Function of ARPU × company_size × seats. |
| `expansion_mrr` | float | 0–1500 USD | Upsell/cross-sell MRR added in period. High expansion = strong engagement. |
| `payment_failures_6m` | int | 0–8 | Count of failed payments in past 6 months. Strong financial distress signal. |

---

## Category 3: Product Usage

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `tenure_months` | int | 1–72 | Months as a customer. Short tenure = high churn probability. |
| `dau_wau_ratio` | float | 0.01–0.99 | Daily Active Users / Weekly Active Users. Core stickiness metric. Low = disengaged. |
| `feature_adoption_rate` | float | 0–1 | Fraction of available features used. Higher = deeper product dependency. |
| `login_frequency_monthly` | float | 0–90 | Average monthly logins per user. Proxy for daily engagement. |
| `api_calls_monthly` | float | 0–150,000 | API call volume per month. Technical integration depth; dropping = churn signal. |
| `integrations_active` | int | 0–12 | Number of 3rd-party integrations active. Each integration increases switching cost. |
| `seats_utilized_pct` | float | 5–100 | % of purchased seats actively used. Low utilization = perceived low ROI. |
| `usage_trend_3m` | float | -1 to +1 | Normalized slope of usage over 3 months. Negative = declining engagement. |
| `usage_trend_6m` | float | -1 to +1 | Normalized slope of usage over 6 months. Longer-term momentum indicator. |

---

## Category 4: Support & Customer Experience

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `support_tickets_6m` | int | 0–20 | Number of support tickets in 6 months. High ticket volume = product friction. |
| `avg_ticket_resolution_days` | float | 0.2–20 | Mean days to resolve support tickets. Slow resolution erodes customer trust. |
| `nps_score` | int | 0–10 | Net Promoter Score (0–10 Likert). ≤6 = detractor; ≥9 = promoter. |
| `csat_score` | float | 1–5 | Customer Satisfaction score. Reflects recent support/product experience. |

---

## Category 5: Relationship Health

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `executive_sponsor` | int | 0=No, 1=Yes | Whether customer has an executive champion for the product internally. Huge retention lever. |
| `decision_maker_change` | int | 0–3 | # of decision-maker changes in last 12mo. New DMs often re-evaluate incumbent tools. |
| `qbr_meetings_yearly` | int | 0–4 | Quarterly Business Reviews held per year. Proactive CSM engagement indicator. |
| `health_score` | float | 5–100 | Composite Customer Health Score (usage + NPS + support + financials). Primary churn predictor. |
| `customer_growth_pct` | float | -30 to +80 | Customer's own YoY revenue growth. Declining customers cut SaaS spend first. |
| `competitor_evaluation` | int | 0=No, 1=Yes | Known active evaluation of a competitor. Highest short-term churn risk flag. |

---

## Category 6: Financial & Contract Signals

| Feature | Type | Range | Business Meaning |
|---|---|---|---|
| `contract_renewal_days` | int | -30 to +365 | Days until contract renewal (negative = overdue/post-churn). Negative values = already at risk. |

---

## Target Variable

| Feature | Type | Values | Description |
|---|---|---|---|
| `churned` | int | 0=Retained, 1=Churned | Whether the customer churned in the observation period. |

---

## Statistical Design Notes

- **Churn segment**: Beta(2,8) for engagement metrics → right-skewed low, health_score uniform(10,55), NPS 0–7
- **Retained segment**: Beta(5,5) for engagement → symmetric moderate-high, health_score uniform(50,100), NPS 5–10
- **Noise**: Gaussian noise (σ=0.05) added to continuous features to prevent perfect separability
- **Seed**: `SEED=42` for full reproducibility
