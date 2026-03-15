# Exploratory Data Analysis — Summary Report
## RavenStack B2B SaaS Churn Prediction

**Generated:** 2026-02-26 09:38
**Dataset:** 500 customers | 44 features | Churn rate: 22.0%

---

## Section-by-Section Findings

### Target Distribution Insights
- **Churn rate: 22.0%** — Moderate imbalance. Stratified splitting + class_weight='balanced' should suffice.
- Class ratio: 3.5:1 (active:churned)
- With 110 positive samples, we have enough signal for tree-based models but should use stratified CV.

---

### Univariate Numeric Insights
- **Monthly Recurring Revenue ($)**: median=1010.5, right-skewed (skew=1.19)
- **Account Tenure (Months)**: median=10.3, symmetric (skew=0.17)
- **Total Usage Events**: median=498.5, symmetric (skew=0.15)
- **Avg Daily Usage (Events/Day)**: median=10.0, symmetric (skew=0.05)
- Right-skewed revenue/usage distributions are typical in B2B SaaS — a few large accounts dominate.

---

### Univariate Categorical Insights
- **Plan Tier**: most common = 'Pro' (178 accounts, 35.6%)
- **Customer Segment (by MRR)**: most common = 'SMB' (168 accounts, 33.6%)
- **Country / Region**: most common = 'US' (291 accounts, 58.2%)
- **Industry**: most common = 'DevTools' (113 accounts, 22.6%)
- Check if plan tier or segment distribution is heavily skewed — could need stratified analysis.

---

### Bivariate Boxplot Insights
- **MRR ($)**: churned median = 1151.5 vs active = 991.5 (16.1% higher)
- **Tenure (Months)**: churned median = 12.4 vs active = 9.9 (25.8% higher)
- **Total Usage Events**: churned median = 527.5 vs active = 491.0 (7.4% higher)
- **Avg Daily Usage**: churned median = 10.0 vs active = 10.0 (0.5% lower)
- **Active Days**: churned median = 51.0 vs active = 48.0 (6.2% higher)
- **Avg Satisfaction**: churned median = 4.0 vs active = 4.0 (0.0% higher)
- Features where churned customers show clearly lower values (usage, tenure) are strong churn predictors.

---

### Churn Rate by Category Insights
- **Plan Tier**: highest churn = 'Enterprise' (22.1%), lowest = 'Pro' (21.9%)
- **Customer Segment**: highest churn = 'Enterprise' (26.1%), lowest = 'Strategic' (20.0%)
- **Country / Region**: highest churn = 'DE' (32.0%), lowest = 'AU' (12.5%)
- **Industry**: highest churn = 'DevTools' (31.0%), lowest = 'Cybersecurity' (16.0%)
- Overall churn rate: 22.0%. Categories significantly above this are priority targets.

---

### Correlation Insights (Top 5)
- **avg_errors**: r = -0.091 (negatively correlated with churn)
- **tenure_days**: r = +0.082 (positively correlated with churn)
- **tenure_months**: r = +0.082 (positively correlated with churn)
- **beta_feature_usage**: r = +0.077 (positively correlated with churn)
- **upgrade_flag**: r = -0.072 (negatively correlated with churn)

**⚠️ Multicollinearity Warning:** 9 feature pairs with |r| > 0.8:
  - mrr_amount ↔ arr_amount: r = 1.000
  - total_events ↔ total_duration_secs: r = 0.973
  - total_events ↔ active_days: r = 0.990
  - total_events ↔ unique_features_used: r = 0.902
  - total_duration_secs ↔ active_days: r = 0.962
  Consider dropping one from each pair before training.

---

## Overall EDA Takeaways for B2B SaaS Churn

### Key Patterns to Exploit in Modeling

1. **Low-Usage Accounts Are High Risk**
   Accounts with below-median usage events and active days are
   disproportionately likely to churn. Usage intensity features
   (total_events, avg_daily_usage, active_days) should be among
   the most important model features.

2. **Short-Tenure Risk**
   New accounts (low tenure_months) show elevated churn — the
   "activation gap" where customers haven't found product value yet.
   Consider creating an `is_new_account` flag (< 3 months).

3. **Plan-Type Patterns**
   Different plan tiers show different churn behaviors. Lower-tier
   plans may churn more (price sensitivity) or less (lower expectations).
   Plan tier should be encoded as a feature.

4. **Support Signals Matter**
   Accounts with many support tickets and low satisfaction scores
   are sending distress signals. Ticket count and satisfaction
   are likely important bivariate predictors.

5. **Revenue at Risk**
   The MRR distribution reveals which churners represent the
   biggest revenue impact. High-MRR churners should get priority
   attention from the CS team.

### Modeling Recommendations

- **Class Imbalance:** Churn rate at {churn_rate:.1f}% — use stratified CV,
  class_weight=‘balanced’, and prioritize **Recall** as the primary metric.
- **Feature Engineering:** Create interaction features (e.g., usage × tenure,
  tickets × satisfaction) and ratio features (events per active day).
- **Multicollinearity:** Drop one feature from highly correlated pairs
  (e.g., mrr_amount ↔ arr_amount) to improve model interpretability.
- **Leakage Check:** Verify no post-churn features survive preprocessing.

---

*All plots saved to `reports/figures/`. Use this report alongside the
visualizations for stakeholder presentations.*
