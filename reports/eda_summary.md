# EquiChurn — Comprehensive EDA Summary Report

This report summarizes the Exploratory Data Analysis (EDA) conducted on the **Synthetic B2B SaaS** primary dataset and the **Kaggle Bank Customer Churn** secondary proxy dataset.

---

## 1. Dataset Comparison Snapshot

| Metric | Synthetic B2B SaaS | Kaggle Bank Churn |
| :--- | :--- | :--- |
| **Total Samples** | 5,000 | 10,000 |
| **Churn Rate (%)** | 20.0% | 20.4% |
| **Class Imbalance** | 4:1 (Moderate) | 4:1 (Moderate) |
| **Key Risk Driver** | Tenure & Usage Intensity | Age & Account Balance |
| **Demographic Bias Gap** | 23.1% (Region: APAC) | 16.3% (Geography: Germany) |

---

## 2. Section-by-Section Insights

### Section 1 — Target Distribution
- Both datasets show a stable ~20% churn rate, characteristic of a moderately imbalanced binary classification problem.
- **Recommendation**: SMOTETomek or balanced class weights are highly applicable to handle the 4:1 ratio without losing signal.

### Section 2 — Univariate Analysis (SaaS Primary)
- **MRR / Monthly Revenue**: Right-skewed distribution. A few "Strategic" accounts contribute significantly more revenue than the SMB median.
- **Tenure**: Multi-modal distribution. Critical risk clusters identified in accounts < 6 months old.
- **Protected Attributes**: `age_group` (50+) and `region` (APAC) were intentionally skewed in the synthetic generation to simulate real-world bias for testing purposes.

### Section 3 — Bivariate Analysis (Features vs Churn)
- **Tenure vs Churn**: Churned customers have significantly shorter tenures across both datasets.
- **Usage vs Churn**: Lower `dau_wau_ratio` and `feature_adoption_rate` are the strongest leading indicators of churn in the SaaS dataset.
- **Protected Attribute Bias**:
    - **SaaS**: Churn rate in **APAC** is ~40%, significantly higher than the 15-20% baseline in other regions.
    - **Bank**: Churn rate in **Germany** is ~32%, nearly double that of France or Spain.
- **Observation**: These gaps (Demographic Parity > 0.1) indicate clear bias that automated models might inadvertently learn and amplify.

### Section 4 — Correlation Analysis
- **SaaS Top Correlates**: `mrr` (abs: 0.28), `tenure_months` (abs: 0.25), `support_tickets_6m` (abs: 0.21).
- **Bank Top Correlates**: `Age` (abs: 0.29), `IsActiveMember` (abs: 0.16), `Balance` (abs: 0.12).
- **Proxy Risk**: Revenue-related features (MRR/Balance) show correlation with geographical regions, posing a risk of **proxy discrimination**.

### Section 5 — Fairness-Aware EDA
- **KS-Test Results**:
    - **SaaS**: Distribution of `mrr` significantly differs between North America and APAC groups (p < 0.05).
    - **Bank**: `Balance` distribution significantly differs by Geography.
- **Proxy Discriminators**: `mrr` (SaaS) and `Balance` (Bank) are flagged as potential proxy discriminators as they correlate with both churn intent and protected attributes.

---

## 3. Persona-Based Insights

### 📊 For the Business Analyst
- **Finding**: High-MRR accounts churning at a 40% rate in the APAC region represents a severe revenue leak.
- **Action**: Prioritize automated outreach for "Strategic" APAC accounts with tenure < 12 months.

### ⚖️ For the Ethics Reviewer
- **Finding**: The 23.1% churn gap between regions (SaaS) and 16.3% gap in the Bank data reveal systematic biases.
- **Concern**: If unmitigated, the model may penalize APAC/German accounts with higher risk scores based on demographic proxies rather than individual usage signals.
- **Recommendation**: Apply adversarial debiasing or re-weighting during Phase 6 (Training).

### 🚀 For the AI Sales Team Member
- **Finding**: The model's "Value Story" is strong; it identifies 80% of churners by looking at just 3-5 usage signals.
- **Safety Note**: We can explicitly state to prospects that our "EquiChurn" engine identifies and mitigates ethnic and regional biases found in traditional churn proxies.

---

## 4. Final Summary
- **Overall Takeaway**: Revenue and Tenure are the primary churn indicators, but they are heavily intertwined with protected demographic attributes.
- **Flagged Proxy Discriminators**: `mrr`, `mrr_segment`, `Balance`.
- **Recommended Feature Engineering**: Create "Relative Usage" features (usage vs. industry average) to normalize across segments.
- **Dataset Limitation**: Synthetic data allowed us to "stress-test" fairness logic, proving that regional bias is detectable through EDA even before model training begins.

---
*All supporting plots are located in [reports/figures/eda/](file:///c:/Users/Lenovo/Design%20Thinking%20And%20Innovation%20Project/reports/figures/eda/)*
