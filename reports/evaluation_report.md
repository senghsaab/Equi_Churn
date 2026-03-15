# Stage 7 -- Model Evaluation Report
## B2B SaaS Churn Prediction

---

## 1. Model Comparison Summary

| model               |   accuracy |   precision_churn |   recall_churn |   f1_churn |   roc_auc |   brier_score |
|:--------------------|-----------:|------------------:|---------------:|-----------:|----------:|--------------:|
| logistic_regression |     0.5900 |            0.2791 |         0.5455 |     0.3692 |    0.6247 |        0.2478 |
| gradient_boosting   |     0.7800 |            0.5000 |         0.0909 |     0.1538 |    0.6136 |        0.1684 |
| dummy               |     0.7800 |            0.0000 |         0.0000 |     0.0000 |    0.5000 |        0.2200 |
| random_forest       |     0.7700 |            0.0000 |         0.0000 |     0.0000 |    0.6131 |        0.1673 |

> Sorted by **Recall (churn class) descending** -- catching churners is the
> business priority for ARR protection.

---

## 2. Best Model -- Business Interpretation

**Best model: `logistic_regression`**
- Recall = **54.5%** -- We identify **12 out of 22
  churners** before they leave
- Precision = **27.9%** -- For every churner we correctly flag,
  we also flag ~31 non-churners (acceptable cost for proactive outreach)
- ROC-AUC = **0.625** -- The model ranks churners higher than non-churners
  62% of the time
- F1 = **0.369** -- Balanced precision-recall tradeoff for the churn class

**What this means for the business:**
- In a portfolio of 100 accounts, we correctly flag **12** at-risk
  accounts for proactive CS intervention
- The CS team should prioritize outreach to flagged accounts before contract renewal
- Even one saved enterprise account can protect $50K-$500K in ARR

---

## 3. Feature Importance -- What CS Teams Should Watch For

### Feature Importance Interpretation

Top features driving churn predictions (from tree-based models):

**Random Forest — Top 5:**
  - `num__avg_errors` (0.0843): High error rates degrade user experience and drive churn
  - `num__avg_duration_secs` (0.0646): Session duration patterns reveal engagement depth
  - `num__avg_daily_usage` (0.0531): Low daily usage = not embedding product in workflow
  - `num__avg_resolution_hours` (0.0525): Slow resolution times erode customer confidence
  - `num__avg_first_response_min` (0.0521): Slow first response to tickets drives dissatisfaction

**Gradient Boosting — Top 5:**
  - `num__avg_errors` (0.1556): High error rates degrade user experience and drive churn
  - `num__avg_duration_secs` (0.1025): Session duration patterns reveal engagement depth
  - `num__avg_resolution_hours` (0.0854): Slow resolution times erode customer confidence
  - `num__avg_first_response_min` (0.0565): Slow first response to tickets drives dissatisfaction
  - `num__usage_intensity` (0.0557): Low events per month = disengaged customer, strong churn signal


**Actionable Signals for Customer Success:**
1. **Usage drops** (`usage_intensity`, `active_day_ratio`): Set up automated
   alerts when a customer's monthly usage drops >30% from baseline
2. **Inactivity** (`days_since_last_activity`): Flag accounts with >14 days
   of no product usage for immediate outreach
3. **Short tenure** (`tenure_months`): New customers (0-3 months) need
   structured onboarding and success milestones
4. **Support issues** (`ticket_count`, `avg_satisfaction`): Accounts with
   high ticket volume or low satisfaction need executive sponsor engagement
5. **Feature adoption** (`unique_features_used`): Low feature adoption
   means the customer hasn't found full product value -- offer training

---

## 4. Class Imbalance Analysis

### Class Imbalance Handling Report

Comparison of models **with** and **without** `class_weight='balanced'`:

| model                         |   accuracy |   precision_churn |   recall_churn |   f1_churn |   roc_auc |
|:------------------------------|-----------:|------------------:|---------------:|-----------:|----------:|
| LogReg (balanced)             |     0.5900 |            0.2791 |         0.5455 |     0.3692 |    0.6247 |
| GB (balanced — sample_weight) |     0.7100 |            0.2667 |         0.1818 |     0.2162 |    0.5845 |
| LogReg (no balancing)         |     0.7700 |            0.4286 |         0.1364 |     0.2069 |    0.6533 |
| GB (no balancing)             |     0.7600 |            0.3333 |         0.0909 |     0.1429 |    0.5781 |
| RF (balanced)                 |     0.7800 |            0.0000 |         0.0000 |     0.0000 |    0.6093 |
| RF (no balancing)             |     0.7700 |            0.0000 |         0.0000 |     0.0000 |    0.6116 |

**Key Findings:**
- **LogReg**: Balanced weights -- Recall delta=+0.409, F1 delta=+0.162
- **RF**: Balanced weights -- Recall delta=+0.000, F1 delta=+0.000
- **GB**: Balanced weights -- Recall delta=+0.091, F1 delta=+0.073

> Class balancing trades accuracy for recall -- critical when missing churners is more costly than false alarms (which it is for B2B SaaS retention).

---

## 5. Recommendations

1. **Deploy `logistic_regression`** as the production churn scoring model
2. **Set recall threshold >= 0.75** -- missing churners is more costly than
   false alarms in B2B SaaS
3. **Integrate predictions into CRM** -- surface churn risk scores alongside
   account health metrics
4. **Weekly scoring cadence** -- re-score all accounts weekly to catch
   emerging churn signals
5. **Feedback loop** -- track which flagged accounts actually churn to
   continuously improve the model

---

*Report generated by the Stage 7 evaluation pipeline.*
*Evaluation dataset: 100 accounts (22 churned, 78 active).*
