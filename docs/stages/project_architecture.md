# 📁 Project Architecture — Complete Guide
# ===========================================================================
# PURPOSE  : This document explains EVERY folder and file in the project.
#            Use it to onboard new teammates quickly.
# AUDIENCE : Developers, Data Scientists, and anyone joining the team.
# UPDATED  : 2026-02-26
# STATUS   : Stage 1 ✅ | Stage 2 ✅ | Stages 3–10 pending
# ===========================================================================

---

## Table of Contents

1. [Full Project Tree (Annotated)](#1-full-project-tree-annotated)
2. [data/ — All Data Lives Here](#2-data--all-data-lives-here)
3. [src/ — The Pipeline Code](#3-src--the-pipeline-code)
4. [configs/ — Pipeline Settings](#4-configs--pipeline-settings)
5. [models/ — Trained Model Artifacts](#5-models--trained-model-artifacts)
6. [notebooks/ — Jupyter Notebooks](#6-notebooks--jupyter-notebooks)
7. [reports/ — Output Reports & Figures](#7-reports--output-reports--figures)
8. [tests/ — Automated Tests](#8-tests--automated-tests)
9. [docs/ — Project Documentation](#9-docs--project-documentation)
10. [Root Support Files](#10-root-support-files)
11. [Pipeline Flow — How Everything Connects](#11-pipeline-flow--how-everything-connects)
12. [What Is Complete vs What Is Pending](#12-what-is-complete-vs-what-is-pending)
13. [Quick Reference — Column Schemas](#13-quick-reference--column-schemas)
14. [Glossary](#14-glossary)

---

## 1. Full Project Tree (Annotated)

```
Project-1/
│
│   # -----------------------------------------------------------------------
│   # ROOT FILES — Project setup, configuration, and documentation
│   # -----------------------------------------------------------------------
│
├── README.md                  # Project overview. Start here. Contains:
│                              #   - What the project does
│                              #   - How to set it up (clone → install → download data)
│                              #   - How to run each pipeline step via CLI
│                              #   - Stage progress tracker
│
├── requirements.txt           # Every Python package we need, with version ranges.
│                              #   numpy, pandas, scikit-learn, matplotlib, seaborn,
│                              #   joblib, pyyaml, jupyter, pytest.
│                              #   Install with: pip install -r requirements.txt
│
├── .gitignore                 # Tells Git which files to NEVER commit:
│                              #   - data/raw/ (CSVs are large, download from Kaggle)
│                              #   - models/*.pkl (binary model files, too big for Git)
│                              #   - logs/, __pycache__/, .env (secrets & temp files)
│
├── .env.example               # TEMPLATE for environment variables (not the real one).
│                              #   Copy to .env and fill in your Kaggle credentials.
│                              #   The real .env is in .gitignore — never committed.
│
│   # -----------------------------------------------------------------------
│   # DATA FOLDER — The single source of truth for all data
│   # -----------------------------------------------------------------------
│
├── data/
│   │
│   ├── raw/                               # 🔴 NEVER modify these files. Read-only.
│   │   │                                  # This is the original Kaggle dataset.
│   │   │                                  # Downloaded from: kaggle.com/rivalytics/
│   │   │                                  #   saas-subscription-and-churn-analytics-dataset
│   │   │
│   │   ├── ravenstack_accounts.csv        # 500 rows × 10 cols
│   │   │   # WHO are our customers?
│   │   │   # Columns: account_id, account_name, industry, country,
│   │   │   #          signup_date, referral_source, plan_tier, seats,
│   │   │   #          is_trial, churn_flag
│   │   │   # This is the master customer table. Every other table
│   │   │   # links back to this via account_id.
│   │   │   # churn_flag HERE is the ultimate target variable (0 or 1).
│   │   │
│   │   ├── ravenstack_subscriptions.csv   # 5,000 rows × 14 cols
│   │   │   # WHAT are they paying for?
│   │   │   # Columns: subscription_id, account_id, start_date, end_date,
│   │   │   #          plan_tier, seats, mrr_amount, arr_amount, is_trial,
│   │   │   #          upgrade_flag, downgrade_flag, churn_flag,
│   │   │   #          billing_frequency, auto_renew_flag
│   │   │   # One account can have MULTIPLE subscriptions over time
│   │   │   # (upgrades, renewals, downgrades). We take the LATEST
│   │   │   # subscription per account during ingestion.
│   │   │   # ⚠️ LEAKAGE WARNING: end_date is filled AFTER churn.
│   │   │   #    We DROP this column in preprocessing.
│   │   │
│   │   ├── ravenstack_feature_usage.csv   # 25,000 rows × 8 cols
│   │   │   # HOW are they using the product?
│   │   │   # Columns: usage_id, subscription_id, usage_date,
│   │   │   #          feature_name, usage_count, usage_duration_secs,
│   │   │   #          error_count, is_beta_feature
│   │   │   # Daily logs — one row per feature per day per subscription.
│   │   │   # This is the LARGEST table (25K rows).
│   │   │   # We AGGREGATE this to account-level (mean, sum, std)
│   │   │   # during ingestion. Joins via subscription_id → account_id.
│   │   │
│   │   ├── ravenstack_support_tickets.csv # 2,000 rows × 9 cols
│   │   │   # ARE they having problems?
│   │   │   # Columns: ticket_id, account_id, submitted_at, closed_at,
│   │   │   #          resolution_time_hours, priority, 
│   │   │   #          first_response_time_minutes, satisfaction_score,
│   │   │   #          escalation_flag
│   │   │   # Each row = one support ticket. We AGGREGATE to account-
│   │   │   # level: ticket_count, avg_satisfaction, etc.
│   │   │
│   │   ├── ravenstack_churn_events.csv    # 600 rows × 9 cols
│   │   │   # WHY did they leave?
│   │   │   # Columns: churn_event_id, account_id, churn_date,
│   │   │   #          reason_code, refund_amount_usd,
│   │   │   #          preceding_upgrade_flag, preceding_downgrade_flag,
│   │   │   #          is_reactivation, feedback_text
│   │   │   # Only 600 rows = only 600 accounts churned (out of 500
│   │   │   # unique accounts — some accounts have multiple churn events,
│   │   │   # meaning they churned, came back, then churned again).
│   │   │   # ⚠️ LEAKAGE WARNING: ALL columns here are post-outcome.
│   │   │   #    We ONLY use this to derive the churn_flag target.
│   │   │   #    Then we DROP: churn_date, reason_code, refund_amount, etc.
│   │   │
│   │   └── README.md                      # Kaggle's original dataset description
│   │
│   ├── processed/                         # 🟢 Pipeline outputs go here
│   │   # These files are GENERATED by running the pipeline.
│   │   # They do NOT exist yet — created when you run:
│   │   #   python -m src.data_ingestion   → abt_merged.csv
│   │   #   python -m src.preprocessing    → abt_cleaned.csv
│   │   #   python -m src.feature_engineering → abt_features.csv
│   │   #
│   │   # Files that WILL exist after running:
│   │   #   abt_merged.csv    — 5 tables merged into 1 (1 row per account)
│   │   #   abt_cleaned.csv   — after removing leakage, dedup, type fixes
│   │   #   abt_features.csv  — final model-ready feature matrix + target
│   │
│   ├── interim/                           # 🟡 Scratch space
│   │   # For any intermediate files during development/debugging.
│   │   # Example: partially merged tables, exploratory subsets.
│   │   # Not used by the main pipeline — purely optional.
│   │
│   └── external/                          # 🟡 External enrichment data
│       # For data NOT from Kaggle that enriches predictions.
│       # Example: industry churn benchmarks, economic indicators.
│       # Not used yet — reserved for future stages.
│
│   # -----------------------------------------------------------------------
│   # SOURCE CODE — The actual ML pipeline logic
│   # -----------------------------------------------------------------------
│
├── src/
│   │
│   ├── __init__.py                        # Makes src/ a Python package.
│   │   # So you can write: from src.train import run_training
│   │   # Contains: package version and module descriptions.
│   │   # You'll never need to edit this file.
│   │
│   ├── data_ingestion.py                  # ◀ STEP 1 — Load & Merge
│   │   # WHAT: Loads all 5 CSV files, validates schemas, and JOINs
│   │   #       them into a single "Analytical Base Table" (ABT) with
│   │   #       one row per customer account.
│   │   #
│   │   # HOW IT WORKS:
│   │   #   1. Loads each CSV, logs row count and memory usage
│   │   #   2. Validates all tables have account_id column
│   │   #   3. Takes the LATEST subscription per account
│   │   #   4. Aggregates feature_usage (mean/sum/std per account)
│   │   #   5. Aggregates support_tickets (count, avg satisfaction)
│   │   #   6. LEFT JOINs everything on account_id
│   │   #   7. Creates churn_flag from churn_events (1 if ever churned)
│   │   #
│   │   # KEY FUNCTIONS:
│   │   #   load_all_tables()  → loads 5 CSVs into dict
│   │   #   validate_schemas() → checks for account_id, duplicates
│   │   #   merge_abt()        → the big JOIN operation
│   │   #   run_ingestion()    → runs everything, saves output
│   │   #
│   │   # INPUT:  data/raw/ravenstack_*.csv (5 files)
│   │   # OUTPUT: data/processed/abt_merged.csv
│   │   # RUN:    python -m src.data_ingestion
│   │
│   ├── preprocessing.py                   # ◀ STEP 2 — Clean the Data
│   │   # WHAT: Takes the merged ABT and makes it clean and safe
│   │   #       for model training.
│   │   #
│   │   # HOW IT WORKS (in order):
│   │   #   1. LEAKAGE GUARD — Drops columns like churn_date,
│   │   #      churn_reason, refund_amount that only exist AFTER
│   │   #      churn has happened. Using them would be "cheating".
│   │   #   2. DEDUPLICATION — Removes duplicate account_ids.
│   │   #   3. TYPE CONVERSION — date strings → datetime objects,
│   │   #      categorical strings → category dtype.
│   │   #   4. MISSING VALUES — Imputes numeric (median) and
│   │   #      categorical (mode). Also creates is_missing_* flags
│   │   #      because missingness itself can be a signal.
│   │   #   5. OUTLIER CLIPPING — Clips extreme values to 1st/99th
│   │   #      percentile to prevent skewed distributions.
│   │   #   6. LEAKAGE SCANNER — Checks if any single feature has
│   │   #      AUC > 0.95 vs target (a red flag for hidden leakage).
│   │   #
│   │   # ⚠️ WHY LEAKAGE MATTERS:
│   │   #   If we accidentally leave churn_date as a feature, the
│   │   #   model learns "if churn_date is not null → customer churned"
│   │   #   → AUC 0.99 in training → ZERO value in production because
│   │   #   we won't know churn_date for active customers!
│   │   #
│   │   # INPUT:  data/processed/abt_merged.csv
│   │   # OUTPUT: data/processed/abt_cleaned.csv
│   │   # RUN:    python -m src.preprocessing
│   │
│   ├── feature_engineering.py             # ◀ STEP 3 — Create Features
│   │   # WHAT: Transforms raw columns into predictive features the
│   │   #       model can learn from. This is where domain knowledge
│   │   #       gets encoded into data.
│   │   #
│   │   # FEATURES CREATED:
│   │   #
│   │   #   TENURE (how long they've been a customer):
│   │   #     - account_age_days       → days since signup
│   │   #     - subscription_tenure_days → days on current plan
│   │   #     - is_new_account         → 1 if < 30 days old
│   │   #
│   │   #   USAGE INTENSITY (are they actually using the product?):
│   │   #     - usage_engagement_score → sum of mean usage metrics
│   │   #     - usage_variability      → avg std of usage (erratic?)
│   │   #     - is_low_usage           → 1 if below 25th percentile
│   │   #
│   │   #   REVENUE / MRR (how much money is at risk?):
│   │   #     - mrr_segment            → low / mid / high / enterprise
│   │   #     - is_high_value          → 1 if top-25% by MRR
│   │   #
│   │   #   SUPPORT SIGNALS (are they unhappy?):
│   │   #     - has_support_tickets    → 1 if any tickets filed
│   │   #     - is_high_ticket_volume  → 1 if above median
│   │   #     - is_low_satisfaction    → 1 if below 25th percentile
│   │   #
│   │   #   ENCODED CATEGORICALS:
│   │   #     - One-hot encoding for cols with ≤ 10 unique values
│   │   #     - Label encoding for cols with > 10 unique values
│   │   #
│   │   # INPUT:  data/processed/abt_cleaned.csv
│   │   # OUTPUT: data/processed/abt_features.csv
│   │   # RUN:    python -m src.feature_engineering
│   │
│   ├── train.py                           # ◀ STEP 4 — Train Models
│   │   # WHAT: Trains 4 progressively stronger classifiers.
│   │   #       Each model should beat the one before it.
│   │   #
│   │   # MODELS (simple → complex):
│   │   #   1. DummyClassifier        → Always predicts "not churned"
│   │   #      Why: Sets the FLOOR. If your real model can't beat
│   │   #           this, something is fundamentally wrong.
│   │   #      Recall: 0.00 (never catches any churner)
│   │   #
│   │   #   2. LogisticRegression     → Linear model, balanced weights
│   │   #      Why: Interpretable. Produces coefficients that explain
│   │   #           WHICH features drive churn. Great for stakeholders.
│   │   #
│   │   #   3. RandomForestClassifier → Bagging ensemble, 200 trees
│   │   #      Why: Captures non-linear patterns, robust to noise,
│   │   #           provides feature importance rankings.
│   │   #
│   │   #   4. GradientBoostingClassifier → Sequential boosting ⭐
│   │   #      Why: Our PRODUCTION candidate. Most powerful model.
│   │   #           Builds trees sequentially, each one correcting
│   │   #           the previous one's mistakes.
│   │   #      NOTE: We use sklearn's GradientBoosting, NOT XGBoost
│   │   #            or LightGBM (project scope decision).
│   │   #
│   │   # HOW IT WORKS:
│   │   #   1. Reads hyperparameters from configs/model_config.yaml
│   │   #   2. Splits data: 80% train / 20% test (stratified — same
│   │   #      churn ratio in both sets)
│   │   #   3. Scales features with StandardScaler (fit on train only!)
│   │   #   4. Trains all 4 models
│   │   #   5. Runs 5-fold stratified cross-validation
│   │   #   6. Saves each model as a .pkl bundle to models/
│   │   #
│   │   # INPUT:  data/processed/abt_features.csv + configs/model_config.yaml
│   │   # OUTPUT: models/dummy.pkl, logistic_regression.pkl,
│   │   #         random_forest.pkl, gradient_boosting.pkl
│   │   # RUN:    python -m src.train
│   │
│   ├── evaluate.py                        # ◀ STEP 5 — Evaluate & Compare
│   │   # WHAT: Computes metrics, generates plots, and produces a
│   │   #       side-by-side comparison of all 4 models.
│   │   #
│   │   # OUTPUTS GENERATED:
│   │   #   TEXT REPORTS:
│   │   #     - reports/model_comparison.csv         → metrics table
│   │   #     - reports/<model>_classification_report.txt → per-model
│   │   #
│   │   #   PLOTS:
│   │   #     - reports/figures/<model>_confusion_matrix.png
│   │   #       → Shows true positives, false positives, etc.
│   │   #         Tells us: "How many churners did we correctly catch,
│   │   #          and how many healthy accounts did we wrongly flag?"
│   │   #
│   │   #     - reports/figures/roc_curves_comparison.png
│   │   #       → All 4 models on one plot. Higher curve = better.
│   │   #         AUC = 0.5 is random, AUC = 1.0 is perfect.
│   │   #
│   │   #     - reports/figures/precision_recall_curves.png
│   │   #       → More honest than ROC when classes are imbalanced.
│   │   #         Shows the trade-off: catch more churners (recall)
│   │   #         vs. fewer false alarms (precision).
│   │   #
│   │   #     - reports/figures/<model>_feature_importance.png
│   │   #       → Top-20 features driving the model's decisions.
│   │   #         Only for tree-based models (RF, GradientBoosting).
│   │   #
│   │   # INPUT:  models/*.pkl + test data
│   │   # OUTPUT: reports/ (CSVs + PNGs)
│   │   # RUN:    python -m src.evaluate
│   │
│   └── inference.py                       # ◀ STEP 6 — Score New Accounts
│       # WHAT: The PRODUCTION module. Loads a saved model and scores
│       #       new customer accounts that the model has never seen.
│       #       Outputs a prioritized risk list for the CS team.
│       #
│       # HOW IT WORKS:
│       #   1. Loads the .pkl bundle (model + scaler + feature list)
│       #   2. Aligns input features to what the model expects
│       #   3. Scales the data using the saved scaler
│       #   4. Predicts churn probability for each account
│       #   5. Assigns a RISK TIER based on probability:
│       #      🟢 Low      (0–25%)  → No action
│       #      🟡 Medium   (25–50%) → Monitor at next QBR
│       #      🟠 High     (50–75%) → CS outreach within 1 week
│       #      🔴 Critical (75–100%) → Executive escalation NOW
│       #   6. Generates a sorted risk report CSV for the CS team
│       #
│       # INPUT:  models/gradient_boosting.pkl + new customer CSV
│       # OUTPUT: reports/risk_report.csv
│       # RUN:    python -m src.inference --model models/gradient_boosting.pkl
│       #                                --input new_customers.csv
│
│   # -----------------------------------------------------------------------
│   # CONFIGURATION
│   # -----------------------------------------------------------------------
│
├── configs/
│   └── model_config.yaml                  # 🎛️ The single control panel
│       # Instead of changing Python code, change THIS file to adjust:
│       #
│       # [data]           → file paths for data/raw and data/processed
│       # [preprocessing]  → how to handle missing values (median/mean/zero)
│       #                    outlier clipping bounds, leakage AUC threshold
│       # [features]       → encoding method (onehot vs label), cardinality
│       #                    limits, new-account threshold (30 days)
│       # [models]         → hyperparameters for ALL 4 models:
│       #                    - n_estimators, max_depth, learning_rate
│       #                    - class_weight, subsample, min_samples
│       # [cross_validation] → number of folds (5), scoring metrics
│       # [evaluation]     → classification threshold, figure settings
│       # [inference]      → default model, risk tier boundaries
│       #
│       # 💡 TIP: To try different model settings, edit this file
│       #         and re-run train.py — no code changes needed.
│
│   # -----------------------------------------------------------------------
│   # TRAINED MODELS
│   # -----------------------------------------------------------------------
│
├── models/                                # Saved model artifacts
│   # EMPTY NOW — populated when you run train.py
│   # Each .pkl file is a BUNDLE containing:
│   #   {
│   #     "model":         <fitted sklearn model object>,
│   #     "model_name":    "gradient_boosting",
│   #     "scaler":        <fitted StandardScaler>,
│   #     "feature_names": ["feature_1", "feature_2", ...]
│   #   }
│   #
│   # Files that WILL exist after training:
│   #   dummy.pkl               → baseline (always predicts 0)
│   #   logistic_regression.pkl → interpretable linear model
│   #   random_forest.pkl       → bagging ensemble
│   #   gradient_boosting.pkl   → ⭐ production model
│   #
│   # WHY bundle model + scaler + features?
│   #   Because at inference time, we need to:
│   #   1. Know which features the model expects (feature_names)
│   #   2. Scale the new data the same way training data was scaled (scaler)
│   #   3. Make predictions (model)
│   #   All in one file = no chance of mismatched components.
│
│   # -----------------------------------------------------------------------
│   # NOTEBOOKS
│   # -----------------------------------------------------------------------
│
├── notebooks/
│   ├── exploratory/                       # EDA (Exploratory Data Analysis)
│   │   # EMPTY NOW — will be populated in Stage 3
│   │   # Purpose: Jupyter notebooks to explore and understand the data
│   │   #   - Distribution of churn vs non-churn
│   │   #   - Correlation heatmaps
│   │   #   - Missing value patterns
│   │   #   - Feature distributions by churn class
│   │   #   - Time-based trends
│   │
│   └── modeling/                          # Model experimentation
│       # EMPTY NOW — will be populated in later stages
│       # Purpose: Interactive model building and comparison
│       #   - Trying different feature combinations
│       #   - Hyperparameter sweeps
│       #   - Error analysis on misclassified accounts
│
│   # -----------------------------------------------------------------------
│   # REPORTS & FIGURES
│   # -----------------------------------------------------------------------
│
├── reports/
│   │   # EMPTY NOW — populated by evaluate.py
│   │   # Files that WILL exist after evaluation:
│   │   #   model_comparison.csv          → side-by-side metrics
│   │   #   <model>_classification_report.txt → per-model detail
│   │   #   risk_report.csv               → inference output for CS team
│   │
│   └── figures/                           # Generated plots
│       # EMPTY NOW — populated by evaluate.py
│       # Files that WILL exist:
│       #   <model>_confusion_matrix.png   → 4 heatmaps
│       #   roc_curves_comparison.png      → all models overlaid
│       #   precision_recall_curves.png    → all models overlaid
│       #   <model>_feature_importance.png → for RF and GB
│
│   # -----------------------------------------------------------------------
│   # TESTS
│   # -----------------------------------------------------------------------
│
├── tests/
│   ├── __init__.py                        # Makes tests/ a Python package
│   │
│   ├── test_preprocessing.py              # 8 tests across 5 test classes
│   │   # Tests: leakage removal, deduplication, missing value
│   │   #        imputation, type conversion, outlier clipping
│   │   # Example: "Does remove_leakage_columns() actually drop
│   │   #           churn_date but keep churn_flag?"
│   │
│   ├── test_feature_engineering.py        # 10 tests across 6 test classes
│   │   # Tests: tenure features, usage features, revenue features,
│   │   #        support features, encoding, feature matrix
│   │   # Example: "Does compute_tenure_features() create
│   │   #           account_age_days with positive values?"
│   │
│   └── test_train.py                      # 5 tests across 3 test classes
│       # Tests: data splitting, feature scaling, model training
│       # Example: "Does train_single_model('unknown') raise ValueError?"
│       #
│       # RUN ALL TESTS: pytest tests/ -v
│
│   # -----------------------------------------------------------------------
│   # DOCUMENTATION
│   # -----------------------------------------------------------------------
│
├── docs/
│   ├── stage_1_problem_definition.md      # ✅ Stage 1 deliverable
│   │   # Contains: Problem statement, success metrics (recall ≥ 0.75),
│   │   #   DummyClassifier baseline, 5 risk categories with mitigations,
│   │   #   ER diagram of the 5-table dataset
│   │
│   └── project_architecture.md            # ✅ THIS document
│       # Contains: Everything you're reading right now — full
│       #   explanation of every file, folder, and function
│
│   # -----------------------------------------------------------------------
│   # LOGS
│   # -----------------------------------------------------------------------
│
└── logs/                                  # Runtime execution logs
    # EMPTY NOW. Each pipeline module logs to console.
    # You can redirect to files here:
    #   python -m src.train > logs/training_20260226.log 2>&1
```

---

## 2. data/ — All Data Lives Here

### Why 4 subfolders?

This follows the **Cookiecutter Data Science** convention, which separates data by processing stage to prevent accidental overwrites and make the pipeline reproducible:

```
raw/          →  Immutable. Downloaded once. Never edited.
    ↓ (data_ingestion.py)
processed/    →  Pipeline outputs. Can be regenerated anytime.
    ↓ (intermediate steps if needed)
interim/      →  Scratch. Temporary. Disposable.

external/     →  Data from other sources (not Kaggle).
```

### What's inside data/raw/ RIGHT NOW

| File | Rows | Columns | Key Question It Answers |
|---|---|---|---|
| `ravenstack_accounts.csv` | 500 | 10 | **WHO** are our customers? (name, industry, country, plan) |
| `ravenstack_subscriptions.csv` | 5,000 | 14 | **WHAT** are they paying? (MRR, ARR, plan tier, billing) |
| `ravenstack_feature_usage.csv` | 25,000 | 8 | **HOW** are they using the product? (feature, count, duration) |
| `ravenstack_support_tickets.csv` | 2,000 | 9 | **ARE** they having problems? (priority, satisfaction, escalation) |
| `ravenstack_churn_events.csv` | 600 | 9 | **WHY** did they leave? (reason, refund, reactivation) |

> **Total:** 33,100 rows across 5 tables, all linked by `account_id`

---

## 3. src/ — The Pipeline Code

### How the 6 modules connect

```
Module 1               Module 2              Module 3
data_ingestion.py  →  preprocessing.py  →  feature_engineering.py
"Load & Merge"        "Clean & Guard"       "Create Features"
                                                    │
                                                    ▼
Module 6               Module 5              Module 4
inference.py       ←  evaluate.py       ←  train.py
"Score Accounts"      "Metrics & Plots"     "Train Models"
```

### Every function explained

#### `data_ingestion.py` — Load & Merge

| Function | What It Does | When It's Called |
|---|---|---|
| `load_single_csv(path)` | Reads one CSV, logs its size and memory | Called 5 times by load_all_tables |
| `load_all_tables(raw_dir)` | Loads all 5 CSVs into a dictionary | Called by run_ingestion |
| `validate_schemas(tables)` | Checks: does every table have account_id? Any duplicates? | Called by run_ingestion |
| `merge_abt(tables)` | The big operation — LEFT JOINs all tables on account_id, aggregates usage and tickets, creates churn_flag | Called by run_ingestion |
| `run_ingestion()` | **Main entrypoint** — calls all above in order, saves result | Called from CLI or notebooks |

#### `preprocessing.py` — Clean the Data

| Function | What It Does | Why It Matters |
|---|---|---|
| `remove_leakage_columns(df)` | Drops churn_date, churn_reason, refund_amount, end_date | **THE MOST IMPORTANT FUNCTION.** Without this, the model "cheats" |
| `detect_leakage_features(df)` | Scans remaining features for suspiciously high AUC vs target | Catches SUBTLE leakage that column-name checks miss |
| `handle_missing_values(df)` | Fills nulls: numeric→median, categorical→mode. Creates is_missing_* flags | Missingness can be a signal (missing usage = inactive?) |
| `remove_duplicates(df)` | Removes duplicate rows by account_id | Prevents same account being in train AND test |
| `convert_types(df)` | Converts date strings→datetime, categories→category dtype | Needed for tenure calculations and memory efficiency |
| `clip_outliers(df)` | Clips extreme values to 1st/99th percentile | Prevents a single extreme account from dominating |
| `run_preprocessing()` | **Main entrypoint** — calls all above in order | Called from CLI or notebooks |

#### `feature_engineering.py` — Create Features

| Function | Features Created | Business Logic |
|---|---|---|
| `compute_tenure_features()` | account_age_days, subscription_tenure_days, is_new_account | Older accounts are more invested; new accounts are riskier |
| `compute_usage_features()` | engagement_score, variability, is_low_usage | Low/declining usage = strongest churn signal in SaaS |
| `compute_revenue_features()` | mrr_segment, is_high_value | Prioritize saving high-value accounts |
| `compute_support_features()` | has_tickets, high_volume, low_satisfaction | Frequent tickets + low satisfaction = unhappy customer |
| `encode_categoricals()` | One-hot or label encoded columns | ML models need numbers, not text strings |
| `prepare_feature_matrix()` | Splits into X (features) and y (churn_flag) | Drops IDs, dates, target — returns model-ready arrays |
| `run_feature_engineering()` | **Main entrypoint** | Called from CLI or notebooks |

#### `train.py` — Train Models

| Function | What It Does |
|---|---|
| `load_config()` | Reads configs/model_config.yaml for hyperparameters |
| `apply_config_overrides()` | Merges YAML overrides into the model registry defaults |
| `split_data()` | 80/20 stratified split (CRITICAL: same churn ratio in both halves) |
| `scale_features()` | StandardScaler fit on TRAIN ONLY, transform both — prevents data leakage |
| `train_single_model(name)` | Trains one model by name from the registry |
| `train_all_models()` | Trains all 4 models sequentially |
| `cross_validate_models()` | 5-fold stratified CV measuring recall, F1, ROC-AUC |
| `save_model()` / `save_all_models()` | Bundles model + scaler + feature_names into .pkl |
| `run_training()` | **Main entrypoint** — runs the full pipeline |

#### `evaluate.py` — Metrics & Plots

| Function | Output |
|---|---|
| `compute_metrics()` | accuracy, precision, recall, F1, ROC-AUC, Brier score |
| `evaluate_all_models()` | Comparison DataFrame — all models side by side |
| `print_classification_report()` | Detailed per-class precision/recall/F1 |
| `plot_confusion_matrix()` | Seaborn heatmap → `<model>_confusion_matrix.png` |
| `plot_roc_curves()` | All models overlaid → `roc_curves_comparison.png` |
| `plot_precision_recall_curves()` | All models overlaid → `precision_recall_curves.png` |
| `plot_feature_importance()` | Top-20 bar chart → `<model>_feature_importance.png` |
| `run_evaluation()` | **Main entrypoint** — runs everything, saves all outputs |

#### `inference.py` — Score New Accounts

| Function | What It Does |
|---|---|
| `load_model_artifact()` | Loads the .pkl bundle from disk |
| `predict(artifact, X)` | Scores accounts, assigns churn_probability + risk_tier |
| `predict_single()` | Scores ONE account (convenience wrapper) |
| `generate_risk_report()` | Creates sorted CSV for CS team with account_id, probability, tier |
| `run_inference()` | **Main entrypoint** — load model → score → generate report |

---

## 4. configs/ — Pipeline Settings

`model_config.yaml` is organized into 7 sections:

| Section | What You Can Tune | Example |
|---|---|---|
| `project` | Name, version, random seed | `random_state: 42` |
| `data` | Input/output file paths | `raw_dir: "data/raw"` |
| `preprocessing` | Missing value strategy, outlier bounds | `missing_numeric_strategy: "median"` |
| `features` | Encoding method, cardinality limit | `encoding_method: "onehot"` |
| `models` | **Hyperparameters for all 4 models** | `learning_rate: 0.1, n_estimators: 200` |
| `cross_validation` | Number of folds, scoring metrics | `n_folds: 5` |
| `inference` | Default model, risk tier boundaries | `threshold: 0.5` |

---

## 5. models/ — Trained Model Artifacts

**Status:** Empty — populated after running `train.py`

Each `.pkl` file contains everything needed for inference:
```python
{
    "model":         RandomForestClassifier(...),   # The trained model
    "model_name":    "random_forest",               # For identification
    "scaler":        StandardScaler(...),            # For feature scaling
    "feature_names": ["account_age_days", ...]      # Expected input columns
}
```

---

## 6. notebooks/ — Jupyter Notebooks

**Status:** Both folders empty — will be created in upcoming stages

| Folder | Stage | What Will Go Here |
|---|---|---|
| `exploratory/` | Stage 3 | EDA: distributions, correlations, missing patterns, churn analysis |
| `modeling/` | Stage 5–7 | Model experiments, hyperparameter tuning, error analysis |

---

## 7. reports/ — Output Reports & Figures

**Status:** Empty — populated after running `evaluate.py` and `inference.py`

| File | Generated By | Purpose |
|---|---|---|
| `model_comparison.csv` | evaluate.py | Side-by-side metrics for all 4 models |
| `*_classification_report.txt` | evaluate.py | Per-model precision/recall/F1 |
| `figures/*_confusion_matrix.png` | evaluate.py | Visual true vs predicted |
| `figures/roc_curves_comparison.png` | evaluate.py | ROC curves overlaid |
| `figures/precision_recall_curves.png` | evaluate.py | PR curves overlaid |
| `figures/*_feature_importance.png` | evaluate.py | Top-20 features |
| `risk_report.csv` | inference.py | Prioritized list for CS team |

---

## 8. tests/ — Automated Tests

| File | What It Tests | # Tests | Run With |
|---|---|---|---|
| `test_preprocessing.py` | Leakage removal, dedup, missing values, types, outliers | 8 | `pytest tests/test_preprocessing.py -v` |
| `test_feature_engineering.py` | All feature categories, encoding, matrix prep | 10 | `pytest tests/test_feature_engineering.py -v` |
| `test_train.py` | Splitting, scaling, training, error handling | 5 | `pytest tests/test_train.py -v` |
| **All** | Everything | 23 | `pytest tests/ -v` |

---

## 9. docs/ — Project Documentation

| File | Stage | What It Contains |
|---|---|---|
| `stage_1_problem_definition.md` | 1 ✅ | Problem statement, success metrics, DummyClassifier baseline, 5 risk categories, ER diagram |
| `project_architecture.md` | 2 ✅ | **This document** — every file/folder/function explained |

---

## 10. Root Support Files

| File | Purpose | When to Use |
|---|---|---|
| `README.md` | Project overview + setup guide | First thing anyone reads when they open the repo |
| `requirements.txt` | Python dependencies list | `pip install -r requirements.txt` — run once during setup |
| `.gitignore` | Files Git should never track | Automatic — Git reads this file on every commit |
| `.env.example` | Template for secrets | Copy to `.env`, fill in Kaggle credentials |

---

## 11. Pipeline Flow — How Everything Connects

```
┌─────────────────────────────────────────────────────────┐
│                    YOU START HERE                         │
│                                                          │
│  1. pip install -r requirements.txt                      │
│  2. Download data → data/raw/                            │  ← ✅ DONE
│  3. (Optional) edit configs/model_config.yaml            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌──── STEP 1 ───────────────────────────────────────────────┐
│  python -m src.data_ingestion                             │
│                                                           │
│  "Load 5 CSVs → Validate → JOIN into 1 table"            │
│  Input:  data/raw/ravenstack_*.csv                        │
│  Output: data/processed/abt_merged.csv                    │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌──── STEP 2 ───────────────────────────────────────────────┐
│  python -m src.preprocessing                              │
│                                                           │
│  "Remove leakage → Dedup → Fix types → Impute → Clip"    │
│  Input:  data/processed/abt_merged.csv                    │
│  Output: data/processed/abt_cleaned.csv                   │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌──── STEP 3 ───────────────────────────────────────────────┐
│  python -m src.feature_engineering                        │
│                                                           │
│  "Create tenure/usage/revenue/support features + encode"  │
│  Input:  data/processed/abt_cleaned.csv                   │
│  Output: data/processed/abt_features.csv                  │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌──── STEP 4 ───────────────────────────────────────────────┐
│  python -m src.train                                      │
│                                                           │
│  "Split → Scale → Train 4 models → CV → Save .pkl"       │
│  Input:  data/processed/abt_features.csv                  │
│  Output: models/*.pkl (4 files)                           │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌──── STEP 5 ───────────────────────────────────────────────┐
│  python -m src.evaluate                                   │
│                                                           │
│  "Metrics → Confusion matrix → ROC → PR → Importance"    │
│  Input:  models/*.pkl + test data                         │
│  Output: reports/ (CSVs + PNGs)                           │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌──── STEP 6 ───────────────────────────────────────────────┐
│  python -m src.inference --input new_customers.csv        │
│                                                           │
│  "Load model → Score accounts → Assign risk tier"         │
│  Input:  models/gradient_boosting.pkl + new data          │
│  Output: reports/risk_report.csv                          │
│                                                           │
│  🟢 Low | 🟡 Medium | 🟠 High | 🔴 Critical              │
└───────────────────────────────────────────────────────────┘
```

---

## 12. What Is Complete vs What Is Pending

### ✅ COMPLETE — Ready to Use Right Now

| Item | Files | Lines of Code |
|---|---|---|
| Project structure | 12 directories | — |
| data_ingestion.py | 1 file | ~290 lines |
| preprocessing.py | 1 file | ~300 lines |
| feature_engineering.py | 1 file | ~310 lines |
| train.py | 1 file | ~330 lines |
| evaluate.py | 1 file | ~310 lines |
| inference.py | 1 file | ~250 lines |
| Config | model_config.yaml | ~100 lines |
| Tests | 3 test files | ~230 lines |
| Support | requirements.txt, .gitignore, .env.example, README.md | ~200 lines |
| Docs | problem_definition.md + this file | ~800 lines |
| **Raw Data** | **5 CSV files downloaded** | **33,100 rows** |

### 🔴 STILL NEEDED — Upcoming Stages

| What | Stage | Why It's Not Done Yet |
|---|---|---|
| Run the pipeline | 3+ | Need to actually execute each step and verify outputs |
| EDA notebooks | 3 | Need to explore data distributions, patterns, imbalances |
| Hyperparameter tuning | 6 | GridSearch to find optimal model settings |
| Model interpretation | 7 | SHAP values, partial dependence plots |
| Pipeline orchestration | 8 | End-to-end automation, error handling, monitoring |
| Dashboard / API | 9 | Streamlit or FastAPI for CS team access |
| Final documentation | 10 | User guide, model card, deployment runbook |

---

## 13. Quick Reference — Column Schemas

### ravenstack_accounts.csv (500 rows)
```
account_id          → Unique customer ID (primary key, links to all tables)
account_name        → Company name
industry            → Customer's industry (Tech, Finance, Healthcare, etc.)
country             → Customer's country
signup_date         → When they first signed up
referral_source     → How they found us (organic, paid, referral, etc.)
plan_tier           → Current plan (Basic, Pro, Enterprise)
seats               → Number of licensed seats
is_trial            → 1 if currently on trial, 0 if paid
churn_flag          → ⭐ TARGET VARIABLE (0 = active, 1 = churned)
```

### ravenstack_subscriptions.csv (5,000 rows)
```
subscription_id     → Unique subscription ID
account_id          → Links to accounts table
start_date          → Subscription start
end_date            → ⚠️ LEAKAGE — only filled after cancellation. WE DROP THIS.
plan_tier           → Plan at time of this subscription
seats               → Seats for this subscription
mrr_amount          → Monthly Recurring Revenue ($$)
arr_amount          → Annual Recurring Revenue ($$)
is_trial            → Trial subscription?
upgrade_flag        → Did they upgrade to this plan?
downgrade_flag      → Did they downgrade to this plan?
churn_flag          → Churned during this subscription?
billing_frequency   → Monthly / Annual
auto_renew_flag     → Auto-renewal enabled?
```

### ravenstack_feature_usage.csv (25,000 rows)
```
usage_id            → Unique usage record ID
subscription_id     → Links to subscriptions (→ accounts via subscription)
usage_date          → Date of usage
feature_name        → Which product feature was used
usage_count         → How many times used that day
usage_duration_secs → Total seconds of usage
error_count         → Errors encountered
is_beta_feature     → Was this a beta feature?
```

### ravenstack_support_tickets.csv (2,000 rows)
```
ticket_id                    → Unique ticket ID
account_id                   → Links to accounts table
submitted_at                 → When ticket was filed
closed_at                    → When ticket was resolved
resolution_time_hours        → Hours to resolve
priority                     → Low / Medium / High / Critical
first_response_time_minutes  → Minutes until first response
satisfaction_score            → Customer satisfaction (1–5)
escalation_flag              → Was it escalated to management?
```

### ravenstack_churn_events.csv (600 rows)
```
churn_event_id         → Unique churn event ID
account_id             → Links to accounts table
churn_date             → ⚠️ LEAKAGE — WE DROP THIS
reason_code            → ⚠️ LEAKAGE — WE DROP THIS (only known post-churn)
refund_amount_usd      → ⚠️ LEAKAGE — WE DROP THIS
preceding_upgrade_flag → ⚠️ LEAKAGE — WE DROP THIS
preceding_downgrade_flag → ⚠️ LEAKAGE — WE DROP THIS
is_reactivation        → Did they come back after churning?
feedback_text          → ⚠️ LEAKAGE — Free text from exit survey
```

---

## 14. Glossary

| Term | What It Means | Why It Matters |
|---|---|---|
| **ABT** | Analytical Base Table — one merged table, one row per account | This is what the model actually trains on |
| **ARR** | Annual Recurring Revenue | Total yearly subscription revenue at risk |
| **MRR** | Monthly Recurring Revenue | Monthly revenue per account |
| **Churn** | Customer cancels their subscription | What we're predicting (churn_flag = 1) |
| **Feature Leakage** | Using post-outcome data as input features | Causes fake-perfect results that fail in production |
| **Stratified Split** | Train/test split preserving class ratio | Ensures both halves have the same % of churners |
| **ROC-AUC** | Area Under ROC Curve (0.5 = random, 1.0 = perfect) | Measures overall discriminative power |
| **PR-AUC** | Area Under Precision-Recall Curve | More honest than ROC when classes are imbalanced |
| **Recall** | Of all actual churners, what % did we catch? | Our #1 metric — missing a churner is costly |
| **Precision** | Of accounts we flagged, what % actually churned? | Too low = CS team wastes time on false alarms |
| **F1 Score** | Harmonic mean of precision and recall | Balances catching churners vs avoiding false alarms |
| **Brier Score** | How well-calibrated are our probabilities? | Lower = predicted 70% churn actually means ~70% |
| **CS Team** | Customer Success team | Primary users of this model's output |
| **Risk Tier** | Low / Medium / High / Critical | Action level for each account |
| **StandardScaler** | Centers data (mean=0, std=1) | Required for LogisticRegression; good practice for all |
| **.pkl** | Python pickle file — serialized object | How we save and load trained models |
| **Cross-Validation** | Train/test on different folds to check stability | Ensures the model isn't just lucky on one specific split |

---

*This document is the permanent team reference. Update it as new stages are completed.*
