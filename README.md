# RavenStack — SaaS Churn Prediction Engine

> **An end-to-end machine learning pipeline for predicting B2B SaaS customer churn, enabling proactive ARR/MRR protection through data-driven interventions.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## Overview

This project builds a **binary classification model** that produces a daily prioritized list of at-risk customer accounts, enabling Customer Success and Revenue teams to intervene **before** churn occurs.

| Component | Details |
|---|---|
| **Dataset** | [RavenStack — SaaS Subscription & Churn Analytics](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset) (5 CSV tables, MIT License) |
| **Target** | `churn_flag` (0 = active, 1 = churned) |
| **Primary Metric** | Recall ≥ 0.75 (catching churners is priority #1) |
| **Models** | DummyClassifier → Logistic Regression → Random Forest → **Gradient Boosting** |
| **Stakeholders** | Customer Success · Revenue Ops · Finance |

---

## Project Structure

```
Project-1/
│
├── configs/
│   └── model_config.yaml         # Hyperparameters & pipeline settings
│
├── data/
│   ├── raw/                      # Original Kaggle CSV files
│   ├── processed/                # Cleaned & engineered datasets
│   ├── interim/                  # Intermediate transformations
│   └── external/                 # External / enrichment data
│
├── docs/
│   └── stage_1_problem_definition.md  # Problem statement & success criteria
│
├── models/                       # Saved model artifacts (.pkl)
│
├── notebooks/
│   ├── exploratory/              # EDA notebooks
│   └── modeling/                 # Modeling experiment notebooks
│
├── reports/
│   └── figures/                  # Generated plots (ROC, confusion matrix, etc.)
│
├── src/
│   ├── __init__.py               # Package init
│   ├── data_ingestion.py         # Load & merge 5 CSV tables into ABT
│   ├── preprocessing.py          # Clean, deduplicate, type-cast, leakage guard
│   ├── feature_engineering.py    # Tenure, usage, MRR segments, encoding
│   ├── train.py                  # Train 4 models with CV & persistence
│   ├── evaluate.py               # Metrics, confusion matrix, ROC/PR curves
│   └── inference.py              # Score new accounts, generate risk reports
│
├── tests/
│   ├── test_preprocessing.py     # Preprocessing unit tests
│   ├── test_feature_engineering.py  # Feature engineering unit tests
│   └── test_train.py             # Training module unit tests
│
├── logs/                         # Runtime logs
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # ← You are here
```

---

## Setup

### 1. Clone & Create Virtual Environment

```bash
git clone <repo-url>
cd Project-1

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

**Option A** — Kaggle CLI:
```bash
pip install kaggle
# Copy your kaggle.json to ~/.kaggle/ (or set KAGGLE_USERNAME and KAGGLE_KEY)
kaggle datasets download rivalytics/saas-subscription-and-churn-analytics-dataset -p data/raw/ --unzip
```

**Option B** — Manual download:
1. Go to [the dataset page](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset)
2. Download and extract all CSV files into `data/raw/`

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Kaggle credentials and path preferences
```

---

## How to Run

### Full Pipeline (CLI)

```bash
# Step 1: Ingest & merge raw tables
python -m src.data_ingestion --raw-dir data/raw --output data/processed/abt_merged.csv

# Step 2: Preprocess
python -m src.preprocessing --input data/processed/abt_merged.csv --output data/processed/abt_cleaned.csv

# Step 3: Feature engineering
python -m src.feature_engineering --input data/processed/abt_cleaned.csv --output data/processed/abt_features.csv

# Step 4: Train models
python -m src.train --input data/processed/abt_features.csv --config configs/model_config.yaml

# Step 5: Evaluate
python -m src.evaluate --models-dir models --test-data data/processed/abt_features.csv

# Step 6: Inference on new data
python -m src.inference --model models/gradient_boosting.pkl --input data/processed/new_accounts.csv
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## Project Stages

| # | Stage | Status |
|---|---|---|
| 1 | Problem Definition & Success Criteria | ✅ |
| 2 | Project Structure & Setup | ✅ |
| 3 | Data Acquisition & EDA | ⏳ Next |
| 4 | Data Cleaning & Preprocessing | |
| 5 | Feature Engineering | |
| 6 | Model Selection & Training | |
| 7 | Hyperparameter Tuning | |
| 8 | Model Evaluation & Interpretation | |
| 9 | Dashboard / API / Deployment | |
| 10 | Documentation & Handoff | |

---

## Models

| Model | Purpose |
|---|---|
| `DummyClassifier` | Majority-class baseline (floor to beat) |
| `LogisticRegression` | Interpretable linear model, balanced weights |
| `RandomForestClassifier` | Bagging ensemble, balanced weights |
| `GradientBoostingClassifier` | **Primary candidate** — sequential boosting |

---

## Key Design Decisions

1. **Recall-first optimization** — Missing a churner costs more than a false alarm
2. **Leakage guard** — Automatic detection of post-outcome features via AUC scanning
3. **Risk tiering** — Accounts scored into Low / Medium / High / Critical tiers
4. **Pipeline modularity** — Each stage is independently runnable via CLI
5. **No XGBoost/LightGBM** — `GradientBoostingClassifier` from scikit-learn only

---

## License

This project uses the [MIT Licensed](https://www.mit.edu/~amini/LICENSE.md) RavenStack dataset by Rivalytics.
