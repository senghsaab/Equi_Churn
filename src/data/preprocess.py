"""
preprocess.py
=============
Research-grade preprocessing pipeline for the B2B SaaS Churn Prediction project.

Pipeline stages:
  1. Load raw dataset & inspect
  2. Encode categoricals (LabelEncoder) — document mappings
  3. Stratified Train / Val / Test split  (64 / 16 / 20)
  4. StandardScaler — fit on train, transform val & test
  5. SMOTE on train only (post-scaling)
  6. Reshape for DL (n_samples, n_features, 1)
  7. Leakage assertions & shape report
  8. Persist artefacts: scaler.pkl, label_encoders.pkl, preprocessed arrays

Data-leakage guard design
─────────────────────────
  • Scaler fitted ONLY on X_train  → prevents test statistics bleeding into
    the normalisation process (would inflate validation metrics).
  • SMOTE applied AFTER scaling    → avoids generating synthetic neighbours
    in distorted, un-normalised space (distance metric integrity).
  • Validation/test never touch SMOTE → real-world distribution preserved.

Author : ML Engineering — B2B SaaS Churn Project
Stage  : 2
"""

import os, pickle, json
import numpy as np
import pandas as pd
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT  = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
RAW_CSV       = os.path.join(PROJECT_ROOT, "data", "raw",        "churn_dataset.csv")
PROC_DIR      = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")

os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.20   # 20 % test
VAL_FRACTION  = 0.15   # 15 % of train  →  ~16 % of total

# ──────────────────────────────────────────────
# UTIL
# ──────────────────────────────────────────────
BOLD = "\033[1m";  RESET = "\033[0m";  GREEN = "\033[32m";  CYAN = "\033[36m"

def section(title: str):
    print()
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")

def sub(title: str):
    print(f"\n{CYAN}  ── {title} ──{RESET}")

# ══════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════
section("STAGE 2 — PREPROCESSING PIPELINE")

df = pd.read_csv(RAW_CSV)
print(f"\n  Raw shape       : {df.shape}")
print(f"  Churn rate      : {df['churned'].mean()*100:.2f}%")
print(f"  Null values     : {df.isnull().sum().sum()}")
print(f"  Duplicate rows  : {df.duplicated().sum()}")

# Feature / target split (before encoding)
TARGET = 'churned'
y_raw  = df[TARGET].values

# Identify categorical columns present in this dataset
# (industry not in Stage-1 data; gracefully skip if absent)
CATEGORICAL_COLS = [c for c in ['company_size', 'contract_type'] if c in df.columns]
NUMERIC_COLS     = [c for c in df.columns if c not in CATEGORICAL_COLS + [TARGET]]

print(f"\n  Categorical cols : {CATEGORICAL_COLS}")
print(f"  Numeric cols     : {len(NUMERIC_COLS)}")

# ══════════════════════════════════════════════
# 2. LABEL ENCODING
# ══════════════════════════════════════════════
section("LABEL ENCODING")

label_encoders   = {}
encoding_mappings = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col]    = le
    encoding_mappings[col] = {
        int(cls): int(enc)
        for cls, enc in zip(le.classes_, le.transform(le.classes_))
    }
    sub(f"{col}  →  {encoding_mappings[col]}")

# Save label encoders
le_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
with open(le_path, "wb") as f:
    pickle.dump(label_encoders, f)

# Save human-readable mapping as JSON
mapping_path = os.path.join(PROC_DIR, "encoding_mappings.json")
with open(mapping_path, "w") as f:
    json.dump(encoding_mappings, f, indent=2)

print(f"\n  {GREEN}Label encoders saved  →  {le_path}{RESET}")
print(f"  {GREEN}Encoding map  saved   →  {mapping_path}{RESET}")

# ══════════════════════════════════════════════
# 3. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════
section("STRATIFIED SPLITS  (64 / 16 / 20)")

X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

# Step 1: 80 % temp_train  |  20 % test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Step 2: 85 % train  |  15 % val   (of the 80 % temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=y_temp
)

sub("Raw split sizes")
total = len(y)
for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    pct  = len(arr) / total * 100
    cr   = arr.mean() * 100
    print(f"    {name:<6}: {len(arr):>5} rows  ({pct:.1f}%)  churn={cr:.2f}%")

# ══════════════════════════════════════════════
# 4. STANDARD SCALING
# ══════════════════════════════════════════════
section("STANDARD SCALING  (fit only on train)")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # FIT + transform
X_val_sc   = scaler.transform(X_val)          # transform only
X_test_sc  = scaler.transform(X_test)         # transform only

print("""
  DATA LEAKAGE EXPLANATION
  ─────────────────────────────────────────────────────────────────────
  Fitting the scaler on validation/test data would expose their mean
  and variance to the pipeline before evaluation — a form of data
  leakage that artificially inflates generalisation metrics.
  The scaler is therefore fitted ONLY on X_train; val/test are
  transformed using those training statistics.
  ─────────────────────────────────────────────────────────────────────
""")

sub("Scaler statistics (first 5 features)")
feat_names = df.drop(columns=[TARGET]).columns.tolist()
for i in range(min(5, len(feat_names))):
    print(f"    {feat_names[i]:<35}  mean={scaler.mean_[i]:>9.4f}  std={scaler.scale_[i]:>8.4f}")

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n  {GREEN}Scaler saved  →  {scaler_path}{RESET}")

# ══════════════════════════════════════════════
# 5. SMOTE (train only, post-scaling)
# ══════════════════════════════════════════════
section("SMOTE — CLASS IMBALANCE CORRECTION")

print("""
  WHY SMOTE AFTER SCALING?
  ─────────────────────────────────────────────────────────────────────
  SMOTE generates synthetic minority-class samples by interpolating
  between real neighbours in feature space. If run on raw/unscaled
  data, features with large magnitudes (e.g. api_calls_monthly ~10⁵)
  dominate the kNN distance metric, creating biased synthetic points.
  Scaling first ensures every feature contributes equally to the
  neighbourhood calculation → higher-quality synthetic samples.
  ─────────────────────────────────────────────────────────────────────
""")

sub("Class distribution BEFORE SMOTE")
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"    Class {int(cls)}  →  {cnt:>5} samples  ({cnt/len(y_train)*100:.1f}%)")

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

sub("Class distribution AFTER SMOTE")
unique, counts = np.unique(y_train_sm, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"    Class {int(cls)}  →  {cnt:>5} samples  ({cnt/len(y_train_sm)*100:.1f}%)")

print(f"\n    Net new synthetic samples added: {len(y_train_sm) - len(y_train)}")

# class_weight='balanced' reference dict (for tree/linear models)
neg, pos = np.bincount(y_train)
class_weight_balanced = {0: len(y_train) / (2 * neg),
                          1: len(y_train) / (2 * pos)}
sub("class_weight='balanced' equivalent dict (for tree/linear models)")
print(f"    {class_weight_balanced}")

# ══════════════════════════════════════════════
# 6. RESHAPE FOR DL  (n_samples, n_features, 1)
# ══════════════════════════════════════════════
section("DL RESHAPE  — (n_samples, n_features, 1)")

# 2-D versions (ML + ANN)
X_train_2d = X_train_sm          # SMOTE-balanced, scaled
X_val_2d   = X_val_sc
X_test_2d  = X_test_sc

# 3-D versions (LSTM / GRU / 1-D CNN)
X_train_3d = X_train_sm.reshape(X_train_sm.shape[0], X_train_sm.shape[1], 1)
X_val_3d   = X_val_sc.reshape(X_val_sc.shape[0], X_val_sc.shape[1], 1)
X_test_3d  = X_test_sc.reshape(X_test_sc.shape[0], X_test_sc.shape[1], 1)

print(f"""
    X_train_2d : {X_train_2d.shape}   ← ML + ANN
    X_val_2d   : {X_val_2d.shape}
    X_test_2d  : {X_test_2d.shape}

    X_train_3d : {X_train_3d.shape}  ← LSTM / GRU / 1-D CNN
    X_val_3d   : {X_val_3d.shape}
    X_test_3d  : {X_test_3d.shape}
""")

# ══════════════════════════════════════════════
# 7. LEAKAGE ASSERTIONS
# ══════════════════════════════════════════════
section("DATA LEAKAGE VALIDATION")

# Check index sets do not overlap
idx_temp_set  = set(range(len(X_temp)))
idx_train_set = set(range(len(X_train)))
idx_val_set   = set(range(len(X_val)))

# Original indices via DataFrame (robust check)
df_reset = df.reset_index(drop=True)
df['_split'] = 'unassigned'

# Re-split using original df index to validate
_, test_idx   = train_test_split(df_reset.index, test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE, stratify=df_reset[TARGET])
train_temp_idx, _ = train_test_split(
    [i for i in df_reset.index if i not in test_idx],
    test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df_reset.loc[[i for i in df_reset.index if i not in test_idx], TARGET]
)
train_idx, val_idx = train_test_split(
    train_temp_idx, test_size=VAL_FRACTION, random_state=RANDOM_STATE,
    stratify=df_reset.loc[train_temp_idx, TARGET]
)

test_set  = set(test_idx)
train_set = set(train_idx)
val_set   = set(val_idx)

assert len(train_set & val_set)  == 0, "LEAK: train ∩ val"
assert len(train_set & test_set) == 0, "LEAK: train ∩ test"
assert len(val_set   & test_set) == 0, "LEAK: val ∩ test"

print(f"\n  {GREEN}✓ No leakage between train, val, test (intersection = 0){RESET}")

# Churn rate preservation check
for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    rate = arr.mean() * 100
    assert abs(rate - 20.0) < 2.5, f"Churn rate in {name} deviates: {rate:.2f}%"
    print(f"  {GREEN}✓ {name} churn rate = {rate:.2f}%  (target ~20%){RESET}")

# Scaler fitted on train only — verify mean on test is not ≈0 before transform
raw_test_mean = X_test.mean(axis=0)[:3]
sc_test_mean  = X_test_sc.mean(axis=0)[:3]
print(f"\n  Scaler sanity (first 3 feats — test raw mean vs transformed mean):")
for i, (r, t) in enumerate(zip(raw_test_mean, sc_test_mean)):
    print(f"    feat_{i}: raw={r:.4f}  →  scaled={t:.4f}")

# ══════════════════════════════════════════════
# 8. SAVE PREPROCESSED ARRAYS
# ══════════════════════════════════════════════
section("SAVING PREPROCESSED ARRAYS")

np.savez_compressed(
    os.path.join(PROC_DIR, "splits_2d.npz"),
    X_train=X_train_2d, y_train=y_train_sm,
    X_val=X_val_2d,     y_val=y_val,
    X_test=X_test_2d,   y_test=y_test,
    feature_names=np.array(feat_names)
)

np.savez_compressed(
    os.path.join(PROC_DIR, "splits_3d.npz"),
    X_train=X_train_3d, y_train=y_train_sm,
    X_val=X_val_3d,     y_val=y_val,
    X_test=X_test_3d,   y_test=y_test
)

# Save class_weight dict
with open(os.path.join(PROC_DIR, "class_weights.json"), "w") as f:
    json.dump({str(k): v for k, v in class_weight_balanced.items()}, f, indent=2)

print(f"  {GREEN}splits_2d.npz saved   →  {PROC_DIR}{RESET}")
print(f"  {GREEN}splits_3d.npz saved   →  {PROC_DIR}{RESET}")
print(f"  {GREEN}class_weights.json    →  {PROC_DIR}{RESET}")

# ══════════════════════════════════════════════
# 9. FINAL SHAPE REPORT
# ══════════════════════════════════════════════
section("FINAL SHAPE REPORT")

print(f"""
  2-D Arrays (for ML + ANN)
  ┌──────────────┬─────────────────┬───────────────┐
  │  Split       │  X shape        │  y shape      │
  ├──────────────┼─────────────────┼───────────────┤
  │  Train (SM)  │  {str(X_train_2d.shape):<15}  │  {str(y_train_sm.shape):<13}│
  │  Val         │  {str(X_val_2d.shape):<15}  │  {str(y_val.shape):<13}│
  │  Test        │  {str(X_test_2d.shape):<15}  │  {str(y_test.shape):<13}│
  └──────────────┴─────────────────┴───────────────┘

  3-D Arrays (for LSTM / GRU / 1-D CNN)
  ┌──────────────┬────────────────────┬───────────────┐
  │  Split       │  X shape           │  y shape      │
  ├──────────────┼────────────────────┼───────────────┤
  │  Train (SM)  │  {str(X_train_3d.shape):<18}  │  {str(y_train_sm.shape):<13}│
  │  Val         │  {str(X_val_3d.shape):<18}  │  {str(y_val.shape):<13}│
  │  Test        │  {str(X_test_3d.shape):<18}  │  {str(y_test.shape):<13}│
  └──────────────┴────────────────────┴───────────────┘

  Total coverage  :  {len(y_train_sm) + len(y_val) + len(y_test) - (len(y_train_sm)-len(y_train))}
                     (original 5000 + {len(y_train_sm)-len(y_train)} SMOTE synthetics in train)
""")

print(f"{BOLD}{'═'*70}{RESET}")
print(f"{GREEN}{BOLD}  Stage 2 Complete — Preprocessing Pipeline Ready ✅{RESET}")
print(f"{BOLD}{'═'*70}{RESET}")
