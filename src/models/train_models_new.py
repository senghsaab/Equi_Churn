import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, classification_report)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Fairness Mitigation (AIF360)
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

# Local imports
PROJECT_ROOT = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
sys.path.append(PROJECT_ROOT)
from src.models.train_utils import get_training_data, compute_fairness_metrics, encode_protected

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("equichurn_fairness_prediction")

def log_all_metrics(model, X_test, y_test, p_test_df, feature_names, run_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Standard Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    # Fairness Metrics (Age & Region)
    for attr in ['age_group', 'region']:
        if attr in p_test_df.columns:
            p_encoded = encode_protected(p_test_df[attr], unprivileged_regex='50+|APAC')
            f_metrics = compute_fairness_metrics(y_test, y_pred, p_encoded)
            metrics[f"demographic_parity_{attr}"] = f_metrics['demographic_parity_diff']
            metrics[f"equalized_odds_{attr}"] = f_metrics['equalized_odds_diff']
            metrics[f"is_fair_dp_{attr}"] = int(abs(f_metrics['demographic_parity_diff']) < 0.1)

    mlflow.log_metrics(metrics)
    print(f"  [METRICS] {run_name} ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics

# ──────────────────────────────────────────────────────────────────────────
# MAIN TRAINING WORKFLOW
# ──────────────────────────────────────────────────────────────────────────

def train_equichurn():
    # 1. Prepare Data
    X_train_bal, y_train_bal, X_test, y_test, p_train, p_test, feature_names = get_training_data('saas')
    
    # For AIF360, we need p_train that aligns with X_train_bal.
    # Since SMOTE created synthetic samples, we'll re-extract protected attributes from nearest neighbors.
    # For simplicity in this script, we'll use original p_train matched to balanced set indices,
    # or just use the balanced feature matrix if those columns were kept (they aren't).
    # We'll re-encode p_train_bal based on the feature matrix's proximity to original samples if needed.
    # ALTERNATIVE: Use the original X_train for mitigation reweighing.
    
    print("\nStarting Model Training Pipeline...")
    results = {}

    # MODEL 1 — Random Forest with Optuna
    with mlflow.start_run(run_name="RandomForest_Optuna"):
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced'
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            return cross_val_score(model, X_train_bal, y_train_bal, cv=3, scoring='roc_auc').mean()
        
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(rf_objective, n_trials=10) # Reduced for speed
        best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
        best_rf.fit(X_train_bal, y_train_bal)
        
        mlflow.log_params(study_rf.best_params)
        results['RF'] = (best_rf, log_all_metrics(best_rf, X_test, y_test, p_test, feature_names, "RandomForest"))
        mlflow.sklearn.log_model(best_rf, "rf_model")

    # MODEL 2 — LightGBM with Optuna
    with mlflow.start_run(run_name="LightGBM_Optuna"):
        def lgbm_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'verbose': -1,
                'n_jobs': -1
            }
            model = LGBMClassifier(**params, random_state=42)
            return cross_val_score(model, X_train_bal, y_train_bal, cv=3, scoring='roc_auc').mean()
        
        study_lgbm = optuna.create_study(direction='maximize')
        study_lgbm.optimize(lgbm_objective, n_trials=10)
        best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42)
        best_lgbm.fit(X_train_bal, y_train_bal)
        
        mlflow.log_params(study_lgbm.best_params)
        results['LGBM'] = (best_lgbm, log_all_metrics(best_lgbm, X_test, y_test, p_test, feature_names, "LightGBM"))
        mlflow.lightgbm.log_model(best_lgbm, "lgbm_model")

    # MODEL 3 — XGBoost Baseline with Optuna
    with mlflow.start_run(run_name="XGBoost_Baseline_Optuna"):
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'use_label_encoder': False,
                'eval_metric': 'auc'
            }
            model = XGBClassifier(**params, random_state=42)
            return cross_val_score(model, X_train_bal, y_train_bal, cv=3, scoring='roc_auc').mean()
        
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(xgb_objective, n_trials=10)
        best_xgb_base = XGBClassifier(**study_xgb.best_params, random_state=42)
        best_xgb_base.fit(X_train_bal, y_train_bal)
        
        mlflow.log_params(study_xgb.best_params)
        results['XGB_Baseline'] = (best_xgb_base, log_all_metrics(best_xgb_base, X_test, y_test, p_test, feature_names, "XGB_Baseline"))
        mlflow.xgboost.log_model(best_xgb_base, "xgb_baseline_model")

    # MODEL 4 — XGBoost Mitigated via AIF360
    with mlflow.start_run(run_name="XGBoost_Mitigated_AIF360"):
        # For Reweighing, we need protected attributes aligned with balanced features
        from sklearn.neighbors import KNeighborsClassifier
        
        # Align p_train with X_train_bal (for SMOTE samples, take majority of 3 nearest neighbors)
        print("Aligning protected attributes for balanced training set...")
        # Encode for neighbor classification
        p_age_enc = encode_protected(p_train['age_group'])
        nn = KNeighborsClassifier(n_neighbors=1).fit(X_train_bal[:len(p_train)], p_age_enc)
        p_train_bal_age = nn.predict(X_train_bal)
        
        df_aif = pd.DataFrame(X_train_bal, columns=feature_names).reset_index(drop=True)
        df_aif['age_group_enc'] = p_train_bal_age
        df_aif['churn_flag'] = y_train_bal
        
        aif_ds = BinaryLabelDataset(
            df=df_aif[['age_group_enc', 'churn_flag']], 
            label_names=['churn_flag'], 
            protected_attribute_names=['age_group_enc']
        )
        
        rw = Reweighing(
            unprivileged_groups=[{'age_group_enc': 0}], 
            privileged_groups=[{'age_group_enc': 1}]
        )
        transformed_ds = rw.fit_transform(aif_ds)
        weights = transformed_ds.instance_weights
        
        best_xgb_mitigated = XGBClassifier(**study_xgb.best_params, random_state=42)
        best_xgb_mitigated.fit(X_train_bal, y_train_bal, sample_weight=weights)
        
        # Calibration
        print("Applying Isotonic Calibration...")
        calibrated_model = CalibratedClassifierCV(best_xgb_mitigated, method='isotonic', cv=3)
        calibrated_model.fit(X_train_bal, y_train_bal)
        
        results['XGB_Mitigated'] = (calibrated_model, log_all_metrics(calibrated_model, X_test, y_test, p_test, feature_names, "XGB_Mitigated"))
        mlflow.sklearn.log_model(calibrated_model, "production_model")
        
        # Calibration Curve
        prob_pos = calibrated_model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated XGB")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.title("Calibration Curve (Isotonic)")
        plt.xlabel("Mean predicted value")
        plt.ylabel("Fraction of positives")
        plt.legend()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/calibration_curve.png")
        print("  [OK] Saved reports/figures/calibration_curve.png")

    # ──────────────────────────────────────────────────────────────────────
    # COMBINED ROC PLOT
    # ──────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    for name, (model, metrics) in results.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Threshold=0.5')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves — EquiChurn Models')
    plt.legend(loc='lower right')
    plt.savefig('reports/figures/roc_curves_all_models.png')
    print("  [OK] Saved reports/figures/roc_curves_all_models.png")

    print("\n" + "="*80)
    print("  MODEL TRAINING COMPLETE")
    print("="*80)
    print("  View MLflow UI: run 'mlflow ui' in terminal")
    print("  Open: http://127.0.0.1:5000")
    print(f"\n  Production model: XGBoost Mitigated (Calibrated)")
    print(f"  Accuracy Drop vs LGBM: {results['LGBM'][1]['accuracy'] - results['XGB_Mitigated'][1]['accuracy']:.4%}")

if __name__ == "__main__":
    train_equichurn()
