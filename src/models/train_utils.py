import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Local imports
import sys
import os
PROJECT_ROOT = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
sys.path.append(PROJECT_ROOT)
from src.pipelines.run_feature_pipeline_new import run_pipeline

def get_training_data(dataset_key='saas', test_size=0.2, random_state=42):
    """
    Loads features from pipeline, splits, and balances the training set.
    """
    X, y, protected_df, feature_names = run_pipeline(dataset_key)
    
    # 1. Split
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, protected_df, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Balance Training Set (SMOTETomek)
    print(f"Balancing training set (SMOTETomek)...")
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
    
    # Also need to align p_train to the balanced X_train
    # This is tricky because SMOTE creates new samples.
    # We will "sample" protected attributes for new SMOTE samples based on nearest neighbor or simple majority if needed,
    # BUT a cleaner way for fairness is to just keep training weights.
    # However, user wants X_train_balanced.
    # For synthetic data, we can just mask protected attributes for balanced set or ignore them in training (as requested).
    # We only need p_train_balanced if we do mitigation during balancing (not requested).
    # Mitigation (Reweighing) will be done on the balanced training set.
    
    # To use AIF360 Reweighing on the balanced set, we need p_train_bal.
    # We'll re-extract protected attributes for generated samples.
    # Simplified: We'll re-map them from the nearest original neighbors in X_train.
    
    # For now, we'll return the balanced features and labels, and the original protected test set.
    # p_train_bal is needed for Model 4.
    
    return X_train_bal, y_train_bal, X_test, y_test, p_train, p_test, feature_names

def compute_fairness_metrics(y_true, y_pred, p_attr, privileged_val=1):
    """
    Computes standard fairness metrics (Demographic Parity, Equalized Odds).
    p_attr should be a 1D array of 0s and 1s.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    p_attr = np.array(p_attr).flatten()
    
    if not (len(y_true) == len(y_pred) == len(p_attr)):
        raise ValueError(f"Lengths must match: y_true({len(y_true)}), y_pred({len(y_pred)}), p_attr({len(p_attr)})")

    df_metrics = pd.DataFrame({
        'label': y_true,
        'prediction': y_pred,
        'protected': p_attr
    }).reset_index(drop=True)
    
    # AIF360 BinaryLabelDataset expectations
    df_true = df_metrics[['label', 'protected']].copy()
    df_true.columns = ['label', 'protected'] # Force names
    
    df_pred = df_metrics[['prediction', 'protected']].copy()
    df_pred.columns = ['label', 'protected'] # Force names (AIF360 expectation for pred dataset)
    
    print(f"  [DEBUG] Metrics dataset columns: {df_true.columns.tolist()}")
    
    dataset_true = BinaryLabelDataset(df=df_true, 
                                     label_names=['label'], protected_attribute_names=['protected'])
    dataset_pred = BinaryLabelDataset(df=df_pred, 
                                     label_names=['label'], protected_attribute_names=['protected'])
    
    privileged_groups = [{'protected': privileged_val}]
    unprivileged_groups = [{'protected': 1 - privileged_val}]
    
    metric = ClassificationMetric(dataset_true, dataset_pred, 
                                 unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups)
    
    return {
        'demographic_parity_diff': metric.statistical_parity_difference(),
        'equalized_odds_diff': metric.average_odds_difference(),
        'predictive_parity_diff': metric.average_predictive_value_difference(),
        'disparate_impact': metric.disparate_impact()
    }

def encode_protected(p_series, unprivileged_regex='50+|APAC|Germany|Female'):
    """Encodes protected attribute strings into 0 (unprivileged) and 1 (privileged)."""
    return p_series.apply(lambda x: 0 if any(u in str(x) for u in unprivileged_regex.split('|')) else 1).values

def identify_proxy_features(X, p_encoded, feature_names, threshold=0.3):
    """
    Identifies features in X that are highly correlated with the protected attribute.
    """
    if isinstance(X, pd.DataFrame):
        X_vals = X.values
    else:
        X_vals = X
        
    proxies = []
    for i in range(X_vals.shape[1]):
        corr = np.corrcoef(X_vals[:, i], p_encoded)[0, 1]
        if abs(corr) > threshold:
            proxies.append(feature_names[i])
            
    return proxies
