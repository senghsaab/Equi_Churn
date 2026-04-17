import os
import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Local imports
import sys
PROJECT_ROOT = r"C:\Users\Lenovo\Design Thinking And Innovation Project"
sys.path.append(PROJECT_ROOT)
from src.features.feature_engineering import EquiChurnFeatureEngineer

def run_pipeline(dataset_key='saas'):
    # 1. Load Config
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'feature_config.yaml')
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    cfg = full_config[dataset_key]
    
    # 2. Load Raw Data
    if dataset_key == 'saas':
        data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'churn_dataset.csv')
    else:
        data_path = os.path.join(PROJECT_ROOT, 'data', 'external', 'Churn_Modelling.csv')
    
    df = pd.read_csv(data_path)
    print(f"\n--- Processing {cfg['dataset_name']} ---")
    
    # 3. Create Pipeline Components
    engineer = EquiChurnFeatureEngineer(dataset_name=dataset_key, config=cfg)
    
    # Fit/Transform engineering first to get the columns for ColumnTransformer
    df_engineered = engineer.transform(df)
    
    numeric_features = cfg['numeric_features']
    categorical_features = cfg['categorical_features']
    binary_flags = cfg.get('binary_flags', [])
    
    # Note: binary_flags are usually just label encoded or passed through if already 0/1
    # We will treat them as numeric for StandardScaler to keep it simple, or passthrough
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_flags)
    ], remainder='drop')

    # 4. Extract Target and Protected Attributes
    y = df[cfg['target_column']].values
    protected_cols = cfg['protected_attributes']
    protected_df = df_engineered[protected_cols].copy()
    
    # 5. Fit Preprocessor on Engineered Data
    X_transformed = preprocessor.fit_transform(df_engineered)
    
    # Get feature names after OHE
    try:
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = list(numeric_features) + list(cat_feature_names) + list(binary_flags)
    except:
        feature_names = ["Feature_" + str(i) for i in range(X_transformed.shape[1])]

    X_final = pd.DataFrame(X_transformed, columns=feature_names)

    # 6. LEAKAGE AUDIT
    print("\n[LEAKAGE AUDIT]")
    leakage_cols = ['age_group', 'region', 'Geography', 'Gender', 'cancellation_reason', 'churn_date']
    found_leakage = [col for col in leakage_cols if col in X_final.columns]
    
    assert len(found_leakage) == 0, f"Leakage detected: {found_leakage}"
    for p_col in protected_cols:
        assert p_col not in X_final.columns, f"Protected attribute {p_col} found in feature matrix!"
    
    print("  [OK] Leakage audit passed - no protected attributes or post-churn features in feature matrix")

    # 7. Save Artifacts
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(PROJECT_ROOT, 'models', f'preprocessor_{dataset_key}_{ts}.pkl')
    joblib.dump(preprocessor, model_path)
    print(f"  [OK] Fitted preprocessor saved to {model_path}")
    
    return X_final, y, protected_df, feature_names

if __name__ == "__main__":
    X_s, y_s, prot_s, names_s = run_pipeline('saas')
    print(f"Final SaaS feature matrix shape: {X_s.shape}")
    
    X_b, y_b, prot_b, names_b = run_pipeline('bank')
    print(f"Final Bank feature matrix shape: {X_b.shape}")
