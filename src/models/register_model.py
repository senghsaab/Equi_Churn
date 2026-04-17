import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "file:///C:/Users/Lenovo/Design%20Thinking%20And%20Innovation%20Project/mlruns"
EXPERIMENT_NAME = "equichurn_fairness_prediction"
MODEL_NAME = "equichurn_xgb_mitigated"
RUN_NAME_TARGET = "XGBoost_Mitigated_AIF360"

def register_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # 1. Get Experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment {EXPERIMENT_NAME} not found.")
        return

    # 2. Find the best run for XGBoost_Mitigated
    # We look for the run that finished successfully and has the specific target name
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME_TARGET}'",
        order_by=["metrics.roc_auc DESC"]
    )
    
    if not runs:
        print(f"No runs found with name {RUN_NAME_TARGET}")
        return
        
    best_run = runs[0]
    run_id = best_run.info.run_id
    print(f"Found best run: {run_id} (AUC: {best_run.data.metrics.get('roc_auc', 'N/A')})")

    # 3. Register the model
    model_uri = f"runs:/{run_id}/production_model" # We named the artifact 'production_model' in training
    # Check if we should use 'xgb_mitigated_model' or 'production_model'
    # Actually, in train_models_new.py: mlflow.sklearn.log_model(calibrated_model, "production_model")
    
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered model {MODEL_NAME} version {result.version}")

    # 4. Promote to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {MODEL_NAME} version {result.version} promoted to Production stage.")

if __name__ == "__main__":
    register_production_model()
