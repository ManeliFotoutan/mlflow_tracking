import mlflow
from mlflow.tracking import MlflowClient

model_name = "RandomForestModel"
client = MlflowClient()

model_versions = client.get_latest_versions(model_name)
if model_versions:
    latest_version = model_versions[0]
    
    run_id = latest_version.run_id
    
    accuracy_metric = client.get_metric_history(run_id, "accuracy")[-1].value
    
    if accuracy_metric > 0.8: 
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production"
        )
        print(f"Model version {latest_version.version} transitioned to Production.")
    else:
        print("Model does not meet the required accuracy threshold for production.")
