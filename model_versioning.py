import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np
from mlflow.tracking import MlflowClient

# Generating data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Parameters
learning_rate = 0.02
batch_size = 40

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log a new version of the model
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    input_example = np.array(X_train[:1]) 
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    # Register the model in MLflow
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "RandomForestModel")
    
    print(f"New model version registered with accuracy: {accuracy}")

# Now compare the new version with the previous versions
client = MlflowClient()

# Fetch the versions of the model
model_versions = client.get_latest_versions("RandomForestModel")

# Iterate through all versions and print their accuracy
for version in model_versions:
    run_id = version.run_id  # Get the run_id associated with the model version
    accuracy_metric = client.get_metric_history(run_id, "accuracy")[-1].value  # Fetch accuracy metric from the run
    print(f"Version: {version.version}, Accuracy: {accuracy_metric}")
    
    # You can implement additional logic here for comparing and promoting models
