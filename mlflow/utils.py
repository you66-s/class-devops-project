import mlflow
import os
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

def log_training_run(model, params, metrics, X_sample, model_name="SpamClassifier"):
    with mlflow.start_run() as run:
        # 1. Generate Signature
        signature = infer_signature(X_sample, model.predict(X_sample))

        # 2. Log Data
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # 3. Log Model & Register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature
        )

        if os.path.exists("confusion_matrix.png"):
            mlflow.log_artifact("confusion_matrix.png")

        # 4. Get URI for the Backend
        model_uri = f"models:/{model_name}/latest" # Points to the latest version
        print(f"Run logged. Model URI: {model_uri}")
        return model_uri

def promote_to_production(model_name, version):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )