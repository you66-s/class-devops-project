import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# MLflow configuration
# =========================
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("spam-classification")


# =========================
# Load dataset
# =========================
DATA_PATH = "/data/spam.csv"   # dataset monté par Docker volume

data = pd.read_csv(DATA_PATH)

# Adapter si nécessaire selon ton dataset
X = data["text"]
y = data["label"]


# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# Model training
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")


# =========================
# MLflow logging
# =========================
with mlflow.start_run():

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("max_features", 5000)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Save vectorizer for inference
    joblib.dump(vectorizer, "vectorizer.joblib")
    mlflow.log_artifact("vectorizer.joblib")


print("Training completed successfully")
print(f"Accuracy : {accuracy:.4f}")
