import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from mlflow_manager import MLflowManager

class SpamTrainer:
    def __init__(self):
        self.config = Config
        self.mlflow = MLflowManager(
            tracking_uri=self.config.MLFLOW_TRACKING_URI,
            experiment_name=self.config.EXPERIMENT_NAME
        )

    def load_data(self):
        data = pd.read_csv(self.config.DATA_PATH)
        X = data["text"]
        y = data["label"]
        return X, y

    def preprocess(self, X_train, X_test):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.config.MAX_FEATURES
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    def getmodel(self):
        # Load data
       


        # Model
        model = LogisticRegression(max_iter=self.config.MAX_ITER)
        return model
