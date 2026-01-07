import pandas as pd
import numpy as np

class InferenceService:    
    def __init__(self, model):
        self.model = model

    def _preprocess(self, data: dict) -> pd.DataFrame:
        """
        Prépare les données d'entrée pour le modèle.
        Args:
            data (dict): données brutes d'entrée
        Returns:
            pd.DataFrame
        """
        return pd.DataFrame([data])

    def _postprocess(self, predictions):
        """
        Nettoie la sortie du modèle pour la rendre sérialisable en JSON.
        """
        if isinstance(predictions, (np.ndarray, list)):
            return predictions[0]
        return predictions

    def predict(self, data: dict):
        """
        Pipeline complet d'inférence.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")

        # 1. Preprocess
        input_df = self._preprocess(data)

        # 2. Predict
        preds = self.model.predict(input_df)

        # 3. Postprocess
        result = self._postprocess(preds)

        return result