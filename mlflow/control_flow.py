import sys
import os
import mlflow
import pandas as pd

# Ajoute le dossier parent au chemin de recherche de Python
# Cela permet d'accéder au dossier voisin 'ml'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.train import train_model  # Importation depuis le dossier ml
from utils import log_training_run

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Email_Spam_Detection")

def run_automation():
    # Le chemin doit remonter d'un cran pour trouver /data
    data_path = "../data/spam_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    print(f"Démarrage de l'entraînement...")
    
    # Appel de la fonction du collègue
    model, params, metrics, X_test = train_model(data_path)
    
    # Logging
    uri = log_training_run(model, params, metrics, X_test)
    print(f"Succès ! Modèle enregistré : {uri}")

if __name__ == "__main__":
    run_automation()