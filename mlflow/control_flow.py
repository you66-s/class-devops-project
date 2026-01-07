import sys
import os
from utils import MLflowManager

# Import dynamique du dossier 'ml'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.train import train_model

def main():
    # Initialisation de ton gestionnaire
    manager = MLflowManager()

    # Déterminer le fichier CSV à utiliser
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/spam_dataset.csv"

    if os.path.exists(data_path):
        print(f"⚙️ Exécution du pipeline pour : {data_path}")
        
        # 1. Entraînement (Code du collègue)
        model, params, metrics, X_test = train_model(data_path)
        
        # 2. Logging et Enregistrement (Ton code)
        manager.log_training_run(model, params, metrics, X_test)
    else:
        print(f"❌ Erreur : {data_path} introuvable.")

if __name__ == "__main__":
    main()