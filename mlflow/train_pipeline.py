# Import dynamique du dossier 'ml'
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.train import SpamTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow


def run_training():
    # 1. Initialisation
    trainer = SpamTrainer()
    
    print("📦 Chargement des données...")
    X, y = trainer.load_data()
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Prétraitement (Vectorisation)
    print("🔍 Prétraitement des données (TF-IDF)...")
    X_train_vec, X_test_vec = trainer.preprocess(X_train, X_test)
    
    # 4. Récupération du modèle
    model = trainer.getmodel()
    
    # 5. Entraînement et Logging via ta classe MLflowManager
    print("🚀 Entraînement et enregistrement MLflow...")
    
    # On définit les hyperparamètres pour le log
    params = {
        "max_iter": trainer.config.MAX_ITER,
        "max_features": trainer.config.MAX_FEATURES,
        "solver": "lbfgs"
    }

    # Entraînement local pour calcul des métriques
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='spam'),
        "recall": recall_score(y_test, y_pred, pos_label='spam'),
        "f1_score": f1_score(y_test, y_pred, pos_label='spam')
    }

    # Utilisation de ta méthode de classe pour le logging
    # Note : On passe X_test_vec pour la signature du modèle
    model_uri = trainer.mlflow.log_training_run(
        model=model,
        params=params,
        metrics=metrics,
        X_sample=X_test_vec,
        model_name="SpamClassifier"
    )

    print(f"✅ Terminé ! Modèle disponible ici : {model_uri}")
    return model_uri

if __name__ == "__main__":
    run_training()