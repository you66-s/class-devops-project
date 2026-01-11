import time
import os
import mlflow
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from spam_trainer import SpamTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

class MLOpsHandler(FileSystemEventHandler):
    def __init__(self):
        self.trainer = SpamTrainer()
        self.last_run = 0  # Pour éviter les déclenchements multiples trop rapides

    def on_modified(self, event):
        # 1. On vérifie si c'est le dataset ou le code du modèle qui a changé
        is_data = event.src_path.endswith(os.path.basename(self.trainer.config.DATA_PATH))
        is_model_code = "spam_trainer.py" in event.src_path
        
        if (is_data or is_model_code) and (time.time() - self.last_run > 5):
            self.last_run = time.time()
            reason = "DONNÉES" if is_data else "CODE MODÈLE"
            print(f"🔄 Changement détecté ({reason}) : {event.src_path}")
            self.run_pipeline()

    def run_pipeline(self):
        print("🚀 Démarrage du pipeline MLflow...")
        try:
            # --- 1. Préparation ---
            X, y = self.trainer.load_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- 2. Preprocessing & Entraînement ---
            # Important : Le vectorizer fait partie du modèle !
            X_train_vec, X_test_vec = self.trainer.preprocess(X_train, X_test)
            model = self.trainer.getmodel()
            model.fit(X_train_vec, y_train)

            # --- 3. Évaluation ---
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)

            # --- 4. Logging MLflow (Via ton Manager) ---
            # On ouvre un bloc mlflow ici pour logger les artefacts manuellement
            with mlflow.start_run(nested=True):
                # Enregistre le dataset utilisé pour la traçabilité
                mlflow.log_artifact(self.trainer.config.DATA_PATH, artifact_path="dataset_used")
                
                # Enregistre le vectorizer (obligatoire pour le backend)
                import joblib
                joblib.dump(self.trainer.vectorizer, "vectorizer.pkl")
                mlflow.log_artifact("vectorizer.pkl", artifact_path="model")

                # Utilisation de ton manager pour enregistrer le modèle proprement
                uri = self.trainer.mlflow.log_training_run(
                    model=model,
                    params={"max_iter": self.trainer.config.MAX_ITER, "source": "automation_v2"},
                    metrics={"accuracy": acc},
                    X_sample=X_test_vec
                )
            
            print(f"✅ Pipeline réussi. Accuracy: {acc:.4f} | URI: {uri}")
            # Dans automation.py, à la fin de run_pipeline
            run_id = self.trainer.mlflow.get_latest_run_id("SpamClassifier")
            print(f"🆔 ID unique du run pour le backend : {run_id}")
            
        except Exception as e:
            print(f"❌ Erreur critique : {e}")

if __name__ == "__main__":
    handler = MLOpsHandler()
    # Lancement initial pour s'assurer que MLflow est à jour
    handler.run_pipeline()
    
    observer = Observer()
    
    # Surveillance du dossier DATA
    data_dir = os.path.dirname(os.path.abspath(handler.trainer.config.DATA_PATH))
    observer.schedule(handler, path=data_dir, recursive=False)
    
    # Surveillance du dossier CODE (où se trouve spam_trainer.py)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    observer.schedule(handler, path=code_dir, recursive=False)
    
    print(f"📡 Surveillance active sur :\n - Données: {data_dir}\n - Code: {code_dir}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()