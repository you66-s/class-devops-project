from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import os

class AutoTrainHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Déclenche si le CSV dans /data est modifié
        if event.src_path.endswith(".csv"):
            print(f"Nouvelle donnée détectée : {event.src_path}")
            # On lance le control_flow qui gère l'entraînement et mlflow
            subprocess.run(["python", "control_flow.py"])

if __name__ == "__main__":
    # S'assurer que le dossier data existe
    if not os.path.exists("./data"):
        os.makedirs("./data")

    observer = Observer()
    observer.schedule(AutoTrainHandler(), path="./data", recursive=False)
    print("Watcher actif : En attente de modifications dans le dossier /data...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()