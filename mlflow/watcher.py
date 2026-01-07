import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoTrainHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            print(f"🔍 Changement détecté : {event.src_path}")
            # On passe le chemin du fichier au control_flow
            subprocess.run(["python", "control_flow.py", event.src_path])

class ModelWatcher:
    def __init__(self, path_to_watch="../data"):
        self.path = path_to_watch
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.observer = Observer()
        self.handler = AutoTrainHandler()

    def start(self):
        self.observer.schedule(self.handler, path=self.path, recursive=False)
        self.observer.start()
        print(f"📡 Watcher actif sur : {self.path}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            print("🛑 Watcher arrêté.")
        self.observer.join()

if __name__ == "__main__":
    watcher = ModelWatcher(path_to_watch="./data")
    watcher.start()