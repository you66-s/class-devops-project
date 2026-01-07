from backend.app.services.storage import Storage

def test_store_prediction():
    storage = Storage()
    try:
        storage.store_prediction("Hello this not a spam email", 0, 0.95)
        print("Test passed: Prediction stored successfully.")
    except Exception as e:
        print("Test failed:", e)

if __name__ == "__main__":
    test_store_prediction()