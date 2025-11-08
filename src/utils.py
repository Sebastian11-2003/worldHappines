# src/utils.py

import joblib
import os

def simpan_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model disimpan ke {path}")

def load_model(path):
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"Model dimuat dari {path}")
        return model
    else:
        print("File model tidak ditemukan!")
        return None
