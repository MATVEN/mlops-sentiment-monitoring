#!/usr/bin/env python3
import sys
import os
# patch per import src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import json
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from fastapi.testclient import TestClient
from src.api import app

# Configurazione
COMPANY_QUERY = "MachineInnovators Inc"
NUM_TWEETS = 100
THRESHOLD = 0.1
HISTORY_FILE = os.path.join(project_root, "sentiment_history.json")

client = TestClient(app)

def fetch_tweets(query, count):
    """
    STUB di sviluppo: ritorna testi fittizi invece di collezionare da Twitter.
    """
    return [f"Sample tweet about {query} #{i}" for i in range(count)]

def predict_sentiments(texts):
    labels = []
    for txt in texts:
        resp = client.post("/predict", json={"text": txt})
        labels.append(resp.json()["label"])
    return labels

def compute_distribution(labels):
    return pd.Series(labels).value_counts(normalize=True).to_dict()

def load_history():
    if os.path.exists(HISTORY_FILE):
        return json.load(open(HISTORY_FILE))
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def check_drift(old_dist, new_dist, threshold):
    all_keys = set(old_dist) | set(new_dist)
    diff = sum(abs(new_dist.get(k,0)-old_dist.get(k,0)) for k in all_keys)
    return diff > threshold

def main():
    texts    = fetch_tweets(COMPANY_QUERY, NUM_TWEETS)
    labels   = predict_sentiments(texts)
    new_dist = compute_distribution(labels)
    today    = datetime.utcnow().date().isoformat()

    history  = load_history()
    old_dist = history.get("last_dist", {})
    history.setdefault("timeline", {})[today] = new_dist
    history["last_dist"] = new_dist
    save_history(history)

    # Stampa per debug
    print("🗓", today, "– distribution:", new_dist)

    if old_dist and check_drift(old_dist, new_dist, THRESHOLD):
        print(f"⚠️ Drift detected Δ={sum(abs(new_dist.get(k,0)-old_dist.get(k,0)) for k in set(old_dist)|set(new_dist)):.3f}, triggering retraining")
        # subprocess.run([...])  # commentato in Colab
    else:
        print("✅ No drift detected.")

if __name__ == "__main__":
    main()
