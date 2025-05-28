#!/usr/bin/env python3
import os, sys, json, subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Config & patch per sviluppo Colab
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Disabilita SSL in Colab (rimuovere o proteggere in produzione)
if "/content" in project_root:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

# Costanti
COMPANY_QUERY = "MachineInnovators Inc"
NUM_TWEETS    = 100
THRESHOLD     = 0.1
HISTORY_FILE  = os.path.join(project_root, "sentiment_history.json")

# Client API
from src.api import app
client = TestClient(app)

# Funzioni
def fetch_tweets(query, count):
    # STUB: testi fittizi per sviluppo
    return [f"Sample tweet about {query} #{i}" for i in range(count)]

def predict_sentiments(texts):
    return [client.post("/predict", json={"text": t}).json()["label"] for t in texts]

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
    return sum(abs(new_dist.get(k,0) - old_dist.get(k,0)) for k in all_keys) > threshold

# Main
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

    print("🗓", today, "– distribution:", new_dist)
    if old_dist and check_drift(old_dist, new_dist, THRESHOLD):
        print("⚠️ Drift detected, triggering retraining (stub)")
        # subprocess.run([...])
    else:
        print("✅ No drift detected.")

if __name__ == "__main__":
    main()
