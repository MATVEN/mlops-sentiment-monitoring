#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import argparse
import requests
import ssl
from datetime import datetime

import numpy as np
import pandas as pd

# -----------------------------------
# Disabilitazione della verifica SSL in ambienti come Colab/Notebook
# -----------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------------
# Definizione degli argomenti da linea di comando
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Monitoraggio sentiment e trigger retraining"
)
parser.add_argument(
    "--use_testclient",
    action="store_true",
    help="Utilizzare TestClient per chiamare l’API invece di HTTP"
)
parser.add_argument(
    "--api_url",
    type=str,
    default="http://localhost:8000/predict",
    help="URL dell’endpoint /predict se --use_testclient è disabilitato"
)
parser.add_argument(
    "--company_query",
    type=str,
    default="MachineInnovators Inc",
    help="Query da utilizzare per estrarre i tweet"
)
parser.add_argument(
    "--num_tweets",
    type=int,
    default=100,
    help="Numero massimo di tweet da estrarre"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="Soglia per il rilevamento del drift (somma delle differenze assolute)"
)
parser.add_argument(
    "--history_file",
    type=str,
    default=None,
    help="Percorso del file JSON per salvare lo storico (default: sentiment_history.json in root)"
)
parser.add_argument(
    "--trigger_branch_prefix",
    type=str,
    default="retrain",
    help="Prefisso da usare per la branch di retraining su GitHub"
)
args = parser.parse_args()

# -----------------------------------
# Configurazione dei percorsi e del file di storico
# -----------------------------------
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, ".."))

if args.history_file:
    HISTORY_FILE = args.history_file
else:
    HISTORY_FILE = os.path.join(project_root, "sentiment_history.json")

COMPANY_QUERY         = args.company_query
NUM_TWEETS            = args.num_tweets
THRESHOLD             = args.threshold
USE_TESTCLIENT        = args.use_testclient
API_URL               = args.api_url
TRIGGER_BRANCH_PREFIX = args.trigger_branch_prefix

# -----------------------------------
# Import di TestClient se richiesto da --use_testclient
# -----------------------------------
if USE_TESTCLIENT:
    from fastapi.testclient import TestClient
    sys.path.append(project_root)
    from src.api import app
    client = TestClient(app)

# -----------------------------------
# Funzioni di supporto
# -----------------------------------
def fetch_tweets(query: str, count: int) -> list[str]:
    """
    Estrazione di tweet reali utilizzando snscrape.
    Se si verifica un errore (es. SSL), vengono generati testi fittizi.
    """
    try:
        import snscrape.modules.twitter as sntwitter
        tweets = []
        scraper = sntwitter.TwitterSearchScraper(f"{query} lang:en")
        for i, tweet in enumerate(scraper.get_items()):
            if i >= count:
                break
            tweets.append(tweet.content)
        return tweets
    except Exception as e:
        print(f"⚠️ Fallita estrazione tweet ({e}); utilizzo dati fittizi")
        return [f"Sample tweet about {query} #{i}" for i in range(count)]

def predict_sentiments(texts: list[str]) -> list[str]:
    """
    Invio dei testi all’API per ottenere le label di sentiment.
    Usa TestClient se --use_testclient è True, altrimenti richieste HTTP.
    """
    labels = []
    for text in texts:
        if USE_TESTCLIENT:
            response = client.post("/predict", json={"text": text})
            if response.status_code != 200:
                raise RuntimeError(f"API error {response.status_code}: {response.text}")
            data = response.json()
        else:
            resp = requests.post(API_URL, json={"text": text}, timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
            data = resp.json()
        labels.append(data["label"])
    return labels

def compute_distribution(labels: list[str]) -> dict[str, float]:
    """
    Calcolo della distribuzione normalizzata delle label.
    Ritorna un dizionario con frazione di ogni classe.
    """
    return pd.Series(labels).value_counts(normalize=True).to_dict()

def load_history() -> dict:
    """
    Caricamento dello storico dal file JSON.
    Se il file non esiste o è corrotto, ritorna un dizionario vuoto.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_history(history: dict) -> None:
    """
    Salvataggio del dizionario storico nel file JSON con indentazione.
    """
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def check_drift(old_dist: dict[str, float], new_dist: dict[str, float], threshold: float) -> bool:
    """
    Confronto tra due distribuzioni di label.
    Ritorna True se la somma delle differenze supera la soglia.
    """
    all_keys = set(old_dist) | set(new_dist)
    diff = sum(abs(new_dist.get(k, 0) - old_dist.get(k, 0)) for k in all_keys)
    return diff > threshold

def trigger_retraining(today: str) -> None:
    """
    Creazione di una branch Git per il retraining automatico in caso di drift.
    Clona il repository, genera un commit con un file marker e fa push.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️ Variabile GITHUB_TOKEN non definita; skip retraining.")
        return

    branch_name = f"{TRIGGER_BRANCH_PREFIX}/{today}"
    tmp_dir = os.path.join("/tmp", f"repo_{today.replace('-', '')}")
    repo_url = f"https://{github_token}@github.com/MATVEN/mlops-sentiment-monitoring.git"
    subprocess.run(["git", "clone", repo_url, tmp_dir], check=True)

    os.chdir(tmp_dir)
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    marker_filename = f"retrain_{today}.txt"
    with open(marker_filename, "w") as f:
        f.write(f"Trigger retraining on {today}\n")

    subprocess.run(["git", "add", marker_filename], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Trigger retraining: {today}"], check=True
    )
    subprocess.run(
        ["git", "push", "-u", "origin", branch_name], check=True
    )

    print(f"✅ Branch '{branch_name}' creata per retraining.")
    os.chdir(project_root)

def main():
    """
    Funzione principale che coordina l’estrazione tweet, predizione,
    calcolo distribuzione, rilevamento drift e eventuale retraining.
    """
    try:
        texts = fetch_tweets(COMPANY_QUERY, NUM_TWEETS)
    except Exception as e:
        print(f"❌ Errore in fetch_tweets: {e}")
        sys.exit(1)

    try:
        labels = predict_sentiments(texts)
    except Exception as e:
        print(f"❌ Errore in predict_sentiments: {e}")
        sys.exit(1)

    new_dist = compute_distribution(labels)
    today = datetime.utcnow().date().isoformat()

    history = load_history()
    old_dist = history.get("last_dist", {})

    history.setdefault("timeline", {})[today] = new_dist
    history["last_dist"] = new_dist
    save_history(history)

    print("🗓", today, "– Distribution:", new_dist)

    if old_dist and check_drift(old_dist, new_dist, THRESHOLD):
        print("⚠️ Drift rilevato; inizio retraining.")
        try:
            trigger_retraining(today)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Errore in trigger_retraining: {e}")
            sys.exit(1)
    else:
        print("✅ Nessun drift rilevato.")
        sys.exit(0)

if __name__ == "__main__":
    main()
