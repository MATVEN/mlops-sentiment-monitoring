from transformers import AutoModelForSequenceClassification
import torch
import os

def build_model(model_name: str, num_labels: int = 3):
    """
    Carica un modello pre-addestrato per classificazione del sentiment.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model

def save_model(model, path: str):
    """
    Salva il modello in una directory specificata.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"✅ Modello salvato in: {path}")

def load_model(path: str):
    """
    Carica un modello salvato da una directory.
    """
    model = AutoModelForSequenceClassification.from_pretrained(path)
    print(f"✅ Modello caricato da: {path}")
    return model
