from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

# Configurazione: tokenizer dal hub, pesi del modello dal locale
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = "/content/model_checkpoints/test_model"

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} non trovato: esegui prima train.py e salva il modello lì")

# Carica il tokenizer (scarica i file dal hub)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Carica i pesi dal filesystem locale
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model.eval()

class TextIn(BaseModel):
    text: str

class SentimentOut(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=SentimentOut)
def predict(payload: TextIn):
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        probs = torch.softmax(logits, dim=-1).tolist()
    labels = ["negative", "neutral", "positive"]
    idx = int(torch.argmax(logits))
    return SentimentOut(label=labels[idx], score=probs[idx])
