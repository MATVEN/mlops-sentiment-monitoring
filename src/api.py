import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# -----------------------------------
# Configurazione percorso modello
# -----------------------------------
# Calcola il path relativo dalla cartella src al root del progetto
here = os.path.dirname(__file__)                       # src/
project_root = os.path.abspath(os.path.join(here, ".."))  # root del repo

# Configurazione: tokenizer dal hub, pesi del modello dal locale
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = "/content/model_checkpoints/test_model"

if not os.path.isdir(MODEL_DIR):
    raise HTTPException(status_code=500, detail=f"Model directory not found: {MODEL_DIR}")

# -----------------------------------
# Gestione DEVICE
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carica tokenizer dal hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Carica i pesi del modello dal filesystem locale
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(device)
model.eval()

# -----------------------------------
# Definizioni Pydantic
# -----------------------------------
class TextIn(BaseModel):
    text: str

class SentimentOut(BaseModel):
    label: str
    score: float

# -----------------------------------
# Endpoint di predizione
# -----------------------------------
@app.post("/predict", response_model=SentimentOut)
def predict(payload: TextIn):
    # Tokenizzazione
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inferenza
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        probs = torch.softmax(logits, dim=-1).tolist()

    labels = ["negative", "neutral", "positive"]
    idx = int(torch.argmax(logits))
    return SentimentOut(label=labels[idx], score=probs[idx])
