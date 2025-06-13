import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# -----------------------------------
# Configurazione percorso modello
# -----------------------------------
here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, ".."))

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Rileva automaticamente l’ultima cartella best_model_*
best_dirs = sorted(
    glob for glob in [
        os.path.join(project_root, "model_checkpoints", d)
        for d in os.listdir(os.path.join(project_root, "model_checkpoints"))
        if d.startswith("best_model_")
    ]
)
if not best_dirs:
    raise HTTPException(
        status_code=500,
        detail="Nessun best_model trovato in model_checkpoints"
    )
MODEL_DIR = best_dirs[-1]

# Verifica che la directory contenga i pesi
if not os.path.isdir(MODEL_DIR):
    raise HTTPException(
        status_code=500,
        detail=f"Model directory not found: {MODEL_DIR}"
    )

# -----------------------------------
# Gestione DEVICE
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Caricamento tokenizer (fallback locale o hub)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Caricamento modello
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    ).to(device)
except Exception:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
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
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        probs = torch.softmax(logits, dim=-1).tolist()

    labels = ["negative", "neutral", "positive"]
    idx = int(torch.argmax(logits))
    return SentimentOut(label=labels[idx], score=probs[idx])
