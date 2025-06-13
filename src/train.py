import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import evaluate

# -----------------------------
# Assicura che la root del progetto sia nel PYTHONPATH
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import get_dataloaders
from src.model_utils import save_model

# -----------------------------------
# Definizione degli argomenti da linea di comando
# -----------------------------------
parser = argparse.ArgumentParser(description="Train sentiment model MLOps")
parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche di training")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size per i DataLoader")
parser.add_argument("--max_length", type=int, default=64, help="Lunghezza massima dei token")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate per l'ottimizzatore")
parser.add_argument("--output_dir", type=str, default="model_checkpoints/best_model", help="Directory di destinazione per il modello migliore")
parser.add_argument("--seed", type=int, default=42, help="Seed per garantire la riproducibilità")
parser.add_argument("--patience", type=int, default=2, help="Numero di epoche di pazienza per early stopping")
args, unknown = parser.parse_known_args()

# -----------------------------------
# Impostazione della riproducibilità tramite seed
# -----------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# -----------------------------------
# Configurazione del logger
# -----------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------------
# Parametri di configurazione
# -----------------------------------
MODEL_NAME    = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE    = args.batch_size
MAX_LENGTH    = args.max_length
NUM_EPOCHS    = args.epochs
LEARNING_RATE = args.lr
WARMUP_RATIO  = 0.1
PATIENCE      = args.patience
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creazione di una directory con timestamp per il modello migliore
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(args.output_dir + f"_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# Configurazione dei DataLoader
# -----------------------------------
train_loader, val_loader = get_dataloaders(MODEL_NAME, BATCH_SIZE, MAX_LENGTH)

# -----------------------------------
# Preparazione di modello, tokenizer, ottimizzatore e scheduler
# -----------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_loader)
num_warmup_steps   = int(num_training_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# -----------------------------------
# Configurazione della mixed precision se disponibile
# -----------------------------------
from torch.cuda.amp import GradScaler
scaler = GradScaler() if torch.cuda.is_available() else None

# -----------------------------------
# Caricamento delle metriche
# -----------------------------------
accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

def compute_metrics(preds, labels):
    """
    Calcolo di accuracy e F1-macro dati predizioni e label.
    """
    preds_flat = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds_flat, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds_flat, references=labels, average="macro")["f1"]
    }

# -----------------------------------
# Loop di training con early stopping basato su F1-macro
# -----------------------------------
best_f1    = 0.0
no_improve = 0

logger.info(
    f"🎬 Inizio training: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}"
)

for epoch in range(NUM_EPOCHS):
    logger.info(f"\n🔁 Epoch [{epoch+1}/{NUM_EPOCHS}] --------------------------")
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss    = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Fase di validazione
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    preds  = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(preds, labels)

    logger.info(
        f"📊 Train Loss: {avg_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}"
    )

    # Early stopping basato sul miglioramento di F1-macro
    if metrics["f1_macro"] > best_f1:
        best_f1    = metrics["f1_macro"]
        no_improve = 0
        save_model(model, OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"✅ Nuovo best model salvato in: {OUTPUT_DIR}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            logger.info(f"ℹ️ Early stopping dopo {PATIENCE} epoche senza miglioramento.")
            break

logger.info(f"🏁 Training completato. Best F1-macro: {best_f1:.4f}")
