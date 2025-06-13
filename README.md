![CI status](https://github.com/MATVEN/mlops-sentiment-monitoring/actions/workflows/train_eval.yml/badge.svg)

# 🚀 MLOps Sentiment Monitoring

Questo repository implementa un'infrastruttura **MLOps completa** per monitorare in modo continuo la reputazione aziendale sui social media, con analisi automatizzata del sentiment e **retraining del modello basato sul concetto di drift**.

## 🔧 Caratteristiche principali

1. **Modello di sentiment analysis** basato su [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
2. **Pipeline CI/CD** per testing, linting, type checking e salvataggio artefatti.
3. **API FastAPI** containerizzata via Docker per predizione in tempo reale.
4. **Monitoraggio giornaliero** del sentiment + trigger automatico di retraining.

### 🔑 Accesso al repository privato

Per eseguire il notebook è necessario avere accesso al repository GitHub `mlops-sentiment-monitoring`.

#### Passaggi:

1. Assicurati di essere **collaboratore del repository** (il proprietario deve aggiungerti).
2. Genera un **GitHub Personal Access Token (classic)** da [questa pagina](https://github.com/settings/tokens) con almeno i permessi:
   - `repo` (per leggere il repository)
3. Quando esegui il notebook, **inserisci il token in modo sicuro** nella cella iniziale (non sarà visibile né salvato).

#### Esempio nel notebook:
```python
from getpass import getpass
import os

GITHUB_TOKEN = getpass("Inserisci il tuo GitHub Token: ")
os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN

## 📦 1. Setup e Addestramento

### Requisiti
- Python 3.9+
- Git, GitHub CLI (opzionale)
- CUDA (opzionale, per accelerazione GPU)

### Installazione
```bash
git clone https://github.com/MATVEN/mlops-sentiment-monitoring.git
cd mlops-sentiment-monitoring

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Addestramento modello
```bash
python src/train.py \
  --epochs 5 \
  --batch_size 16 \
  --max_length 64 \
  --lr 3e-5 \
  --output_dir model_checkpoints/best_model
```

Output: `model_checkpoints/best_model_<timestamp>/` con:
- `config.json`, `pytorch_model.bin`, `tokenizer` files

### Valutazione finale
```bash
python src/eval_model.py \
  --model_dir model_checkpoints/best_model_<timestamp> \
  --batch_size 16 \
  --max_length 64
```
Restituisce **accuracy** e **macro F1-score** su `tweet_eval/sentiment`.

---

## 🐳 2. Deploy API con Docker

### Build e run
```bash
docker build -t mlops-sentiment-api .
docker run -d -p 8000:8000 \
  -v $(pwd)/model_checkpoints/test_model:/app/model_checkpoints/test_model \
  mlops-sentiment-api
```

📍 L’API sarà disponibile su: http://localhost:8000/predict`

### Test dell’API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love MachineInnovators Inc! Their AI product is amazing."}'
```

✅ Output:
```json
{"label":"positive","score":0.9876}
```

---

**Nota:** il Dockerfile è multi-stage e non include i pesi del modello all’interno dell’immagine.
Per fornire i pesi al container, monta la cartella locale model_checkpoints/test_model come volume (opzione -v).

## ⚙️ 3. CI/CD Workflow

### train_eval.yml (su `push` su `main`)
- Setup ambiente e cache pip/HF
- `flake8` e `mypy` su `src/`
- Addestramento + valutazione modello
- Upload artefatti del modello (miglior checkpoint)

### monitor.yml (cron: ogni giorno a mezzanotte UTC)
- Esegue `monitor.py`
- Valuta drift rispetto alla distribuzione storica
- Se rilevato drift → crea nuova branch `retrain/YYYY-MM-DD`

---

## 📈 4. Monitoraggio e Retraining

### 4.1 Esecuzione monitoraggio
```bash
python src/monitor.py --use_testclient
```
Oppure:
```bash
python src/monitor.py --api_url http://localhost:8000/predict
```

Verifica i nuovi tweet con `snscrape`, predice il sentiment e calcola distribuzione:
```json
{
  "timeline": {
    "2025-06-04": {"positive": 0.55, "neutral": 0.30, "negative": 0.15}
  },
  "last_dist": {"positive": 0.55, "neutral": 0.30, "negative": 0.15}
}

```

### 4.2 Trigger retraining
Se la distanza fra la distribuzione attuale e quella precedente supera la soglia:
- Nuova branch: `retrain/YYYY-MM-DD`
- File marker: `retrain_YYYY-MM-DD.txt`
- Push automatico via `GITHUB_TOKEN`

---

## 📁 5. Struttura del repository

```text
mlops-sentiment-monitoring/
├── .github/workflows/
│   ├── train_eval.yml
│   └── monitor.yml
├── model_checkpoints/
├── src/
│   ├── api.py           # API FastAPI
│   ├── data_utils.py    # Preprocessing
│   ├── model_utils.py   # Load/save model
│   ├── train.py         # Addestramento
│   ├── eval_model.py    # Valutazione
│   └── monitor.py       # Drift detection e retraining
├── requirements.txt
└── README.md

```

## ✅ Note finali

- 🔁 **Riproducibilità garantita** con seed fisso
- 💾 **Caching** pip e HF datasets in CI
- 🧪 **Qualità del codice** con `flake8` e `mypy`
- 📊 **Drift detection** automatizzata
- 🔒 **Autenticazione sicura** via `GITHUB_TOKEN`

---

## 🧠 Credits

Progetto sviluppato da [MATVEN](https://github.com/MATVEN) per dimostrare un ciclo MLOps completo su use case reali.

---

## 📄 License

MIT License - vedi `LICENSE`.
