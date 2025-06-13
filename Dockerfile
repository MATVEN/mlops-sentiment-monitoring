# ---------- STAGE 1: BUILD & COLLECT DEPENDENCIES ----------
FROM python:3.9-slim AS builder

WORKDIR /build

# Copia le dipendenze di sviluppo e training
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copia il codice (se hai script di preprocessing o simili)
COPY src/ ./src

# ---------- STAGE 2: RUNTIME LEGGERO ----------
FROM python:3.9-slim AS runtime

WORKDIR /app

# Installa solo le dipendenze necessarie all’API
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn \
        transformers \
        torch \
        requests

# Copia la logica dell’API
COPY src/ ./src

# I modelli non sono inclusi: andranno montati come volume
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
