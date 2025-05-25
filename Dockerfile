# Usa una base Python leggera
FROM python:3.9-slim

# Crea e imposta la working dir
WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il modello e il codice dell'API
COPY model_checkpoints/test_model model_checkpoints/test_model
COPY src/api.py src/api.py

# Espone la porta
EXPOSE 8000

# Comando di avvio
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
