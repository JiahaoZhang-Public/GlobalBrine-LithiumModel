FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install system deps (optional: add libgomp if torch needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt setup.py pyproject.toml /app/
# Install deps (handles `-e .` in requirements)
COPY src /app/src
RUN pip install -r /app/requirements.txt

# App code + model artifacts
COPY web/backend /app/web/backend
COPY models /app/models
COPY data/processed /app/data/processed
COPY data/predictions /app/data/predictions

# Defaults (can be overridden in DO App Platform)
ENV GLB_MAE_PATH=/app/models/mae_pretrained.pth \
    GLB_HEAD_PATH=/app/models/downstream_head.pth \
    GLB_SCALER_PATH=/app/data/processed/feature_scaler.joblib \
    GLB_MODEL_VERSION=0.2.1 \
    GLB_PREDICTIONS_CSV=/app/data/predictions/brines_with_predictions.csv \
    GLB_JOB_STORAGE_DIR=/app/web/backend/jobs

EXPOSE 8000

CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
