# GlobalBrine Web Platform

End-to-end web experience for the GlobalBrine-LithiumModel: FastAPI backend + React/Tailwind frontend + MapLibre explorer.

## Version

- Web UI: **v0.2.2** — February 2, 2026
- Backend/API: aligned to model artifacts `models/mae_pretrained.pth` + `models/downstream_head.pth`
- Key endpoints: `/api/v1/model`, `/api/v1/data/points`, `/api/v1/predict`, `/api/v1/predict/batch`
- Theme: dark glassmorphism, MapLibre map layers, CSV batch upload with in-memory job store

## Quick start (local)

```bash
# 1) API (from repo root)
python -m pip install -r requirements.txt
uvicorn web.backend.main:app --reload

# 2) Frontend (new terminal)
cd web/app
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`.

## Public site

- Production (DigitalOcean): https://globalbrine-web-gwvzz.ondigitalocean.app/

Homepage preview:

![GlobalBrine homepage](app/public/homepage.png)

## Environment switches

- `GLB_MODEL_VERSION` (default `0.2.2`)
- `GLB_MAE_PATH`, `GLB_HEAD_PATH`, `GLB_SCALER_PATH` (defaults to `models/` and `data/processed/feature_scaler.joblib`)
- `GLB_PREDICTIONS_CSV` (defaults to `data/predictions/brines_with_predictions.csv`; auto-generated if missing)
- `GLB_JOB_STORAGE_DIR` (defaults to `web/backend/jobs`)
- `GLB_ALLOW_ORIGINS` (comma list)

## API surface

- `GET /api/v1/model` — model metadata, git tag/commit, checksums
- `GET /api/v1/data/points` — GeoJSON of brine points + summary stats
- `POST /api/v1/predict` — single-point JSON prediction (partial inputs OK)
- `POST /api/v1/predict/batch` — CSV upload → async job
- `GET /api/v1/predict/batch/{job_id}/status|result`

## Frontend routes

- `/` landing dashboard
- `/map` global explorer (MapLibre)
- `/predict` single-point form
- `/batch` async upload + polling
- `/model` reproducibility + checksums

## Notes

- Batch jobs persist under `web/backend/jobs/` for 48h by default.
- Geo endpoint falls back to running `src/models/predict_brines.py` if the predictions CSV is missing.

## Deploy to DigitalOcean App Platform (concise playbook)

**Components**
- Static Site: `web/app` (build: `npm ci && npm run build`, output `dist`, enable SPA routing)
- Web Service: `web/backend` (Dockerfile build, port 8000, health check `/api/v1/model`)

**Backend Dockerfile (place at `web/backend/Dockerfile`)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY web/backend /app/web/backend
COPY src /app/src
COPY models /app/models
COPY data/processed/feature_scaler.joblib /app/data/processed/feature_scaler.joblib
ENV PYTHONUNBUFFERED=1 \
    GLB_MAE_PATH=/app/models/mae_pretrained.pth \
    GLB_HEAD_PATH=/app/models/downstream_head.pth \
    GLB_SCALER_PATH=/app/data/processed/feature_scaler.joblib
EXPOSE 8000
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Env vars (Backend service)**
```
GLB_MODEL_VERSION=0.2.2
GLB_MAE_PATH=/app/models/mae_pretrained.pth
GLB_HEAD_PATH=/app/models/downstream_head.pth
GLB_SCALER_PATH=/app/data/processed/feature_scaler.joblib
GLB_ALLOW_ORIGINS=https://<app>.ondigitalocean.app
GLB_PREDICTIONS_CSV=data/predictions/brines_with_predictions.csv   # optional
```

**DO App setup**
1) Create App → connect repo/branch.  
2) Add Static Site component: source `web/app`, build `npm ci && npm run build`, output `dist`, enable SPA.  
3) Add Web Service component: source `web/backend`, build via Dockerfile, HTTP port 8000, health path `/api/v1/model`.  
4) Deploy; default URL `https://<app>-<hash>.ondigitalocean.app`.
   - You can use the provided `.do/app.yaml` to auto-create both components.

**Verify**
- `GET /api/v1/model` → 200 JSON
- `GET /api/v1/data/points` → GeoJSON
- UI routes `/`, `/map`, `/predict`, `/batch`, `/model` load (SPA fallback on)
- `POST /api/v1/predict` returns predictions

**MVP caveat**: Batch jobs use local filesystem (`web/backend/jobs`) and will reset on redeploy/scale. For durability, back with Spaces (S3) + Redis/DB in a follow-up.
