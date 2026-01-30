# GlobalBrine Web Platform

End-to-end web experience for the GlobalBrine-LithiumModel: FastAPI backend + React/Tailwind frontend + MapLibre explorer.

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

## Environment switches

- `GLB_MODEL_VERSION` (default `0.1.0`)
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
