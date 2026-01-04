# UI Demo (Gradio + FastAPI)

This folder contains a Gradio frontend and a FastAPI backend that wrap the project's existing model inference utilities in `src/models/inference.py`.

## Install dependencies

```bash
python -m pip install -r ui/backend/requirements.txt
python -m pip install -r ui/frontend/requirements.txt
```

## Run locally (no Docker)

Backend:

```bash
export MODEL_DIR=models
export DATA_DIR=data/processed
uvicorn ui.backend.app:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
export BACKEND_URL=http://localhost:8000
python ui/frontend/app.py
```

Open:
- Gradio: `http://localhost:7860`
- Backend: `http://localhost:8000`

## Run with Docker Compose

Requires Docker to be installed (e.g. Docker Desktop). If `docker` is not found, use the “Run locally” commands above.

From repo root:

```bash
docker compose -f ui/docker-compose.yml up --build
```

## Data prerequisites (for full functionality)

- Model artifacts:
  - `models/mae_pretrained.pth`
  - `models/downstream_head.pth`
- Scaler:
  - `data/processed/feature_scaler.joblib` (produced by `python src/features/build_features.py data/processed`)
- For Map + Latent tabs (recommended):
  - `data/predictions/brines_imputed.csv`
  - `data/predictions/brines_with_predictions.csv`

Generate the cached prediction files with:

```bash
python src/models/predict_brines.py --processed-dir data/processed --out-dir data/predictions
```

If you run in Docker, paths are usually under `/repo/...`, e.g.:
- `PREDICTIONS_CSV=/repo/data/predictions/brines_with_predictions.csv`

## Environment variables

- `MODEL_DIR`: path to `models/` (default `models`)
- `DATA_DIR`: path to `data/processed/` (default `data/processed`)
- `BACKEND_URL`: frontend → backend URL (default `http://localhost:8000`)
- `PREDICT_API_TOKEN`: optional bearer token; if set, backend requires `Authorization: Bearer <token>`
- `PREDICTIONS_CSV`: path to a predictions CSV for map/aggregation endpoints (default `data/predictions/brines_with_predictions.csv`)
- `BRINES_CSV`: frontend default dataset path for Data Explorer (default `data/processed/brines.csv`)
- `UI_CACHE_DIR`: cache directory for `/latent_embeddings` (default `ui/backend/cached/`)
- `DEVICE`: `auto` (default) or `cpu`/`cuda`/`mps` for backend inference
- `KNN_K`: K for `impute_strategy=knn` (default `5`)
- `IG_STEPS`: Integrated Gradients steps for attributions (default `50`)

Light inference:
- `LIGHT_GEOTIFF_PATH`: optional GeoTIFF path for `/infer_light` (requires `rasterio` installed)
- `LIGHT_SCALE`: optional scale for GeoTIFF units → `Light_kW_m2`

## Backend endpoints

- `POST /predict`: single prediction + attributions (Integrated Gradients over `TDS_gL`, `MLR`, `Light_kW_m2`)
- `POST /batch_predict`: multipart CSV → returns annotated CSV, or JSON → returns JSON results
- `POST /interpolate_heatmap`: build a global continuous field by interpolating existing sample points in `PREDICTIONS_CSV` (recommended for “global heatmap from finite samples”)
- `POST /grid_heatmap`: compute a dense global grid of predictions from a fixed chemistry “scenario” (advanced; requires `light_source=geotiff` for spatial variation)
- `GET /models`: list model versions
- `GET /infer_light`: GeoTIFF-based light sampling (best-effort)
- `POST /impute`: optional imputation helper
- `POST /aggregate`: summarize predictions within a bounding box (uses `data/predictions/brines_with_predictions.csv` by default)
- `GET /map_points`: return map-ready points with `lat/lon`, `metric_value`, and tooltip metadata
- `POST /aggregate_region`: aggregate a metric over a polygon or `sample_ids`
- `GET /latent_embeddings`: project MAE latents to 2D (PCA built-in; UMAP/t-SNE optional) with cached results

## Tests

Tests run the backend in `UI_TEST_MODE=1` (dummy model) and validate the API contract / caching behavior.

```bash
UI_TEST_MODE=1 pytest -q ui/tests
```

## Map + Latent Space tabs (Gradio)

The frontend now includes two analysis tabs:

- **Data Explorer**: upload/preview raw data, apply backend imputation, then run batch prediction and persist:
  - `data/predictions/brines_imputed.csv` (used by Latent Space)
  - `data/predictions/brines_with_predictions.csv` (used by Map + interpolation)
- **Map**: Plotly-based world map (no tile basemap needed) with sample-point tooltips and an optional global heatmap overlay. Heatmap supports:
  - point-cloud (fast): interpolated grid points rendered as a transparent layer
  - raster (paper-style): interpolated grid rendered as square “pixels” on a planar world map, with sites overlaid
- **Latent Space**: calls `/latent_embeddings` (cached), plots it (Plotly; falls back to Matplotlib), and supports sample inspection + nearest-neighbor lookup.

### Why did the old heatmap look constant?

`/grid_heatmap` holds brine chemistry constant (scenario) and the model does not use `lat/lon` directly; if you also use `light_source=constant`, predictions are identical at every grid cell. For “infer a global field from finite sample points”, use `/interpolate_heatmap` (the Map tab defaults to this).

## Heatmap interpolation API (recommended)

`POST /interpolate_heatmap` parameters (JSON body):

- `metric`: `missing_rate`, any numeric column, or prediction target (`Selectivity`, `Li_Crystallization_mg_m2_h`, `Evap_kg_m2_h`, or `Pred_*`)
- `bbox`: `{lon_min, lat_min, lon_max, lat_max}` (optional; defaults to global)
- `step_deg`: grid resolution in degrees (e.g. `2.0`)
- `method`: `idw` (inverse-distance weighting) or `knn` (k-nearest mean)
- `k`: neighbors used for interpolation (e.g. `8`)
- `power`: IDW power (only for `method=idw`)
- `max_distance_km`: mask extrapolation beyond this distance from the nearest sample (`null` disables masking)
- `return_grid`: `true` to include `{lons,lats,values}` for raster-style rendering

## Adding new metrics

`/map_points?metric=...` supports:

- `missing_rate` (computed from common brine feature columns)
- any numeric column present in the predictions CSV (e.g. `TDS_gL`, `MLR`)
- prediction columns by either name:
  - `Selectivity` (resolved to `Pred_Selectivity`)
  - `Pred_Selectivity` (direct)

If you add a new prediction target, ensure it is written to `data/predictions/brines_with_predictions.csv` as `Pred_<TargetName>` so it can be used for map coloring and aggregation.

## Caching

`/latent_embeddings` caches computed 2D embeddings to disk:

- default: `ui/backend/cached/`
- override with `UI_CACHE_DIR=/path/to/cache`
