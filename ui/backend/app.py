from __future__ import annotations

import io
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ui.backend.cache import cache_key, load_cached, save_cached
from ui.backend.geo_helpers import (
    parse_bbox_param,
    point_in_polygon,
)
from ui.backend.model_server import DummyModelWrapper, ModelServerError, ModelWrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("UI_TEST_MODE", "").strip() == "1":
        app.state.wrapper = DummyModelWrapper()
        yield
        return

    model_dir = Path(os.environ.get("MODEL_DIR", "models")).resolve()
    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).resolve()
    device = os.environ.get("DEVICE", "auto")
    app.state.wrapper = ModelWrapper(model_dir=model_dir, data_dir=data_dir, device=device)
    yield


app = FastAPI(title="Nature-Water UI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_token(authorization: str | None) -> None:
    token = os.environ.get("PREDICT_API_TOKEN", "").strip()
    if not token:
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization.removeprefix("Bearer ").strip()
    if provided != token:
        raise HTTPException(status_code=403, detail="Invalid token")


def get_wrapper() -> ModelWrapper:
    wrapper = getattr(app.state, "wrapper", None)
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return wrapper


class PredictRequest(BaseModel):
    features: dict[str, Any]
    model_version: str = "downstream_head_latest"
    impute_strategy: str = Field(default="model", pattern="^(none|mean|knn|model)$")
    request_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models(
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    _require_token(authorization)
    return {"models": [m.__dict__ for m in wrapper.list_models()]}


@app.get("/infer_light")
def infer_light(
    lat: float,
    lon: float,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    _require_token(authorization)
    try:
        return wrapper.infer_light(lat=lat, lon=lon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelServerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict(
    req: PredictRequest,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    _require_token(authorization)
    try:
        return wrapper.predict_one(
            features=req.features,
            model_version=req.model_version,
            impute_strategy=req.impute_strategy,
            request_id=req.request_id,
        )
    except ModelServerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class BatchPredictJSONRequest(BaseModel):
    rows: list[dict[str, Any]]
    model_version: str = "downstream_head_latest"
    impute_strategy: str = Field(default="model", pattern="^(none|mean|knn|model)$")
    request_id: str | None = None


@app.post("/batch_predict")
async def batch_predict(request: Request) -> Any:
    """Batch prediction endpoint.

    - If Content-Type is multipart/form-data, expects `file` upload and returns an annotated CSV file.
    - Otherwise expects JSON and returns JSON results.
    """
    wrapper: ModelWrapper = get_wrapper()
    _require_token(request.headers.get("authorization"))

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploaded: UploadFile | None = form.get("file")  # type: ignore[assignment]
        if uploaded is None:
            raise HTTPException(status_code=400, detail="Missing multipart field: file")
        model_version = str(form.get("model_version") or "downstream_head_latest")
        impute_strategy = str(form.get("impute_strategy") or "model")

        import pandas as pd

        data = await uploaded.read()
        df = pd.read_csv(io.BytesIO(data))
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            res = wrapper.predict_one(
                features=row.to_dict(),
                model_version=model_version,
                impute_strategy=impute_strategy,
            )
            results.append(res)

        out_rows: list[dict[str, Any]] = []
        for res in results:
            merged: dict[str, Any] = {}
            merged.update(res.get("input_features", {}))
            merged.update({f"Imputed_{k}": v for k, v in res.get("imputed_features", {}).items()})
            for t, obj in res.get("predictions", {}).items():
                merged[f"Pred_{t}"] = obj.get("value")
            # Top-3 attribution features per target.
            for t, feats in res.get("attributions", {}).items():
                if not isinstance(feats, dict):
                    continue
                items = sorted(feats.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:3]
                merged[f"AttrTop3_{t}"] = ";".join([f"{k}={float(v):.4g}" for k, v in items])
            out_rows.append(merged)

        out_df = pd.DataFrame(out_rows)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"},
        )

    payload = await request.json()
    req = BatchPredictJSONRequest.model_validate(payload)
    results = [
        wrapper.predict_one(
            features=row,
            model_version=req.model_version,
            impute_strategy=req.impute_strategy,
        )
        for row in req.rows
    ]
    return JSONResponse(
        {
            "request_id": req.request_id or "batch",
            "model_version": req.model_version,
            "results": results,
            "meta": {"warnings": []},
        }
    )


class ImputeRequest(BaseModel):
    features: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    model_version: str = "downstream_head_latest"
    impute_strategy: str = Field(default="model", pattern="^(none|mean|knn|model)$")


@app.post("/impute")
def impute(
    req: ImputeRequest,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    _require_token(authorization)
    if req.features is None and req.rows is None:
        raise HTTPException(status_code=400, detail="Provide features or rows")
    if req.features is not None and req.rows is not None:
        raise HTTPException(status_code=400, detail="Provide only one of features or rows")

    if req.features is not None:
        imp = wrapper.impute(
            features=req.features,
            impute_strategy=req.impute_strategy,
            model_version=req.model_version,
        )
        return {"imputed_features": imp.imputed_features, "meta": {"warnings": imp.warnings}}

    outs: list[dict[str, Any]] = []
    warnings: list[str] = []
    assert req.rows is not None
    for row in req.rows:
        imp = wrapper.impute(
            features=row,
            impute_strategy=req.impute_strategy,
            model_version=req.model_version,
        )
        outs.append(imp.imputed_features)
        warnings.extend(imp.warnings)
    return {"imputed_rows": outs, "meta": {"warnings": warnings}}


class AggregateBBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class AggregateRequest(BaseModel):
    bbox: AggregateBBox
    metric: str = "Pred_Selectivity"
    predictions_csv: str | None = None


@app.post("/aggregate")
def aggregate(
    req: AggregateRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_token(authorization)
    import pandas as pd

    path = _resolve_predictions_path(req.predictions_csv)
    df = pd.read_csv(path)
    for col in ("Latitude", "Longitude", req.metric):
        if col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Missing column: {col}")

    sub = df[
        (df["Latitude"] >= req.bbox.lat_min)
        & (df["Latitude"] <= req.bbox.lat_max)
        & (df["Longitude"] >= req.bbox.lon_min)
        & (df["Longitude"] <= req.bbox.lon_max)
    ]
    vals = pd.to_numeric(sub[req.metric], errors="coerce").dropna()
    if len(vals) == 0:
        return {"metric": req.metric, "count": 0, "stats": None}

    return {
        "metric": req.metric,
        "count": int(len(vals)),
        "stats": {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        },
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_repo_root() / p)


def _resolve_predictions_path(predictions_csv: str | None = None) -> Path:
    """
    Resolve the predictions CSV path with a few best-effort fallbacks.

    Primary sources:
    - explicit `predictions_csv` argument (if provided)
    - env `PREDICTIONS_CSV`
    - default `data/predictions/brines_with_predictions.csv` (repo-relative)

    Fallback:
    - if an absolute path is provided but missing and contains a `.../data/...` segment,
      also try mapping that suffix onto the repo root (common Docker mount mismatch).
    """
    raw = (predictions_csv or "").strip()
    if not raw:
        raw = os.environ.get("PREDICTIONS_CSV", "data/predictions/brines_with_predictions.csv").strip()

    tried: list[Path] = []
    p = Path(raw)
    if p.is_absolute():
        tried.append(p)
        if not p.exists():
            parts = list(p.parts)
            if "data" in parts:
                idx = parts.index("data")
                tried.append(_repo_root().joinpath(*parts[idx:]))
    else:
        tried.append(_resolve_path(raw))

    for candidate in tried:
        if candidate.exists():
            return candidate

    hint = (
        "Generate it with `python src/models/predict_brines.py --processed-dir data/processed --out-dir data/predictions` "
        "or set `PREDICTIONS_CSV` to the correct path (inside Docker: usually `/repo/data/predictions/brines_with_predictions.csv`)."
    )
    tried_str = ", ".join(str(t) for t in tried)
    raise HTTPException(status_code=404, detail=f"Missing predictions CSV (tried: {tried_str}). {hint}")


def _load_predictions_df(predictions_csv: str | None = None):
    import pandas as pd

    resolved = _resolve_predictions_path(predictions_csv)
    return pd.read_csv(resolved)


def _load_imputed_brines_df():
    import pandas as pd

    data_dir = os.environ.get("DATA_DIR", "data/processed")
    # Prefer the imputed file (no NaNs) if present.
    candidate = _resolve_path("data/predictions/brines_imputed.csv")
    if candidate.exists():
        return pd.read_csv(candidate)
    fallback = _resolve_path(f"{data_dir.rstrip('/')}/brines.csv")
    if not fallback.exists():
        raise HTTPException(status_code=404, detail=f"Missing brines CSV: {fallback}")
    return pd.read_csv(fallback)


def _row_sample_id(idx: int, row: dict[str, Any]) -> str:
    loc = str(row.get("Location") or "").strip()
    if loc:
        return f"{loc}__{idx}"
    return f"row_{idx}"


def _missing_rate(row: dict[str, Any]) -> float:
    # Consider numeric brine features + Light as missing-sensitive columns.
    cols = [
        "Li_gL",
        "Mg_gL",
        "Na_gL",
        "K_gL",
        "Ca_gL",
        "SO4_gL",
        "Cl_gL",
        "MLR",
        "TDS_gL",
        "Light_kW_m2",
    ]
    missing = 0
    total = 0
    for c in cols:
        if c not in row:
            continue
        total += 1
        v = row.get(c)
        if v is None:
            missing += 1
            continue
        try:
            import pandas as pd

            if pd.isna(v):
                missing += 1
        except Exception:
            # best-effort
            if str(v).strip().lower() in ("", "nan"):
                missing += 1
    return float(missing) / float(total or 1)


def _resolve_metric_value(metric: str, row: dict[str, Any]) -> float | None:
    if metric == "missing_rate":
        return _missing_rate(row)
    # Allow direct raw feature metrics.
    if metric in row and isinstance(row.get(metric), (int, float)):
        return float(row[metric])
    # Allow either Pred_* or user-friendly names.
    pred_key = metric if metric.startswith("Pred_") else f"Pred_{metric}"
    if pred_key in row:
        try:
            return float(row[pred_key])
        except Exception:
            return None
    return None


def _grid_centers(min_v: float, max_v: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    # Center points: [min+step/2, ..., max-step/2]
    start = min_v + step / 2.0
    end = max_v - step / 2.0
    if end < start:
        return []
    n = int(math.floor((end - start) / step)) + 1
    return [start + i * step for i in range(n)]


def _target_index(metric: str) -> tuple[str, int]:
    """Map a metric name to (canonical_target_name, index)."""
    m = metric.removeprefix("Pred_").strip()
    targets = ["Selectivity", "Li_Crystallization_mg_m2_h", "Evap_kg_m2_h"]
    if m not in targets:
        raise ValueError(
            f"metric must be one of {targets} (or Pred_<name>); got {metric!r}"
        )
    return m, targets.index(m)


class GridHeatmapBBox(BaseModel):
    lon_min: float = -180.0
    lat_min: float = -60.0
    lon_max: float = 180.0
    lat_max: float = 80.0


class GridHeatmapRequest(BaseModel):
    metric: str = "Selectivity"
    bbox: GridHeatmapBBox = Field(default_factory=GridHeatmapBBox)
    step_deg: float = 1.0
    model_version: str = "downstream_head_latest"
    impute_strategy: str = Field(default="model", pattern="^(none|mean|knn|model)$")
    scenario_features: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional brine chemistry scenario; missing fields are imputed.",
    )
    light_source: str = Field(
        default="geotiff",
        description="geotiff (requires LIGHT_GEOTIFF_PATH) or constant",
    )
    light_value: float | None = None
    max_points: int = 75000
    recompute: bool = False
    return_grid: bool = False


@app.post("/grid_heatmap")
def grid_heatmap(
    req: GridHeatmapRequest,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    """
    Compute a global grid of predictions and return as point samples.

    Intended to be rendered as a continuous heatmap layer (dense grid points).
    Chemistry is held constant via `scenario_features` while `Light_kW_m2` varies
    by location (from GeoTIFF or constant).
    """
    _require_token(authorization)
    try:
        target_name, target_idx = _target_index(req.metric)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    step = float(req.step_deg)
    if step <= 0 or step > 20:
        raise HTTPException(status_code=400, detail="step_deg must be in (0, 20].")

    lons = _grid_centers(req.bbox.lon_min, req.bbox.lon_max, step)
    lats = _grid_centers(req.bbox.lat_min, req.bbox.lat_max, step)
    n_points = int(len(lons) * len(lats))
    if n_points <= 0:
        raise HTTPException(status_code=400, detail="Empty grid (check bbox/step_deg).")
    if n_points > int(req.max_points):
        raise HTTPException(
            status_code=422,
            detail=f"Grid too large ({n_points} points). Increase step_deg or reduce bbox, or raise max_points.",
        )

    # Cache key depends on grid params + scenario + model version + light source.
    geotiff = os.environ.get("LIGHT_GEOTIFF_PATH", "").strip()
    geotiff_path = Path(geotiff) if geotiff else None
    geotiff_mtime = int(geotiff_path.stat().st_mtime) if geotiff_path and geotiff_path.exists() else 0
    key = cache_key(
        "grid_heatmap",
        params={
            "metric": target_name,
            "bbox": req.bbox.model_dump(),
            "step_deg": step,
            "model_version": req.model_version,
            "impute_strategy": req.impute_strategy,
            "scenario": req.scenario_features,
            "light_source": req.light_source,
            "light_value": req.light_value,
            "geotiff_mtime": geotiff_mtime,
            "light_scale": os.environ.get("LIGHT_SCALE", "1.0"),
        },
    )
    if not req.recompute:
        cached = load_cached(key)
        if cached is not None:
            meta, arr = cached
            pts = [
                {"lon": float(r[0]), "lat": float(r[1]), "value": float(r[2])}
                for r in arr.tolist()
            ]
            out: dict[str, Any] = {
                "n_points": int(meta.get("n_points", len(pts))),
                "points": pts,
                "meta": meta.get("meta", {}),
            }
            if bool(req.return_grid):
                # Reconstruct a dense grid (with NaNs where missing).
                lons_full = _grid_centers(req.bbox.lon_min, req.bbox.lon_max, step)
                lats_full = _grid_centers(req.bbox.lat_min, req.bbox.lat_max, step)
                if lons_full and lats_full:
                    import numpy as np

                    grid = np.full((len(lats_full), len(lons_full)), np.nan, dtype=np.float32)
                    lon0 = lons_full[0]
                    lat0 = lats_full[0]
                    for r in pts:
                        lon = float(r["lon"])
                        lat = float(r["lat"])
                        i = int(round((lat - lat0) / step))
                        j = int(round((lon - lon0) / step))
                        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                            grid[i, j] = float(r["value"])
                    out["grid"] = {"lons": lons_full, "lats": lats_full, "values": grid.tolist()}
            return out

    # Prepare scenario (impute missing chemistry once).
    is_test_mode = os.environ.get("UI_TEST_MODE", "").strip() == "1"
    effective_impute = req.impute_strategy
    # Dummy wrapper doesn't have MAE artifacts; treat "model" as "mean" for test mode.
    if is_test_mode and effective_impute == "model":
        effective_impute = "mean"

    imp = wrapper.impute(
        features=req.scenario_features,
        impute_strategy=effective_impute,
        model_version=req.model_version,
    )

    # Build grid points and sample Light.
    points_lon: list[float] = []
    points_lat: list[float] = []
    light_vals: list[float] = []

    light_source = (req.light_source or "geotiff").lower().strip()
    if light_source == "constant":
        if req.light_value is None:
            raise HTTPException(status_code=400, detail="light_value is required for light_source=constant.")
        lv = float(req.light_value)
        for lat in lats:
            for lon in lons:
                points_lon.append(float(lon))
                points_lat.append(float(lat))
                light_vals.append(lv)
    elif light_source == "geotiff":
        if not geotiff_path or not geotiff_path.exists():
            raise HTTPException(
                status_code=404,
                detail="LIGHT_GEOTIFF_PATH not configured or file missing; use light_source=constant or configure GeoTIFF.",
            )
        from src.data.light import GeoTiffSampler

        scale = float(os.environ.get("LIGHT_SCALE", "1.0"))
        sampler = GeoTiffSampler(geotiff_path, band=1)
        try:
            for lat in lats:
                for lon in lons:
                    v = sampler.sample_lat_lon(float(lat), float(lon))
                    if v is None:
                        continue
                    points_lon.append(float(lon))
                    points_lat.append(float(lat))
                    light_vals.append(float(v) * scale)
        finally:
            sampler.close()
    else:
        raise HTTPException(status_code=400, detail="light_source must be geotiff or constant.")

    if not points_lon:
        raise HTTPException(status_code=422, detail="No grid points had usable Light_kW_m2 (GeoTIFF coverage/mask).")

    # Predict in batch (delegated to ModelWrapper; avoid torch usage in this endpoint).
    import numpy as np

    from src.constants import BRINE_FEATURE_COLUMNS

    x_raw = np.zeros((len(points_lon), len(BRINE_FEATURE_COLUMNS)), dtype=np.float32)
    # Fill from imputed scenario (full brine feature vector).
    for j, name in enumerate(BRINE_FEATURE_COLUMNS):
        try:
            x_raw[:, j] = float(imp.imputed_features[name])
        except Exception:
            x_raw[:, j] = 0.0
    # Override Light by location.
    light_idx = list(BRINE_FEATURE_COLUMNS).index("Light_kW_m2")
    x_raw[:, light_idx] = np.asarray(light_vals, dtype=np.float32)

    if is_test_mode:
        y_raw = np.zeros((len(points_lon), 3), dtype=np.float32)
    else:
        y_raw = wrapper.predict_brine_matrix(x_raw=x_raw, model_version=req.model_version)
    values = y_raw[:, int(target_idx)].astype(np.float32)

    arr = np.stack(
        [
            np.asarray(points_lon, dtype=np.float32),
            np.asarray(points_lat, dtype=np.float32),
            values.astype(np.float32),
        ],
        axis=1,
    )

    meta = {
        "n_points": int(arr.shape[0]),
        "metric": target_name,
        "bbox": req.bbox.model_dump(),
        "step_deg": step,
        "light_source": light_source,
        "model_version": req.model_version,
        "impute_strategy": req.impute_strategy,
        "scenario_features": imp.imputed_features,
    }
    save_cached(key, meta={"n_points": int(arr.shape[0]), "meta": meta}, array=arr)

    pts = [{"lon": float(r[0]), "lat": float(r[1]), "value": float(r[2])} for r in arr.tolist()]
    out: dict[str, Any] = {"n_points": int(len(pts)), "points": pts, "meta": meta}
    if bool(req.return_grid):
        lons_full = _grid_centers(req.bbox.lon_min, req.bbox.lon_max, step)
        lats_full = _grid_centers(req.bbox.lat_min, req.bbox.lat_max, step)
        if lons_full and lats_full:
            import numpy as np

            grid = np.full((len(lats_full), len(lons_full)), np.nan, dtype=np.float32)
            lon0 = lons_full[0]
            lat0 = lats_full[0]
            for r in pts:
                lon = float(r["lon"])
                lat = float(r["lat"])
                i = int(round((lat - lat0) / step))
                j = int(round((lon - lon0) / step))
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                    grid[i, j] = float(r["value"])
            out["grid"] = {"lons": lons_full, "lats": lats_full, "values": grid.tolist()}
    return out


class InterpolateHeatmapBBox(BaseModel):
    lon_min: float = -180.0
    lat_min: float = -60.0
    lon_max: float = 180.0
    lat_max: float = 80.0


class InterpolateHeatmapRequest(BaseModel):
    metric: str = "Selectivity"
    bbox: InterpolateHeatmapBBox = Field(default_factory=InterpolateHeatmapBBox)
    step_deg: float = 1.0
    method: str = Field(default="idw", description="idw or knn")
    k: int = 8
    power: float = 2.0  # for IDW
    max_distance_km: float | None = 1500.0  # mask far extrapolation
    predictions_csv: str | None = None
    return_grid: bool = False
    recompute: bool = False


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np

    r = 6371.0
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _interpolate_knn_idw(
    *,
    sample_lat: "np.ndarray",
    sample_lon: "np.ndarray",
    sample_val: "np.ndarray",
    grid_lat: "np.ndarray",
    grid_lon: "np.ndarray",
    method: str,
    k: int,
    power: float,
    max_distance_km: float | None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Return (pred_values, nn_distance_km).
    Uses haversine distance and inverse-distance weighting over k nearest samples.
    """
    import numpy as np

    # Prefer sklearn BallTree if available for speed.
    try:
        from sklearn.neighbors import BallTree  # type: ignore

        X = np.stack([np.deg2rad(sample_lat), np.deg2rad(sample_lon)], axis=1)
        Q = np.stack([np.deg2rad(grid_lat), np.deg2rad(grid_lon)], axis=1)
        tree = BallTree(X, metric="haversine")
        kk = min(int(k), int(X.shape[0]))
        dist_rad, ind = tree.query(Q, k=kk)
        dist_km = dist_rad * 6371.0
        nn_km = dist_km[:, 0]
        vals = sample_val[ind]  # [M, k]
        if method == "knn":
            pred = vals.mean(axis=1)
        else:
            w = 1.0 / (np.power(dist_km, float(power)) + 1e-6)
            pred = (w * vals).sum(axis=1) / w.sum(axis=1)
    except Exception:
        # Fallback: full distance matrix (O(N*M)); OK for small grids.
        lat1 = np.deg2rad(grid_lat).reshape(-1, 1)
        lon1 = np.deg2rad(grid_lon).reshape(-1, 1)
        lat2 = np.deg2rad(sample_lat).reshape(1, -1)
        lon2 = np.deg2rad(sample_lon).reshape(1, -1)
        d = _haversine_km(lat1, lon1, lat2, lon2)
        nn_km = d.min(axis=1)
        kk = min(int(k), d.shape[1])
        idx = np.argpartition(d, kth=kk - 1, axis=1)[:, :kk]
        d_k = np.take_along_axis(d, idx, axis=1)
        v_k = sample_val[idx]
        if method == "knn":
            pred = v_k.mean(axis=1)
        else:
            w = 1.0 / (np.power(d_k, float(power)) + 1e-6)
            pred = (w * v_k).sum(axis=1) / w.sum(axis=1)

    if max_distance_km is not None:
        pred = np.where(nn_km <= float(max_distance_km), pred, np.nan)
    return pred.astype(np.float32), nn_km.astype(np.float32)


@app.post("/interpolate_heatmap")
def interpolate_heatmap(
    req: InterpolateHeatmapRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Interpolate a continuous field from existing sample points (predictions CSV).

    This is the intended way to get a "global heatmap from finite samples".
    """
    _require_token(authorization)
    method = (req.method or "idw").lower().strip()
    if method not in ("idw", "knn"):
        raise HTTPException(status_code=400, detail="method must be idw or knn")
    step = float(req.step_deg)
    if step <= 0 or step > 20:
        raise HTTPException(status_code=400, detail="step_deg must be in (0, 20].")

    # Cache key depends on params + predictions CSV mtime.
    pred_path = _resolve_predictions_path(req.predictions_csv)
    mtime = int(pred_path.stat().st_mtime) if pred_path.exists() else 0
    key = cache_key(
        "interpolate_heatmap",
        params={
            "metric": req.metric,
            "bbox": req.bbox.model_dump(),
            "step_deg": step,
            "method": method,
            "k": int(req.k),
            "power": float(req.power),
            "max_distance_km": req.max_distance_km,
            "pred_mtime": mtime,
        },
    )
    if not req.recompute:
        cached = load_cached(key)
        if cached is not None:
            meta, arr = cached
            pts = [{"lon": float(r[0]), "lat": float(r[1]), "value": float(r[2])} for r in arr.tolist()]
            out: dict[str, Any] = {
                "n_points": int(meta.get("n_points", len(pts))),
                "points": pts,
                "meta": meta.get("meta", {}),
            }
            if bool(req.return_grid):
                out["grid"] = meta.get("grid", None)
            return out

    import numpy as np
    import pandas as pd

    df = pd.read_csv(pred_path)
    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Missing column: {col}")

    # Extract sample values.
    rows = df.to_dict(orient="records")
    sample_lat: list[float] = []
    sample_lon: list[float] = []
    sample_val: list[float] = []
    for r in rows:
        mv = _resolve_metric_value(req.metric, r)
        if mv is None:
            continue
        try:
            lat = float(r["Latitude"])
            lon = float(r["Longitude"])
            v = float(mv)
        except Exception:
            continue
        if not math.isfinite(lat) or not math.isfinite(lon) or not math.isfinite(v):
            continue
        sample_lat.append(lat)
        sample_lon.append(lon)
        sample_val.append(v)

    if len(sample_val) < 3:
        raise HTTPException(status_code=422, detail="Not enough valid sample points to interpolate (need >= 3).")

    s_lat = np.asarray(sample_lat, dtype=np.float32)
    s_lon = np.asarray(sample_lon, dtype=np.float32)
    s_val = np.asarray(sample_val, dtype=np.float32)

    # Collapse duplicate coordinates by mean value (important for stable interpolation).
    key2 = np.stack([np.round(s_lat, 5), np.round(s_lon, 5)], axis=1)
    uniq, inv = np.unique(key2, axis=0, return_inverse=True)
    if uniq.shape[0] != key2.shape[0]:
        sums = np.zeros((uniq.shape[0],), dtype=np.float64)
        counts = np.zeros((uniq.shape[0],), dtype=np.int64)
        for i, g in enumerate(inv):
            sums[g] += float(s_val[i])
            counts[g] += 1
        s_lat = uniq[:, 0].astype(np.float32)
        s_lon = uniq[:, 1].astype(np.float32)
        s_val = (sums / np.maximum(1, counts)).astype(np.float32)

    lons = _grid_centers(req.bbox.lon_min, req.bbox.lon_max, step)
    lats = _grid_centers(req.bbox.lat_min, req.bbox.lat_max, step)
    if not lons or not lats:
        raise HTTPException(status_code=400, detail="Empty grid (check bbox/step_deg).")
    n_points = int(len(lons) * len(lats))
    if n_points > 200000:
        raise HTTPException(status_code=422, detail=f"Grid too large ({n_points}). Increase step_deg or reduce bbox.")

    glon, glat = np.meshgrid(np.asarray(lons, dtype=np.float32), np.asarray(lats, dtype=np.float32))
    grid_lat = glat.reshape(-1)
    grid_lon = glon.reshape(-1)

    pred, nn_km = _interpolate_knn_idw(
        sample_lat=s_lat,
        sample_lon=s_lon,
        sample_val=s_val,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        method=method,
        k=int(req.k),
        power=float(req.power),
        max_distance_km=req.max_distance_km,
    )

    valid = np.isfinite(pred)
    arr = np.stack([grid_lon[valid], grid_lat[valid], pred[valid]], axis=1).astype(np.float32)

    meta = {
        "metric": req.metric,
        "bbox": req.bbox.model_dump(),
        "step_deg": step,
        "method": method,
        "k": int(req.k),
        "power": float(req.power),
        "max_distance_km": req.max_distance_km,
        "n_samples": int(s_val.shape[0]),
        "n_grid": int(n_points),
        "n_returned": int(arr.shape[0]),
        "predictions_csv": str(pred_path),
    }
    cache_meta: dict[str, Any] = {"n_points": int(arr.shape[0]), "meta": meta}
    if bool(req.return_grid):
        Z = pred.reshape((len(lats), len(lons))).astype(np.float32)
        cache_meta["grid"] = {"lons": lons, "lats": lats, "values": Z.tolist()}
    save_cached(key, meta=cache_meta, array=arr)

    out: dict[str, Any] = {
        "n_points": int(arr.shape[0]),
        "points": [{"lon": float(r[0]), "lat": float(r[1]), "value": float(r[2])} for r in arr.tolist()],
        "meta": meta,
    }
    if bool(req.return_grid):
        out["grid"] = cache_meta["grid"]
    return out
@app.get("/map_points")
def map_points(
    metric: str = "missing_rate",
    limit: int = 5000,
    bbox: str | None = None,
    model_version: str | None = None,  # noqa: ARG001 - reserved for future use
    include_attributions: bool = False,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    """
    Return geolocated points for map rendering.

    Uses the precomputed predictions CSV by default for fast loading.
    """
    _require_token(authorization)
    df = _load_predictions_df()
    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Missing column: {col}")

    if bbox:
        try:
            bb = parse_bbox_param(bbox)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        df = df[
            (df["Longitude"] >= bb.lon_min)
            & (df["Longitude"] <= bb.lon_max)
            & (df["Latitude"] >= bb.lat_min)
            & (df["Latitude"] <= bb.lat_max)
        ]

    df = df.head(max(1, int(limit)))
    points: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        record = row.to_dict()
        sample_id = _row_sample_id(idx, record)
        lat = float(record["Latitude"])
        lon = float(record["Longitude"])
        metric_value = _resolve_metric_value(metric, record)

        prediction = {}
        for name in ("Selectivity", "Li_Crystallization_mg_m2_h", "Evap_kg_m2_h"):
            pred_key = f"Pred_{name}"
            if pred_key in record:
                try:
                    prediction[name] = float(record[pred_key])
                except Exception:
                    pass

        top_attributions: dict[str, list[list[Any]]] = {}
        if include_attributions:
            try:
                pred = wrapper.predict_one(
                    features={
                        "TDS_gL": record.get("TDS_gL"),
                        "MLR": record.get("MLR"),
                        "Light_kW_m2": record.get("Light_kW_m2"),
                        "lat": lat,
                        "lon": lon,
                        "sample_id": sample_id,
                    },
                    model_version="downstream_head_latest",
                    impute_strategy="model",
                )
                attrs = pred.get("attributions", {}) or {}
                for tname, feat_map in attrs.items():
                    if not isinstance(feat_map, dict):
                        continue
                    items = sorted(feat_map.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:3]
                    top_attributions[tname] = [[k, float(v)] for k, v in items]
            except Exception:
                top_attributions = {}

        points.append(
            {
                "sample_id": sample_id,
                "lat": lat,
                "lon": lon,
                "raw": {
                    "TDS_gL": record.get("TDS_gL"),
                    "MLR": record.get("MLR"),
                    "Light_kW_m2": record.get("Light_kW_m2"),
                    "Location": record.get("Location"),
                },
                "imputed": {
                    k: record.get(f"Imputed_{k}")
                    for k in ("TDS_gL", "MLR", "Light_kW_m2")
                    if f"Imputed_{k}" in record and record.get(f"Imputed_{k}") is not None
                },
                "prediction": prediction,
                "top_attributions": {k: v for k, v in top_attributions.items()},
                "metric_value": metric_value,
            }
        )

    return {
        "n_points": int(len(points)),
        "points": points,
        "meta": {
            "metric": metric,
            "limit": int(limit),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


class AggregateRegionRequest(BaseModel):
    polygon: list[list[float]] | None = None
    sample_ids: list[str] | None = None
    metric: str = "missing_rate"
    model_version: str = "downstream_head_latest"


@app.post("/aggregate_region")
def aggregate_region(
    req: AggregateRegionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_token(authorization)
    if req.polygon is None and req.sample_ids is None:
        raise HTTPException(status_code=400, detail="Provide polygon or sample_ids")
    if req.polygon is not None and req.sample_ids is not None:
        raise HTTPException(status_code=400, detail="Provide only one of polygon or sample_ids")

    df = _load_predictions_df()
    # Build map-like records with a stable sample_id.
    rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        record = row.to_dict()
        record["sample_id"] = _row_sample_id(idx, record)
        rows.append(record)

    selected: list[dict[str, Any]] = []
    if req.sample_ids is not None:
        wanted = set(req.sample_ids)
        selected = [r for r in rows if r.get("sample_id") in wanted]
    else:
        polygon = [(float(lon), float(lat)) for lon, lat in req.polygon or []]
        for r in rows:
            try:
                lon = float(r["Longitude"])
                lat = float(r["Latitude"])
            except Exception:
                continue
            if point_in_polygon((lon, lat), polygon):
                selected.append(r)

    values: list[float] = []
    samples: list[dict[str, Any]] = []
    for r in selected:
        mv = _resolve_metric_value(req.metric, r)
        if mv is None:
            continue
        values.append(float(mv))
        samples.append(
            {
                "sample_id": r.get("sample_id"),
                "lat": float(r["Latitude"]),
                "lon": float(r["Longitude"]),
                "metric_value": float(mv),
            }
        )

    if not values:
        return {"n_points": 0, "aggregates": None, "samples": [], "meta": {"metric": req.metric}}

    import numpy as np

    arr = np.asarray(values, dtype=float)
    aggregates = {
        "metric": req.metric,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "sum": float(arr.sum()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    return {
        "n_points": int(len(samples)),
        "aggregates": aggregates,
        "samples": samples,
        "meta": {
            "metric": req.metric,
            "model_version": req.model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@app.get("/latent_embeddings")
def latent_embeddings(
    method: str = "pca",
    n_components: int = 2,
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    model_version: str = "downstream_head_latest",
    recompute: bool = False,
    authorization: str | None = Header(default=None),
    wrapper: ModelWrapper = Depends(get_wrapper),
) -> dict[str, Any]:
    _require_token(authorization)
    method = method.lower().strip()
    if method not in ("pca", "umap", "tsne"):
        raise HTTPException(status_code=400, detail="method must be one of pca|umap|tsne")
    if n_components != 2:
        raise HTTPException(status_code=400, detail="Only n_components=2 is supported currently.")

    # Cache key depends on method+params+data mtime.
    data_path = _resolve_path("data/predictions/brines_imputed.csv")
    mtime = int(data_path.stat().st_mtime) if data_path.exists() else 0
    key = cache_key(
        "latent2d",
        params={
            "method": method,
            "n_components": n_components,
            "n_neighbors": int(n_neighbors),
            "perplexity": float(perplexity),
            "model_version": model_version,
            "data_mtime": mtime,
        },
    )
    if not recompute:
        cached = load_cached(key)
        if cached is not None:
            meta, emb = cached
            return {
                "n_samples": int(meta.get("n_samples", len(meta.get("sample_ids", [])))),
                "embeddings": emb.tolist(),
                "sample_ids": meta.get("sample_ids", []),
                "metadata": meta.get("metadata", []),
                "meta": meta.get("meta", {}),
            }

    # Load data for latents and metadata.
    import numpy as np
    import pandas as pd

    brines = _load_imputed_brines_df()
    preds = _load_predictions_df()
    # Align by row order (same source CSV); if misaligned, merge on lat/lon/location where possible.
    if len(preds) == len(brines):
        merged = pd.concat([brines.reset_index(drop=True), preds.filter(like="Pred_").reset_index(drop=True)], axis=1)
    else:
        merged = brines

    # Build a numeric matrix from available brine feature columns.
    feature_cols = [
        "Li_gL",
        "Mg_gL",
        "Na_gL",
        "K_gL",
        "Ca_gL",
        "SO4_gL",
        "Cl_gL",
        "MLR",
        "TDS_gL",
        "Light_kW_m2",
    ]
    X = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)

    # Try MAE latents; fallback to standardized features if artifacts unavailable.
    Z = None
    try:
        artifacts = wrapper._artifacts(model_version)  # type: ignore[attr-defined]
        # Standardize using wrapper's scaler (same as inference).
        from src.constants import BRINE_FEATURE_COLUMNS
        from src.features.scaler import get_stats

        mean, std = get_stats(wrapper.scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS)
        X_std = ((X - mean) / std).astype(np.float32)
        import torch

        with torch.no_grad():
            xt = torch.from_numpy(X_std).to(device=wrapper.device, dtype=torch.float32)
            zt = artifacts.mae.encode(xt)
            Z = zt.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        # Fallback: use standardized features as "latents" (good enough for UI test mode).
        X_center = X - np.nanmean(X, axis=0, keepdims=True)
        X_std = X_center / (np.nanstd(X_center, axis=0, keepdims=True) + 1e-6)
        Z = X_std.astype(np.float32)

    assert Z is not None

    # Project to 2D.
    if method == "pca":
        Zc = Z - Z.mean(axis=0, keepdims=True)
        # SVD PCA
        u, s, vt = np.linalg.svd(Zc, full_matrices=False)
        emb2 = (u[:, :2] * s[:2]).astype(np.float32)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=503, detail="scikit-learn is required for t-SNE") from exc
        emb2 = TSNE(n_components=2, perplexity=float(perplexity), init="pca", random_state=0).fit_transform(Z).astype(
            np.float32
        )
    else:  # umap
        try:
            import umap  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=503, detail="umap-learn is required for UMAP") from exc
        emb2 = (
            umap.UMAP(n_components=2, n_neighbors=int(n_neighbors), random_state=0).fit_transform(Z).astype(np.float32)
        )

    # Compute nearest neighbors in latent space (Z).
    k = 10
    # squared distances via dot products
    norms = (Z * Z).sum(axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :k]

    sample_ids = [_row_sample_id(i, merged.iloc[i].to_dict()) for i in range(len(merged))]

    metadata: list[dict[str, Any]] = []
    for i in range(len(merged)):
        row = merged.iloc[i].to_dict()
        sid = sample_ids[i]
        md: dict[str, Any] = {
            "sample_id": sid,
            "lat": float(row.get("Latitude")) if row.get("Latitude") is not None else None,
            "lon": float(row.get("Longitude")) if row.get("Longitude") is not None else None,
            "TDS_gL": row.get("TDS_gL"),
            "MLR": row.get("MLR"),
            "Light_kW_m2": row.get("Light_kW_m2"),
            "Location": row.get("Location"),
            "country": row.get("Country") or row.get("country"),
            "predicted_Li": row.get("Pred_Li_Crystallization_mg_m2_h"),
            "Selectivity": row.get("Pred_Selectivity"),
        }
        md["neighbors"] = [sample_ids[j] for j in nn_idx[i].tolist()]
        metadata.append(md)

    meta = {
        "method": method,
        "params": {"n_neighbors": int(n_neighbors), "perplexity": float(perplexity)},
        "cached_from": datetime.now(timezone.utc).isoformat(),
    }
    save_cached(
        key,
        meta={
            "n_samples": int(len(sample_ids)),
            "sample_ids": sample_ids,
            "metadata": metadata,
            "meta": meta,
        },
        array=emb2,
    )

    return {
        "n_samples": int(len(sample_ids)),
        "embeddings": emb2.tolist(),
        "sample_ids": sample_ids,
        "metadata": metadata,
        "meta": meta,
    }
