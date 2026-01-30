from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.constants import BRINE_FEATURE_COLUMNS

from .config import Settings, get_settings
from .data_access import ensure_predictions_csv, load_geojson
from .jobs import JobRunner, JobStore
from .predict import PredictionResult, Predictor
from .schemas import (
    BatchJobStatus,
    GeoResponse,
    ModelArtifact,
    ModelMetadata,
    SinglePredictRequest,
    SinglePredictResponse,
)

logger = logging.getLogger("globalbrine.api")
logging.basicConfig(level=logging.INFO)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        return None


def _git_tag() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
            .decode()
            .strip()
        )
    except Exception:
        return None


def build_model_metadata(settings: Settings) -> ModelMetadata:
    artifacts = []
    for name, path in [
        ("mae", settings.mae_path),
        ("head", settings.head_path),
        ("scaler", settings.scaler_path),
    ]:
        if not path.exists():
            continue
        artifacts.append(
            ModelArtifact(
                name=name,
                path=str(path),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
                modified_at=datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ),
            )
        )
    return ModelMetadata(
        version=settings.model_version,
        git_commit=_git_commit(),
        git_tag=settings.git_tag or _git_tag(),
        artifacts=artifacts,
        scaler_path=str(settings.scaler_path),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading model artifacts…")
    predictor = Predictor(
        mae_path=settings.mae_path,
        head_path=settings.head_path,
        scaler_path=settings.scaler_path,
    )
    ensure_predictions_csv(
        predictions_path=settings.predictions_csv,
        processed_dir=settings.processed_dir,
        mae_path=settings.mae_path,
        head_path=settings.head_path,
        scaler_path=settings.scaler_path,
    )
    geo = load_geojson(settings.predictions_csv)
    store = JobStore(settings.job_storage_dir, ttl_hours=settings.job_ttl_hours)
    runner = JobRunner(store)

    app.state.settings = settings
    app.state.predictor = predictor
    app.state.geo = geo
    app.state.job_store = store
    app.state.job_runner = runner
    app.state.model_metadata = build_model_metadata(settings)
    yield
    store.cleanup_expired()


app = FastAPI(
    title="GlobalBrine-LithiumModel API",
    version="0.1.0",
    description="FastAPI service for GlobalBrine lithium predictions.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> Predictor:
    return app.state.predictor


def get_settings_dep() -> Settings:
    return app.state.settings


def get_job_store() -> JobStore:
    return app.state.job_store


def get_job_runner() -> JobRunner:
    return app.state.job_runner


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "version": app.state.settings.model_version,
    }


@app.get("/api/v1/model", response_model=ModelMetadata)
async def model_metadata() -> ModelMetadata:
    return app.state.model_metadata


@app.get("/api/v1/data/points", response_model=GeoResponse)
async def data_points() -> GeoResponse:
    return app.state.geo


@app.post("/api/v1/predict", response_model=SinglePredictResponse)
async def predict_single(
    payload: SinglePredictRequest,
    predictor: Predictor = Depends(get_predictor),
) -> SinglePredictResponse:
    try:
        result: PredictionResult = predictor.predict_single(
            payload.dict(), impute_missing=payload.impute_missing_chemistry
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SinglePredictResponse(
        predictions=result.predictions,
        imputed_input=result.imputed_input,
        metadata=app.state.model_metadata,
    )


@app.post("/api/v1/predict/batch", response_model=BatchJobStatus)
async def predict_batch(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    impute_missing_chemistry: bool = True,
    settings: Settings = Depends(get_settings_dep),
    store: JobStore = Depends(get_job_store),
    runner: JobRunner = Depends(get_job_runner),
    predictor: Predictor = Depends(get_predictor),
) -> BatchJobStatus:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    job = store.create(input_path=settings.job_storage_dir / "pending")
    job_dir = settings.job_storage_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.csv"
    store.update_input_path(job.id, input_path)

    content = await file.read()
    input_path.write_bytes(content)

    async def _run_job():
        try:
            df = pd.read_csv(input_path)
            if len(df) > settings.max_batch_rows:
                raise ValueError(
                    f"Too many rows ({len(df)}). Limit: {settings.max_batch_rows}."
                )
            out_df = predictor.predict_dataframe(
                df, impute_missing=impute_missing_chemistry
            )
            output_path = job_dir / "result.csv"
            out_df.to_csv(output_path, index=False)
            store.set_completed(job.id, output_path)
        except Exception as exc:
            logger.exception("Batch job failed")
            store.set_failed(job.id, str(exc))

    asyncio.create_task(runner.run(job.id, _run_job))

    return BatchJobStatus(
        job_id=job.id,
        status=job.status,
        submitted_at=job.submitted_at,
        completed_at=job.completed_at,
        download_url=None,
    )


@app.get("/api/v1/predict/batch/{job_id}/status", response_model=BatchJobStatus)
async def batch_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings_dep),
) -> BatchJobStatus:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    download_url = (
        f"{settings.api_prefix}/predict/batch/{job.id}/result"
        if job.status == "completed"
        else None
    )
    return BatchJobStatus(
        job_id=job.id,
        status=job.status,
        submitted_at=job.submitted_at,
        completed_at=job.completed_at,
        error=job.error,
        download_url=download_url,
    )


@app.get("/api/v1/predict/batch/{job_id}/result")
async def batch_result(
    job_id: str, store: JobStore = Depends(get_job_store)
) -> StreamingResponse:
    job = store.get(job_id)
    if job is None or job.output_path is None:
        raise HTTPException(status_code=404, detail="Job not found or not finished.")
    if not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file missing.")

    def iterfile():
        with open(job.output_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="result-{job_id}.csv"'},
    )


@app.get("/api/v1")
async def api_index(settings: Settings = Depends(get_settings_dep)) -> JSONResponse:
    return JSONResponse(
        {
            "service": "GlobalBrine-LithiumModel API",
            "version": settings.model_version,
            "endpoints": [
                "/api/v1/model",
                "/api/v1/data/points",
                "/api/v1/predict",
                "/api/v1/predict/batch",
            ],
        }
    )
