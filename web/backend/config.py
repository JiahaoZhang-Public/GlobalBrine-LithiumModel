from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the web API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GLB_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    api_prefix: str = "/api/v1"
    model_version: str = "0.2.2"
    git_tag: str | None = None

    mae_path: Path = Path("models/mae_pretrained.pth")
    head_path: Path = Path("models/downstream_head.pth")
    scaler_path: Path = Path("data/processed/feature_scaler.joblib")
    processed_dir: Path = Path("data/processed")
    predictions_csv: Path = Path("data/predictions/brines_with_predictions.csv")

    job_storage_dir: Path = Path("web/backend/jobs")
    job_ttl_hours: int = 48
    max_batch_rows: int = 200_000

    allow_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ]
    )

    map_style_url: str = "https://demotiles.maplibre.org/style.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
