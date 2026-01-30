from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS


class ModelArtifact(BaseModel):
    name: str
    path: str
    sha256: str
    size_bytes: int
    modified_at: datetime


class ModelMetadata(BaseModel):
    version: str
    git_commit: str | None = None
    git_tag: str | None = None
    artifacts: List[ModelArtifact]
    feature_schema: List[str] = list(BRINE_FEATURE_COLUMNS)
    targets: List[str] = list(EXPERIMENTAL_TARGET_COLUMNS)
    scaler_path: str


class SinglePredictRequest(BaseModel):
    Li_gL: float | None = Field(default=None, description="Lithium concentration (g/L)")
    Mg_gL: float | None = None
    Na_gL: float | None = None
    K_gL: float | None = None
    Ca_gL: float | None = None
    SO4_gL: float | None = None
    Cl_gL: float | None = None
    MLR: float | None = Field(
        default=None, description="Magnesium-to-lithium ratio (dimensionless)"
    )
    TDS_gL: float | None = Field(default=None, description="Total dissolved solids (g/L)")
    Light_kW_m2: float | None = Field(
        default=None, description="Solar irradiance (kW/m²)"
    )
    impute_missing_chemistry: bool = Field(
        default=True,
        description="Use the MAE encoder to impute missing chemistry values.",
    )


class PredictionValues(BaseModel):
    Selectivity: float
    Li_Crystallization_mg_m2_h: float
    Evap_kg_m2_h: float


class SinglePredictResponse(BaseModel):
    predictions: PredictionValues
    imputed_input: Dict[str, float | None]
    metadata: ModelMetadata


class BatchJobRequest(BaseModel):
    impute_missing_chemistry: bool = True


class BatchJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    submitted_at: datetime
    completed_at: datetime | None = None
    download_url: str | None = None
    error: str | None = None


class DataPointProperties(BaseModel):
    id: int
    brine: str | None = None
    location: str | None = None
    country: str | None = None
    Type_of_water: str | None = None
    MLR: float | None = None
    TDS_gL: float | None = None
    Light_kW_m2: float | None = None
    Pred_Selectivity: float | None = None
    Pred_Li_Crystallization_mg_m2_h: float | None = None
    Pred_Evap_kg_m2_h: float | None = None
    Li_gL: float | None = None
    Mg_gL: float | None = None
    Na_gL: float | None = None
    K_gL: float | None = None
    Ca_gL: float | None = None
    SO4_gL: float | None = None
    Cl_gL: float | None = None


class GeoFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: Dict[str, Any]
    properties: DataPointProperties


class GeoResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoFeature]
    meta: Dict[str, Any]

