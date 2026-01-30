from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.predict_brines import predict_brines

from .schemas import GeoFeature, GeoResponse, DataPointProperties


def ensure_predictions_csv(
    *, predictions_path: Path, processed_dir: Path, mae_path: Path, head_path: Path, scaler_path: Path
) -> Path:
    """Ensure brine predictions CSV exists; generate if missing."""
    if predictions_path.exists():
        return predictions_path
    out_dir = predictions_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    predict_brines(
        processed_dir=processed_dir,
        out_dir=out_dir,
        mae_path=mae_path,
        head_path=head_path,
        scaler_path=scaler_path,
        device="auto",
    )
    return predictions_path


def load_geojson(predictions_path: Path) -> GeoResponse:
    df = pd.read_csv(predictions_path)
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise RuntimeError("Predictions CSV missing Latitude/Longitude columns.")

    features: List[GeoFeature] = []
    numeric_cols = [
        "MLR",
        "TDS_gL",
        "Light_kW_m2",
        "Pred_Selectivity",
        "Pred_Li_Crystallization_mg_m2_h",
        "Pred_Evap_kg_m2_h",
    ]
    for idx, row in df.iterrows():
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        props = DataPointProperties(
            id=int(idx),
            brine=row.get("Brine"),
            location=row.get("Location"),
            country=row.get("Country") if "Country" in row else None,
            Type_of_water=row.get("Type_of_water")
            if "Type_of_water" in row
            else row.get("Type_of_water"),
            MLR=_maybe_float(row.get("MLR")),
            TDS_gL=_maybe_float(row.get("TDS_gL")),
            Light_kW_m2=_maybe_float(row.get("Light_kW_m2")),
            Pred_Selectivity=_maybe_float(row.get("Pred_Selectivity")),
            Pred_Li_Crystallization_mg_m2_h=_maybe_float(
                row.get("Pred_Li_Crystallization_mg_m2_h")
            ),
            Pred_Evap_kg_m2_h=_maybe_float(row.get("Pred_Evap_kg_m2_h")),
            Li_gL=_maybe_float(row.get("Li_gL")),
            Mg_gL=_maybe_float(row.get("Mg_gL")),
            Na_gL=_maybe_float(row.get("Na_gL")),
            K_gL=_maybe_float(row.get("K_gL")),
            Ca_gL=_maybe_float(row.get("Ca_gL")),
            SO4_gL=_maybe_float(row.get("SO4_gL")),
            Cl_gL=_maybe_float(row.get("Cl_gL")),
        )
        feature = GeoFeature(
            geometry={"type": "Point", "coordinates": [float(lon), float(lat)]},
            properties=props,
        )
        features.append(feature)

    meta: Dict[str, object] = {
        "count": len(features),
        "updated_at": datetime.fromtimestamp(
            predictions_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            meta[col] = {
                "min": float(np.nanmin(series)) if series.notna().any() else None,
                "max": float(np.nanmax(series)) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
            }
    return GeoResponse(features=features, meta=meta)


def _maybe_float(val):
    try:
        if val is None:
            return None
        f = float(val)
        return None if np.isnan(f) else float(f)
    except Exception:
        return None

