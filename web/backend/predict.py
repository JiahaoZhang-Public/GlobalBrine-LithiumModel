from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
from src.features.scaler import (
    destandardize_preserve_nan,
    get_stats,
    load_scaler,
    standardize_preserve_nan,
)
from src.models.inference import (
    InferenceArtifacts,
    load_artifacts,
    mae_impute_brine_features,
    predict_labels,
)


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        out = float(value)
        if np.isnan(out):  # type: ignore[arg-type]
            return None
        return float(out)
    except Exception:
        return None


def _head_in_dim(head) -> int:
    for module in getattr(head, "net", []):
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    raise RuntimeError("Unable to infer regression head input dimension.")


@dataclass
class PredictionResult:
    predictions: dict[str, float]
    imputed_input: dict[str, float | None]


class Predictor:
    """Wraps model loading and prediction helpers for the API."""

    def __init__(
        self,
        *,
        mae_path: Path,
        head_path: Path,
        scaler_path: Path,
        device: str = "auto",
    ) -> None:
        self.artifacts = load_artifacts(
            mae_path=mae_path,
            head_path=head_path,
            scaler_path=scaler_path,
            device=torch.device(device) if device != "auto" else None,
        )
        self.scaler = self.artifacts.scaler
        self.device = next(self.artifacts.mae.parameters()).device

    def predict_single(
        self, payload: Mapping[str, float | int | None], *, impute_missing: bool
    ) -> PredictionResult:
        raw = np.full((1, len(BRINE_FEATURE_COLUMNS)), np.nan, dtype=np.float32)
        for idx, col in enumerate(BRINE_FEATURE_COLUMNS):
            val = _safe_float(payload.get(col))
            if val is not None:
                raw[0, idx] = float(val)

        if impute_missing:
            imputed_raw, imputed_std = mae_impute_brine_features(
                self.artifacts.mae,
                brine_raw=raw,
                scaler=self.scaler,
                preserve_observed=True,
                device=self.device,
            )
        else:
            mean, std = get_stats(
                self.scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS
            )
            imputed_std = standardize_preserve_nan(raw, mean, std)
            imputed_raw = destandardize_preserve_nan(imputed_std, mean, std)

        z = self.artifacts.mae.encode(
            torch.from_numpy(imputed_std).to(device=self.device, dtype=torch.float32)
        )
        head_dim = _head_in_dim(self.artifacts.head)
        if head_dim == int(z.shape[1]):
            head_in = z
        elif head_dim == int(z.shape[1]) + 1:
            light_idx = list(BRINE_FEATURE_COLUMNS).index("Light_kW_m2")
            light = torch.from_numpy(imputed_std[:, light_idx : light_idx + 1]).to(
                device=self.device, dtype=torch.float32
            )
            head_in = torch.cat([z, light], dim=1)
        else:  # pragma: no cover
            raise RuntimeError(
                f"Unexpected head input dimension {head_dim}; expected {z.shape[1]} or {int(z.shape[1]) + 1}."
            )

        with torch.no_grad():
            y_std_pred = (
                self.artifacts.head(head_in).detach().cpu().numpy().astype(np.float32)
            )

        y_mean, y_std = get_stats(
            self.scaler,
            key="experimental_targets",
            feature_names=EXPERIMENTAL_TARGET_COLUMNS,
        )
        y_pred = (y_std_pred * y_std + y_mean).astype(np.float32)
        y_pred = np.clip(y_pred, 0.0, None)

        preds = {
            "Selectivity": float(y_pred[0, 0]),
            "Li_Crystallization_mg_m2_h": float(y_pred[0, 1]),
            "Evap_kg_m2_h": float(y_pred[0, 2]),
        }
        imputed_map = {
            col: (None if np.isnan(imputed_raw[0, idx]) else float(imputed_raw[0, idx]))
            for idx, col in enumerate(BRINE_FEATURE_COLUMNS)
        }
        return PredictionResult(predictions=preds, imputed_input=imputed_map)

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        *,
        impute_missing: bool,
    ) -> pd.DataFrame:
        missing = [c for c in EXPERIMENTAL_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing required columns: {', '.join(missing)}"
            )

        samples: list[dict[str, float | int | None]] = []
        for _, row in df.iterrows():
            sample: dict[str, float | int | None] = {}
            for col in EXPERIMENTAL_FEATURE_COLUMNS:
                sample[col] = _safe_float(row[col])
            # attach optional extra chemistry if present
            for col in BRINE_FEATURE_COLUMNS:
                if col in df.columns:
                    sample[col] = _safe_float(row[col])
            samples.append(sample)

        preds = predict_labels(
            self.artifacts,
            samples=samples,
            impute_missing_chemistry=impute_missing,
        )

        df_out = df.copy()
        df_out["Pred_Selectivity"] = preds[:, 0]
        df_out["Pred_Li_Crystallization_mg_m2_h"] = preds[:, 1]
        df_out["Pred_Evap_kg_m2_h"] = preds[:, 2]
        return df_out

