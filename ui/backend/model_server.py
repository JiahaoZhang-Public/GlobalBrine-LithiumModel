from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
from src.data.light import GeoTiffError, sample_light_kW_m2
from src.features.scaler import get_stats, load_scaler
from src.interpretability.attributions import integrated_gradients_for_prediction
from src.models.inference import (
    InferenceArtifacts,
    auto_device,
    load_artifacts,
    mae_impute_brine_features,
)

from ui.backend.imputer import ImputationResult, impute_knn, impute_mean, impute_none


class ModelServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    version: str
    path: str
    created_at: float | None
    meta: dict[str, Any]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).resolve()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def _zscore_warnings(
    *,
    scaler: dict[str, Any],
    features: Mapping[str, Any],
    z_threshold: float = 3.0,
) -> list[str]:
    mean, std = get_stats(scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS)
    warnings: list[str] = []
    for i, name in enumerate(BRINE_FEATURE_COLUMNS):
        if name not in features:
            continue
        val = _to_float(features.get(name))
        if val is None:
            continue
        z = (float(val) - float(mean[i])) / float(std[i] or 1.0)
        if abs(z) > z_threshold:
            warnings.append(f"out_of_range:{name}:z={z:.2f}")
    return warnings


class ModelWrapper:
    def __init__(
        self,
        *,
        model_dir: Path,
        data_dir: Path,
        device: str = "auto",
    ) -> None:
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = auto_device() if device == "auto" else torch.device(device)

        self.scaler_path = self.data_dir / "feature_scaler.joblib"
        self.scaler: dict[str, Any] = load_scaler(self.scaler_path)

        # Model registry.
        self._models: dict[str, dict[str, Path]] = {
            "downstream_head_latest": {
                "mae": self.model_dir / "mae_pretrained.pth",
                "head": self.model_dir / "downstream_head.pth",
            }
        }

        self.artifacts_by_version: dict[str, InferenceArtifacts] = {}
        self._reference_brines: np.ndarray | None = None

    def list_models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for version, paths in self._models.items():
            head = paths.get("head")
            created = None
            if head is not None and head.exists():
                created = head.stat().st_mtime
            out.append(
                ModelInfo(
                    version=version,
                    path=str(head) if head is not None else "",
                    created_at=created,
                    meta={
                        "mae_path": str(paths.get("mae", "")),
                        "scaler_path": str(self.scaler_path),
                    },
                )
            )
        return out

    def _artifacts(self, model_version: str) -> InferenceArtifacts:
        if model_version not in self._models:
            raise ModelServerError(f"Unknown model_version: {model_version}")
        if model_version not in self.artifacts_by_version:
            paths = self._models[model_version]
            self.artifacts_by_version[model_version] = load_artifacts(
                mae_path=paths["mae"],
                head_path=paths["head"],
                scaler_path=self.scaler_path,
                device=self.device,
            )
        return self.artifacts_by_version[model_version]

    def _reference_brines(self) -> np.ndarray:
        if self._reference_brines is None:
            brines_csv = self.data_dir / "brines.csv"
            if not brines_csv.exists():
                raise ModelServerError(f"Missing reference brines.csv at {brines_csv}")
            import pandas as pd

            df = pd.read_csv(brines_csv)
            self._reference_brines = df[list(BRINE_FEATURE_COLUMNS)].to_numpy(
                dtype=np.float32, copy=True
            )
        return self._reference_brines

    def infer_light(self, *, lat: float, lon: float) -> dict[str, Any]:
        geotiff = os.environ.get("LIGHT_GEOTIFF_PATH", "").strip()
        if not geotiff:
            raise FileNotFoundError("LIGHT_GEOTIFF_PATH not configured")
        geotiff_path = Path(geotiff)
        if not geotiff_path.exists():
            raise FileNotFoundError(str(geotiff_path))
        scale = float(os.environ.get("LIGHT_SCALE", "1.0"))
        try:
            value = sample_light_kW_m2(geotiff_path, lat=lat, lon=lon, scale=scale)
        except GeoTiffError as exc:
            raise ModelServerError(str(exc)) from exc
        if value is None:
            raise ModelServerError("Point outside raster or masked")
        return {
            "lat": float(lat),
            "lon": float(lon),
            "Light_kW_m2": float(value),
            "source": "GHI_GeoTIFF",
            "confidence": 0.85,
        }

    def impute(
        self,
        *,
        features: Mapping[str, Any],
        impute_strategy: str,
        model_version: str,
    ) -> ImputationResult:
        strategy = impute_strategy.lower()
        if strategy == "none":
            return impute_none(features)
        if strategy == "mean":
            return impute_mean(features, scaler=self.scaler)
        if strategy == "knn":
            return impute_knn(
                features,
                reference_matrix=self._reference_brines(),
                scaler=self.scaler,
                k=int(os.environ.get("KNN_K", "5")),
            )
        if strategy == "model":
            artifacts = self._artifacts(model_version)
            x_raw = np.full((1, len(BRINE_FEATURE_COLUMNS)), np.nan, dtype=np.float32)
            for j, name in enumerate(BRINE_FEATURE_COLUMNS):
                if name in features:
                    val = _to_float(features.get(name))
                    x_raw[0, j] = np.nan if val is None else float(val)

            imputed_raw, _imputed_std = mae_impute_brine_features(
                artifacts.mae,
                brine_raw=x_raw,
                scaler=self.scaler,
                preserve_observed=True,
                device=self.device,
            )
            out = dict(features)
            for j, name in enumerate(BRINE_FEATURE_COLUMNS):
                out[name] = float(imputed_raw[0, j])
            return ImputationResult(
                imputed_features=out,
                warnings=["model_impute:mae"],
            )
        raise ModelServerError(
            f"Unknown impute_strategy={impute_strategy!r}; expected none|mean|knn|model"
        )

    def predict_one(
        self,
        *,
        features: Mapping[str, Any],
        model_version: str,
        impute_strategy: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        rid = request_id or str(uuid.uuid4())
        warnings: list[str] = []

        # Optional: infer Light if missing and lat/lon are provided and GeoTIFF configured.
        lat = _to_float(features.get("lat"))
        lon = _to_float(features.get("lon"))
        feat = dict(features)
        if feat.get("Light_kW_m2") in (None, "", "nan") and lat is not None and lon is not None:
            try:
                light = self.infer_light(lat=lat, lon=lon)
            except Exception:
                pass
            else:
                feat["Light_kW_m2"] = light["Light_kW_m2"]
                warnings.append("light_inferred:geotiff")

        imp = self.impute(
            features=feat,
            impute_strategy=impute_strategy,
            model_version=model_version,
        )
        warnings.extend(imp.warnings)
        warnings.extend(_zscore_warnings(scaler=self.scaler, features=imp.imputed_features))

        artifacts = self._artifacts(model_version)

        # Build a single brine feature vector in raw space.
        x_raw = np.full((1, len(BRINE_FEATURE_COLUMNS)), np.nan, dtype=np.float32)
        for j, name in enumerate(BRINE_FEATURE_COLUMNS):
            val = _to_float(imp.imputed_features.get(name))
            x_raw[0, j] = np.nan if val is None else float(val)

        # If we still have NaNs after "none", we can't run the MAE+head reliably.
        if np.isnan(x_raw).any():
            raise ModelServerError(
                "Missing required brine features for prediction after imputation. "
                "Use impute_strategy=mean|knn|model or provide all features."
            )

        mean, std = get_stats(
            self.scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS
        )
        x_std = ((x_raw - mean) / std).astype(np.float32)
        xt = torch.from_numpy(x_std).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            z = artifacts.mae.encode(xt)
            # Match inference.py behavior: support head with or without Light concatenation.
            in_dim = None
            for m in artifacts.head.modules():
                if isinstance(m, torch.nn.Linear):
                    in_dim = int(m.in_features)
                    break
            if in_dim is None:
                raise ModelServerError("Unable to infer regression head input dim.")
            if in_dim == int(z.shape[1]):
                y_std_pred = artifacts.head(z)
            elif in_dim == int(z.shape[1]) + 1:
                light_idx = list(BRINE_FEATURE_COLUMNS).index("Light_kW_m2")
                light = xt[:, light_idx : light_idx + 1]
                y_std_pred = artifacts.head(torch.cat([z, light], dim=1))
            else:
                raise ModelServerError(
                    f"Unexpected head input dim={in_dim}; expected {int(z.shape[1])} "
                    f"or {int(z.shape[1]) + 1}."
                )

        y_mean, y_std = get_stats(
            self.scaler,
            key="experimental_targets",
            feature_names=EXPERIMENTAL_TARGET_COLUMNS,
        )
        y_raw = (y_std_pred.detach().cpu().numpy().astype(np.float32) * y_std + y_mean)[
            0
        ]
        y_raw = np.clip(y_raw, 0.0, None)

        preds: dict[str, Any] = {}
        for i, name in enumerate(EXPERIMENTAL_TARGET_COLUMNS):
            preds[name] = {"value": float(y_raw[i]), "ci": None}

        attrs_out: dict[str, dict[str, float]] = {}
        try:
            ig = integrated_gradients_for_prediction(
                artifacts,
                sample={
                    "TDS_gL": float(imp.imputed_features.get("TDS_gL")),
                    "MLR": float(imp.imputed_features.get("MLR")),
                    "Light_kW_m2": float(imp.imputed_features.get("Light_kW_m2")),
                },
                steps=int(os.environ.get("IG_STEPS", "50")),
            )
            for tname, feat_map in ig.attributions.items():
                attrs_out[tname] = {k: float(v) for k, v in feat_map.items()}
        except Exception as exc:
            warnings.append(f"attributions_unavailable:{type(exc).__name__}")

        runtime_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "request_id": rid,
            "input_features": dict(features),
            "imputed_features": imp.imputed_features,
            "predictions": preds,
            "attributions": attrs_out,
            "model_version": model_version,
            "meta": {"runtime_ms": runtime_ms, "warnings": warnings},
        }

    def predict_brine_matrix(self, *, x_raw: np.ndarray, model_version: str) -> np.ndarray:
        """Predict all targets for a batch of brine-feature rows.

        Args:
            x_raw: raw brine features, shape [N, len(BRINE_FEATURE_COLUMNS)], no NaNs.
        Returns:
            y_raw: raw predictions, shape [N, 3], clipped to >=0.
        """
        if x_raw.ndim != 2 or x_raw.shape[1] != len(BRINE_FEATURE_COLUMNS):
            raise ModelServerError(
                f"x_raw must be shape [N, {len(BRINE_FEATURE_COLUMNS)}], got {tuple(x_raw.shape)}"
            )
        if np.isnan(x_raw).any():
            raise ModelServerError("x_raw contains NaNs; impute/fill before calling predict_brine_matrix().")

        artifacts = self._artifacts(model_version)
        mean, std = get_stats(self.scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS)
        y_mean, y_std = get_stats(
            self.scaler, key="experimental_targets", feature_names=EXPERIMENTAL_TARGET_COLUMNS
        )

        x_std = ((x_raw.astype(np.float32) - mean) / std).astype(np.float32)
        xt = torch.from_numpy(x_std).to(device=self.device, dtype=torch.float32)
        light_idx = list(BRINE_FEATURE_COLUMNS).index("Light_kW_m2")

        with torch.no_grad():
            z = artifacts.mae.encode(xt)
            in_dim = None
            for m in artifacts.head.modules():
                if isinstance(m, torch.nn.Linear):
                    in_dim = int(m.in_features)
                    break
            if in_dim is None:
                raise ModelServerError("Unable to infer regression head input dim.")
            if in_dim == int(z.shape[1]):
                y_std_pred = artifacts.head(z)
            elif in_dim == int(z.shape[1]) + 1:
                y_std_pred = artifacts.head(torch.cat([z, xt[:, light_idx : light_idx + 1]], dim=1))
            else:
                raise ModelServerError(
                    f"Unexpected head input dim={in_dim}; expected {int(z.shape[1])} or {int(z.shape[1]) + 1}."
                )

        y_raw = y_std_pred.detach().cpu().numpy().astype(np.float32) * y_std + y_mean
        y_raw = np.clip(y_raw, 0.0, None)
        return y_raw.astype(np.float32, copy=False)


class DummyModelWrapper(ModelWrapper):
    def __init__(self) -> None:
        self.model_dir = Path(".")
        self.data_dir = Path(".")
        self.device = torch.device("cpu")
        self.scaler_path = Path(".")
        self.scaler = {"brine_features": {"feature_names": list(BRINE_FEATURE_COLUMNS), "mean": [0.0]*len(BRINE_FEATURE_COLUMNS), "std": [1.0]*len(BRINE_FEATURE_COLUMNS)},
                       "experimental_targets": {"feature_names": list(EXPERIMENTAL_TARGET_COLUMNS), "mean": [0.0]*len(EXPERIMENTAL_TARGET_COLUMNS), "std": [1.0]*len(EXPERIMENTAL_TARGET_COLUMNS)}}
        self._models = {"downstream_head_latest": {"mae": Path(""), "head": Path("")}}
        self.artifacts_by_version = {}
        self._reference_brines = None

    def _artifacts(self, model_version: str) -> InferenceArtifacts:  # pragma: no cover
        raise ModelServerError("DummyModelWrapper has no artifacts.")

    def predict_one(self, *, features: Mapping[str, Any], model_version: str, impute_strategy: str, request_id: str | None = None) -> dict[str, Any]:
        rid = request_id or "demo"
        preds = {n: {"value": 0.0, "ci": None} for n in EXPERIMENTAL_TARGET_COLUMNS}
        attrs = {n: {"TDS_gL": 0.0, "MLR": 0.0, "Light_kW_m2": 0.0} for n in EXPERIMENTAL_TARGET_COLUMNS}
        return {
            "request_id": rid,
            "input_features": dict(features),
            "imputed_features": dict(features),
            "predictions": preds,
            "attributions": attrs,
            "model_version": model_version,
            "meta": {"runtime_ms": 1, "warnings": ["dummy_model"]},
        }
