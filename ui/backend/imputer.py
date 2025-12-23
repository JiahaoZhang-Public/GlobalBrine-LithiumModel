from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.constants import BRINE_FEATURE_COLUMNS
from src.features.scaler import get_stats


class ImputationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImputationResult:
    imputed_features: dict[str, Any]
    warnings: list[str]


def _to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def _extract_brine_vector(features: Mapping[str, Any]) -> np.ndarray:
    x = np.full((len(BRINE_FEATURE_COLUMNS),), np.nan, dtype=np.float32)
    for i, name in enumerate(BRINE_FEATURE_COLUMNS):
        if name in features:
            x[i] = _to_float_or_nan(features.get(name))
    return x


def impute_none(features: Mapping[str, Any]) -> ImputationResult:
    out = dict(features)
    return ImputationResult(imputed_features=out, warnings=[])


def impute_mean(
    features: Mapping[str, Any],
    *,
    scaler: dict[str, Any],
) -> ImputationResult:
    mean, _std = get_stats(scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS)
    x = _extract_brine_vector(features)
    filled = np.where(np.isnan(x), mean.astype(np.float32), x).astype(np.float32)
    out = dict(features)
    for i, name in enumerate(BRINE_FEATURE_COLUMNS):
        out[name] = float(filled[i])
    return ImputationResult(imputed_features=out, warnings=[])


def impute_knn(
    features: Mapping[str, Any],
    *,
    reference_matrix: np.ndarray,
    scaler: dict[str, Any],
    k: int = 5,
) -> ImputationResult:
    if reference_matrix.ndim != 2 or reference_matrix.shape[1] != len(BRINE_FEATURE_COLUMNS):
        raise ImputationError("reference_matrix must be shape [N, num_brine_features].")
    if k <= 0:
        raise ImputationError("k must be > 0.")

    mean, std = get_stats(scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS)
    x = _extract_brine_vector(features).astype(np.float32)
    obs = ~np.isnan(x)
    if not obs.any():
        # No observed features; fall back to mean.
        return impute_mean(features, scaler=scaler)

    x_std = (x - mean) / std
    ref = reference_matrix.astype(np.float32)
    ref_std = (ref - mean) / std

    # Nan-aware distance over overlapping observed dims.
    diffs = ref_std[:, obs] - x_std[obs]
    d2 = np.nanmean(diffs * diffs, axis=1)
    if np.isnan(d2).all():
        return impute_mean(features, scaler=scaler)

    nn_idx = np.argsort(d2)[: min(k, ref.shape[0])]
    nn = ref[nn_idx]  # raw space

    filled = x.copy()
    missing = ~obs
    if missing.any():
        filled[missing] = np.nanmean(nn[:, missing], axis=0)

    # If any column couldn't be imputed (all neighbors nan), fall back to mean.
    if np.isnan(filled).any():
        filled = np.where(np.isnan(filled), mean.astype(np.float32), filled)

    out = dict(features)
    for i, name in enumerate(BRINE_FEATURE_COLUMNS):
        out[name] = float(filled[i])
    return ImputationResult(
        imputed_features=out,
        warnings=[f"knn_impute:k={int(k)}"],
    )

