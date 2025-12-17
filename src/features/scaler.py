from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ScalerError(RuntimeError):
    pass


def load_scaler(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import joblib  # type: ignore
    except ModuleNotFoundError:
        with path.open("rb") as handle:
            return pickle.load(handle)
    else:
        return joblib.load(path)


def get_stats(
    scaler: dict[str, Any],
    *,
    key: str,
    feature_names: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    if key not in scaler:
        raise ScalerError(f"Missing scaler key: {key}")
    stats = scaler[key]
    if not isinstance(stats, dict):
        raise ScalerError(f"Unexpected scaler format for key: {key}")

    expected = list(stats.get("feature_names", []))
    names = list(feature_names)
    if expected and expected != names:
        raise ScalerError(
            f"Feature order mismatch for scaler key '{key}'. "
            f"Expected {expected}, got {names}."
        )

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    if mean.shape[0] != len(names) or std.shape[0] != len(names):
        raise ScalerError(f"Scaler stats shape mismatch for key: {key}")
    return mean, std


def standardize_preserve_nan(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    scaled = (x - mean) / std
    scaled[np.isnan(x)] = np.nan
    return scaled.astype(np.float32)


def destandardize_preserve_nan(
    x_std: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    x = x_std * std + mean
    x[np.isnan(x_std)] = np.nan
    return x.astype(np.float32)
