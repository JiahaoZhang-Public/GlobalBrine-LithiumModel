from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.constants import (
    BRINE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    EXPERIMENTAL_TARGET_COLUMNS,
)
from src.features.scaler import (
    destandardize_preserve_nan,
    get_stats,
    load_scaler,
    standardize_preserve_nan,
)
from src.models.mae import TabularMAE, TabularMAEConfig
from src.models.regression_head import RegressionHead, RegressionHeadConfig


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class InferenceArtifacts:
    mae: TabularMAE
    head: RegressionHead
    scaler: dict[str, Any]


def load_mae(path: Path, *, device: torch.device | None = None) -> TabularMAE:
    ckpt = torch.load(path, map_location="cpu")
    cfg = TabularMAEConfig(**ckpt.get("mae_config", {}))
    model = TabularMAE(num_features=int(ckpt["num_features"]), config=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model


def _infer_head_in_dim(state_dict: Mapping[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2:
            continue
        if key.endswith(".weight") or key.endswith("weight"):
            return int(value.shape[1])
    raise ValueError("Unable to infer regression head input dim from checkpoint.")


def load_head(
    path: Path,
    *,
    out_dim: int,
    device: torch.device | None = None,
) -> RegressionHead:
    ckpt = torch.load(path, map_location="cpu")
    cfg = RegressionHeadConfig(**ckpt.get("head_config", {}))
    in_dim = int(ckpt.get("in_dim", 0))
    if in_dim <= 0:
        in_dim = _infer_head_in_dim(ckpt["head_state_dict"])
    cfg = RegressionHeadConfig(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        out_dim=out_dim,
    )
    head = RegressionHead(in_dim=in_dim, config=cfg)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    if device is not None:
        head.to(device)
    return head


def load_artifacts(
    *,
    mae_path: Path,
    head_path: Path,
    scaler_path: Path,
    device: torch.device | None = None,
) -> InferenceArtifacts:
    device = device or auto_device()
    scaler = load_scaler(scaler_path)
    mae = load_mae(mae_path, device=device)
    head = load_head(
        head_path,
        out_dim=len(EXPERIMENTAL_TARGET_COLUMNS),
        device=device,
    )
    return InferenceArtifacts(mae=mae, head=head, scaler=scaler)


def _as_matrix(
    rows: Sequence[Mapping[str, float | int | None]],
    columns: Iterable[str],
) -> np.ndarray:
    cols = list(columns)
    mat = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            val = row.get(col, None)
            if val is None:
                continue
            mat[i, j] = float(val)
    return mat


@torch.no_grad()
def mae_impute_brine_features(
    mae: TabularMAE,
    *,
    brine_raw: np.ndarray,
    scaler: dict[str, Any],
    preserve_observed: bool = True,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing brine chemistry features using the MAE.

    Returns (imputed_raw, imputed_normalized).
    """
    mean, std = get_stats(
        scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS
    )
    x_std = standardize_preserve_nan(brine_raw.astype(np.float32), mean, std)
    dev = device or next(mae.parameters()).device

    xt = torch.from_numpy(x_std).to(dtype=torch.float32, device=dev)
    pred_std = mae(xt).detach().cpu().numpy().astype(np.float32)

    missing = np.isnan(x_std)
    imputed_std = np.where(missing, pred_std, x_std) if preserve_observed else pred_std
    imputed_raw = destandardize_preserve_nan(imputed_std, mean, std)
    imputed_raw = np.clip(imputed_raw, 0.0, None)
    return imputed_raw, imputed_std


@torch.no_grad()
def predict_labels(
    artifacts: InferenceArtifacts,
    *,
    samples: Sequence[Mapping[str, float | int | None]],
    impute_missing_chemistry: bool = False,
) -> np.ndarray:
    """Predict experimental targets from raw inputs.

    Required keys per sample: `TDS_gL`, `MLR`, `Light_kW_m2` (can be None).

    If impute_missing_chemistry=True, we first MAE-impute the full brine-chemistry
    vector using only provided chemistry fields, then encode.
    """
    scaler = artifacts.scaler
    device = next(artifacts.mae.parameters()).device

    x_exp_raw = _as_matrix(samples, EXPERIMENTAL_FEATURE_COLUMNS)

    br_mean, br_std = get_stats(
        scaler, key="brine_features", feature_names=BRINE_FEATURE_COLUMNS
    )
    y_mean, y_std = get_stats(
        scaler, key="experimental_targets", feature_names=EXPERIMENTAL_TARGET_COLUMNS
    )

    x_exp_std = x_exp_raw.astype(np.float32)
    tds_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("TDS_gL")
    mlr_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("MLR")
    light_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("Light_kW_m2")

    br_cols = list(BRINE_FEATURE_COLUMNS)
    br_tds = br_cols.index("TDS_gL")
    br_mlr = br_cols.index("MLR")
    br_light = br_cols.index("Light_kW_m2")

    x_exp_std[:, tds_idx] = (x_exp_std[:, tds_idx] - br_mean[br_tds]) / br_std[br_tds]
    x_exp_std[:, mlr_idx] = (x_exp_std[:, mlr_idx] - br_mean[br_mlr]) / br_std[br_mlr]
    x_exp_std[:, light_idx] = (x_exp_std[:, light_idx] - br_mean[br_light]) / br_std[
        br_light
    ]

    # Build chemistry vector for MAE encoder.
    chem_std = np.full(
        (x_exp_std.shape[0], len(BRINE_FEATURE_COLUMNS)), np.nan, dtype=np.float32
    )
    chem_std[:, br_tds] = x_exp_std[:, tds_idx]
    chem_std[:, br_mlr] = x_exp_std[:, mlr_idx]
    chem_std[:, br_light] = x_exp_std[:, light_idx]

    if impute_missing_chemistry:
        xt = torch.from_numpy(chem_std).to(device=device, dtype=torch.float32)
        pred_std = artifacts.mae(xt).detach().cpu().numpy().astype(np.float32)
        chem_std = np.where(np.isnan(chem_std), pred_std, chem_std)

    z = artifacts.mae.encode(
        torch.from_numpy(chem_std).to(device=device, dtype=torch.float32)
    )
    head_in_dim: int | None = None
    for module in artifacts.head.net:
        if isinstance(module, nn.Linear):
            head_in_dim = int(module.in_features)
            break
    if head_in_dim is None:
        raise ValueError("Unable to infer regression head input dim.")

    if head_in_dim == int(z.shape[1]):
        head_in = z
    elif head_in_dim == int(z.shape[1]) + 1:
        light = torch.from_numpy(x_exp_std[:, light_idx : light_idx + 1]).to(
            device=device, dtype=torch.float32
        )
        head_in = torch.cat([z, light], dim=1)
    else:
        raise ValueError(
            f"Unexpected regression head input dim={head_in_dim}; "
            f"expected {int(z.shape[1])} or {int(z.shape[1]) + 1}."
        )
    y_std_pred = artifacts.head(head_in).detach().cpu().numpy().astype(np.float32)
    y_pred = y_std_pred * y_std + y_mean
    y_pred = np.clip(y_pred, 0.0, None)
    return y_pred.astype(np.float32)
