from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mae import TabularMAE


@dataclass(frozen=True)
class FeatureReconstructionResult:
    feature_index: int
    mse: float
    n: int


@torch.no_grad()
def masked_mse_dataset(
    mae: TabularMAE,
    x: np.ndarray,
    *,
    mask_ratio: float,
    batch_size: int = 256,
    seed: int = 42,
    device: Optional[torch.device] = None,
    fallback_to_cpu: bool = True,
) -> float:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a non-empty 2D array [n, d].")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device or next(mae.parameters()).device
    original_device = next(mae.parameters()).device
    mae.eval().to(dev)
    xt = torch.from_numpy(x).to(dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt), batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(dev, non_blocking=True)
        loss = mae.pretrain_loss(batch_x, mask_ratio=mask_ratio)
        losses.append(float(loss.detach().cpu().item()))

    value = float(np.mean(losses)) if losses else float("nan")
    if not np.isfinite(value) and fallback_to_cpu and dev.type != "cpu":
        mae.to(original_device)
        return masked_mse_dataset(
            mae,
            x,
            mask_ratio=mask_ratio,
            batch_size=batch_size,
            seed=seed,
            device=torch.device("cpu"),
            fallback_to_cpu=False,
        )

    mae.to(original_device)
    return value


@torch.no_grad()
def reconstruct_with_feature_mask(
    mae: TabularMAE,
    x: np.ndarray,
    *,
    feature_mask: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    fallback_to_cpu: bool = True,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D [n, d].")
    if feature_mask.shape != x.shape:
        raise ValueError("feature_mask must have same shape as x.")

    dev = device or next(mae.parameters()).device
    original_device = next(mae.parameters()).device
    mae.eval().to(dev)
    xt = torch.from_numpy(x).to(dtype=torch.float32)
    mt = torch.from_numpy(feature_mask.astype(bool))

    preds: list[np.ndarray] = []
    for start in range(0, xt.shape[0], batch_size):
        batch_x = xt[start : start + batch_size].to(dev, non_blocking=True)
        batch_m = mt[start : start + batch_size].to(dev, non_blocking=True)
        pred = mae(batch_x, feature_mask=batch_m)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
    out = np.concatenate(preds, axis=0)
    if not np.isfinite(out).all() and fallback_to_cpu and dev.type != "cpu":
        mae.to(original_device)
        return reconstruct_with_feature_mask(
            mae,
            x,
            feature_mask=feature_mask,
            batch_size=batch_size,
            device=torch.device("cpu"),
            fallback_to_cpu=False,
        )
    mae.to(original_device)
    return out


def per_feature_drop_mse(
    mae: TabularMAE,
    x: np.ndarray,
    *,
    feature_indices: Optional[Iterable[int]] = None,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    fallback_to_cpu: bool = True,
) -> list[FeatureReconstructionResult]:
    if x.ndim != 2:
        raise ValueError("x must be 2D [n, d].")

    n, d = x.shape
    indices = list(range(d)) if feature_indices is None else list(feature_indices)

    results: list[FeatureReconstructionResult] = []
    for j in indices:
        if not (0 <= j < d):
            raise ValueError(f"Invalid feature index: {j}")

        obs = ~np.isnan(x[:, j])
        if int(obs.sum()) == 0:
            results.append(FeatureReconstructionResult(feature_index=j, mse=float("nan"), n=0))
            continue

        mask = np.zeros((n, d), dtype=bool)
        mask[obs, j] = True
        pred = reconstruct_with_feature_mask(
            mae,
            x,
            feature_mask=mask,
            batch_size=batch_size,
            device=device,
            fallback_to_cpu=fallback_to_cpu,
        )
        err = (pred[obs, j] - x[obs, j]) ** 2
        results.append(
            FeatureReconstructionResult(
                feature_index=j,
                mse=float(np.mean(err)),
                n=int(obs.sum()),
            )
        )
    return results


def feature_true_vs_pred(
    mae: TabularMAE,
    x: np.ndarray,
    *,
    feature_index: int,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be 2D [n, d].")
    n, d = x.shape
    if not (0 <= feature_index < d):
        raise ValueError("feature_index out of range.")

    obs = ~np.isnan(x[:, feature_index])
    mask = np.zeros((n, d), dtype=bool)
    mask[obs, feature_index] = True
    pred = reconstruct_with_feature_mask(
        mae,
        x,
        feature_mask=mask,
        batch_size=batch_size,
        device=device,
    )
    return x[obs, feature_index].astype(np.float32), pred[obs, feature_index].astype(
        np.float32
    )
