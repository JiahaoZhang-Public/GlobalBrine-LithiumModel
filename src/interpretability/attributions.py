from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn

from src.constants import (
    BRINE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    EXPERIMENTAL_TARGET_COLUMNS,
)
from src.features.scaler import get_stats
from src.interpretability.integrated_gradients import (
    IntegratedGradientsResult,
    integrated_gradients,
)
from src.models.inference import InferenceArtifacts


@dataclass(frozen=True)
class IGAttributions:
    features: tuple[str, ...]
    targets: tuple[str, ...]
    input_values: dict[str, float]
    baseline_values: dict[str, float]
    attributions: dict[str, dict[str, float]]
    deltas: dict[str, float]


def _to_float(value: float | int | None) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _head_in_dim(head: nn.Module) -> int:
    for m in head.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise ValueError("Unable to infer regression head input dim.")


def _get_stats_any(
    scaler: dict,
    *,
    keys: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    for key in keys:
        if key in scaler:
            return get_stats(scaler, key=key, feature_names=feature_names)
    raise KeyError(f"Missing scaler key (tried: {keys})")


def integrated_gradients_for_prediction(
    artifacts: InferenceArtifacts,
    *,
    sample: Mapping[str, float | int | None],
    baseline: Mapping[str, float | int | None] | None = None,
    steps: int = 50,
) -> IGAttributions:
    """Compute IG feature attributions for the regression prediction.

    Explains the effect of experimental input features:
    `TDS_gL`, `MLR`, `Light_kW_m2` -> predicted targets.

    Notes:
    - Attribution is computed on the *unclamped* raw prediction (no >=0 clip),
      to keep gradients well-defined.
    - Missing input fields default to the baseline value (mean by default).
    """
    device = next(artifacts.mae.parameters()).device
    scaler = artifacts.scaler

    br_mean, br_std = _get_stats_any(
        scaler,
        keys=("brine_chemistry", "brine_features"),
        feature_names=BRINE_FEATURE_COLUMNS,
    )
    exp_mean, exp_std = get_stats(
        scaler, key="experimental_features", feature_names=EXPERIMENTAL_FEATURE_COLUMNS
    )
    y_mean, y_std = get_stats(
        scaler, key="experimental_targets", feature_names=EXPERIMENTAL_TARGET_COLUMNS
    )

    feat_names = tuple(EXPERIMENTAL_FEATURE_COLUMNS)
    target_names = tuple(EXPERIMENTAL_TARGET_COLUMNS)

    tds_idx = feat_names.index("TDS_gL")
    mlr_idx = feat_names.index("MLR")
    light_idx = feat_names.index("Light_kW_m2")

    br_cols = list(BRINE_FEATURE_COLUMNS)
    br_tds = br_cols.index("TDS_gL")
    br_mlr = br_cols.index("MLR")
    br_light = br_cols.index("Light_kW_m2")

    default_baseline = {
        "TDS_gL": float(br_mean[br_tds]),
        "MLR": float(br_mean[br_mlr]),
        "Light_kW_m2": float(br_mean[br_light]),
    }
    baseline_dict = dict(default_baseline)
    if baseline is not None:
        for k in feat_names:
            v = _to_float(baseline.get(k, None))
            if v is not None:
                baseline_dict[k] = v

    input_dict: dict[str, float] = {}
    for k in feat_names:
        v = _to_float(sample.get(k, None))
        input_dict[k] = baseline_dict[k] if v is None else v

    x_raw = np.array([input_dict[n] for n in feat_names], dtype=np.float32)
    b_raw = np.array([baseline_dict[n] for n in feat_names], dtype=np.float32)

    br_mean_t = torch.tensor(br_mean, device=device, dtype=torch.float32)
    br_std_t = torch.tensor(br_std, device=device, dtype=torch.float32)
    exp_mean_t = torch.tensor(exp_mean, device=device, dtype=torch.float32)
    exp_std_t = torch.tensor(exp_std, device=device, dtype=torch.float32)
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, device=device, dtype=torch.float32)

    def forward_raw(batch_raw: torch.Tensor) -> torch.Tensor:
        tds = batch_raw[:, tds_idx]
        mlr = batch_raw[:, mlr_idx]
        light = batch_raw[:, light_idx]

        tds_std = (tds - br_mean_t[br_tds]) / br_std_t[br_tds]
        mlr_std = (mlr - br_mean_t[br_mlr]) / br_std_t[br_mlr]
        light_std_br = (light - br_mean_t[br_light]) / br_std_t[br_light]
        light_std_exp = (light - exp_mean_t[light_idx]) / exp_std_t[light_idx]

        chem = torch.full(
            (batch_raw.shape[0], len(BRINE_FEATURE_COLUMNS)),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        chem[:, br_tds] = tds_std
        chem[:, br_mlr] = mlr_std
        chem[:, br_light] = light_std_br

        z = artifacts.mae.encode(chem)
        in_dim = _head_in_dim(artifacts.head)
        if in_dim == int(z.shape[1]):
            y_std_pred = artifacts.head(z)
        elif in_dim == int(z.shape[1]) + 1:
            y_std_pred = artifacts.head(
                torch.cat([z, light_std_exp.unsqueeze(1)], dim=1)
            )
        else:
            raise ValueError(
                f"Unexpected head input dim={in_dim}; expected {int(z.shape[1])} "
                f"or {int(z.shape[1]) + 1}."
            )
        y_raw_pred = y_std_pred * y_std_t + y_mean_t
        return y_raw_pred

    inputs_t = torch.from_numpy(x_raw).to(device=device, dtype=torch.float32)
    baseline_t = torch.from_numpy(b_raw).to(device=device, dtype=torch.float32)

    attrs: dict[str, dict[str, float]] = {}
    deltas: dict[str, float] = {}
    for ti, tname in enumerate(target_names):
        res: IntegratedGradientsResult = integrated_gradients(
            forward_raw,
            inputs=inputs_t,
            baseline=baseline_t,
            target_index=ti,
            steps=steps,
        )
        attrs[tname] = {
            fname: float(res.attributions[i].cpu().item())
            for i, fname in enumerate(feat_names)
        }
        deltas[tname] = float(res.delta.cpu().item())

    return IGAttributions(
        features=feat_names,
        targets=target_names,
        input_values={k: float(v) for k, v in input_dict.items()},
        baseline_values={k: float(v) for k, v in baseline_dict.items()},
        attributions=attrs,
        deltas=deltas,
    )
