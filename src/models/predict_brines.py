from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

try:
    from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
    from src.data.datasets import read_csv_rows, write_csv_rows
    from src.features.scaler import get_stats, load_scaler
    from src.models.inference import (
        auto_device,
        load_head,
        load_mae,
        mae_impute_brine_features,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
    from src.data.datasets import read_csv_rows, write_csv_rows
    from src.features.scaler import get_stats, load_scaler
    from src.models.inference import (
        auto_device,
        load_head,
        load_mae,
        mae_impute_brine_features,
    )


class PredictionError(RuntimeError):
    pass


def _parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover
        raise PredictionError(f"Non-numeric value: {value!r}") from exc


def _fmt_float(value: float) -> str:
    if value is None or isinstance(value, str):  # pragma: no cover
        return ""
    if math.isnan(float(value)):
        return ""
    if math.isinf(float(value)):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.6g}"


def _head_in_dim(head) -> int:
    for module in getattr(head, "net", []):
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    for module in head.modules():  # pragma: no cover
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    raise PredictionError("Unable to infer regression head input dim.")


@dataclass(frozen=True)
class PredictOutputs:
    imputed_csv: Path
    predictions_csv: Path


def predict_brines(
    *,
    processed_dir: Path,
    out_dir: Path,
    mae_path: Path,
    head_path: Path,
    scaler_path: Path,
    device: str = "auto",
) -> PredictOutputs:
    brines_csv = processed_dir / "brines.csv"
    if not brines_csv.exists():
        raise FileNotFoundError(brines_csv)

    scaler: dict[str, Any] = load_scaler(scaler_path)

    torch_device = auto_device() if device == "auto" else torch.device(device)
    mae = load_mae(mae_path, device=torch_device)
    head = load_head(
        head_path,
        out_dim=len(EXPERIMENTAL_TARGET_COLUMNS),
        device=torch_device,
    )

    rows, header = read_csv_rows(brines_csv)
    missing_cols = [c for c in BRINE_FEATURE_COLUMNS if c not in set(header)]
    if missing_cols:
        raise PredictionError(
            f"{brines_csv.name} missing required columns: {', '.join(missing_cols)}"
        )

    x_raw = np.zeros((len(rows), len(BRINE_FEATURE_COLUMNS)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(BRINE_FEATURE_COLUMNS):
            x_raw[i, j] = _parse_float(row.get(col, ""))

    imputed_raw, imputed_std = mae_impute_brine_features(
        mae,
        brine_raw=x_raw,
        scaler=scaler,
        preserve_observed=True,
        device=torch_device,
    )

    # Write imputed brines CSV (same columns as processed brines.csv).
    imputed_rows: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        out_row = dict(row)
        for j, col in enumerate(BRINE_FEATURE_COLUMNS):
            out_row[col] = _fmt_float(float(imputed_raw[i, j]))
        imputed_rows.append(out_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    imputed_csv = out_dir / "brines_imputed.csv"
    write_csv_rows(imputed_csv, imputed_rows, fieldnames=header)

    # Predict targets from MAE-encoded (imputed) brine chemistry.
    y_mean, y_std = get_stats(
        scaler, key="experimental_targets", feature_names=EXPERIMENTAL_TARGET_COLUMNS
    )

    dev = torch_device or next(mae.parameters()).device
    z = mae.encode(torch.from_numpy(imputed_std).to(device=dev, dtype=torch.float32))
    head_dim = _head_in_dim(head)

    if head_dim == int(z.shape[1]):
        head_in = z
    elif head_dim == int(z.shape[1]) + 1:
        light_idx = list(BRINE_FEATURE_COLUMNS).index("Light_kW_m2")
        light = torch.from_numpy(imputed_std[:, light_idx : light_idx + 1]).to(
            device=dev, dtype=torch.float32
        )
        head_in = torch.cat([z, light], dim=1)
    else:
        raise PredictionError(
            f"Unexpected regression head input dim={head_dim}; "
            f"expected {int(z.shape[1])} or {int(z.shape[1]) + 1}."
        )

    with torch.no_grad():
        y_std_pred = head(head_in).detach().cpu().numpy().astype(np.float32)

    y_pred = (y_std_pred * y_std + y_mean).astype(np.float32)
    y_pred = np.clip(y_pred, 0.0, None)

    pred_cols = [
        "Pred_Selectivity",
        "Pred_Li_Crystallization_mg_m2_h",
        "Pred_Evap_kg_m2_h",
    ]
    pred_header = list(header)
    for col in pred_cols:
        if col not in pred_header:
            pred_header.append(col)

    pred_rows: list[dict[str, str]] = []
    for i, row in enumerate(imputed_rows):
        out_row = dict(row)
        out_row[pred_cols[0]] = _fmt_float(float(y_pred[i, 0]))
        out_row[pred_cols[1]] = _fmt_float(float(y_pred[i, 1]))
        out_row[pred_cols[2]] = _fmt_float(float(y_pred[i, 2]))
        pred_rows.append(out_row)

    predictions_csv = out_dir / "brines_with_predictions.csv"
    write_csv_rows(predictions_csv, pred_rows, fieldnames=pred_header)

    return PredictOutputs(imputed_csv=imputed_csv, predictions_csv=predictions_csv)


@click.command()
@click.option(
    "--processed-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/processed",
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default="data/predictions",
    show_default=True,
)
@click.option(
    "--mae",
    "mae_path",
    type=click.Path(exists=True, dir_okay=False),
    default="models/mae_pretrained.pth",
    show_default=True,
)
@click.option(
    "--head",
    "head_path",
    type=click.Path(exists=True, dir_okay=False),
    default="models/downstream_head.pth",
    show_default=True,
)
@click.option(
    "--scaler",
    "scaler_path",
    type=click.Path(exists=True, dir_okay=False),
    default="data/processed/feature_scaler.joblib",
    show_default=True,
)
@click.option("--device", type=str, default="auto", show_default=True)
def main(
    processed_dir: str,
    out_dir: str,
    mae_path: str,
    head_path: str,
    scaler_path: str,
    device: str,
) -> None:
    out = predict_brines(
        processed_dir=Path(processed_dir),
        out_dir=Path(out_dir),
        mae_path=Path(mae_path),
        head_path=Path(head_path),
        scaler_path=Path(scaler_path),
        device=device,
    )
    click.echo(f"Wrote: {out.imputed_csv}")
    click.echo(f"Wrote: {out.predictions_csv}")


if __name__ == "__main__":
    main()
