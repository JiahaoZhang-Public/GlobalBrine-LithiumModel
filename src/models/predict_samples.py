from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

try:
    from src.constants import EXPERIMENTAL_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
    from src.data.datasets import read_csv_rows, write_csv_rows
    from src.models.inference import (
        auto_device,
        load_artifacts,
        predict_labels,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import EXPERIMENTAL_FEATURE_COLUMNS, EXPERIMENTAL_TARGET_COLUMNS
    from src.data.datasets import read_csv_rows, write_csv_rows
    from src.models.inference import (
        auto_device,
        load_artifacts,
        predict_labels,
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


def _torch_device(device: str) -> torch.device:
    return auto_device() if device == "auto" else torch.device(device)


def predict_samples_csv(
    *,
    input_csv: Path,
    out_csv: Path,
    mae_path: Path,
    head_path: Path,
    scaler_path: Path,
    device: str,
    impute_missing_chemistry: bool,
) -> Path:
    rows, header = read_csv_rows(input_csv)
    if not rows:
        raise PredictionError(f"No rows found in: {input_csv}")

    missing_cols = [c for c in EXPERIMENTAL_FEATURE_COLUMNS if c not in set(header)]
    if missing_cols:
        raise PredictionError(
            f"{input_csv.name} missing required columns: {', '.join(missing_cols)}"
        )

    artifacts = load_artifacts(
        mae_path=mae_path,
        head_path=head_path,
        scaler_path=scaler_path,
        device=_torch_device(device),
    )

    samples: list[dict[str, float | int | None]] = []
    for row in rows:
        sample: dict[str, float | int | None] = {}
        for col in EXPERIMENTAL_FEATURE_COLUMNS:
            value = _parse_float(row.get(col, ""))
            sample[col] = None if np.isnan(value) else float(value)
        samples.append(sample)

    y_pred = predict_labels(
        artifacts,
        samples=samples,
        impute_missing_chemistry=impute_missing_chemistry,
    )

    pred_cols = [f"Pred_{c}" for c in EXPERIMENTAL_TARGET_COLUMNS]
    out_header = list(header)
    for col in pred_cols:
        if col not in out_header:
            out_header.append(col)

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out_row = dict(row)
        for j, col in enumerate(pred_cols):
            out_row[col] = _fmt_float(float(y_pred[i, j]))
        out_rows.append(out_row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_csv, out_rows, fieldnames=out_header)
    return out_csv


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False),
    default="examples/experimental_samples.csv",
    show_default=True,
    help="Input CSV containing experimental features.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default="examples/experimental_samples_with_predictions.csv",
    show_default=True,
    help="Output CSV path.",
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
@click.option(
    "--impute-missing-chemistry/--no-impute-missing-chemistry",
    default=False,
    show_default=True,
    help="Use MAE to impute missing chemistry before encoding.",
)
def main(
    input_path: str,
    out_path: str,
    mae_path: str,
    head_path: str,
    scaler_path: str,
    device: str,
    impute_missing_chemistry: bool,
) -> None:
    """Predict targets for a CSV of samples.

    Required input columns are `TDS_gL`, `MLR`, and `Light_kW_m2`. Predictions are
    appended as `Pred_<target>` columns.
    """
    try:
        out = predict_samples_csv(
            input_csv=Path(input_path),
            out_csv=Path(out_path),
            mae_path=Path(mae_path),
            head_path=Path(head_path),
            scaler_path=Path(scaler_path),
            device=device,
            impute_missing_chemistry=impute_missing_chemistry,
        )
    except PredictionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote: {out}")


if __name__ == "__main__":
    main()
