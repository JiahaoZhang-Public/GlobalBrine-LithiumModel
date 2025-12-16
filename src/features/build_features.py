from __future__ import annotations

import csv
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import click

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'numpy'. Install it (e.g. `pip install numpy`)."
    ) from exc

try:
    from src.constants import (
        BRINE_FEATURE_COLUMNS,
        BRINES_DATASET,
        EXPERIMENTAL_DATASET,
        EXPERIMENTAL_FEATURE_COLUMNS,
        EXPERIMENTAL_TARGET_COLUMNS,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import (
        BRINE_FEATURE_COLUMNS,
        BRINES_DATASET,
        EXPERIMENTAL_DATASET,
        EXPERIMENTAL_FEATURE_COLUMNS,
        EXPERIMENTAL_TARGET_COLUMNS,
    )


class FeatureBuildError(RuntimeError):
    pass


def _fmt_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value: .4f}"


def _print_matrix_stats(title: str, matrix: np.ndarray, columns: Iterable[str]) -> None:
    cols = list(columns)
    if matrix.ndim != 2:
        raise FeatureBuildError("Expected a 2D matrix for stats printing.")
    if matrix.shape[1] != len(cols):
        raise FeatureBuildError("Column count mismatch for stats printing.")

    n_rows = matrix.shape[0]
    click.echo(f"\n[{title}] shape={matrix.shape}")
    click.echo(
        "  {:<24} {:>6} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
            "feature", "n", "missing", "missing%", "mean", "std", "min/max"
        )
    )
    for j, name in enumerate(cols):
        col = matrix[:, j].astype(np.float64, copy=False)
        missing = int(np.isnan(col).sum())
        missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0
        mean = float(np.nanmean(col)) if missing != n_rows else float("nan")
        std = float(np.nanstd(col)) if missing != n_rows else float("nan")
        cmin = float(np.nanmin(col)) if missing != n_rows else float("nan")
        cmax = float(np.nanmax(col)) if missing != n_rows else float("nan")
        click.echo(
            "  {:<24} {:>6} {:>9} {:>9.2f}% {:>10} {:>10} {:>5}/{:<5}".format(
                name[:24],
                n_rows,
                missing,
                missing_pct,
                _fmt_float(mean),
                _fmt_float(std),
                _fmt_float(cmin).strip(),
                _fmt_float(cmax).strip(),
            )
        )


def _parse_float(value: str, *, path: Path, row_index: int, column: str) -> float:
    stripped = value.strip()
    if stripped == "":
        return float("nan")
    try:
        return float(stripped)
    except ValueError as exc:
        raise FeatureBuildError(
            f"Non-numeric value in {path.name} row {row_index} column "
            f"'{column}': {value!r}"
        ) from exc


def load_csv_matrix(path: Path, columns: Iterable[str]) -> np.ndarray:
    column_list = list(columns)
    if not path.exists():
        raise FeatureBuildError(f"Missing CSV file: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FeatureBuildError(f"CSV has no header row: {path}")
        missing = [c for c in column_list if c not in set(reader.fieldnames)]
        if missing:
            raise FeatureBuildError(
                f"{path.name} missing required columns: {', '.join(missing)}"
            )

        rows = list(reader)

    matrix = np.zeros((len(rows), len(column_list)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(column_list):
            matrix[i, j] = _parse_float(
                row.get(col, ""),
                path=path,
                row_index=i + 1,
                column=col,
            )
    return matrix


@dataclass(frozen=True)
class StandardScalerStats:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def fit_standard_scaler(
    matrix: np.ndarray,
    feature_names: Iterable[str],
) -> StandardScalerStats:
    if matrix.ndim != 2:
        raise FeatureBuildError("Expected a 2D matrix to fit scaler.")

    mean = np.nanmean(matrix, axis=0).astype(np.float64)
    std = np.nanstd(matrix, axis=0).astype(np.float64)
    std[std == 0] = 1.0
    return StandardScalerStats(
        feature_names=tuple(feature_names),
        mean=tuple(mean.tolist()),
        std=tuple(std.tolist()),
    )


def transform_standard(
    matrix: np.ndarray,
    feature_names: Iterable[str],
    stats_by_name: dict[str, StandardScalerStats],
    *,
    scaler_key: str,
) -> np.ndarray:
    if scaler_key not in stats_by_name:
        raise FeatureBuildError(f"Missing scaler stats: {scaler_key}")

    stats = stats_by_name[scaler_key]
    feature_list = list(feature_names)
    if tuple(feature_list) != stats.feature_names:
        raise FeatureBuildError(
            f"Feature order mismatch for scaler '{scaler_key}'. "
            f"Expected {stats.feature_names}, got {tuple(feature_list)}."
        )

    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return (matrix - mean) / std


def _stats_map(stats: StandardScalerStats) -> dict[str, tuple[float, float]]:
    return dict(zip(stats.feature_names, zip(stats.mean, stats.std)))


def transform_by_feature(
    matrix: np.ndarray,
    feature_names: Iterable[str],
    feature_stats: dict[str, tuple[float, float]],
    *,
    preserve_nan: bool,
) -> np.ndarray:
    feature_list = list(feature_names)
    mean = np.zeros((len(feature_list),), dtype=np.float32)
    std = np.zeros((len(feature_list),), dtype=np.float32)

    for idx, name in enumerate(feature_list):
        if name not in feature_stats:
            raise FeatureBuildError(f"Missing scaler stats for feature: {name}")
        mean[idx] = float(feature_stats[name][0])
        std[idx] = float(feature_stats[name][1])

    std[std == 0] = 1.0
    scaled = (matrix - mean) / std
    if preserve_nan:
        scaled[np.isnan(matrix)] = np.nan
    return scaled.astype(np.float32)


def _drop_rows_with_any_nan(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise FeatureBuildError("Expected a 2D matrix to drop rows.")
    keep = ~np.isnan(matrix).any(axis=1)
    return matrix[keep], keep


def save_scaler_artifact(path: Path, artifact: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib  # type: ignore
    except ModuleNotFoundError:
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)
    else:
        joblib.dump(artifact, path)


def build_and_save_features(
    processed_dir: Path,
    *,
    missing_features: str = "preserve",
    missing_targets: str = "drop",
    print_stats: bool = False,
) -> dict[str, Path]:
    brines_path = processed_dir / BRINES_DATASET.filename
    experimental_path = processed_dir / EXPERIMENTAL_DATASET.filename

    x_lake = load_csv_matrix(brines_path, BRINE_FEATURE_COLUMNS)
    x_exp = load_csv_matrix(experimental_path, EXPERIMENTAL_FEATURE_COLUMNS)
    y_exp = load_csv_matrix(experimental_path, EXPERIMENTAL_TARGET_COLUMNS)

    if print_stats:
        _print_matrix_stats("X_lake (raw)", x_lake, BRINE_FEATURE_COLUMNS)
        _print_matrix_stats("X_exp (raw)", x_exp, EXPERIMENTAL_FEATURE_COLUMNS)
        _print_matrix_stats("y_exp (raw)", y_exp, EXPERIMENTAL_TARGET_COLUMNS)

    if missing_targets not in {"drop", "error"}:
        raise FeatureBuildError("missing_targets must be 'drop' or 'error'")
    if np.isnan(y_exp).any():
        if missing_targets == "error":
            raise FeatureBuildError(
                "Experimental targets contain missing values; "
                "use --missing-targets drop to drop those rows."
            )
        valid = ~np.isnan(y_exp).any(axis=1)
        x_exp = x_exp[valid]
        y_exp = y_exp[valid]

    stats_by_name: dict[str, StandardScalerStats] = {}
    stats_by_name["brine_chemistry"] = fit_standard_scaler(
        x_lake,
        BRINE_FEATURE_COLUMNS,
    )

    brine_feature_stats = _stats_map(stats_by_name["brine_chemistry"])
    exp_light = x_exp[:, list(EXPERIMENTAL_FEATURE_COLUMNS).index("Light_kW_m2")]
    light_mean = float(np.nanmean(exp_light, dtype=np.float64))
    light_std = float(np.nanstd(exp_light, dtype=np.float64)) or 1.0

    exp_feature_stats = dict(brine_feature_stats)
    exp_feature_stats["Light_kW_m2"] = (light_mean, light_std)
    stats_by_name["experimental_features"] = StandardScalerStats(
        feature_names=EXPERIMENTAL_FEATURE_COLUMNS,
        mean=tuple(exp_feature_stats[n][0] for n in EXPERIMENTAL_FEATURE_COLUMNS),
        std=tuple(exp_feature_stats[n][1] for n in EXPERIMENTAL_FEATURE_COLUMNS),
    )

    stats_by_name["experimental_targets"] = fit_standard_scaler(
        y_exp,
        EXPERIMENTAL_TARGET_COLUMNS,
    )

    if missing_features not in {"preserve", "mean", "zero", "drop", "error"}:
        raise FeatureBuildError(
            "missing_features must be one of: preserve, mean, zero, drop, error"
        )

    if missing_features == "error":
        if np.isnan(x_lake).any() or np.isnan(x_exp).any():
            raise FeatureBuildError(
                "Found missing feature values; "
                "use --missing-features preserve|mean|zero|drop to handle them."
            )

    if missing_features == "drop":
        x_lake, _ = _drop_rows_with_any_nan(x_lake)
        x_exp, _ = _drop_rows_with_any_nan(x_exp)

    preserve_nan = missing_features == "preserve"

    if missing_features == "mean":
        x_lake = np.where(
            np.isnan(x_lake),
            np.asarray(stats_by_name["brine_chemistry"].mean, dtype=np.float32),
            x_lake,
        )
        x_exp = np.where(
            np.isnan(x_exp),
            np.asarray(stats_by_name["experimental_features"].mean, dtype=np.float32),
            x_exp,
        )
        preserve_nan = False

    if missing_features == "zero":
        x_lake = np.nan_to_num(x_lake, nan=0.0)
        x_exp = np.nan_to_num(x_exp, nan=0.0)
        preserve_nan = False

    x_lake_scaled = transform_by_feature(
        x_lake,
        BRINE_FEATURE_COLUMNS,
        brine_feature_stats,
        preserve_nan=preserve_nan,
    )

    x_exp_scaled = transform_by_feature(
        x_exp,
        EXPERIMENTAL_FEATURE_COLUMNS,
        exp_feature_stats,
        preserve_nan=preserve_nan,
    )

    y_stats = stats_by_name["experimental_targets"]
    y_mean = np.asarray(y_stats.mean, dtype=np.float32)
    y_std = np.asarray(y_stats.std, dtype=np.float32)
    y_exp_scaled = ((y_exp - y_mean) / y_std).astype(np.float32)

    if print_stats:
        _print_matrix_stats("X_lake (normalized)", x_lake_scaled, BRINE_FEATURE_COLUMNS)
        _print_matrix_stats(
            "X_exp (normalized)", x_exp_scaled, EXPERIMENTAL_FEATURE_COLUMNS
        )
        _print_matrix_stats(
            "y_exp (normalized)", y_exp_scaled, EXPERIMENTAL_TARGET_COLUMNS
        )

    x_lake_out = processed_dir / "X_lake.npy"
    x_exp_out = processed_dir / "X_exp.npy"
    y_exp_out = processed_dir / "y_exp.npy"
    scaler_out = processed_dir / "feature_scaler.joblib"

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_lake_out, x_lake_scaled.astype(np.float32))
    np.save(x_exp_out, x_exp_scaled.astype(np.float32))
    np.save(y_exp_out, y_exp_scaled.astype(np.float32))
    save_scaler_artifact(
        scaler_out,
        {
            "brine_chemistry": stats_by_name["brine_chemistry"].as_dict(),
            "experimental_features": stats_by_name["experimental_features"].as_dict(),
            "experimental_targets": stats_by_name["experimental_targets"].as_dict(),
        },
    )

    return {
        "X_lake": x_lake_out,
        "X_exp": x_exp_out,
        "y_exp": y_exp_out,
        "feature_scaler": scaler_out,
    }


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--missing-features",
    type=click.Choice(
        ["preserve", "mean", "zero", "drop", "error"], case_sensitive=False
    ),
    default="preserve",
    show_default=True,
    help="How to handle missing feature values in CSVs (blank cells -> NaN).",
)
@click.option(
    "--missing-targets",
    type=click.Choice(["drop", "error"], case_sensitive=False),
    default="drop",
    show_default=True,
    help="How to handle missing target values in experimental.csv.",
)
@click.option(
    "--stats/--no-stats",
    default=False,
    show_default=True,
    help="Print per-column stats before/after normalization.",
)
def main(
    processed_dir: str, missing_features: str, missing_targets: str, stats: bool
) -> None:
    """Build model-ready arrays and scaler from processed CSVs.

    Expects `brines.csv` and `experimental.csv` to exist in PROCESSED_DIR and
    writes normalized arrays `X_lake.npy`, `X_exp.npy`, `y_exp.npy` plus
    `feature_scaler.joblib` (input + target scaler stats) alongside them.
    """
    out = build_and_save_features(
        Path(processed_dir),
        missing_features=missing_features.lower(),
        missing_targets=missing_targets.lower(),
        print_stats=stats,
    )
    click.echo(f"Wrote: {out['X_lake']}")
    click.echo(f"Wrote: {out['X_exp']}")
    click.echo(f"Wrote: {out['y_exp']}")
    click.echo(f"Wrote: {out['feature_scaler']}")


if __name__ == "__main__":
    main()
