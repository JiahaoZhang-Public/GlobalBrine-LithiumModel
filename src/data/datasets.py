from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class CsvDataset:
    path: Path
    required_columns: tuple[str, ...]


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise DatasetError(f"Missing CSV file: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"CSV has no header row: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, str]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_required_columns(
    available: Iterable[str],
    required: Iterable[str],
    *,
    dataset_name: str,
) -> list[str]:
    available_set = set(available)
    missing = [col for col in required if col not in available_set]
    if missing:
        missing_list = ", ".join(missing)
        raise DatasetError(
            f"{dataset_name} is missing required columns: {missing_list}"
        )
    return list(required)


def ordered_columns(
    header: Iterable[str],
    required: Iterable[str],
    *,
    keep_extra: bool = True,
) -> list[str]:
    header_list = list(header)
    required_list = list(required)

    header_set = set(header_list)
    required_set = set(required_list)

    extras = [col for col in header_list if col not in required_set]
    required_present = [col for col in required_list if col in header_set]
    return required_present + extras if keep_extra else required_present


def normalize_rows_to_columns(
    rows: Iterable[Mapping[str, str]],
    columns: Iterable[str],
) -> list[dict[str, str]]:
    column_list = list(columns)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized_row: dict[str, str] = {}
        for col in column_list:
            value = row.get(col, "")
            normalized_row[col] = "" if value is None else str(value)
        normalized.append(normalized_row)
    return normalized


def process_csv_dataset(
    dataset: CsvDataset,
    *,
    output_path: Path,
    keep_extra_columns: bool = True,
) -> Path:
    rows, header = read_csv_rows(dataset.path)
    validate_required_columns(
        header,
        dataset.required_columns,
        dataset_name=dataset.path.name,
    )
    columns = ordered_columns(
        header,
        dataset.required_columns,
        keep_extra=keep_extra_columns,
    )
    normalized = normalize_rows_to_columns(rows, columns)
    write_csv_rows(output_path, normalized, columns)
    return output_path
