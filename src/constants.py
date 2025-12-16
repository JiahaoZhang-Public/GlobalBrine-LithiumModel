from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def raw_data_dir() -> Path:
    return data_dir() / "raw"


def processed_data_dir() -> Path:
    return data_dir() / "processed"


BRINE_FEATURE_COLUMNS: tuple[str, ...] = (
    "Li_gL",
    "Mg_gL",
    "Na_gL",
    "K_gL",
    "Ca_gL",
    "SO4_gL",
    "Cl_gL",
    "MLR",
    "TDS_gL",
)

EXPERIMENTAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "TDS_gL",
    "MLR",
    "Light_kW_m2",
)

EXPERIMENTAL_TARGET_COLUMNS: tuple[str, ...] = (
    "Selectivity",
    "Li_Crystallization_mg_m2_h",
    "Evap_kg_m2_h",
)


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    required_columns: tuple[str, ...]


BRINES_DATASET = DatasetSpec(
    filename="brines.csv",
    required_columns=BRINE_FEATURE_COLUMNS,
)

EXPERIMENTAL_DATASET = DatasetSpec(
    filename="experimental.csv",
    required_columns=EXPERIMENTAL_FEATURE_COLUMNS + EXPERIMENTAL_TARGET_COLUMNS,
)
