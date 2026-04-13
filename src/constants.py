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


NATURE_STYLE: str = str(project_root() / "scripts" / "py" / "viz" / "nature.mplstyle")

ION_COLUMNS: tuple[str, ...] = (
    "Li_gL",
    "Mg_gL",
    "Na_gL",
    "K_gL",
    "Ca_gL",
    "SO4_gL",
    "Cl_gL",
)

TDS_MAX_GL: float = 450.0
TDS_UNIT_RATIO_THRESHOLD: float = 50.0

BRINE_CHEMISTRY_COLUMNS: tuple[str, ...] = (
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

# v0.3.0: Light_kW_m2 removed from brine features — it enters via FiLM
# conditioning in the regression head, not the MAE encoder.
BRINE_FEATURE_COLUMNS: tuple[str, ...] = BRINE_CHEMISTRY_COLUMNS

LIGHT_COLUMN: str = "Light_kW_m2"

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
    required_columns=BRINE_CHEMISTRY_COLUMNS,
)

EXPERIMENTAL_DATASET = DatasetSpec(
    filename="experimental.csv",
    required_columns=EXPERIMENTAL_FEATURE_COLUMNS + EXPERIMENTAL_TARGET_COLUMNS,
)

# ---------------------------------------------------------------------------
# Display labels for publication figures (Nature style)
# ---------------------------------------------------------------------------
# LaTeX-formatted labels for matplotlib (use raw strings).
DISPLAY_LABELS: dict[str, str] = {
    # Brine chemistry features
    "Li_gL": r"Li$^+$ (g L$^{-1}$)",
    "Mg_gL": r"Mg$^{2+}$ (g L$^{-1}$)",
    "Na_gL": r"Na$^+$ (g L$^{-1}$)",
    "K_gL": r"K$^+$ (g L$^{-1}$)",
    "Ca_gL": r"Ca$^{2+}$ (g L$^{-1}$)",
    "SO4_gL": r"SO$_4^{2-}$ (g L$^{-1}$)",
    "Cl_gL": r"Cl$^-$ (g L$^{-1}$)",
    "MLR": r"Mg$^{2+}$/Li$^+$ ratio",
    "TDS_gL": r"TDS (g L$^{-1}$)",
    # Experimental features
    "Light_kW_m2": r"Solar irradiance (kW m$^{-2}$)",
    # Targets
    "Selectivity": r"Li$^+$/Mg$^{2+}$ selectivity",
    "Li_Crystallization_mg_m2_h": r"Li$^+$ flux (mg m$^{-2}$ h$^{-1}$)",
    "Evap_kg_m2_h": r"Evaporation rate (kg m$^{-2}$ h$^{-1}$)",
    # Predicted targets
    "Pred_Selectivity": r"Li$^+$/Mg$^{2+}$ selectivity",
    "Pred_Li_Crystallization_mg_m2_h": r"Li$^+$ flux (mg m$^{-2}$ h$^{-1}$)",
    "Pred_Evap_kg_m2_h": r"Evaporation rate (kg m$^{-2}$ h$^{-1}$)",
}


def display_label(column: str) -> str:
    """Return the publication-ready display label for a column name."""
    return DISPLAY_LABELS.get(column, column)
