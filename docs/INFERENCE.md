# Inference Guide

Two supported inference paths: batch prediction for all brines, or prediction for custom samples.

## 1. Predict for All Processed Brines

```bash
python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions
```

This uses checkpoints from `models/` and the scaler from `data/processed/`. Outputs:

- `data/predictions/brines_imputed.csv` — MAE-imputed brine chemistry (missing ions filled)
- `data/predictions/brines_with_predictions.csv` — all brines with predicted targets

### How it works (v0.3.0)

1. Load MAE encoder + FiLM regression head + scaler
2. For each brine: feed 9 chemistry features through MAE encoder (missing features auto-imputed)
3. Apply FiLM conditioning with Light_kW_m2 (geographic GHI or provided value)
4. Regression head outputs 3 predictions (Selectivity, Li crystallization, Evaporation)
5. De-standardize using saved scaler statistics
6. Clamp predictions to non-negative (physical constraint)

## 2. Predict for Custom Samples

Input CSV must contain these columns:

| Column | Description | Unit |
|--------|-------------|------|
| `TDS_gL` | Total dissolved solids | g/L |
| `MLR` | Mg/Li mass ratio | dimensionless |
| `Light_kW_m2` | Solar irradiance | kW/m2 |

Example input: `examples/experimental_samples.csv`

```bash
python src/models/predict_samples.py \
  --input examples/experimental_samples.csv \
  --out examples/experimental_samples_with_predictions.csv
```

### Output Columns

The output CSV appends three prediction columns:

| Column | Description | Unit |
|--------|-------------|------|
| `Selectivity` | Li+/Mg2+ selectivity | dimensionless |
| `Li_Crystallization_mg_m2_h` | Li+ crystallization rate | mg m-2 h-1 |
| `Evap_kg_m2_h` | Evaporation rate | kg m-2 h-1 |

## Programmatic Usage

```python
from src.models.inference import load_artifacts, predict_labels

artifacts = load_artifacts(
    mae_path="models/mae_pretrained.pth",
    head_path="models/downstream_head.pth",
    scaler_path="data/processed/feature_scaler.joblib",
)

samples = [
    {"TDS_gL": 300.0, "MLR": 20.0, "Light_kW_m2": 1.0},
    {"TDS_gL": 150.0, "MLR": 10.0, "Light_kW_m2": 0.5},
]

predictions = predict_labels(artifacts, samples=samples)
# predictions shape: (2, 3) — [Selectivity, Li_Crystallization, Evap]
```
