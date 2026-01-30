# Inference tutorial

There are two supported inference paths:

1. **Batch prediction for all brines** (uses `data/processed/brines.csv`)
2. **Prediction for your own samples** (a CSV with `TDS_gL`, `MLR`, `Light_kW_m2`)

## 1) Predict for all processed brines

```bash
python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions
```

## 2) Predict for your own samples (CSV)

Input CSV must contain these columns:

- `TDS_gL`
- `MLR`
- `Light_kW_m2`

Example input file: `examples/experimental_samples.csv`

```bash
python src/models/predict_samples.py \
  --input examples/experimental_samples.csv \
  --out examples/experimental_samples_with_predictions.csv
```

