# GlobalBrine-LithiumModel

Global ML model for predicting lithium extraction metrics from brine chemistry and solar irradiance.

> **Version 0.3.0** &mdash; [Changelog](CHANGELOG.md) &middot; [Release checklist](RELEASING.md)

## Overview

This project predicts three lithium extraction targets for any brine composition under a given solar irradiance:

| Target | Unit |
|--------|------|
| Li+/Mg2+ Selectivity | dimensionless |
| Li+ Crystallization Rate | mg m-2 h-1 |
| Evaporation Rate | kg m-2 h-1 |

### Model Architecture (v0.3.0)

A two-stage approach addresses the extreme label scarcity (~30 labeled experiments vs ~1000 unlabeled brine samples):

```
Stage 1 — MAE Pretraining (unsupervised)
  ~1000 brine samples (9 chemistry features)
  -> Masked Autoencoder learns latent brine representation z [128-dim]

Stage 2 — Supervised Fine-tuning
  ~30 labeled experiments (TDS, MLR, Light)
  -> MAE encoder (frozen) produces z
  -> FiLM layer modulates z with Light_kW_m2:  z' = gamma(light) * z + beta(light)
  -> Regression head maps z' -> 3 targets
```

**Key design decision**: Light_kW_m2 is *not* part of the MAE encoder input. Lab irradiance (0.5-1.5 kW/m2) and geographic GHI (0.1-0.3 kW/m2) have fundamentally different distributions. FiLM conditioning avoids this mismatch while preserving light's physical influence on predictions.

## Quickstart

### 1. Environment Setup

```bash
# Option A: Conda (recommended)
make create_environment
conda activate global-brine-lithium-model
make requirements

# Option B: venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Prerequisites: Python >= 3.10. Optional: `conda`/`mamba`, `make`.

### 2. Reproduce Predictions (using included checkpoints)

```bash
python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions
```

This uses the pre-trained checkpoints in `models/` and outputs:
- `data/predictions/brines_imputed.csv` — MAE-imputed brine chemistry
- `data/predictions/brines_with_predictions.csv` — predictions for all brines

### 3. Full Pipeline (train from scratch)

```bash
# Step 1: Process raw CSVs
python src/data/make_dataset.py data/raw data/processed

# Step 2: Build feature arrays + scaler
python src/features/build_features.py data/processed

# Step 3: Pretrain MAE on brine chemistry
python src/models/train_mae.py data/processed --out models/mae_pretrained.pth

# Step 4: Fine-tune regression head (with FiLM conditioning)
python src/models/finetune_regression.py data/processed \
  --mae models/mae_pretrained.pth --out models/downstream_head.pth

# Step 5: Generate predictions
python src/models/predict_brines.py \
  --processed-dir data/processed --out-dir data/predictions
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for detailed instructions including GeoTIFF light data setup.

### 4. Predict for Custom Samples

```bash
python src/models/predict_samples.py \
  --input examples/experimental_samples.csv \
  --out examples/experimental_samples_with_predictions.csv
```

Input CSV must contain columns: `TDS_gL`, `MLR`, `Light_kW_m2`. See [docs/INFERENCE.md](docs/INFERENCE.md).

## Project Structure

```
GlobalBrine-LithiumModel/
├── src/
│   ├── constants.py              # Canonical column names, dataset specs
│   ├── data/
│   │   ├── make_dataset.py       # Raw CSV -> processed CSV (+ optional GeoTIFF light)
│   │   ├── datasets.py           # Schema validation helpers
│   │   ├── location.py           # Lat/lon parsing from Location text
│   │   └── light.py              # GeoTIFF irradiance sampling
│   ├── features/
│   │   ├── build_features.py     # Processed CSV -> .npy arrays + scaler
│   │   └── scaler.py             # Feature standardization utilities
│   ├── models/
│   │   ├── mae.py                # Masked Autoencoder architecture
│   │   ├── film.py               # FiLM conditioning layer (v0.3.0)
│   │   ├── regression_head.py    # MLP regression head
│   │   ├── train_mae.py          # MAE pretraining script
│   │   ├── finetune_regression.py# Supervised fine-tuning with FiLM
│   │   ├── inference.py          # Load artifacts + predict
│   │   ├── predict_brines.py     # Batch prediction for all brines
│   │   ├── predict_samples.py    # Prediction for custom CSV input
│   │   └── sweep_mae.py          # W&B hyperparameter sweep
│   └── interpretability/         # Integrated gradients, attributions
├── data/
│   ├── raw/                      # Immutable input CSVs
│   ├── processed/                # Generated arrays, scaler, validated CSVs
│   └── predictions/              # Inference outputs
├── models/                       # Saved checkpoints (.pth)
├── evaluations/                  # LOO-CV evaluation scripts
├── web/                          # FastAPI backend + React frontend
│   ├── backend/                  # API: /api/predict, /api/brines, ...
│   └── app/                      # React + Vite + Tailwind UI
├── notebooks/                    # Jupyter notebooks for exploration
├── docs/                         # Reproducibility & inference guides
├── tests/                        # pytest test suite (12 files)
└── Makefile                      # Pipeline shortcuts
```

## Feature Columns

| Group | Columns | Used By |
|-------|---------|---------|
| Brine chemistry (9) | Li_gL, Mg_gL, Na_gL, K_gL, Ca_gL, SO4_gL, Cl_gL, MLR, TDS_gL | MAE encoder |
| Experimental features (3) | TDS_gL, MLR, Light_kW_m2 | Regression input (TDS/MLR via encoder, Light via FiLM) |
| Targets (3) | Selectivity, Li_Crystallization_mg_m2_h, Evap_kg_m2_h | Regression output |

## Testing

```bash
pytest tests/          # Full test suite
pytest tests/ -q       # Quiet mode
make lint              # black + ruff
```

## Web Platform

```bash
# Backend
uvicorn web.backend.main:app --reload

# Frontend
cd web/app && npm install && npm run dev
```

Production: https://globalbrine-web-gwvzz.ondigitalocean.app/

![Homepage](web/app/public/homepage.png)

## License

MIT
