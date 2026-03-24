# GlobalBrine-LithiumModel

## Project Overview
Global ML model for predicting lithium extraction metrics (Li⁺/Mg²⁺ selectivity, Li⁺ crystallization rate, evaporation rate) from brine chemistry and solar irradiance. Uses a two-stage approach: MAE pretraining on ~1000 unlabeled brine samples, then supervised fine-tuning on ~10 labeled experiments.

## Architecture
- **Masked Autoencoder (MAE)**: Learns latent brine chemistry representations from unlabeled data (`src/models/mae.py`, `src/models/train_mae.py`)
- **Regression Head**: Small MLP fine-tuned on labeled experimental data (`src/models/regression_head.py`, `src/models/finetune_regression.py`)
- **Inference**: Combines encoder + head for predictions (`src/models/inference.py`, `src/models/predict_brines.py`, `src/models/predict_samples.py`)
- **Web UI**: FastAPI backend (`web/backend/`) + React/Tailwind frontend (`web/app/`)

## Key Directories
```
src/
  constants.py          # Canonical column lists, dataset specs, path helpers
  data/                 # Data loading: make_dataset.py, datasets.py, location.py, light.py
  features/             # Feature engineering: build_features.py, scaler.py
  models/               # MAE training, fine-tuning, inference, W&B sweeps
  interpretability/     # Integrated gradients, attributions
  visualization/        # Latent space viz, prediction plots
data/
  raw/                  # Immutable input CSVs (brines.csv, experimental.csv)
  processed/            # Generated: validated CSVs, .npy arrays, scaler .joblib
  predictions/          # Inference outputs
web/
  backend/              # FastAPI: main.py, predict.py, data_access.py, schemas.py
  app/                  # React + Vite + Tailwind frontend
tests/                  # pytest test suite
models/                 # Saved model checkpoints (.pth)
```

## Pipeline Commands
```bash
# 1. Process raw data
python src/data/make_dataset.py data/raw data/processed

# 2. Build features (normalized arrays + scaler)
python src/features/build_features.py data/processed

# 3. Pretrain MAE
python src/models/train_mae.py data/processed --out models/mae_pretrained.pth

# 4. Fine-tune regression head
python src/models/finetune_regression.py data/processed --mae models/mae_pretrained.pth --out models/downstream_head.pth

# 5. Run predictions
python src/models/predict_brines.py --processed-dir data/processed --out-dir data/predictions
```

## Feature Columns (from src/constants.py)
- **Brine chemistry**: Li_gL, Mg_gL, Na_gL, K_gL, Ca_gL, SO4_gL, Cl_gL, MLR, TDS_gL
- **Brine features** (MAE input): above + Light_kW_m2
- **Experimental features**: TDS_gL, MLR, Light_kW_m2
- **Targets**: Selectivity, Li_Crystallization_mg_m2_h, Evap_kg_m2_h

## Testing
```bash
pytest tests/                    # Run full test suite
pytest tests/test_mae.py         # Individual test file
```

## Web Platform
```bash
# Backend
uvicorn web.backend.main:app --reload

# Frontend
cd web/app && npm install && npm run dev
```
Production: https://globalbrine-web-gwvzz.ondigitalocean.app/

## Environment
- Python >= 3.10
- Setup: `make create_environment && conda activate global-brine-lithium-model && make requirements`
- Or: `python3 -m venv .venv && pip install -r requirements.txt`

## Conventions
- Cookiecutter data science project structure
- W&B logging (offline by default) for training scripts
- NaN values in features are intentional — MAE treats them as masked inputs
- Feature scaling fits on brine data, then applied to experimental data
- `src/constants.py` is the single source of truth for column names and dataset specs
