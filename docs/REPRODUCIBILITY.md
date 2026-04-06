# Reproducibility Guide

Full pipeline for reproducing results from raw data to predictions.

## 0. Environment

```bash
# Conda (recommended)
make create_environment
conda activate global-brine-lithium-model
make requirements

# Or venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel && pip install -r requirements.txt
```

Sanity check:

```bash
make test_environment
pytest -q
```

## 1. Prepare Datasets (raw -> processed)

```bash
python src/data/make_dataset.py data/raw data/processed
```

### (Optional) Populate Light_kW_m2 from Global Solar Atlas

Download GHI GeoTIFF from [Global Solar Atlas](https://globalsolaratlas.info/download/world) and run:

```bash
python src/data/make_dataset.py \
  --light-geotiff path/to/GHI_*.tif \
  --light-scale 0.0416666667 \
  data/raw data/processed
```

The scale factor `0.0416666667 = 1/24` converts kWh/m2/day to average kW/m2.

## 2. Build Features + Scaler

```bash
python src/features/build_features.py data/processed --stats
```

Outputs in `data/processed/`:

| File | Description |
|------|-------------|
| `X_lake.npy` | Brine features, shape `[n_brines, 9]`, may contain NaN |
| `X_exp.npy` | Experimental features, shape `[n_experiments, 3]` |
| `y_exp.npy` | Standardized targets, shape `[n_experiments, 3]` |
| `feature_scaler.joblib` | Scaling statistics for de-normalization |

**Scaling strategy (v0.3.0)**: Brine chemistry (9 features) fitted on brine data. For experimental features, TDS/MLR reuse brine statistics while Light_kW_m2 uses experimental statistics (since lab irradiance and geographic GHI have different distributions).

## 3. Train Models

### Quick smoke test

```bash
scripts/sh/test/smoke_train.sh
```

### Full training

```bash
# Stage 1: MAE pretraining on 9 brine chemistry features
python src/models/train_mae.py data/processed --out models/mae_pretrained.pth

# Stage 2: Fine-tune regression head with FiLM conditioning on Light
python src/models/finetune_regression.py data/processed \
  --mae models/mae_pretrained.pth --out models/downstream_head.pth
```

The regression head uses **FiLM (Feature-wise Linear Modulation)** to condition the MAE latent representation on Light_kW_m2. The MAE encoder is frozen during fine-tuning.

Shell script wrappers with recommended hyperparameters:

```bash
scripts/sh/run/train_mae.sh data/processed
scripts/sh/run/finetune_regression.sh data/processed
```

## 4. Run Inference

```bash
python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions \
  --mae models/mae_pretrained.pth \
  --head models/downstream_head.pth \
  --scaler data/processed/feature_scaler.joblib
```

Outputs:
- `data/predictions/brines_imputed.csv` — MAE-imputed brine chemistry
- `data/predictions/brines_with_predictions.csv` — predictions for all brines

## 5. Figures (optional)

Figure scripts require extra dependencies (`cartopy`, etc.):

```bash
mamba install -c conda-forge cartopy
```

```bash
python scripts/py/viz/figure1_map.py
python scripts/py/viz/figure2_map.py --no-ghi
```

## Notes on Determinism

- Training sets `torch.manual_seed(...)` and `numpy.random.seed(...)`.
- Exact bitwise reproducibility across hardware is not guaranteed (CUDA kernels, BLAS, driver versions).
- Pre-trained checkpoints in `models/` provide exact reproducibility for inference.
