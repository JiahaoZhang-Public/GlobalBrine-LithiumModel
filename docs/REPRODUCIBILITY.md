# Reproducibility guide

This repository provides a full pipeline for:

1. preparing brine + experimental datasets (`src/data/make_dataset.py`)
2. building model-ready arrays + scalers (`src/features/build_features.py`)
3. MAE pretraining (`src/models/train_mae.py`)
4. regression head fine-tuning (`src/models/finetune_regression.py`)
5. inference on brines (`src/models/predict_brines.py`)

## 0) Environment

Recommended:

```bash
make create_environment
conda activate global-brine-lithium-model
make requirements
```

Sanity check:

```bash
make test_environment
pytest -q
```

## 1) Prepare datasets (raw -> processed)

```bash
python src/data/make_dataset.py data/raw data/processed
```

### (Optional) populate `Light_kW_m2` from Global Solar Atlas

Download GHI GeoTIFF from Global Solar Atlas and run:

```bash
python src/data/make_dataset.py \
  --light-geotiff path/to/GHI_*.tif \
  --light-scale 0.0416666667 \
  data/raw data/processed
```

(`0.0416666667` = 1/24, converting kWh/m²/day to average kW/m².)

## 2) Build features + scaler

```bash
python src/features/build_features.py data/processed --stats
```

Outputs are written into `data/processed/` (`X_lake.npy`, `X_exp.npy`, `y_exp.npy`,
`feature_scaler.joblib`).

## 3) Train models (slow)

Quick smoke test (tiny synthetic data, CPU, ~seconds):

```bash
scripts/sh/test/smoke_train.sh
```

Full run entrypoints (these are long by default; adjust flags as needed):

```bash
scripts/sh/run/train_mae.sh data/processed
scripts/sh/run/finetune_regression.sh data/processed
```

## 4) Run inference (reproduce predictions)

Generate predictions for all processed brines:

```bash
python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions \
  --mae models/mae_pretrained.pth \
  --head models/downstream_head.pth \
  --scaler data/processed/feature_scaler.joblib
```

This writes:

- `data/predictions/brines_imputed.csv`
- `data/predictions/brines_with_predictions.csv`

## 5) Figures (optional)

Figure scripts live in `scripts/py/viz/` and expect
`data/predictions/brines_with_predictions.csv`.

They require extra dependencies (`cartopy` in particular). Prefer conda-forge:

```bash
mamba install -c conda-forge cartopy
```

Then:

```bash
python scripts/py/viz/figure1_map.py
python scripts/py/viz/figure2_map.py --no-ghi
```

## Notes on determinism

- Training sets `torch.manual_seed(...)` and `numpy.random.seed(...)`.
- Exact bitwise reproducibility across hardware/accelerators is not guaranteed
  (CUDA kernels, BLAS, and driver versions can introduce small differences).

