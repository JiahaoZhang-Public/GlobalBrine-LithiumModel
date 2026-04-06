# Source Code Overview

Two-stage learning pipeline for predicting lithium extraction metrics from brine chemistry and solar irradiance.

## Core Methodology

### Stage 1: MAE Pretraining (Unsupervised)

**Input**: ~1000 brine samples with 9 chemistry features:
```
Li_gL, Mg_gL, Na_gL, K_gL, Ca_gL, SO4_gL, Cl_gL, MLR, TDS_gL
```

- Each feature is a continuous token.
- MAE masks a subset of observed features and learns to reconstruct them.
- Missing values (NaN) are treated as pre-masked — never included in reconstruction loss.

**Masking strategy**:
- `obs_mask`: True where feature has a real value (not NaN)
- `mae_mask`: randomly selected from observed features
- `input_mask = (~obs_mask) | mae_mask`
- `loss_mask = obs_mask` (reconstruct all observed values)
- At least `min_visible=1` features remain unmasked per sample.

**Output**: `mae_pretrained.pth` — a chemistry-aware encoder producing 128-dim latent vectors.

### Stage 2: Supervised Fine-tuning with FiLM Conditioning

**Input**: ~30 labeled experiments with features `TDS_gL`, `MLR`, `Light_kW_m2` and targets `Selectivity`, `Li_Crystallization_mg_m2_h`, `Evap_kg_m2_h`.

**Architecture (v0.3.0)**:
```
Experimental sample
  ├─ Chemistry (TDS, MLR) ──> MAE encoder (frozen) ──> z [128-dim]
  │                            (missing ions treated as masked)
  │
  └─ Light_kW_m2 ──> FiLM layer ──> gamma, beta [128-dim each]
                                       │
                         z' = gamma * z + beta
                                       │
                              Regression Head (MLP)
                                       │
                              3 target predictions
```

**Why FiLM?** Lab irradiance (0.5-1.5 kW/m2) and geographic GHI (0.1-0.3 kW/m2) have fundamentally different distributions. Concatenating a 1-dim light signal to a 128-dim latent is also dimensionally imbalanced. FiLM (Feature-wise Linear Modulation) addresses both issues by learning affine transforms that modulate the entire latent space.

**Identity initialization**: gamma=1, beta=0 at init, so the model starts as if light has no effect and gradually learns the conditioning.

**Output**: `downstream_head.pth` — contains FiLM weights + regression head weights.

### Stage 3: Inference

Using encoder + FiLM head, predict for new brines:
1. Feed chemistry features through MAE encoder (missing ions auto-imputed)
2. Apply FiLM conditioning with light intensity
3. Regression head produces 3 target predictions
4. De-standardize using saved scaler statistics

## Directory Structure

```
src/
├── constants.py                # Canonical column lists, dataset specs, path helpers
├── data/
│   ├── make_dataset.py         # Raw CSV -> processed CSV (schema validation, optional GeoTIFF)
│   ├── datasets.py             # DatasetSpec, column validation, normalization
│   ├── location.py             # Lat/lon parsing from Location text
│   └── light.py                # GeoTIFF irradiance sampling (requires rasterio)
├── features/
│   ├── build_features.py       # Processed CSV -> .npy arrays + feature_scaler.joblib
│   └── scaler.py               # Standardization utilities
├── models/
│   ├── mae.py                  # TabularMAE, TabularMAEConfig, build_effective_masks
│   ├── film.py                 # FiLMLayer, FiLMConfig, FiLMRegressionHead
│   ├── regression_head.py      # RegressionHead, RegressionHeadConfig
│   ├── train_mae.py            # MAE pretraining (CLI entry point)
│   ├── finetune_regression.py  # Supervised fine-tuning with FiLM (CLI entry point)
│   ├── inference.py            # InferenceArtifacts, load_artifacts, predict_labels
│   ├── predict_brines.py       # Batch prediction for all brines (CLI entry point)
│   ├── predict_samples.py      # Custom sample prediction (CLI entry point)
│   ├── sweep_mae.py            # W&B hyperparameter sweep (CLI entry point)
│   └── mae_metrics.py          # MAE reconstruction evaluation utilities
└── interpretability/           # Integrated gradients, feature attributions
```

## Pipeline Commands

```bash
# 1. Raw -> processed CSVs
python src/data/make_dataset.py data/raw data/processed

# 2. Build arrays + scaler
python src/features/build_features.py data/processed

# 3. Pretrain MAE
python src/models/train_mae.py data/processed --out models/mae_pretrained.pth

# 4. Fine-tune regression head (FiLM conditioning on light)
python src/models/finetune_regression.py data/processed \
  --mae models/mae_pretrained.pth --out models/downstream_head.pth

# 5. Predict
python src/models/predict_brines.py --processed-dir data/processed --out-dir data/predictions
```

## Scaling Strategy

- **Brine features**: mean/std computed from the ~1000 brine samples (9 chemistry columns)
- **Experimental features**: TDS and MLR use brine statistics; Light_kW_m2 uses experimental statistics
- **Targets**: standardized using experimental target statistics; de-standardized at inference time
- All statistics saved in `feature_scaler.joblib`
