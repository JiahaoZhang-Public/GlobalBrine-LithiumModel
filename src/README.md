# **README — Source Code Overview**

This directory contains all code for data loading, feature building, model pretraining, downstream fine-tuning, and visualization.

The project follows the **cookiecutter-data-science** structure and implements a two-stage learning approach:

---

# **Project Idea**

We aim to develop a **global predictive model** for:

- **Li⁺/Mg²⁺ selectivity**
- **Li⁺ crystallization rate**
- **Evaporation rate**

across **brines from salt lakes, seawater, and other natural water systems**.

## **Challenge**

- Only few(~10) **laboratory samples** contain labels (selectivity, crystallization, evaporation).
- about **1000 brine samples** from literature contain **features but no labels**.

## **Our Solution:**

We use a **Masked Autoencoder (MAE)** to **learn a latent representation of brine chemistry** from the large unlabeled dataset, then **fine-tune** this representation using the limited labeled experimental data.

---

# **Core Methodology**

## **1. MAE Pretraining (Unsupervised Learning)**

Input: data/raw/brines.csv

- Each brine sample has features such as:

```
Li_gL, Mg_gL, Na_gL, K_gL, Ca_gL, SO4_gL, Cl_gL, MLR, TDS_gL
```

- 
- Each feature is treated as a **continuous token**.
- MAE masks a subset of these tokens and learns to **reconstruct the missing chemical values**.
- Missing values (`NaN`) are treated as already-masked inputs and are **never** included in the reconstruction loss (no ground truth).

### **MAE masking + loss (obs_mask vs mae_mask)**

- **obs_mask**: whether a feature is observed (1=has value, 0=missing/NaN)
- **mae_mask**: features randomly masked by MAE for this step (only sampled from observed features)

Model input masking:

```
input_mask = (not obs_mask) OR mae_mask
```

Loss masking (only learn to reconstruct values we deliberately hid):

```
loss_mask = obs_mask AND mae_mask
loss = ((recon - x_true)^2 * loss_mask).sum() / loss_mask.sum()
```

Mask ratio rule (per sample): apply `mask_ratio` on the **observed** feature set, and keep at least `min_visible` observed features unmasked (current default: `min_visible=1`) to avoid over-masking.
- The encoder therefore learns:
    - The **chemical manifold** of natural brines
    - Relationships between ions
    - Structure across different brine types worldwide

Output:

- mae_pretrained.pth — a chemical-aware latent encoder

---

## **2. Downstream Fine-tuning (Supervised Learning)**

Input: data/raw/experimental.csv

- 10 labeled samples containing:

```
TDS_gL, MLR, Light_kW_m2, Selectivity,
Li_Crystallization_mg_m2_h,
Evap_kg_m2_h
```

- The MAE encoder generates a latent vector for each sample.
- A small MLP (“downstream head”) is trained to predict the three outputs.

Output:

- downstream_head.pth

---

## **3. Inference / Prediction**

Using the pretrained encoder + downstream head, we can predict lithium-related performance for:

- New brine samples
- Entire salt lakes
- Global geospatial grids (irradiance + chemistry)

---

# **Directory Overview**

```
src/
├── data/
│   ├── make_dataset.py
│   └── __init__.py
├── features/
│   ├── build_features.py
│   └── __init__.py
├── models/
│   ├── train_model.py
│   ├── predict_model.py
│   └── __init__.py
└── visualization/
    ├── visualize.py
    └── __init__.py
```

---

# **Module Details**

## **src/data/make_dataset.py**

This stage turns **raw CSVs** into **processed CSVs** with consistent schema.

Since CSV files are assumed to be already clean, this module:

- Loads data/raw/brines.csv and data/raw/experimental.csv
- Validates required columns exist (see `src/constants.py`)
- Enforces consistent column ordering (required columns first; extra columns preserved)
- Saves them to `data/processed/` for downstream modules

**No data cleaning is performed.**

This module preserves the reproducibility and modularity of the pipeline.

### **Inputs**

- `data/raw/brines.csv` (unlabeled brine chemistry; must include:
  `Li_gL, Mg_gL, Na_gL, K_gL, Ca_gL, SO4_gL, Cl_gL, MLR, TDS_gL`)
- `data/raw/experimental.csv` (labeled experiments; must include:
  `TDS_gL, MLR, Light_kW_m2, Selectivity, Li_Crystallization_mg_m2_h, Evap_kg_m2_h`)

### **Outputs**

- `data/processed/brines.csv`
- `data/processed/experimental.csv`

### **How to run**

```bash
python src/data/make_dataset.py data/raw data/processed
```

---

## **src/features/**

### **build_features.py**

This stage converts processed CSVs into **normalized model-ready arrays** and a saved scaler.

- Extract feature matrices:
    - X_lake (for MAE pretraining)
    - X_exp (for downstream training)
- Extract experiment labels y_exp (and standardize them for stable training)
- Apply feature scaling:
    - Fit scaling stats on brine-chemistry features from `brines.csv` (`X_lake`)
    - Apply those stats to `X_lake`
    - For `X_exp`, reuse brine stats for shared chemistry features (`TDS_gL`, `MLR`)
      and fit `Light_kW_m2` stats from the experimental dataset
- Missing values:
    - Blank feature cells are kept as `NaN` by default (so the MAE can treat them as missing)
    - Rows with missing targets in `experimental.csv` are dropped by default
- Save arrays into `data/processed/`:
    - X_lake.npy
    - X_exp.npy
    - y_exp.npy
    - feature_scaler.joblib (a serialized dict of scaler statistics)

This isolates **feature engineering** from model code.

### **Outputs**

- `data/processed/X_lake.npy` (float32, standardized, shape `[n_brines, 9]`, may contain `NaN`)
- `data/processed/X_exp.npy` (float32, standardized, shape `[n_experiments, 3]`, may contain `NaN`)
- `data/processed/y_exp.npy` (float32, standardized targets, shape `[n_experiments, 3]`)
- `data/processed/feature_scaler.joblib` (scaler stats for inputs + targets, used for inference de-normalization)

### **How to run**

```bash
python src/features/build_features.py data/processed
```

Optional flags:

```bash
python src/features/build_features.py data/processed --missing-features error
python src/features/build_features.py data/processed --missing-features drop
python src/features/build_features.py data/processed --missing-features zero
python src/features/build_features.py data/processed --missing-features mean
python src/features/build_features.py data/processed --missing-targets error
```

---

# **Quickstart: data → features**

```bash
python src/data/make_dataset.py data/raw data/processed
python src/features/build_features.py data/processed
```

---

## **src/models/**

### **train_model.py**

Runs the **two-stage training pipeline**:

### **Stage 1 — MAE Pretraining**

- Trains the Masked Autoencoder on brine chemistry
- Learns a shared representation of global brine composition
- Saves encoder weights as mae_pretrained.pth

### **Stage 2 — Downstream Fine-tuning**

- Freezes the MAE encoder
- Trains a small regression head using experimental labels
- Outputs downstream_head.pth

### **How to run (separated scripts)**

Pretrain MAE:

```bash
python src/models/train_mae.py data/processed --out models/mae_pretrained.pth
```

Fine-tune regression head (encoder frozen by default):

```bash
python src/models/finetune_regression.py data/processed --mae models/mae_pretrained.pth --out models/downstream_head.pth
```

### **W&B tracking**

Both scripts support optional W&B logging. Offline mode is the default to avoid requiring credentials.

```bash
python src/models/train_mae.py data/processed --wandb --wandb-mode offline
python src/models/finetune_regression.py data/processed --wandb --wandb-mode offline
```

To log online, set `WANDB_API_KEY` and use `--wandb-mode online`.

Note: the regression fine-tuning uses the MAE encoder on the brine-chemistry feature space; with the current `X_exp` schema (`TDS_gL`, `MLR`, `Light_kW_m2`), only `TDS_gL` and `MLR` are provided to the encoder and the remaining chemistry features are treated as missing. `Light_kW_m2` is concatenated to the latent vector before the regression head.

---

### **predict_model.py**

Implements model inference:

- Loads pretrained encoder + regression head
- Loads feature scaler
- Accepts new brine chemistry as input
- Returns predictions for:
    - Selectivity
    - Li crystallization
    - Evaporation

Designed for batch predictions over:

- Global salt lakes
- Geospatial grids
- Synthetic brine compositions

---

## **src/visualization/**

### **visualize.py**

Provides tools for:

- Latent space visualization (PCA, UMAP)
- MAE reconstruction evaluation
- Correlation plots
- Brine clustering
- Prediction vs. ground truth comparison
- Mapping global lithium extraction potential

Useful for reports, publications, and EDA.

---

# **Summary**

This codebase implements a modern, scalable, and data-efficient architecture for global lithium extraction modeling:

- **MAE learns chemical representation from unlabeled data**
- **Fine-tuned regression learns performance mapping from only 10 experiments**
- **Predictive model scales to global brine systems**

The pipeline is modular, reproducible, and extendable—and aligns with best practices in scientific machine learning.
