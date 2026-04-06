# Model Evaluation Results

Generated automatically by `evaluations/run_evaluation.py`

---

## 1. MAE Pretrain Evaluation

### 1a. Per-Feature Reconstruction Error

| Feature | MSE (normalized) | Observed Samples |
|---------|-----------------|------------------|
| Li_gL | 0.799381 | 946 |
| Mg_gL | 0.930898 | 946 |
| Na_gL | 0.557574 | 939 |
| K_gL | 0.929127 | 938 |
| Ca_gL | 0.970847 | 937 |
| SO4_gL | 0.906118 | 889 |
| Cl_gL | 0.545792 | 899 |
| MLR | 1.196074 | 946 |
| TDS_gL | 0.464199 | 656 |

![Per-Feature Reconstruction](mae_pretrain/per_feature_reconstruction.png)

**True vs Predicted scatter plots** (per feature):

<details><summary>Li_gL</summary>

![Li_gL](mae_pretrain/feature_true_vs_pred_Li_gL.png)
</details>

<details><summary>Mg_gL</summary>

![Mg_gL](mae_pretrain/feature_true_vs_pred_Mg_gL.png)
</details>

<details><summary>Na_gL</summary>

![Na_gL](mae_pretrain/feature_true_vs_pred_Na_gL.png)
</details>

<details><summary>K_gL</summary>

![K_gL](mae_pretrain/feature_true_vs_pred_K_gL.png)
</details>

<details><summary>Ca_gL</summary>

![Ca_gL](mae_pretrain/feature_true_vs_pred_Ca_gL.png)
</details>

<details><summary>SO4_gL</summary>

![SO4_gL](mae_pretrain/feature_true_vs_pred_SO4_gL.png)
</details>

<details><summary>Cl_gL</summary>

![Cl_gL](mae_pretrain/feature_true_vs_pred_Cl_gL.png)
</details>

<details><summary>MLR</summary>

![MLR](mae_pretrain/feature_true_vs_pred_MLR.png)
</details>

<details><summary>TDS_gL</summary>

![TDS_gL](mae_pretrain/feature_true_vs_pred_TDS_gL.png)
</details>

### 1b. Validation Loss Curve

- Final train loss: **0.084789**

- Final val loss: **0.443299**

![Validation Loss Curve](mae_pretrain/val_loss_curve.png)

### 1c. Latent Space Visualization (t-SNE)

![Latent Space t-SNE](mae_pretrain/latent_space_tsne.png)

---

## 2. Regression Fine-tuning Evaluation

### 2a. LOO-CV Results

| Target | MAE | RMSE | R2 |
|--------|-----|------|-----|
| Selectivity | 0.4016 | 0.7541 | 0.9603 |
| Li_Crystallization_mg_m2_h | 0.0372 | 0.0576 | 0.6991 |
| Evap_kg_m2_h | 0.0141 | 0.0190 | 0.9148 |

![LOO-CV Scatter](regression/loo_cv_scatter.png)

### 2b. Physical Bounds Check

| Target | Min | Max | Mean | Negative Count |
|--------|-----|-----|------|---------------|
| Selectivity | 4.0924 | 23.3012 | 11.8281 | 0 |
| Li_Crystallization_mg_m2_h | 0.1497 | 0.4867 | 0.2921 | 0 |
| Evap_kg_m2_h | 1.1788 | 1.5129 | 1.2573 | 0 |

### 2c. Baseline Comparison

| Method | Target | MAE | RMSE | R2 |
|--------|--------|-----|------|-----|
| MAE+Head | Selectivity | 0.4016 | 0.7541 | 0.9603 |
| MAE+Head | Li_Crystallization_mg_m2_h | 0.0372 | 0.0576 | 0.6991 |
| MAE+Head | Evap_kg_m2_h | 0.0141 | 0.0190 | 0.9148 |
| Mean | Selectivity | 2.3662 | 3.8315 | -0.0258 |
| Mean | Li_Crystallization_mg_m2_h | 0.0978 | 0.1064 | -0.0258 |
| Mean | Evap_kg_m2_h | 0.0338 | 0.0661 | -0.0258 |
| Linear | Selectivity | 0.8630 | 1.3902 | 0.8650 |
| Linear | Li_Crystallization_mg_m2_h | 0.0820 | 0.0900 | 0.2665 |
| Linear | Evap_kg_m2_h | 0.0279 | 0.0411 | 0.6031 |

![Baseline Comparison](regression/baseline_comparison.png)
