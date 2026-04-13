#!/usr/bin/env python3
"""Compute-only evaluation pipeline — no plotting.

Runs MAE pretrain and regression evaluations, saves all intermediate
results as metadata files (JSON + NPZ) so plots can be regenerated
without re-running the models.

Usage:
    python evaluations/run_compute.py
    python evaluations/run_compute.py --skip-mae
    python evaluations/run_compute.py --skip-regression
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    BRINE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    EXPERIMENTAL_TARGET_COLUMNS,
)
from src.features.scaler import (  # noqa: E402
    destandardize_preserve_nan,
    get_stats,
    load_scaler,
)
from src.models.finetune_regression import (  # noqa: E402
    FinetuneConfig,
    build_encoder_from_checkpoint,
    finetune_regression_head,
    load_mae_checkpoint,
    mae_feature_names_from_checkpoint,
)
from src.models.mae import TabularMAE, TabularMAEConfig  # noqa: E402
from src.models.mae_metrics import (  # noqa: E402
    feature_true_vs_pred,
    per_feature_drop_mse,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluations"
METADATA_DIR = EVAL_DIR / "metadata"
REGRESSION_DIR = EVAL_DIR / "regression"

FEATURE_NAMES = list(BRINE_FEATURE_COLUMNS)
TARGET_NAMES = list(EXPERIMENTAL_TARGET_COLUMNS)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path.name}")


# ===================================================================
# Category 1: MAE Pretrain
# ===================================================================


def compute_mae_pretrain(device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("Category 1: MAE Pretrain Evaluation")
    print("=" * 60)

    x_lake = np.load(PROCESSED_DIR / "X_lake.npy")
    ckpt = load_mae_checkpoint(MODELS_DIR / "mae_pretrained.pth")
    mae = build_encoder_from_checkpoint(ckpt)
    mae.to(device)

    brines_df = pd.read_csv(PROCESSED_DIR / "brines.csv")

    # --- 1a. Per-feature reconstruction error ---
    print("\n[1a] Per-feature reconstruction error...")
    drop_results = per_feature_drop_mse(mae, x_lake, device=device)

    feat_mse: dict = {}
    for r in drop_results:
        name = FEATURE_NAMES[r.feature_index]
        feat_mse[name] = {"mse": r.mse, "n_obs": r.n}
        print(f"  {name:25s}  MSE={r.mse:.6f}  (n={r.n})")
    _save_json(feat_mse, METADATA_DIR / "mae_feature_mse.json")

    # Scatter data per feature
    print("  Computing true-vs-pred arrays...")
    scatter_kw: dict = {"feature_names": np.array(FEATURE_NAMES)}
    for idx, fname in enumerate(FEATURE_NAMES):
        true_vals, pred_vals = feature_true_vs_pred(
            mae, x_lake, feature_index=idx, device=device
        )
        scatter_kw[f"{fname}_true"] = true_vals
        scatter_kw[f"{fname}_pred"] = pred_vals
    np.savez_compressed(METADATA_DIR / "mae_scatter.npz", **scatter_kw)
    print("  Saved mae_scatter.npz")

    # --- 1b. Validation loss curve ---
    print("\n[1b] Training with val split for loss curve...")
    rng = np.random.default_rng(42)
    idx = np.arange(x_lake.shape[0])
    rng.shuffle(idx)
    split = int(x_lake.shape[0] * 0.9)
    x_train = x_lake[idx[:split]]
    x_val = x_lake[idx[split:]]

    mae_config = TabularMAEConfig(**ckpt.get("mae_config", {}))
    mae_fresh = TabularMAE(num_features=x_lake.shape[1], config=mae_config).to(device)
    torch.manual_seed(42)
    np.random.seed(42)

    epochs = 50
    batch_size = 128
    lr = 1e-3
    mask_ratio = 0.2
    optimizer = torch.optim.AdamW(mae_fresh.parameters(), lr=lr, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train).to(dtype=torch.float32, device=device)
    x_val_t = torch.from_numpy(x_val).to(dtype=torch.float32, device=device)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for ep in range(epochs):
        mae_fresh.train()
        perm = torch.randperm(x_train_t.shape[0])
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, x_train_t.shape[0], batch_size):
            batch = x_train_t[perm[start : start + batch_size]]
            loss = mae_fresh.pretrain_loss(batch, mask_ratio=mask_ratio)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        train_losses.append(ep_loss / max(n_batches, 1))

        mae_fresh.train(False)
        with torch.no_grad():
            val_loss_sum = 0.0
            val_n = 0
            for start in range(0, x_val_t.shape[0], batch_size):
                batch = x_val_t[start : start + batch_size]
                vl = mae_fresh.pretrain_loss(batch, mask_ratio=mask_ratio)
                val_loss_sum += vl.item()
                val_n += 1
            val_losses.append(val_loss_sum / max(val_n, 1))

        if (ep + 1) % 10 == 0:
            print(
                f"  Epoch {ep+1:3d}/{epochs}  "
                f"train_loss={train_losses[-1]:.6f}  val_loss={val_losses[-1]:.6f}"
            )

    _save_json(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
        },
        METADATA_DIR / "mae_loss_curve.json",
    )

    # --- 1c. Latent space t-SNE ---
    print("\n[1c] Latent space t-SNE...")
    mae.train(False)
    with torch.no_grad():
        x_lake_t = torch.from_numpy(x_lake).to(dtype=torch.float32, device=device)
        latent = mae.encode(x_lake_t).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_2d = tsne.fit_transform(latent)
    water_types = brines_df["Type_of_water"].fillna("Unknown").values

    np.savez_compressed(
        METADATA_DIR / "mae_tsne.npz",
        latent_2d=latent_2d,
        water_types=np.array(water_types, dtype=object),
    )
    print("  Saved mae_tsne.npz")
    print("  Done: MAE pretrain compute complete.")


# ===================================================================
# Category 2: Regression
# ===================================================================


def compute_regression(device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("Category 2: Regression Fine-tuning Evaluation")
    print("=" * 60)

    x_exp = np.load(PROCESSED_DIR / "X_exp.npy")
    y_exp = np.load(PROCESSED_DIR / "y_exp.npy")
    n_samples = x_exp.shape[0]

    scaler = load_scaler(PROCESSED_DIR / "feature_scaler.joblib")
    y_mean, y_std = get_stats(
        scaler, key="experimental_targets", feature_names=EXPERIMENTAL_TARGET_COLUMNS
    )

    ckpt = load_mae_checkpoint(MODELS_DIR / "mae_pretrained.pth")
    mae_feature_names = mae_feature_names_from_checkpoint(ckpt)

    # --- 2a. Leave-One-Out CV ---
    print(f"\n[2a] Leave-One-Out CV ({n_samples} folds)...")
    finetune_cfg = FinetuneConfig(
        epochs=200, batch_size=16, lr=1e-3, device=str(device), seed=42
    )

    loo_preds_std = np.zeros_like(y_exp)

    for i in range(n_samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Fold {i+1}/{n_samples}...")
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        x_train = x_exp[train_mask]
        y_train = y_exp[train_mask]
        x_test = x_exp[i : i + 1]

        encoder = build_encoder_from_checkpoint(ckpt)
        encoder.to(device)

        film_head = finetune_regression_head(
            x_train,
            y_train,
            encoder,
            finetune_config=finetune_cfg,
            freeze_encoder=True,
            mae_feature_names=mae_feature_names,
        )
        film_head.to(device)

        encoder_features = list(mae_feature_names)
        tds_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("TDS_gL")
        mlr_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("MLR")
        light_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("Light_kW_m2")

        chem = torch.full(
            (1, encoder.num_features), float("nan"), dtype=torch.float32, device=device
        )
        x_t = torch.from_numpy(x_test).to(dtype=torch.float32, device=device)
        if "TDS_gL" in encoder_features:
            chem[0, encoder_features.index("TDS_gL")] = x_t[0, tds_idx]
        if "MLR" in encoder_features:
            chem[0, encoder_features.index("MLR")] = x_t[0, mlr_idx]

        with torch.no_grad():
            z = encoder.encode(chem)
            light = x_t[:, light_idx : light_idx + 1]
            pred = film_head(z, light)
            loo_preds_std[i] = pred.cpu().numpy().flatten()

    # De-normalize
    y_true_raw = destandardize_preserve_nan(y_exp, y_mean, y_std)
    y_pred_raw = destandardize_preserve_nan(loo_preds_std, y_mean, y_std)

    np.savez_compressed(
        METADATA_DIR / "reg_loo_predictions.npz",
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        target_names=np.array(TARGET_NAMES),
    )
    print("  Saved reg_loo_predictions.npz")

    # --- 2b. Per-target metrics ---
    print("\n[2b] Per-target metrics (de-normalized)...")
    metrics: dict = {}
    for j, tname in enumerate(TARGET_NAMES):
        true_j = y_true_raw[:, j]
        pred_j = y_pred_raw[:, j]
        mae_val = float(mean_absolute_error(true_j, pred_j))
        rmse_val = float(np.sqrt(mean_squared_error(true_j, pred_j)))
        r2_val = float(r2_score(true_j, pred_j))
        metrics[tname] = {"MAE": mae_val, "RMSE": rmse_val, "R2": r2_val}
        print(f"  {tname:35s}  MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  R2={r2_val:.4f}")

    _save_json(metrics, METADATA_DIR / "reg_loo_metrics.json")
    # Backward-compatible copy
    _save_json(metrics, REGRESSION_DIR / "loo_cv_metrics.json")

    # --- 2c. Physical bounds check ---
    print("\n[2c] Physical bounds check...")
    bounds_check: dict = {}
    for j, tname in enumerate(TARGET_NAMES):
        pred_j = y_pred_raw[:, j]
        n_neg = int((pred_j < 0).sum())
        bounds_check[tname] = {
            "min": float(pred_j.min()),
            "max": float(pred_j.max()),
            "mean": float(pred_j.mean()),
            "n_negative": n_neg,
        }
        print(
            f"  {tname:35s}  min={pred_j.min():.4f}  max={pred_j.max():.4f}  "
            f"n_negative={n_neg}"
        )

    _save_json(bounds_check, METADATA_DIR / "reg_physical_bounds.json")
    # Backward-compatible copy
    _save_json(bounds_check, REGRESSION_DIR / "physical_bounds_check.json")

    # --- 2d. Baseline comparison ---
    print("\n[2d] Baseline comparison (LOO-CV)...")
    mean_preds_raw = np.zeros_like(y_true_raw)
    lr_preds_raw = np.zeros_like(y_true_raw)

    for i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False
        x_tr = x_exp[train_mask]
        y_tr_raw = y_true_raw[train_mask]
        x_te = x_exp[i : i + 1]
        mean_preds_raw[i] = y_tr_raw.mean(axis=0)
        for j in range(y_true_raw.shape[1]):
            lr_model = LinearRegression()
            lr_model.fit(x_tr, y_tr_raw[:, j])
            lr_preds_raw[i, j] = lr_model.predict(x_te)[0]

    all_methods_metrics: dict = {"MAE+Head": metrics}
    for method_name, method_preds in [
        ("Mean", mean_preds_raw),
        ("Linear", lr_preds_raw),
    ]:
        m_metrics: dict = {}
        for j, tname in enumerate(TARGET_NAMES):
            true_j = y_true_raw[:, j]
            pred_j = method_preds[:, j]
            m_metrics[tname] = {
                "MAE": float(mean_absolute_error(true_j, pred_j)),
                "RMSE": float(np.sqrt(mean_squared_error(true_j, pred_j))),
                "R2": float(r2_score(true_j, pred_j)),
            }
        all_methods_metrics[method_name] = m_metrics
        print(f"\n  {method_name}:")
        for tname in TARGET_NAMES:
            m = m_metrics[tname]
            print(
                f"    {tname:35s}  MAE={m['MAE']:.4f}  "
                f"RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}"
            )

    _save_json(all_methods_metrics, METADATA_DIR / "reg_baseline_metrics.json")
    print("  Done: Regression compute complete.")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation (compute only).")
    parser.add_argument("--skip-mae", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.skip_mae:
        compute_mae_pretrain(device)
    if not args.skip_regression:
        compute_regression(device)

    print("\n" + "=" * 60)
    print(f"  Compute complete. Metadata saved to {METADATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
