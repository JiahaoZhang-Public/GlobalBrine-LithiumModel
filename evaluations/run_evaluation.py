#!/usr/bin/env python3
"""Comprehensive model evaluation pipeline.

Produces two categories of outputs:
  1. MAE pretrain   -> evaluations/mae_pretrain/
  2. Regression     -> evaluations/regression/

Usage:
    python evaluations/run_evaluation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    BRINE_FEATURE_COLUMNS,
    EXPERIMENTAL_FEATURE_COLUMNS,
    EXPERIMENTAL_TARGET_COLUMNS,
    NATURE_STYLE,
    display_label,
)

plt.style.use(NATURE_STYLE)
from src.features.scaler import destandardize_preserve_nan, get_stats, load_scaler
from src.models.finetune_regression import (
    FinetuneConfig,
    build_encoder_from_checkpoint,
    finetune_regression_head,
    load_mae_checkpoint,
    mae_feature_names_from_checkpoint,
)
from src.models.mae import TabularMAE, TabularMAEConfig
from src.models.mae_metrics import feature_true_vs_pred, per_feature_drop_mse

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluations"
MAE_PRETRAIN_DIR = EVAL_DIR / "mae_pretrain"
REGRESSION_DIR = EVAL_DIR / "regression"

for d in [MAE_PRETRAIN_DIR, REGRESSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = list(BRINE_FEATURE_COLUMNS)
TARGET_NAMES = list(EXPERIMENTAL_TARGET_COLUMNS)
EXP_FEATURE_NAMES = list(EXPERIMENTAL_FEATURE_COLUMNS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nature-style colour palette (muted, colour-blind-safe subset)
C_BLUE = "#4878CF"       # primary data
C_RED = "#C44E52"        # secondary / emphasis
C_GREEN = "#6ACC65"      # tertiary
C_GREY = "#8C8C8C"       # baselines / neutral
C_ORANGE = "#D65F5F"     # category accent
C_TEAL = "#59A89C"       # category accent


# ===================================================================
# Category 1: MAE Pretrain Evaluation
# ===================================================================


def run_mae_pretrain_evaluation() -> dict:
    """Run MAE pretrain evaluation and save outputs."""
    print("\n" + "=" * 60)
    print("Category 1: MAE Pretrain Evaluation")
    print("=" * 60)

    x_lake = np.load(PROCESSED_DIR / "X_lake.npy")
    ckpt = load_mae_checkpoint(MODELS_DIR / "mae_pretrained.pth")
    mae = build_encoder_from_checkpoint(ckpt)
    mae.to(DEVICE)

    brines_df = pd.read_csv(PROCESSED_DIR / "brines.csv")

    results: dict = {}

    # --- 1a. Per-feature reconstruction error ---
    print("\n[1a] Per-feature reconstruction error...")
    drop_results = per_feature_drop_mse(mae, x_lake, device=DEVICE)

    feat_mse = {}
    for r in drop_results:
        name = FEATURE_NAMES[r.feature_index]
        feat_mse[name] = {"mse": r.mse, "n_obs": r.n}
        print(f"  {name:25s}  MSE={r.mse:.6f}  (n={r.n})")
    results["per_feature_mse"] = feat_mse

    # Bar chart
    names = [FEATURE_NAMES[r.feature_index] for r in drop_results]
    labels = [display_label(n) for n in names]
    mses = [r.mse for r in drop_results]
    colors = [C_RED if n == "TDS_gL" else C_BLUE for n in names]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(range(len(names)), mses, color=colors, edgecolor="none")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Reconstruction MSE (normalised)")
    ax.set_title("Per-feature reconstruction error (MAE)")
    ax.legend(
        [bars[names.index("TDS_gL")], bars[0]],
        [display_label("TDS_gL") + " (focus)", "Other features"],
        loc="upper left",
    )
    for i, v in enumerate(mses):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=5)
    fig.savefig(MAE_PRETRAIN_DIR / "per_feature_reconstruction.png")
    plt.close(fig)

    # Scatter plots per feature
    print("  Generating true-vs-pred scatter plots...")
    for idx, fname in enumerate(FEATURE_NAMES):
        true_vals, pred_vals = feature_true_vs_pred(
            mae, x_lake, feature_index=idx, device=DEVICE
        )
        if len(true_vals) == 0:
            continue
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(true_vals, pred_vals, alpha=0.4, s=8, color=C_BLUE, linewidths=0)
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        ax.plot([lo, hi], [lo, hi], "--", color=C_RED, linewidth=0.8, label="y = x")
        ax.set_xlabel(f"True {display_label(fname)} (normalised)")
        ax.set_ylabel(f"Predicted {display_label(fname)} (normalised)")
        ax.set_title(f"MAE reconstruction: {display_label(fname)}")
        ax.legend()
        fig.savefig(MAE_PRETRAIN_DIR / f"feature_true_vs_pred_{fname}.png")
        plt.close(fig)

    # --- 1b. Validation loss curve ---
    print("\n[1b] Training with val split for loss curve...")
    rng = np.random.default_rng(42)
    idx = np.arange(x_lake.shape[0])
    rng.shuffle(idx)
    split = int(x_lake.shape[0] * 0.9)
    x_train = x_lake[idx[:split]]
    x_val = x_lake[idx[split:]]

    mae_config = TabularMAEConfig(**ckpt.get("mae_config", {}))
    mae_fresh = TabularMAE(num_features=x_lake.shape[1], config=mae_config).to(DEVICE)
    torch.manual_seed(42)
    np.random.seed(42)

    epochs = 50
    batch_size = 128
    lr = 1e-3
    mask_ratio = 0.2

    optimizer = torch.optim.AdamW(mae_fresh.parameters(), lr=lr, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train).to(dtype=torch.float32, device=DEVICE)
    x_val_t = torch.from_numpy(x_val).to(dtype=torch.float32, device=DEVICE)

    train_losses = []
    val_losses = []

    for ep in range(epochs):
        # Train
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

        # Val
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

    results["val_loss_curve"] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(range(1, epochs + 1), train_losses, label="Train", color=C_BLUE)
    ax.plot(range(1, epochs + 1), val_losses, label="Validation", color=C_RED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.set_title("MAE pre-training loss curve")
    ax.legend()
    fig.savefig(MAE_PRETRAIN_DIR / "val_loss_curve.png")
    plt.close(fig)

    # --- 1c. Latent space visualization ---
    print("\n[1c] Latent space t-SNE visualization...")
    mae.train(False)
    with torch.no_grad():
        x_lake_t = torch.from_numpy(x_lake).to(dtype=torch.float32, device=DEVICE)
        latent = mae.encode(x_lake_t).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_2d = tsne.fit_transform(latent)

    water_types = brines_df["Type_of_water"].fillna("Unknown").values
    unique_types = sorted(set(water_types))
    color_map = {
        "Salt lake brine": C_BLUE,
        "seawater": C_TEAL,
        "geothermal brine": C_RED,
        "Oil field brine": C_ORANGE,
        "Unknown": C_GREY,
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for wt in unique_types:
        mask = water_types == wt
        color = color_map.get(wt, C_GREY)
        ax.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            label=f"{wt} (n={mask.sum()})",
            alpha=0.6,
            s=10,
            color=color,
            linewidths=0,
        )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("MAE latent space (coloured by water type)")
    ax.legend(loc="best")
    fig.savefig(MAE_PRETRAIN_DIR / "latent_space_tsne.png")
    plt.close(fig)

    print("  Done: MAE pretrain evaluation complete.")
    return results


# ===================================================================
# Category 2: Regression Evaluation
# ===================================================================


def run_regression_evaluation() -> dict:
    """Run regression LOO-CV evaluation and save outputs."""
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

    results: dict = {}

    # --- 2a. Leave-One-Out CV ---
    print(f"\n[2a] Leave-One-Out CV ({n_samples} folds)...")
    finetune_cfg = FinetuneConfig(
        epochs=200, batch_size=16, lr=1e-3, device=str(DEVICE), seed=42
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
        encoder.to(DEVICE)

        film_head = finetune_regression_head(
            x_train,
            y_train,
            encoder,
            finetune_config=finetune_cfg,
            freeze_encoder=True,
            mae_feature_names=mae_feature_names,
        )
        film_head.to(DEVICE)

        # Forward pass for held-out sample (FiLM: Light conditions the head).
        encoder_features = list(mae_feature_names)

        tds_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("TDS_gL")
        mlr_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("MLR")
        light_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("Light_kW_m2")

        chem = torch.full(
            (1, encoder.num_features), float("nan"), dtype=torch.float32, device=DEVICE
        )
        x_t = torch.from_numpy(x_test).to(dtype=torch.float32, device=DEVICE)
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

    results["loo_cv_metrics"] = metrics
    with open(REGRESSION_DIR / "loo_cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # LOO-CV scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))
    for j, (tname, ax) in enumerate(zip(TARGET_NAMES, axes)):
        true_j = y_true_raw[:, j]
        pred_j = y_pred_raw[:, j]
        ax.scatter(true_j, pred_j, alpha=0.7, s=12, color=C_BLUE, linewidths=0)
        lo = min(true_j.min(), pred_j.min())
        hi = max(true_j.max(), pred_j.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "--",
            color=C_RED,
            linewidth=0.8,
        )
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_title(display_label(tname))
        m = metrics[tname]
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=6,
        )
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(REGRESSION_DIR / "loo_cv_scatter.png")
    plt.close(fig)

    # --- 2c. Physical bounds check ---
    print("\n[2c] Physical bounds check...")
    bounds_check = {}
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
    results["physical_bounds"] = bounds_check
    with open(REGRESSION_DIR / "physical_bounds_check.json", "w") as f:
        json.dump(bounds_check, f, indent=2)

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

        # Mean predictor
        mean_preds_raw[i] = y_tr_raw.mean(axis=0)

        # Linear regression
        for j in range(y_true_raw.shape[1]):
            lr_model = LinearRegression()
            lr_model.fit(x_tr, y_tr_raw[:, j])
            lr_preds_raw[i, j] = lr_model.predict(x_te)[0]

    all_methods_metrics = {"MAE+Head": metrics}

    for method_name, method_preds in [
        ("Mean", mean_preds_raw),
        ("Linear", lr_preds_raw),
    ]:
        m_metrics = {}
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

    results["baseline_comparison"] = all_methods_metrics

    # Grouped bar chart
    methods = list(all_methods_metrics.keys())
    x_pos = np.arange(len(TARGET_NAMES))
    width = 0.25
    method_colors = {"MAE+Head": C_BLUE, "Mean": C_GREY, "Linear": C_GREEN}

    # Short tick labels to avoid overlap
    short_labels = {
        "Selectivity": r"Li$^+$/Mg$^{2+}$" + "\nselectivity",
        "Li_Crystallization_mg_m2_h": r"Li$^+$ flux",
        "Evap_kg_m2_h": "Evaporation\nrate",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))
    for mi, metric_name in enumerate(["MAE", "RMSE", "R2"]):
        ax = axes[mi]
        for k, method in enumerate(methods):
            vals = [all_methods_metrics[method][t][metric_name] for t in TARGET_NAMES]
            ax.bar(
                x_pos + k * width,
                vals,
                width,
                label=method,
                color=method_colors.get(method, C_GREY),
                edgecolor="none",
            )
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(
            [short_labels.get(t, t) for t in TARGET_NAMES],
            ha="center",
            fontsize=6,
        )
        ax.set_title(metric_name)
        if mi == 0:
            ax.legend()
    fig.suptitle("Baseline comparison (leave-one-out CV)", y=1.0)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(REGRESSION_DIR / "baseline_comparison.png")
    plt.close(fig)

    # Store LOO-CV results for use in end-to-end
    results["_loo_y_true_raw"] = y_true_raw
    results["_loo_y_pred_raw"] = y_pred_raw
    results["_x_exp"] = x_exp

    print("  Done: Regression evaluation complete.")
    return results


# ===================================================================
# README Generation
# ===================================================================


def generate_readme(mae_results: dict, reg_results: dict) -> None:
    """Generate evaluations/README.md with summary tables and image refs."""
    lines: list[str] = []
    lines.append("# Model Evaluation Results\n")
    lines.append("Generated automatically by `evaluations/run_evaluation.py`\n")
    lines.append("---\n")

    # --- Category 1 ---
    lines.append("## 1. MAE Pretrain Evaluation\n")

    lines.append("### 1a. Per-Feature Reconstruction Error\n")
    lines.append("| Feature | MSE (normalized) | Observed Samples |")
    lines.append("|---------|-----------------|------------------|")
    for fname, data in mae_results.get("per_feature_mse", {}).items():
        lines.append(f"| {fname} | {data['mse']:.6f} | {data['n_obs']} |")
    lines.append("")
    lines.append(
        "![Per-Feature Reconstruction](mae_pretrain/per_feature_reconstruction.png)\n"
    )

    lines.append("**True vs Predicted scatter plots** (per feature):\n")
    for fname in FEATURE_NAMES:
        p = f"mae_pretrain/feature_true_vs_pred_{fname}.png"
        lines.append(f"<details><summary>{fname}</summary>\n")
        lines.append(f"![{fname}]({p})\n</details>\n")

    lines.append("### 1b. Validation Loss Curve\n")
    vlc = mae_results.get("val_loss_curve", {})
    ftl = vlc.get("final_train_loss", 0)
    fvl = vlc.get("final_val_loss", 0)
    lines.append(f"- Final train loss: **{ftl:.6f}**\n")
    lines.append(f"- Final val loss: **{fvl:.6f}**\n")
    lines.append("![Validation Loss Curve](mae_pretrain/val_loss_curve.png)\n")

    lines.append("### 1c. Latent Space Visualization (t-SNE)\n")
    lines.append("![Latent Space t-SNE](mae_pretrain/latent_space_tsne.png)\n")

    # --- Category 2 ---
    lines.append("---\n")
    lines.append("## 2. Regression Fine-tuning Evaluation\n")

    lines.append("### 2a. LOO-CV Results\n")
    lines.append("| Target | MAE | RMSE | R2 |")
    lines.append("|--------|-----|------|-----|")
    for tname, m in reg_results.get("loo_cv_metrics", {}).items():
        lines.append(f"| {tname} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")
    lines.append("")
    lines.append("![LOO-CV Scatter](regression/loo_cv_scatter.png)\n")

    lines.append("### 2b. Physical Bounds Check\n")
    lines.append("| Target | Min | Max | Mean | Negative Count |")
    lines.append("|--------|-----|-----|------|---------------|")
    for tname, b in reg_results.get("physical_bounds", {}).items():
        lines.append(
            f"| {tname} | {b['min']:.4f} | {b['max']:.4f} | "
            f"{b['mean']:.4f} | {b['n_negative']} |"
        )
    lines.append("")

    lines.append("### 2c. Baseline Comparison\n")
    bc = reg_results.get("baseline_comparison", {})
    lines.append("| Method | Target | MAE | RMSE | R2 |")
    lines.append("|--------|--------|-----|------|-----|")
    for method, method_metrics in bc.items():
        for tname, m in method_metrics.items():
            lines.append(
                f"| {method} | {tname} | {m['MAE']:.4f} | "
                f"{m['RMSE']:.4f} | {m['R2']:.4f} |"
            )
    lines.append("")
    lines.append("![Baseline Comparison](regression/baseline_comparison.png)\n")

    readme_path = EVAL_DIR / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nREADME written to {readme_path}")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    print("=" * 60)
    print("  GlobalBrine Model Evaluation Pipeline")
    print("=" * 60)

    mae_results = run_mae_pretrain_evaluation()
    reg_results = run_regression_evaluation()
    generate_readme(mae_results, reg_results)

    print("\n" + "=" * 60)
    print("  All evaluations complete!")
    print(f"  Outputs saved to: {EVAL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
