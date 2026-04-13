#!/usr/bin/env python3
"""Plot-only evaluation pipeline — reads saved metadata, no models.

Generates all evaluation figures and README from metadata files
produced by ``run_compute.py``.

Usage:
    python evaluations/run_plots.py
    python evaluations/run_plots.py --skip-readme
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    BRINE_FEATURE_COLUMNS,
    C_BLUE,
    C_GREEN,
    C_GREY,
    C_ORANGE,
    C_RED,
    C_TEAL,
    EXPERIMENTAL_TARGET_COLUMNS,
    NATURE_STYLE,
    display_label,
)

plt.style.use(NATURE_STYLE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = ROOT / "evaluations"
METADATA_DIR = EVAL_DIR / "metadata"
MAE_PRETRAIN_DIR = EVAL_DIR / "mae_pretrain"
REGRESSION_DIR = EVAL_DIR / "regression"

FEATURE_NAMES = list(BRINE_FEATURE_COLUMNS)
TARGET_NAMES = list(EXPERIMENTAL_TARGET_COLUMNS)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ===================================================================
# MAE Pretrain Plots
# ===================================================================


def plot_per_feature_bar(feat_mse: dict, out_dir: Path) -> None:
    names = list(feat_mse.keys())
    labels = [display_label(n) for n in names]
    mses = [feat_mse[n]["mse"] for n in names]
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
    fig.savefig(out_dir / "per_feature_reconstruction.png")
    plt.close(fig)
    print("  Saved per_feature_reconstruction.png")


def plot_per_feature_scatter(scatter_path: Path, out_dir: Path) -> None:
    # NPZ contains trusted, self-generated arrays from run_compute.py
    data = np.load(scatter_path, allow_pickle=True)
    feature_names = list(data["feature_names"])

    for fname in feature_names:
        true_key = f"{fname}_true"
        pred_key = f"{fname}_pred"
        if true_key not in data or len(data[true_key]) == 0:
            continue
        true_vals = data[true_key]
        pred_vals = data[pred_key]

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(true_vals, pred_vals, alpha=0.4, s=8, color=C_BLUE, linewidths=0)
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        ax.plot([lo, hi], [lo, hi], "--", color=C_RED, linewidth=0.8, label="y = x")
        ax.set_xlabel(f"True {display_label(fname)} (normalised)")
        ax.set_ylabel(f"Predicted {display_label(fname)} (normalised)")
        ax.set_title(f"MAE reconstruction: {display_label(fname)}")
        ax.legend()
        fig.savefig(out_dir / f"feature_true_vs_pred_{fname}.png")
        plt.close(fig)
    print(f"  Saved {len(feature_names)} scatter plots")


def plot_loss_curve(loss_data: dict, out_dir: Path) -> None:
    train = loss_data["train_losses"]
    val = loss_data["val_losses"]
    epochs = range(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(epochs, train, label="Train", color=C_BLUE)
    ax.plot(epochs, val, label="Validation", color=C_RED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.set_title("MAE pre-training loss curve")
    ax.legend()
    fig.savefig(out_dir / "val_loss_curve.png")
    plt.close(fig)
    print("  Saved val_loss_curve.png")


def plot_tsne(tsne_path: Path, out_dir: Path) -> None:
    # NPZ contains trusted, self-generated arrays from run_compute.py
    data = np.load(tsne_path, allow_pickle=True)
    latent_2d = data["latent_2d"]
    water_types = data["water_types"].astype(str)
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
        ax.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            label=f"{wt} (n={mask.sum()})",
            alpha=0.6,
            s=10,
            color=color_map.get(wt, C_GREY),
            linewidths=0,
        )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("MAE latent space (coloured by water type)")
    ax.legend(loc="best")
    fig.savefig(out_dir / "latent_space_tsne.png")
    plt.close(fig)
    print("  Saved latent_space_tsne.png")


# ===================================================================
# Regression Plots
# ===================================================================


def plot_loo_scatter(pred_path: Path, metrics: dict, out_dir: Path) -> None:
    data = np.load(pred_path)
    y_true = data["y_true_raw"]
    y_pred = data["y_pred_raw"]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))
    for j, (tname, ax) in enumerate(zip(TARGET_NAMES, axes)):
        true_j = y_true[:, j]
        pred_j = y_pred[:, j]
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
    fig.savefig(out_dir / "loo_cv_scatter.png")
    plt.close(fig)
    print("  Saved loo_cv_scatter.png")


def plot_baseline_comparison(all_methods_metrics: dict, out_dir: Path) -> None:
    methods = list(all_methods_metrics.keys())
    x_pos = np.arange(len(TARGET_NAMES))
    width = 0.25
    method_colors = {"MAE+Head": C_BLUE, "Mean": C_GREY, "Linear": C_GREEN}

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
    fig.savefig(out_dir / "baseline_comparison.png")
    plt.close(fig)
    print("  Saved baseline_comparison.png")


# ===================================================================
# README Generation
# ===================================================================


def generate_readme(
    feat_mse: dict,
    loss_curve: dict,
    loo_metrics: dict,
    bounds: dict,
    baseline_metrics: dict,
) -> None:
    lines: list[str] = []
    lines.append("# Model Evaluation Results\n")
    lines.append("Generated automatically by `evaluations/run_plots.py`\n")
    lines.append("---\n")

    # Category 1
    lines.append("## 1. MAE Pretrain Evaluation\n")
    lines.append("### 1a. Per-Feature Reconstruction Error\n")
    lines.append("| Feature | MSE (normalized) | Observed Samples |")
    lines.append("|---------|-----------------|------------------|")
    for fname, data in feat_mse.items():
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
    ftl = loss_curve.get("final_train_loss", 0)
    fvl = loss_curve.get("final_val_loss", 0)
    lines.append(f"- Final train loss: **{ftl:.6f}**\n")
    lines.append(f"- Final val loss: **{fvl:.6f}**\n")
    lines.append("![Validation Loss Curve](mae_pretrain/val_loss_curve.png)\n")

    lines.append("### 1c. Latent Space Visualization (t-SNE)\n")
    lines.append("![Latent Space t-SNE](mae_pretrain/latent_space_tsne.png)\n")

    # Category 2
    lines.append("---\n")
    lines.append("## 2. Regression Fine-tuning Evaluation\n")
    lines.append("### 2a. LOO-CV Results\n")
    lines.append("| Target | MAE | RMSE | R2 |")
    lines.append("|--------|-----|------|-----|")
    for tname, m in loo_metrics.items():
        lines.append(f"| {tname} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")
    lines.append("")
    lines.append("![LOO-CV Scatter](regression/loo_cv_scatter.png)\n")

    lines.append("### 2b. Physical Bounds Check\n")
    lines.append("| Target | Min | Max | Mean | Negative Count |")
    lines.append("|--------|-----|-----|------|---------------|")
    for tname, b in bounds.items():
        lines.append(
            f"| {tname} | {b['min']:.4f} | {b['max']:.4f} | "
            f"{b['mean']:.4f} | {b['n_negative']} |"
        )
    lines.append("")

    lines.append("### 2c. Baseline Comparison\n")
    lines.append("| Method | Target | MAE | RMSE | R2 |")
    lines.append("|--------|--------|-----|------|-----|")
    for method, method_metrics in baseline_metrics.items():
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
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from metadata."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help="Path to metadata directory",
    )
    parser.add_argument("--skip-readme", action="store_true")
    args = parser.parse_args()

    md = args.metadata_dir
    required = [
        "mae_feature_mse.json",
        "mae_scatter.npz",
        "mae_loss_curve.json",
        "mae_tsne.npz",
        "reg_loo_predictions.npz",
        "reg_loo_metrics.json",
        "reg_physical_bounds.json",
        "reg_baseline_metrics.json",
    ]
    missing = [f for f in required if not (md / f).exists()]
    if missing:
        print(f"ERROR: Missing metadata files in {md}/:")
        for f in missing:
            print(f"  - {f}")
        print("Run `python evaluations/run_compute.py` first.")
        sys.exit(1)

    MAE_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    feat_mse = _load_json(md / "mae_feature_mse.json")
    loss_curve = _load_json(md / "mae_loss_curve.json")
    loo_metrics = _load_json(md / "reg_loo_metrics.json")
    bounds = _load_json(md / "reg_physical_bounds.json")
    baseline_metrics = _load_json(md / "reg_baseline_metrics.json")

    print("Generating MAE pretrain plots...")
    plot_per_feature_bar(feat_mse, MAE_PRETRAIN_DIR)
    plot_per_feature_scatter(md / "mae_scatter.npz", MAE_PRETRAIN_DIR)
    plot_loss_curve(loss_curve, MAE_PRETRAIN_DIR)
    plot_tsne(md / "mae_tsne.npz", MAE_PRETRAIN_DIR)

    print("\nGenerating regression plots...")
    plot_loo_scatter(md / "reg_loo_predictions.npz", loo_metrics, REGRESSION_DIR)
    plot_baseline_comparison(baseline_metrics, REGRESSION_DIR)

    if not args.skip_readme:
        generate_readme(feat_mse, loss_curve, loo_metrics, bounds, baseline_metrics)

    print("\n" + "=" * 60)
    print("  All plots generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
