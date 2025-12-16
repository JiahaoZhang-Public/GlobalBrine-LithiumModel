from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.models.mae import TabularMAE, TabularMAEConfig
    from src.models.train_mae import TrainConfig, save_checkpoint, train_mae
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.models.mae import TabularMAE, TabularMAEConfig
    from src.models.train_mae import TrainConfig, save_checkpoint, train_mae


def _parse_int_list(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip() != ""]


def _parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip() != ""]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def generate_config_grid(
    *,
    d_model: Iterable[int],
    n_heads: Iterable[int],
    n_layers: Iterable[int],
    mlp_ratio: Iterable[float],
    dropout: Iterable[float],
    mask_ratio: Iterable[float],
) -> list[TabularMAEConfig]:
    configs: list[TabularMAEConfig] = []
    for vals in itertools.product(
        d_model, n_heads, n_layers, mlp_ratio, dropout, mask_ratio
    ):
        dm, nh, nl, mr, do, mkr = vals
        if dm <= 0 or nh <= 0 or nl <= 0:
            continue
        if dm % nh != 0:
            continue
        if not (0.0 <= do <= 1.0):
            continue
        if not (0.0 < mkr <= 1.0):
            continue
        configs.append(
            TabularMAEConfig(
                d_model=int(dm),
                n_heads=int(nh),
                n_layers=int(nl),
                mlp_ratio=float(mr),
                dropout=float(do),
                mask_ratio=float(mkr),
            )
        )
    return configs


@torch.no_grad()
def estimate_pretrain_loss(
    model: TabularMAE,
    x: np.ndarray,
    *,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    device: str,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = _device(device)
    model.eval().to(dev)
    xt = torch.from_numpy(x).to(dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt), batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(dev, non_blocking=True)
        loss = model.pretrain_loss(batch_x, mask_ratio=mask_ratio)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default="models/mae_sweep",
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--train-fraction", type=float, default=0.9, show_default=True)
@click.option("--epochs", type=int, default=1000, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--eval-batch-size", type=int, default=16, show_default=True)
@click.option("--save-all/--save-best-only", default=False, show_default=True)
@click.option(
    "--d-model", "d_model_list", type=str, default="64,128,256", show_default=True
)
@click.option(
    "--n-heads", "n_heads_list", type=str, default="4,8,16", show_default=True
)
@click.option(
    "--n-layers", "n_layers_list", type=str, default="2,4,6", show_default=True
)
@click.option(
    "--mlp-ratio", "mlp_ratio_list", type=str, default="2.0,4.0", show_default=True
)
@click.option(
    "--dropout", "dropout_list", type=str, default="0.0,0.1,0.2", show_default=True
)
@click.option(
    "--mask-ratio",
    "mask_ratio_list",
    type=str,
    default="0.1,0.2,0.3",
    show_default=True,
)
def main(
    processed_dir: str,
    out_dir: str,
    seed: int,
    train_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    eval_batch_size: int,
    save_all: bool,
    d_model_list: str,
    n_heads_list: str,
    n_layers_list: str,
    mlp_ratio_list: str,
    dropout_list: str,
    mask_ratio_list: str,
) -> None:
    processed = Path(processed_dir)
    x_lake = np.load(processed / "X_lake.npy")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    idx = np.arange(x_lake.shape[0])
    rng.shuffle(idx)
    split = int(x_lake.shape[0] * train_fraction)
    train_idx = idx[:split]
    val_idx = idx[split:] if split < len(idx) else idx[:0]
    x_train = x_lake[train_idx]
    x_val = (
        x_lake[val_idx]
        if val_idx.size
        else x_lake[train_idx[: max(1, len(train_idx) // 10)]]
    )

    configs = generate_config_grid(
        d_model=_parse_int_list(d_model_list),
        n_heads=_parse_int_list(n_heads_list),
        n_layers=_parse_int_list(n_layers_list),
        mlp_ratio=_parse_float_list(mlp_ratio_list),
        dropout=_parse_float_list(dropout_list),
        mask_ratio=_parse_float_list(mask_ratio_list),
    )
    if not configs:
        raise click.ClickException("No valid configurations generated for sweep.")

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        mask_ratio=0.0,  # overridden per-config below
        device=device,
        seed=seed,
    )

    results: list[dict] = []
    best_loss = float("inf")
    best_row: dict | None = None
    best_ckpt_path: Path | None = None

    for i, cfg in enumerate(configs, start=1):
        click.echo(
            f"[{i}/{len(configs)}] d_model={cfg.d_model} n_heads={cfg.n_heads} "
            f"n_layers={cfg.n_layers} mlp_ratio={cfg.mlp_ratio} dropout={cfg.dropout} "
            f"mask_ratio={cfg.mask_ratio}"
        )
        run_train_cfg = TrainConfig(
            **{**train_cfg.as_dict(), "mask_ratio": cfg.mask_ratio}
        )
        model = train_mae(x_train, mae_config=cfg, train_config=run_train_cfg)
        val_loss = estimate_pretrain_loss(
            model,
            x_val,
            batch_size=eval_batch_size,
            mask_ratio=cfg.mask_ratio,
            seed=seed + 1000,
            device=device,
        )

        ckpt_file = out_path / f"mae_run_{i:03d}.pth"
        if save_all:
            save_checkpoint(
                ckpt_file, model, mae_config=cfg, train_config=run_train_cfg
            )

        row = {
            "run": i,
            "val_loss": f"{val_loss:.6f}",
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "mlp_ratio": cfg.mlp_ratio,
            "dropout": cfg.dropout,
            "mask_ratio": cfg.mask_ratio,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        results.append(row)

        if val_loss < best_loss:
            best_loss = val_loss
            best_row = row
            best_ckpt_path = out_path / "best_mae_pretrained.pth"
            save_checkpoint(
                best_ckpt_path, model, mae_config=cfg, train_config=run_train_cfg
            )

    results_sorted = sorted(results, key=lambda r: float(r["val_loss"]))
    write_csv_table(out_path / "sweep_results.csv", results_sorted)
    write_markdown_table(out_path / "sweep_results.md", results_sorted)

    if best_row is not None:
        (out_path / "best_config.json").write_text(
            json.dumps(best_row, indent=2), encoding="utf-8"
        )
        click.echo(f"Best val_loss={best_row['val_loss']} saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
