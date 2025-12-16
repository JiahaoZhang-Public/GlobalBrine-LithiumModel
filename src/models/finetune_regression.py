from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.constants import BRINE_FEATURE_COLUMNS, EXPERIMENTAL_FEATURE_COLUMNS
from src.models.mae import TabularMAE, TabularMAEConfig
from src.models.regression_head import RegressionHead, RegressionHeadConfig
from src.models.wandb_utils import init_wandb, log_wandb


@dataclass(frozen=True)
class FinetuneConfig:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_mae_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def build_encoder_from_checkpoint(ckpt: dict[str, Any]) -> TabularMAE:
    mae_config = TabularMAEConfig(**ckpt.get("mae_config", {}))
    model = TabularMAE(num_features=int(ckpt["num_features"]), config=mae_config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def finetune_regression_head(
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    encoder: TabularMAE,
    *,
    head_config: RegressionHeadConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    freeze_encoder: bool = True,
    wandb_run=None,
) -> RegressionHead:
    head_config = head_config or RegressionHeadConfig(out_dim=y_exp.shape[1])
    finetune_config = finetune_config or FinetuneConfig()

    torch.manual_seed(finetune_config.seed)
    np.random.seed(finetune_config.seed)

    dev = _device(finetune_config.device)
    encoder = encoder.to(dev)
    head = RegressionHead(in_dim=encoder.config.d_model + 1, config=head_config).to(dev)

    x = torch.from_numpy(x_exp).to(dtype=torch.float32)
    y = torch.from_numpy(y_exp).to(dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=finetune_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        encoder.train()

    head.train()
    optim = torch.optim.AdamW(
        (
            head.parameters()
            if freeze_encoder
            else list(head.parameters()) + list(encoder.parameters())
        ),
        lr=finetune_config.lr,
        weight_decay=finetune_config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    tds_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("TDS_gL")
    mlr_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("MLR")
    light_idx = list(EXPERIMENTAL_FEATURE_COLUMNS).index("Light_kW_m2")
    chem_tds = list(BRINE_FEATURE_COLUMNS).index("TDS_gL")
    chem_mlr = list(BRINE_FEATURE_COLUMNS).index("MLR")

    global_step = 0
    for _epoch in range(finetune_config.epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(dev, non_blocking=True)
            batch_y = batch_y.to(dev, non_blocking=True)

            # Map experimental features into the MAE's chemistry feature space.
            # Current experimental features are: [TDS_gL, MLR, Light_kW_m2].
            chem = torch.full(
                (batch_x.shape[0], encoder.num_features),
                float("nan"),
                device=dev,
                dtype=batch_x.dtype,
            )
            chem[:, chem_tds] = batch_x[:, tds_idx]
            chem[:, chem_mlr] = batch_x[:, mlr_idx]

            if freeze_encoder:
                with torch.no_grad():
                    z = encoder.encode(chem)
            else:
                z = encoder.encode(chem)

            light = batch_x[:, light_idx].unsqueeze(1)
            pred = head(torch.cat([z, light], dim=1))
            loss = loss_fn(pred, batch_y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_value = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_value
            epoch_steps += 1
            log_wandb(
                wandb_run,
                {
                    "train/loss": loss_value,
                    "train/lr": finetune_config.lr,
                    "epoch": _epoch,
                },
                step=global_step,
            )
            global_step += 1
        if epoch_steps:
            log_wandb(
                wandb_run,
                {"train/epoch_loss": epoch_loss_sum / epoch_steps, "epoch": _epoch},
                step=global_step,
            )

    return head


def save_head_checkpoint(
    path: Path,
    head: RegressionHead,
    *,
    head_config: RegressionHeadConfig,
    finetune_config: FinetuneConfig,
    mae_checkpoint_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "head_config": head_config.as_dict(),
            "finetune_config": finetune_config.as_dict(),
            "mae_checkpoint": mae_checkpoint_path,
        },
        path,
    )


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--mae",
    "mae_path",
    type=click.Path(exists=True, dir_okay=False),
    default="models/mae_pretrained.pth",
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default="models/downstream_head.pth",
    show_default=True,
)
@click.option("--epochs", type=int, default=200, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--weight-decay", type=float, default=0.0, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--hidden-dim", type=int, default=128, show_default=True)
@click.option("--n-layers", "head_layers", type=int, default=2, show_default=True)
@click.option("--dropout", type=float, default=0.0, show_default=True)
@click.option("--freeze-encoder/--train-encoder", default=True, show_default=True)
@click.option("--wandb/--no-wandb", default=False, show_default=True)
@click.option(
    "--wandb-project", type=str, default="GlobalBrine-LithiumModel", show_default=True
)
@click.option("--wandb-name", type=str, default=None)
@click.option(
    "--wandb-mode",
    type=click.Choice(["offline", "online", "disabled"], case_sensitive=False),
    default="offline",
    show_default=True,
)
@click.option("--wandb-tag", "wandb_tags", multiple=True)
def main(
    processed_dir: str,
    mae_path: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    hidden_dim: int,
    head_layers: int,
    dropout: float,
    freeze_encoder: bool,
    wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    wandb_mode: str,
    wandb_tags: tuple[str, ...],
) -> None:
    processed = Path(processed_dir)
    x_exp = np.load(processed / "X_exp.npy")
    y_exp = np.load(processed / "y_exp.npy")

    ckpt = load_mae_checkpoint(Path(mae_path))
    encoder = build_encoder_from_checkpoint(ckpt)

    head_config = RegressionHeadConfig(
        hidden_dim=hidden_dim,
        n_layers=head_layers,
        dropout=dropout,
        out_dim=y_exp.shape[1],
    )
    finetune_config = FinetuneConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )

    run = init_wandb(
        enabled=wandb and wandb_mode.lower() != "disabled",
        project=wandb_project,
        run_name=wandb_name,
        mode=wandb_mode.lower(),
        config={
            "task": "regression_finetune",
            "freeze_encoder": freeze_encoder,
            "num_features_encoder": int(encoder.num_features),
            "num_features_exp": int(x_exp.shape[1]),
            "num_targets": int(y_exp.shape[1]),
            **ckpt.get("mae_config", {}),
            **head_config.as_dict(),
            **finetune_config.as_dict(),
        },
        tags=wandb_tags,
    )

    head = finetune_regression_head(
        x_exp,
        y_exp,
        encoder,
        head_config=head_config,
        finetune_config=finetune_config,
        freeze_encoder=freeze_encoder,
        wandb_run=run,
    )
    save_head_checkpoint(
        Path(out_path),
        head,
        head_config=head_config,
        finetune_config=finetune_config,
        mae_checkpoint_path=mae_path,
    )
    click.echo(f"Saved regression head to: {out_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
