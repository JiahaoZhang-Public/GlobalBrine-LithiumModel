from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.constants import BRINE_FEATURE_COLUMNS
from src.models.mae import TabularMAE, TabularMAEConfig
from src.models.wandb_utils import init_wandb, log_wandb


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mask_ratio: float = 0.2  # about 2 features are masked out on average
    device: str = "auto"
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_mae(
    x_lake: np.ndarray,
    *,
    mae_config: Optional[TabularMAEConfig] = None,
    train_config: Optional[TrainConfig] = None,
    wandb_run=None,
) -> TabularMAE:
    mae_config = mae_config or TabularMAEConfig()
    train_config = train_config or TrainConfig()

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    dev = _device(train_config.device)
    x = torch.from_numpy(x_lake).to(dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = TabularMAE(num_features=x.shape[1], config=mae_config).to(dev)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    model.train()
    global_step = 0
    for _epoch in range(train_config.epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(dev, non_blocking=True)
            loss = model.pretrain_loss(batch_x, mask_ratio=train_config.mask_ratio)
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
                    "train/lr": train_config.lr,
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

    return model


def save_checkpoint(
    path: Path,
    model: TabularMAE,
    *,
    mae_config: TabularMAEConfig,
    train_config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": model.num_features,
            "feature_names": list(BRINE_FEATURE_COLUMNS),
            "mae_config": mae_config.as_dict(),
            "train_config": train_config.as_dict(),
        },
        path,
    )


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default="models/mae_pretrained.pth",
    show_default=True,
)
@click.option("--epochs", type=int, default=50, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--mask-ratio", type=float, default=0.4, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--d-model", type=int, default=128, show_default=True)
@click.option("--n-heads", type=int, default=8, show_default=True)
@click.option("--n-layers", type=int, default=4, show_default=True)
@click.option("--mlp-ratio", type=float, default=4.0, show_default=True)
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
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    mask_ratio: float,
    device: str,
    seed: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mlp_ratio: float,
    wandb: bool,
    wandb_project: str,
    wandb_name: str | None,
    wandb_mode: str,
    wandb_tags: tuple[str, ...],
) -> None:
    processed = Path(processed_dir)
    x_lake_path = processed / "X_lake.npy"
    x_lake = np.load(x_lake_path)

    mae_config = TabularMAEConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        mask_ratio=mask_ratio,
    )
    train_config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        mask_ratio=mask_ratio,
        device=device,
        seed=seed,
    )

    run = init_wandb(
        enabled=wandb and wandb_mode.lower() != "disabled",
        project=wandb_project,
        run_name=wandb_name,
        mode=wandb_mode.lower(),
        config={
            "task": "mae_pretrain",
            "num_features": int(x_lake.shape[1]),
            "feature_names": list(BRINE_FEATURE_COLUMNS),
            **mae_config.as_dict(),
            **train_config.as_dict(),
        },
        tags=wandb_tags,
    )

    model = train_mae(
        x_lake,
        mae_config=mae_config,
        train_config=train_config,
        wandb_run=run,
    )
    save_checkpoint(
        Path(out_path), model, mae_config=mae_config, train_config=train_config
    )
    click.echo(f"Saved MAE checkpoint to: {out_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
