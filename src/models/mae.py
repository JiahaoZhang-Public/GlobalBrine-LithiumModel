from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class TabularMAEConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    mask_ratio: float = 0.4

    def as_dict(self) -> dict:
        return asdict(self)


def sample_feature_mask(
    batch_size: int,
    num_features: int,
    mask_ratio: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0, 1].")
    if num_features <= 0:
        raise ValueError("num_features must be > 0.")

    probs = torch.full((batch_size, num_features), mask_ratio, device=device)
    return torch.bernoulli(probs).to(dtype=torch.bool)


def build_effective_masks(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    min_visible: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_mask, loss_mask) following obs_mask vs mae_mask semantics.

    obs_mask: True where x has a real value (not NaN)
    mae_mask: True where we randomly mask an observed feature for MAE training

    input_mask = (~obs_mask) OR mae_mask
    loss_mask  = obs_mask AND mae_mask

    mask_ratio applies only over observed features (per-row), with at least
    min_visible observed features kept unmasked.
    """
    if x.ndim != 2:
        raise ValueError("Expected x with shape [batch, num_features].")
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0, 1].")

    if min_visible < 0:
        raise ValueError("min_visible must be >= 0.")

    obs_mask = ~torch.isnan(x)
    batch_size, _num_features = obs_mask.shape
    mae_mask = torch.zeros_like(obs_mask)

    for i in range(batch_size):
        eligible = torch.nonzero(obs_mask[i], as_tuple=False).squeeze(1)
        f_obs = int(eligible.numel())
        if f_obs == 0:
            continue

        max_mask = max(0, f_obs - min_visible)
        num_mask = int(math.floor(f_obs * mask_ratio))
        if mask_ratio > 0 and num_mask == 0 and max_mask > 0:
            num_mask = 1
        num_mask = min(num_mask, max_mask)
        if num_mask <= 0:
            continue

        perm = eligible[torch.randperm(f_obs, device=x.device)]
        mae_mask[i, perm[:num_mask]] = True

    input_mask = (~obs_mask) | mae_mask
    loss_mask = obs_mask & mae_mask
    return input_mask, loss_mask


class TabularMAE(nn.Module):
    def __init__(self, num_features: int, config: Optional[TabularMAEConfig] = None):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0.")

        self.num_features = num_features
        self.config = config or TabularMAEConfig()

        self.value_proj = nn.Linear(1, self.config.d_model)
        self.feature_embed = nn.Embedding(num_features, self.config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(self.config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=int(self.config.d_model * self.config.mlp_ratio),
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.n_layers
        )
        self.reconstruct_head = nn.Linear(self.config.d_model, 1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.feature_embed.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def _tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("Expected x with shape [batch, num_features].")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected num_features={self.num_features}, got {x.shape[1]}."
            )

        missing_mask = torch.isnan(x)
        x_filled = torch.where(missing_mask, torch.zeros_like(x), x)

        token = self.value_proj(x_filled.unsqueeze(-1))
        pos = torch.arange(self.num_features, device=x.device)
        token = token + self.feature_embed(pos).unsqueeze(0)
        return token, missing_mask

    def forward(
        self, x: torch.Tensor, *, feature_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        token, missing_mask = self._tokens(x)
        if feature_mask is None:
            feature_mask = missing_mask
        else:
            if feature_mask.shape != missing_mask.shape:
                raise ValueError("feature_mask must have same shape as x.")
            feature_mask = feature_mask | missing_mask

        mask_token = self.mask_token.view(1, 1, -1)
        token = torch.where(feature_mask.unsqueeze(-1), mask_token, token)

        encoded = self.encoder(token)
        pred = self.reconstruct_head(encoded).squeeze(-1)
        return pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        token, missing_mask = self._tokens(x)
        mask_token = self.mask_token.view(1, 1, -1)
        token = torch.where(missing_mask.unsqueeze(-1), mask_token, token)
        encoded = self.encoder(token)
        return encoded.mean(dim=1)

    def pretrain_loss(
        self, x: torch.Tensor, *, mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """MAE reconstruction loss on observed+masked positions."""
        token, missing_mask = self._tokens(x)
        ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio
        input_mask, loss_mask = build_effective_masks(x, mask_ratio=ratio)

        mask_token = self.mask_token.view(1, 1, -1)
        token = torch.where(input_mask.unsqueeze(-1), mask_token, token)
        encoded = self.encoder(token)
        pred = self.reconstruct_head(encoded).squeeze(-1)

        target = torch.where(missing_mask, pred.detach(), x)
        sq_err = (pred - target) ** 2
        denom = loss_mask.sum()
        if int(denom.item()) == 0:
            return pred.sum() * 0.0
        return (sq_err[loss_mask].sum() / denom).to(dtype=torch.float32)
