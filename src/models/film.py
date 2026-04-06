"""Feature-wise Linear Modulation (FiLM) for conditioning on Light_kW_m2.

Light is an external experimental condition, not an intrinsic brine property.
Instead of feeding Light into the MAE encoder alongside chemistry features,
we use FiLM to modulate the encoder's latent representation:

    z_film = γ(light) ⊙ z + β(light)

where γ and β are learned affine functions of the standardized Light value.
This keeps the MAE's self-supervised pretraining clean (chemistry only) while
allowing Light to influence downstream predictions.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
Layer", AAAI 2018.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from src.models.regression_head import RegressionHead, RegressionHeadConfig


@dataclass(frozen=True)
class FiLMConfig:
    """Configuration for the FiLM conditioning layer."""

    cond_dim: int = 1  # Light is a scalar
    latent_dim: int = 128  # must match encoder d_model

    def as_dict(self) -> dict:
        return asdict(self)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: Light [1] → (γ [D], β [D]).

    Parameters: Linear(cond_dim, 2 * latent_dim) = ~257 params (for D=128).
    Initialized so that γ=1, β=0 (identity modulation) at start of training.
    """

    def __init__(self, config: FiLMConfig | None = None) -> None:
        super().__init__()
        config = config or FiLMConfig()
        self.config = config
        self.net = nn.Linear(config.cond_dim, 2 * config.latent_dim)

        # Identity initialization: γ=1, β=0 so z_film = z initially.
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.net.bias.data[: config.latent_dim] = 1.0

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: [batch, cond_dim] standardized Light value.

        Returns:
            gamma: [batch, latent_dim] multiplicative modulation.
            beta:  [batch, latent_dim] additive modulation.
        """
        out = self.net(cond)  # [batch, 2 * latent_dim]
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class FiLMRegressionHead(nn.Module):
    """Regression head with FiLM conditioning on Light.

    Composes FiLMLayer + RegressionHead:
        z_film = γ(light) ⊙ z + β(light)
        y = head(z_film)
    """

    def __init__(
        self,
        in_dim: int,
        head_config: RegressionHeadConfig | None = None,
        film_config: FiLMConfig | None = None,
    ) -> None:
        super().__init__()
        self.film = FiLMLayer(film_config)
        self.head = RegressionHead(in_dim=in_dim, config=head_config)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:    [batch, latent_dim] encoder output.
            cond: [batch, cond_dim]   standardized Light value.

        Returns:
            [batch, out_dim] predicted targets.
        """
        gamma, beta = self.film(cond)
        z_film = gamma * z + beta
        return self.head(z_film)
