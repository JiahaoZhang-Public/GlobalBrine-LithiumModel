from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RegressionHeadConfig:
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    out_dim: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, config: RegressionHeadConfig | None = None):
        super().__init__()
        config = config or RegressionHeadConfig()
        self.config = config

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, config.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
