"""Feature-wise linear modulation (FiLM) layers."""

from __future__ import annotations

import torch
from torch import nn


class FiLM(nn.Module):
    """Simple FiLM adapter that modulates feature maps via conditioning vectors."""

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.cond_dim = cond_dim
        if cond_dim <= 0:
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, num_channels * 2),
                nn.SiLU(),
                nn.Linear(num_channels * 2, num_channels * 2),
            )

    def forward(self, features: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.mlp is None or cond is None or cond.numel() == 0:
            return features
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(gamma.shape[0], self.num_channels, 1, 1, 1)
        beta = beta.view(beta.shape[0], self.num_channels, 1, 1, 1)
        return features * (1 + gamma) + beta
