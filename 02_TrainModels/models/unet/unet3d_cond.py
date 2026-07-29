"""Conditioned 3D UNet using FiLM adapters."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..common.film import FiLM


class FiLMDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        groups = math.gcd(out_ch, 8) or 1
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.film = FiLM(cond_dim, out_ch) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        if self.film is not None:
            h = self.film(h, cond)
        h = self.act(self.norm2(self.conv2(h)))
        if self.film is not None:
            h = self.film(h, cond)
        return h


class DownBlockCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = FiLMDoubleConv(in_ch, out_ch, cond_dim)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x, cond)
        return self.down(skip), skip


class UpBlockCond(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = FiLMDoubleConv(out_ch + skip_ch, out_ch, cond_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, cond)


class ConditionalUNet3D(nn.Module):
    """UNet variant with FiLM conditioning."""

    def __init__(
        self, in_channels: int, base_channels: int, depth: int, cond_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        channels = [base_channels * (2**i) for i in range(depth)]

        self.downs = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.downs.append(DownBlockCond(prev, ch, cond_dim))
            prev = ch

        self.bottleneck = FiLMDoubleConv(prev, prev * 2, cond_dim)
        bottleneck_ch = prev * 2

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(UpBlockCond(bottleneck_ch, ch, ch, cond_dim))
            bottleneck_ch = ch

        self.dropout = nn.Dropout3d(dropout)
        self.head = nn.Conv3d(bottleneck_ch, 1, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        skips = []
        h = x
        for down in self.downs:
            h, skip = down(h, cond)
            skips.append(skip)
        h = self.bottleneck(h, cond)
        h = self.dropout(h)
        for up, skip in zip(self.ups, reversed(skips), strict=False):
            h = up(h, skip, cond)
        return self.head(h)
