"""Reusable 3D convolutional building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv3D(nn.Module):
    """Two Conv3d + GroupNorm + SiLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        def _group_norm(ch: int) -> nn.GroupNorm:
            # Pick the largest divisor of ch that is <= 8 to satisfy GroupNorm requirement.
            groups = math.gcd(ch, 8) or 1
            return nn.GroupNorm(groups, ch)

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock3D(nn.Module):
    """Conv block followed by stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.down(skip), skip


class UpBlock3D(nn.Module):
    """Transpose-conv upsample followed by conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
