"""Base 3D UNet for voxel segmentation."""

from __future__ import annotations

import torch
from torch import nn

from ..common.blocks3d import DoubleConv3D, DownBlock3D, UpBlock3D


class UNet3D(nn.Module):
    """Vanilla 3D UNet backbone."""

    def __init__(
        self, in_channels: int, base_channels: int, depth: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth

        channels = [base_channels * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.downs.append(DownBlock3D(prev, ch))
            prev = ch

        self.bottleneck = DoubleConv3D(prev, prev * 2)
        bottleneck_ch = prev * 2

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(UpBlock3D(bottleneck_ch, ch, ch))
            bottleneck_ch = ch

        self.dropout = nn.Dropout3d(dropout)
        self.head = nn.Conv3d(bottleneck_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for down in self.downs:
            h, skip = down(h)
            skips.append(skip)
        h = self.bottleneck(h)
        h = self.dropout(h)
        for up, skip in zip(self.ups, reversed(skips), strict=False):
            h = up(h, skip)
        return self.head(h)
