"""UNet backbone for latent diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..common.film import FiLM


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -(
            torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        )
        emb = timesteps.float()[:, None] * exponent.exp()[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DoubleConvCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        groups = math.gcd(out_ch, 8) or 1
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_mlp = (
            nn.Sequential(nn.Linear(time_dim, out_ch), nn.SiLU()) if time_dim > 0 else None
        )
        self.film = FiLM(cond_dim, out_ch) if cond_dim > 0 else None
        self.act = nn.SiLU(inplace=True)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor | None, cond: torch.Tensor | None
    ) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb)[..., None, None, None]
        if self.film is not None:
            h = self.film(h, cond)
        h = self.act(self.norm2(self.conv2(h)))
        if self.film is not None:
            h = self.film(h, cond)
        return h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.block = DoubleConvCond(in_ch, out_ch, time_dim, cond_dim)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor | None, cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x, t_emb, cond)
        return self.down(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = DoubleConvCond(out_ch + skip_ch, out_ch, time_dim, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = torch.nn.functional.pad(
                x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2]
            )
        x = torch.cat([skip, x], dim=1)
        return self.block(x, t_emb, cond)


class DiffusionUNet3D(nn.Module):
    def __init__(  # noqa: PLR0917
        self,
        latent_channels: int,
        base_channels: int,
        depth: int,
        time_dim: int,
        cond_dim: int,
        context_channels: int = 0,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.cond_dim = cond_dim
        self.context_channels = max(int(context_channels), 0)

        channels = [base_channels * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = latent_channels + self.context_channels
        for ch in channels:
            self.downs.append(DownBlock(prev, ch, time_dim, cond_dim))
            prev = ch

        self.bottleneck = DoubleConvCond(prev, prev * 2, time_dim, cond_dim)
        bottleneck_ch = prev * 2

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(UpBlock(bottleneck_ch, ch, ch, time_dim, cond_dim))
            bottleneck_ch = ch

        self.final = nn.Conv3d(bottleneck_ch, latent_channels, kernel_size=1)

    def forward(
        self,
        z_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        h = z_noisy
        if self.context_channels > 0:
            if context is None:
                raise ValueError("DiffusionUNet3D requires context input but received None.")
            if context.dim() != z_noisy.dim():
                raise ValueError(
                    "Context tensor must match latent tensor rank "
                    f"({z_noisy.dim()}), got {context.dim()}."
                )
            if context.size(1) != self.context_channels:
                raise ValueError(
                    f"Expected {self.context_channels} context channels, got {context.size(1)}."
                )
            if context.shape[-3:] != z_noisy.shape[-3:]:
                context = torch.nn.functional.interpolate(
                    context,
                    size=z_noisy.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            h = torch.cat([z_noisy, context], dim=1)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            h, skip = down(h, t_emb, cond)
            skips.append(skip)
        h = self.bottleneck(h, t_emb, cond)
        for up, skip in zip(self.ups, reversed(skips), strict=False):
            h = up(h, skip, t_emb, cond)
        return self.final(h)
