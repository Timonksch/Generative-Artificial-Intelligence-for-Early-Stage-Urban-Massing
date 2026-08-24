"""3D convolutional VAE used for latent diffusion."""

from __future__ import annotations

import torch
from torch import nn

from ..common.blocks3d import DoubleConv3D


class VAE3D(nn.Module):
    """Encoder/decoder for voxel volumes."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        latent_channels: int,
        depth: int,
    ) -> None:
        super().__init__()

        enc_channels = [base_channels * (2**i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for ch in enc_channels[:-1]:
            self.enc_blocks.append(
                nn.Sequential(
                    DoubleConv3D(prev, ch),
                    nn.Conv3d(ch, ch, kernel_size=2, stride=2),
                )
            )
            prev = ch

        self.enc_final = DoubleConv3D(prev, enc_channels[-1])
        self.fc_mu = nn.Conv3d(enc_channels[-1], latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv3d(enc_channels[-1], latent_channels, kernel_size=1)

        # Decoder
        dec_channels = list(reversed(enc_channels[:-1]))
        self.dec_initial = DoubleConv3D(latent_channels, enc_channels[-1])
        self.dec_blocks = nn.ModuleList()
        prev = enc_channels[-1]
        for ch in dec_channels:
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(prev, ch, kernel_size=2, stride=2),
                    DoubleConv3D(ch, ch),
                )
            )
            prev = ch
        self.dec_final = nn.Conv3d(prev, in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_final(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_initial(z)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_final(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
