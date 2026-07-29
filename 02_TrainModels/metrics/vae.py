"""Metrics for VAE stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float
) -> dict[str, torch.Tensor]:
    """Return reconstruction loss, KL divergence, and total."""
    recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.numel()
    total = recon_loss + beta * kl
    return {"recon_loss": recon_loss, "kl_loss": kl, "loss": total}
