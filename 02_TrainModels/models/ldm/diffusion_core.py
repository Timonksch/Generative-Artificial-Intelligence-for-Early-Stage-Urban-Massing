"""Noise schedules and samplers for latent diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def make_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if kind == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    raise ValueError(f"Unknown beta schedule '{kind}'")


@dataclass
class DiffusionCoefficients:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor


def build_coefficients(beta_schedule: str, timesteps: int) -> DiffusionCoefficients:
    betas = make_beta_schedule(beta_schedule, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return DiffusionCoefficients(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
    )


def cfg_predicted_noise(  # noqa: PLR0917
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    cfg_scale: float,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    if cond is None or cfg_scale <= 1.0:
        return model(x_t, t, cond, context)
    eps_uncond = model(x_t, t, torch.zeros_like(cond), context)
    eps_cond = model(x_t, t, cond, context)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


class GaussianDiffusion:
    """Gaussian diffusion process with DDPM and DDIM sampling."""

    def __init__(self, timesteps: int, beta_schedule: str) -> None:
        self.timesteps = timesteps
        self.coefs = build_coefficients(beta_schedule, timesteps)

    def _extract(
        self, coeff: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        batch = t.shape[0]
        out = coeff.to(t.device).gather(0, t)
        return out.reshape(batch, *((1,) * (len(shape) - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas = self._extract(self.coefs.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.coefs.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alphas * x0 + sqrt_one_minus * noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, cond, context)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_ddpm(  # noqa: PLR0917
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor | None,
        cfg_scale: float,
        device: torch.device,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for idx in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), idx, device=device, dtype=torch.long)
            betas = self._extract(self.coefs.betas, t, shape)
            sqrt_one_minus = self._extract(self.coefs.sqrt_one_minus_alphas_cumprod, t, shape)
            sqrt_recip_alpha = self._extract(torch.sqrt(1.0 / self.coefs.alphas), t, shape)
            eps = cfg_predicted_noise(model, x, t, cond, cfg_scale, context=context)
            model_mean = sqrt_recip_alpha * (x - betas * eps / sqrt_one_minus)
            if idx == 0:
                x = model_mean
            else:
                posterior_var = self._extract(self.coefs.posterior_variance, t, shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var) * noise
        return x

    @torch.no_grad()
    def sample_ddim(  # noqa: PLR0917
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor | None,
        cfg_scale: float,
        device: torch.device,
        ddim_steps: int,
        eta: float,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ddim_steps = max(1, min(int(ddim_steps), int(self.timesteps)))
        ddim_seq = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=np.int64)

        x = torch.randn(shape, device=device)
        for idx in reversed(range(ddim_steps)):
            t_int = int(ddim_seq[idx])
            prev_t_int = int(ddim_seq[idx - 1]) if idx > 0 else -1
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            eps = cfg_predicted_noise(model, x, t, cond, cfg_scale, context=context)
            alpha_t = self._extract(self.coefs.alphas_cumprod, t, shape)
            if prev_t_int >= 0:
                prev_t = torch.full((shape[0],), prev_t_int, device=device, dtype=torch.long)
                alpha_prev = self._extract(self.coefs.alphas_cumprod, prev_t, shape)
            else:
                alpha_prev = torch.ones_like(alpha_t)
            x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            sigma_arg = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            sigma = eta * torch.sqrt(torch.clamp(sigma_arg, min=0.0))
            noise = torch.randn_like(x) if eta > 0 else 0.0

            dir_part = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * eps
            x = torch.sqrt(alpha_prev) * x0 + dir_part + sigma * noise
        return x
