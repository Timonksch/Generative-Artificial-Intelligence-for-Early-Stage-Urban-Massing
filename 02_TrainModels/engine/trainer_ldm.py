"""Trainer for 3D latent diffusion models."""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from ..metrics.vae import vae_loss as compute_vae_loss
    from ..models.ldm.diffusion_core import GaussianDiffusion
    from ..utils.device import DeviceConfig
    from ..utils.io import ensure_dir, save_json
    from ..utils.logging import RunLogger, configure_root_logger
    from ..utils.visuals import (
        save_3d_context_visualization,
        save_dual_projection,
        save_max_projection,
    )
except ImportError:
    from metrics.vae import vae_loss as compute_vae_loss  # type: ignore
    from models.ldm.diffusion_core import GaussianDiffusion  # type: ignore
    from utils.device import DeviceConfig  # type: ignore
    from utils.io import ensure_dir, save_json  # type: ignore
    from utils.logging import RunLogger, configure_root_logger  # type: ignore
    from utils.visuals import (  # type: ignore
        save_3d_context_visualization,
        save_dual_projection,
        save_max_projection,
    )

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


class EMA:
    """Simple exponential moving average wrapper."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, p in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = p.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


@dataclass
class LDMTrainOptions:
    epochs: int
    log_every: int
    save_every: int
    accum_steps: int
    grad_clip: float
    mode: str  # "vae_pretrain" | "diffusion" | "joint"
    kl_weight: float
    kl_anneal_epochs: int
    cfg_scale: float
    sampler: str
    sample_steps: int
    ddim_eta: float
    ema_decay: float
    vis_every: int
    cond_drop: float
    sample_every: int
    num_sample_batches: int
    num_sample_conds: int
    ema_sample_start: int
    latent_channels: int
    latent_res: int
    dual_view_samples: bool = False  # Use dual-view (heatmap + binary) for diffusion samples


def _convert_cond_stats(cond_stats: object | None) -> dict[str, np.ndarray] | None:
    if cond_stats is None:
        return None
    if hasattr(cond_stats, "mean") and hasattr(cond_stats, "std"):
        mean = np.asarray(cond_stats.mean, dtype=np.float32)
        std = np.asarray(cond_stats.std, dtype=np.float32)
    elif isinstance(cond_stats, dict):
        if "mean" not in cond_stats or "std" not in cond_stats:
            return None
        mean = np.asarray(cond_stats["mean"], dtype=np.float32)
        std = np.asarray(cond_stats["std"], dtype=np.float32)
    else:
        return None
    return {"mean": mean, "std": std}


def _format_cond_name(prefix: str, values: np.ndarray, cond_dim: int) -> str:
    if cond_dim == 2:
        return f"{prefix}_grz{values[0]:.2f}_h{values[1]:.0f}"
    if cond_dim == 3:
        return f"{prefix}_grz{values[0]:.2f}_gfz{values[1]:.2f}_h{values[2]:.0f}"
    if cond_dim == 1:
        return f"{prefix}_c{values[0]:.2f}"
    joined = "_".join(f"{v:.2f}" for v in values)
    return f"{prefix}_{joined}"


def _denorm_conditioning(cond_vec: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    if stats is None:
        return cond_vec
    std = stats["std"][: cond_vec.shape[0]]
    mean = stats["mean"][: cond_vec.shape[0]]
    return cond_vec * std + mean


def _format_cond_title(
    epoch: int,
    cond_name: str,
    cond_vec: np.ndarray,
    cond_stats: dict[str, np.ndarray] | None,
    cond_dim: int,
) -> str:
    values = _denorm_conditioning(cond_vec, cond_stats)
    if cond_dim == 3:
        return (
            f"Epoch {epoch} | {cond_name} | "
            f"GRZ={values[0]:.2f} GFZ={values[1]:.2f} H={values[2]:.1f}m"
        )
    if cond_dim == 2:
        return f"Epoch {epoch} | {cond_name} | GRZ={values[0]:.2f} H={values[1]:.1f}"
    if cond_dim == 1:
        return f"Epoch {epoch} | {cond_name} | C={values[0]:.2f}"
    return f"Epoch {epoch} | {cond_name}"


def _select_volume(channel_first_volume: np.ndarray) -> np.ndarray:
    vol = channel_first_volume
    if vol.ndim == 5:
        vol = vol[0]
    if vol.ndim == 4:
        vol = vol[0]
    return vol


def _max_projection_panels(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return volume.max(axis=0), volume.max(axis=1), volume.max(axis=2)


def sample_vae_reconstructions(  # noqa: PLR0917
    vae: torch.nn.Module,
    dataset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    num_samples: int = 5,
) -> None:
    if len(dataset) == 0:
        return

    epoch_dir = ensure_dir(out_dir / f"vae_recon_epoch_{epoch:03d}")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        volume = sample.get("target") if isinstance(sample, dict) else sample[0]
        if volume is None:
            continue
        if not torch.is_tensor(volume):
            volume = torch.as_tensor(volume)
        volume = volume.to(dtype=torch.float32)
        input_tensor = volume.unsqueeze(0).to(device)

        with torch.no_grad():
            recon, _, _ = vae(input_tensor)
            recon = torch.sigmoid(recon)

        original_np = volume.cpu().numpy()
        recon_np = recon.squeeze(0).cpu().numpy()

        original_vol = _select_volume(original_np)
        recon_vol = _select_volume(recon_np)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        views = ["XY (Top)", "XZ (Front)", "YZ (Side)"]
        original_projs = _max_projection_panels(original_vol)
        recon_projs = _max_projection_panels(recon_vol)

        # Original row with viridis heatmap
        for col, view in enumerate(views):
            axes[0, col].imshow(original_projs[col], cmap="viridis", origin="lower")
            axes[0, col].set_title(f"Original - {view}")
            axes[0, col].axis("off")

        # Reconstruction row with viridis heatmap
        for col, view in enumerate(views):
            axes[1, col].imshow(recon_projs[col], cmap="viridis", origin="lower")
            axes[1, col].set_title(f"Reconstruction - {view}")
            axes[1, col].axis("off")

        fig.suptitle(f"VAE Reconstruction - Epoch {epoch} - Sample {idx}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_path = epoch_dir / f"recon_sample_{idx}.png"
        plt.savefig(img_path, dpi=150)
        plt.close(fig)


def prepare_conditioning_samples(
    dataset,
    cond_dim: int,
    stats: dict[str, np.ndarray] | None,
    max_samples: int,
) -> list[tuple[np.ndarray, str]]:
    if cond_dim <= 0:
        return [(None, "unconditional")]

    if stats is None:
        mean = np.zeros(cond_dim, dtype=np.float32)
        std = np.ones(cond_dim, dtype=np.float32)
    else:
        mean = stats["mean"][:cond_dim]
        std = stats["std"][:cond_dim]

    samples: list[tuple[np.ndarray, str]] = []
    for level, scale in [("low", -1.0), ("medium", 0.0), ("high", 1.0)]:
        cond_norm = np.full((cond_dim,), scale, dtype=np.float32)
        cond_denorm = cond_norm * std + mean
        name = _format_cond_name(level, cond_denorm, cond_dim)
        samples.append((cond_norm, name))

    if len(dataset) > 0:
        candidate_indices = {0, len(dataset) // 2, len(dataset) - 1}
        for idx in candidate_indices:
            try:
                sample = dataset[idx]
                cond_tensor = sample["cond"] if isinstance(sample, dict) else sample[2]
                cond_np = cond_tensor.cpu().numpy()
                if cond_np.shape[0] > cond_dim:
                    cond_np = cond_np[:cond_dim]
                cond_denorm = cond_np * std + mean
                name = _format_cond_name("real", cond_denorm, cond_dim)
                samples.append((cond_np.astype(np.float32), name))
            except Exception as exc:  # Sample previews must not stop training.
                logging.getLogger("urban3d.trainer_ldm").warning(
                    "Could not read conditioning sample %s: %s", idx, exc
                )
                continue

    # Deduplicate and trim
    seen = set()
    unique_samples: list[tuple[np.ndarray, str]] = []
    for cond_norm, name in samples:
        if name in seen:
            continue
        seen.add(name)
        unique_samples.append((cond_norm, name))

    limit = max(1, max_samples)
    return unique_samples[:limit]


def prepare_context_samples(dataset, max_samples: int) -> list[torch.Tensor]:
    if len(dataset) == 0:
        return []
    candidate_indices = [0, len(dataset) // 2, len(dataset) - 1]
    contexts: list[torch.Tensor] = []
    seen = set()
    for idx in candidate_indices:
        if idx in seen:
            continue
        seen.add(idx)
        try:
            sample = dataset[idx]
            vox = sample["voxels"] if isinstance(sample, dict) else sample[0]
            if not torch.is_tensor(vox):
                vox = torch.as_tensor(vox)
            contexts.append(vox.detach().cpu().float())
        except Exception as exc:  # Sample previews must not stop training.
            logging.getLogger("urban3d.trainer_ldm").warning(
                "Could not read context sample %s: %s", idx, exc
            )
            continue
    if not contexts:
        return []
    return contexts[: max(1, max_samples)]


def _create_overview_image(
    epoch_dir: Path,
    epoch: int,
    conditioning_samples: Sequence[tuple[np.ndarray, str]],
    num_samples: int,
) -> None:
    n_cond = max(1, len(conditioning_samples))
    fig, axes = plt.subplots(n_cond, num_samples, figsize=(num_samples * 3, n_cond * 3))

    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_cond == 1 else axes.reshape(-1, 1)

    for cond_idx, (_, cond_name) in enumerate(conditioning_samples or [(None, "sample")]):
        for i in range(num_samples):
            sample_name = f"{cond_name}_sample{i:02d}"
            img_path = epoch_dir / f"{sample_name}_2d.png"
            ax = axes[cond_idx, i]
            if img_path.exists():
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(cond_name, fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(f"Epoch {epoch} - Generated Samples Overview", fontsize=14)
    plt.tight_layout()
    overview_path = epoch_dir / f"overview_epoch_{epoch:03d}.png"
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_diffusion_samples(  # noqa: PLR0917
    model: torch.nn.Module,
    vae: torch.nn.Module,
    diffusion: GaussianDiffusion,
    conditioning_samples: Sequence[tuple[np.ndarray, str]],
    context_samples: Sequence[torch.Tensor],
    device: torch.device,
    options: LDMTrainOptions,
    epoch: int,
    out_dir: Path,
    cond_stats: dict[str, np.ndarray] | None,
    cond_dim: int,
) -> None:
    if not conditioning_samples:
        conditioning_samples = [(None, "sample")]

    samples_root = ensure_dir(out_dir)
    epoch_dir = ensure_dir(samples_root / f"samples_epoch_{epoch:03d}")

    num_samples = max(1, options.num_sample_batches)

    for cond_idx, (cond_vector, cond_name) in enumerate(conditioning_samples):
        if cond_vector is not None and cond_dim > 0:
            cond_tensor = torch.from_numpy(cond_vector).to(device=device, dtype=torch.float32)
            cond_tensor = cond_tensor.unsqueeze(0).repeat(num_samples, 1)
        else:
            cond_tensor = None

        context_batch = None
        if context_samples:
            ctx = context_samples[cond_idx % len(context_samples)].to(
                device=device, dtype=torch.float32
            )
            if ctx.dim() == 4:
                context_batch = ctx.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
            elif ctx.dim() == 5:
                context_batch = ctx.repeat(num_samples, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unexpected context sample shape: {tuple(ctx.shape)}")

        latent_shape = (
            num_samples,
            options.latent_channels,
            options.latent_res,
            options.latent_res,
            options.latent_res,
        )

        if options.sampler == "ddim":
            z_samples = diffusion.sample_ddim(
                model,
                latent_shape,
                cond_tensor,
                cfg_scale=options.cfg_scale,
                device=device,
                ddim_steps=options.sample_steps,
                eta=options.ddim_eta,
                context=context_batch,
            )
        else:
            z_samples = diffusion.sample_ddpm(
                model,
                latent_shape,
                cond_tensor,
                cfg_scale=options.cfg_scale,
                device=device,
                context=context_batch,
            )

        with torch.no_grad():
            voxel_samples = torch.sigmoid(vae.decode(z_samples)).cpu().numpy()

        for idx in range(num_samples):
            sample_name = f"{cond_name}_sample{idx:02d}"
            volume = _select_volume(voxel_samples[idx])

            # Generate title for visualization
            if cond_vector is not None and cond_dim > 0:
                figure_title = _format_cond_title(
                    epoch, cond_name, cond_vector, cond_stats, cond_dim
                )
            else:
                figure_title = f"Epoch {epoch} | {sample_name}"

            # Choose visualization type based on options
            if options.dual_view_samples:
                # Dual-view: heatmap + binary threshold
                volume_tensor = torch.from_numpy(volume)
                save_dual_projection(
                    out_dir=epoch_dir,
                    name=sample_name,
                    volume=volume_tensor,
                    threshold=0.5,
                )
            else:
                # Standard heatmap-only view
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                projections = _max_projection_panels(volume)
                titles = ["XY (Top)", "XZ (Front)", "YZ (Side)"]

                # Create heatmap visualizations with viridis colormap
                for col in range(3):
                    axes[col].imshow(projections[col], cmap="viridis", origin="lower")
                    axes[col].set_title(titles[col])
                    axes[col].axis("off")

                fig.suptitle(figure_title, fontsize=14)
                plt.tight_layout()
                plt.savefig(epoch_dir / f"{sample_name}_2d.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    _create_overview_image(epoch_dir, epoch, conditioning_samples, num_samples)


class LDMTrainer:
    def __init__(  # noqa: PLR0917
        self,
        vae: torch.nn.Module,
        diff_unet: torch.nn.Module,
        diffusion: GaussianDiffusion,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device_cfg: DeviceConfig,
        run_dir: Path,
        logger: RunLogger,
        options: LDMTrainOptions,
        cond_dim: int,
        ema_enabled: bool,
    ) -> None:
        self.vae = vae
        self.diff_unet = diff_unet
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device_cfg = device_cfg
        self.run_dir = ensure_dir(run_dir)
        self.logger = logger
        self.options = options
        self.cond_dim = cond_dim
        self.cond_drop = max(options.cond_drop, 0.0)
        self._progress_last: dict[str, int] = {}
        self._last_epoch_summary: str = ""
        self.sample_every = max(options.sample_every, 0)
        self.num_sample_batches = max(options.num_sample_batches, 1)
        self.num_sample_conds = max(options.num_sample_conds, 1)
        self.ema_sample_start = max(options.ema_sample_start, 0)
        self.latent_channels = options.latent_channels
        self.latent_res = options.latent_res
        self.best_loss = float("inf")
        self.best_metrics: dict[str, float] = {}

        self.vis_dir = ensure_dir(self.run_dir / "vis")

        if not logging.getLogger("urban3d").handlers:
            configure_root_logger()
        self.console = logging.getLogger("urban3d").getChild("trainer.ldm")

        self.device = device_cfg.device
        self.vae.to(self.device)
        self.diff_unet.to(self.device)
        self.use_amp = device_cfg.device_type == "cuda" and device_cfg.amp_dtype is not None
        self.scaler = GradScaler(enabled=self.use_amp)

        self.ema = EMA(self.diff_unet, options.ema_decay) if ema_enabled else None
        self.console.info(
            "Initialized LDM trainer | mode=%s | device=%s | amp=%s | ema=%s | vis_every=%d",
            self.options.mode,
            self.device.type,
            self.use_amp,
            self.ema is not None,
            self.options.vis_every,
        )

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0
    ) -> dict[str, float]:
        total_epochs = self.options.epochs
        if start_epoch > 0:
            self.console.info(
                "Resuming LDM training from epoch %d/%d",
                start_epoch,
                total_epochs,
            )
        cond_stats_dict = _convert_cond_stats(getattr(train_loader.dataset, "cond_stats", None))
        conditioning_samples: list[tuple[np.ndarray, str]] | None = None
        context_samples: list[torch.Tensor] = []
        if self.sample_every > 0 and self.options.mode in {"diffusion", "joint"}:
            conditioning_samples = prepare_conditioning_samples(
                train_loader.dataset,
                self.cond_dim,
                cond_stats_dict,
                self.num_sample_conds,
            )
            context_samples = prepare_context_samples(
                train_loader.dataset,
                self.num_sample_conds,
            )
            self.console.info(
                "Prepared %d conditioning vectors and %d context samples for sampling",
                len(conditioning_samples),
                len(context_samples),
            )

        self.console.info(
            "Starting LDM training for %d epochs | train_batches=%d | val_batches=%d",
            total_epochs,
            len(train_loader),
            len(val_loader),
        )

        for epoch in range(start_epoch, total_epochs):
            self.console.info("Epoch %d/%d", epoch + 1, total_epochs)
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._eval_epoch(val_loader, epoch)

            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.best_metrics = val_metrics
                self._save_checkpoint(epoch, is_best=True)
                self.console.info(
                    "New best validation loss %.6f at epoch %d | checkpoint updated",
                    self.best_loss,
                    epoch + 1,
                )
            elif self.options.save_every and (epoch + 1) % self.options.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)
                self.console.info("Saved periodic checkpoint for epoch %d", epoch + 1)

            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
            save_json(self.run_dir / "ldm_epoch.json", record)
            summary_msg = _build_ldm_epoch_summary(
                epoch + 1, train_metrics, val_metrics, self.options.mode
            )
            if summary_msg != self._last_epoch_summary:
                self.console.info(summary_msg)
                self._last_epoch_summary = summary_msg

            if self.sample_every > 0 and (epoch + 1) % self.sample_every == 0:
                if self.options.mode in {"vae_pretrain", "joint"}:
                    try:
                        sample_dir = ensure_dir(self.run_dir / "vae_samples")
                        sample_vae_reconstructions(
                            self.vae,
                            val_loader.dataset,
                            self.device,
                            epoch + 1,
                            sample_dir,
                        )
                        self.console.info("Saved VAE reconstructions for epoch %d", epoch + 1)
                    except Exception as exc:  # pragma: no cover - visual export best-effort
                        self.console.warning("VAE sampling failed: %s", exc)
                if self.options.mode in {"diffusion", "joint"}:
                    try:
                        sample_model = self.diff_unet
                        if self.ema is not None and (epoch + 1) >= self.ema_sample_start:
                            sample_model = deepcopy(self.diff_unet).eval()
                            self.ema.copy_to(sample_model)
                            self.console.info("Sampling with EMA weights at epoch %d", epoch + 1)
                        self.console.info(
                            "Generating diffusion samples | epoch=%d | conds=%d | per_cond=%d",
                            epoch + 1,
                            len(conditioning_samples or []),
                            self.num_sample_batches,
                        )
                        samples_dir = ensure_dir(self.run_dir / "samples")
                        generate_diffusion_samples(
                            sample_model,
                            self.vae,
                            self.diffusion,
                            conditioning_samples or [(None, "unconditional")],
                            context_samples,
                            self.device,
                            self.options,
                            epoch + 1,
                            samples_dir,
                            cond_stats_dict,
                            self.cond_dim,
                        )
                    except Exception as exc:  # pragma: no cover - visual export best-effort
                        self.console.warning("Diffusion sampling failed: %s", exc)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

        save_json(self.run_dir / "final_metrics.json", self.best_metrics)
        self.logger.close()
        if self.best_metrics:
            summary = _build_ldm_epoch_summary(
                -1,
                self.best_metrics,
                self.best_metrics,
                self.options.mode,
            )
            self.console.info("Training complete | %s", summary)
        return self.best_metrics

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.vae.train(self.options.mode in {"vae_pretrain", "joint"})
        self.diff_unet.train(self.options.mode in {"diffusion", "joint"})
        total_vae_loss = 0.0
        total_diff_loss = 0.0
        total_loss = 0.0
        batches = len(loader)
        grad_norm_sum = 0.0
        grad_norm_count = 0
        last_grad_norm: float | None = None
        kl_beta_value = self._kl_beta(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()

        for step, batch in enumerate(loader):
            targets = batch["target"].to(self.device, non_blocking=True).float()
            cond = batch["cond"].to(self.device, non_blocking=True) if self.cond_dim > 0 else None
            context = batch["voxels"].to(self.device, non_blocking=True).float()

            amp_ctx = (
                autocast(device_type="cuda", dtype=self.device_cfg.amp_dtype, enabled=True)
                if self.use_amp
                else nullcontext()
            )
            with amp_ctx:
                loss = torch.zeros((), device=self.device)
                vae_loss = None
                diff_loss = None
                if self.options.mode in {"vae_pretrain", "joint"}:
                    recon, mu, logvar = self.vae(targets)
                    loss_dict = compute_vae_loss(recon, targets, mu, logvar, kl_beta_value)
                    vae_loss = loss_dict["loss"]
                    total_vae_loss += vae_loss.item()
                    loss = loss + vae_loss
                else:
                    with torch.no_grad():
                        self.vae.eval()
                        mu, logvar = self.vae.encode(targets)
                        z = self.vae.reparameterize(mu, logvar)

                if self.options.mode in {"diffusion", "joint"}:
                    if self.options.mode == "joint":
                        mu, logvar = self.vae.encode(targets)
                        z = self.vae.reparameterize(mu, logvar)
                    if cond is not None and cond.numel() > 0 and self.cond_drop > 0.0:
                        mask = (torch.rand_like(cond[:, :1]) >= self.cond_drop).float()
                        cond = cond * mask
                    diff_loss = self.diffusion.training_loss(
                        self.diff_unet,
                        z,
                        self._sample_timesteps(z.size(0)),
                        cond,
                        context=context,
                    )
                    total_diff_loss += diff_loss.item()
                    loss = loss + diff_loss

                loss = loss / self.options.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm_value: float | None = None
            if (step + 1) % self.options.accum_steps == 0:
                trainable_params = list(self._trainable_parameters())
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                grad_norm_value = self._grad_norm_from_params(trainable_params)
                if grad_norm_value is not None:
                    grad_norm_sum += grad_norm_value
                    grad_norm_count += 1
                    last_grad_norm = grad_norm_value
                if self.options.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.options.grad_clip)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None and self.options.mode in {"diffusion", "joint"}:
                    self.ema.update(self.diff_unet)

            combined_value = loss.detach().item() * self.options.accum_steps
            total_loss += combined_value

            elapsed = max(time.perf_counter() - epoch_start, 1e-6)
            processed_batches = step + 1
            iters_per_sec = processed_batches / elapsed
            remaining = max(batches - processed_batches, 0)
            eta_seconds = remaining / iters_per_sec if iters_per_sec > 0 else float("inf")

            global_step = epoch * batches + step
            if global_step % self.options.log_every == 0:
                scalars = {
                    "train_loss": combined_value,
                    "train_lr": self.optimizer.param_groups[0]["lr"]
                    if self.optimizer.param_groups
                    else 0.0,
                    "train_kl_beta": kl_beta_value,
                    "train_iter_per_sec": iters_per_sec,
                    "train_eta_sec": eta_seconds,
                }
                if vae_loss is not None:
                    scalars["train_vae_loss"] = vae_loss.detach().item()
                if diff_loss is not None:
                    scalars["train_diff_loss"] = diff_loss.detach().item()
                if last_grad_norm is not None:
                    scalars["train_grad_norm"] = last_grad_norm
                self.logger.log_scalars(global_step, scalars)

            progress_metrics = {
                "loss": combined_value,
                "it_s": iters_per_sec,
                "eta_s": eta_seconds,
            }
            if vae_loss is not None:
                progress_metrics["vae"] = vae_loss.detach().item()
            if diff_loss is not None:
                progress_metrics["diff"] = diff_loss.detach().item()
            self._progress("train", step + 1, batches, progress_metrics, done=(step + 1) == batches)

        # Flush gradients for trailing mini-batches when batches % accum_steps != 0.
        if batches and (batches % self.options.accum_steps) != 0:
            trainable_params = list(self._trainable_parameters())
            grad_norm_value: float | None = None
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm_value = self._grad_norm_from_params(trainable_params)
            if grad_norm_value is not None:
                grad_norm_sum += grad_norm_value
                grad_norm_count += 1
                last_grad_norm = grad_norm_value
            if self.options.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, self.options.grad_clip)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None and self.options.mode in {"diffusion", "joint"}:
                self.ema.update(self.diff_unet)

        avg_vae = total_vae_loss / max(batches, 1) if total_vae_loss else 0.0
        avg_diff = total_diff_loss / max(batches, 1) if total_diff_loss else 0.0
        avg_total = (total_loss / max(batches, 1)) if total_loss else (avg_vae + avg_diff)
        epoch_time = max(time.perf_counter() - epoch_start, 1e-6)
        summary: dict[str, float] = {
            "loss": avg_total,
            "vae_loss": avg_vae,
            "diff_loss": avg_diff,
            "kl_beta": kl_beta_value,
            "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0,
            "iter_per_sec": (batches / epoch_time) if batches else 0.0,
            "epoch_time_sec": epoch_time,
        }
        if grad_norm_count:
            summary["grad_norm"] = grad_norm_sum / grad_norm_count
        return summary

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.vae.eval()
        self.diff_unet.eval()

        total_vae_loss = 0.0
        total_diff_loss = 0.0
        batches = len(loader)
        capture_visual = self.options.vis_every and (epoch + 1) % self.options.vis_every == 0
        vis_sample = None
        vis_context: torch.Tensor | None = None

        for idx, batch in enumerate(loader):
            targets = batch["target"].to(self.device, non_blocking=True).float()
            cond = batch["cond"].to(self.device, non_blocking=True) if self.cond_dim > 0 else None
            context = batch["voxels"].to(self.device, non_blocking=True).float()

            if self.options.mode in {"vae_pretrain", "joint"}:
                recon, mu, logvar = self.vae(targets)
                beta = self._kl_beta(epoch)
                loss_dict = compute_vae_loss(recon, targets, mu, logvar, beta)
                total_vae_loss += loss_dict["loss"].item()
                if capture_visual and vis_sample is None:
                    vis_sample = (recon.detach().cpu(), targets.detach().cpu())

            if self.options.mode in {"diffusion", "joint"}:
                mu, logvar = self.vae.encode(targets)
                # Deterministic validation latent to reduce checkpoint-selection noise.
                z = mu
                diff_loss = self.diffusion.training_loss(
                    self.diff_unet,
                    z,
                    self._sample_timesteps(z.size(0)),
                    cond,
                    context=context,
                )
                total_diff_loss += diff_loss.item()
                if capture_visual and vis_sample is None:
                    with torch.no_grad():
                        decoded = self.vae.decode(mu)
                    vis_sample = (decoded.detach().cpu(), targets.detach().cpu())

            if capture_visual and vis_context is None:
                vis_context = self._extract_context(batch)

            running_loss = (total_vae_loss + total_diff_loss) / max(idx + 1, 1)
            current_metrics = {"loss": running_loss}
            if total_vae_loss:
                current_metrics["vae"] = total_vae_loss / max(idx + 1, 1)
            if total_diff_loss:
                current_metrics["diff"] = total_diff_loss / max(idx + 1, 1)
            self._progress("val", idx + 1, batches, current_metrics, done=(idx + 1) == batches)

        avg_vae = total_vae_loss / max(batches, 1) if total_vae_loss else 0.0
        avg_diff = total_diff_loss / max(batches, 1) if total_diff_loss else 0.0
        total = avg_vae + avg_diff
        summary_metrics = {"loss": total}
        if avg_vae:
            summary_metrics["vae_loss"] = avg_vae
        if avg_diff:
            summary_metrics["diff_loss"] = avg_diff
        log_payload = {f"val_{k}": v for k, v in summary_metrics.items()}
        self.logger.log_scalars(epoch, log_payload)

        if capture_visual and vis_sample is not None:
            pred, target = vis_sample
            pred = torch.sigmoid(pred[0:1])
            target = target[0:1]
            name = f"epoch_{epoch + 1:04d}_sample0"
            thresh = 0.5
            pred_cpu = pred.detach().cpu()
            target_cpu = target.detach().cpu()
            save_max_projection(self.vis_dir, name, target_cpu, pred_cpu, thresh)
            self.console.info(
                "Saved visualization -> %s", (self.vis_dir / f"{name}_projections.png")
            )

            if vis_context is not None:
                context_cpu = vis_context.detach().cpu()
                stride = self._compute_context_stride(pred_cpu, context_cpu)
                save_3d_context_visualization(
                    out_dir=self.vis_dir,
                    name=name,
                    prediction=pred_cpu,
                    context=context_cpu,
                    threshold=thresh,
                    context_stride=stride,
                )
                self.console.info(
                    "Saved 3D context visualization -> %s",
                    (self.vis_dir / f"{name}_3d_context.png"),
                )

        return {"vae_loss": avg_vae, "diff_loss": avg_diff, "loss": total}

    def _progress(
        self,
        phase: str,
        current: int,
        total: int,
        metrics: dict[str, float],
        done: bool = False,
    ) -> None:
        total = max(total, 1)
        current = max(min(current, total), 0)
        percent = (current / total) * 100.0

        def _format_value(value: object) -> str:
            if isinstance(value, (int, float)):
                if math.isfinite(value):
                    return f"{value:.4f}"
                return "nan"
            return str(value)

        metrics_str = " | ".join(f"{k}={_format_value(v)}" for k, v in metrics.items())
        line = f"\r[{phase}] {current:>4}/{total:<4} ({percent:5.1f}%) {metrics_str}"
        prev_len = self._progress_last.get(phase, 0)
        padding = max(prev_len - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        if done or current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_last.pop(phase, None)
        else:
            self._progress_last[phase] = len(line)

    def _extract_context(
        self, batch: dict[str, torch.Tensor], index: int = 0
    ) -> torch.Tensor | None:
        paths = batch.get("path")
        if paths is None:
            return None
        if isinstance(paths, (list, tuple)):
            if not paths:
                return None
            sample_path = paths[index]
        else:
            sample_path = paths
        if isinstance(sample_path, bytes):
            sample_path = sample_path.decode()
        return self._load_context_volume(Path(sample_path))

    def _load_context_volume(self, sample_path: Path) -> torch.Tensor | None:
        try:
            with np.load(sample_path, allow_pickle=True) as data:
                if "Y_neigh" in data:
                    ctx = data["Y_neigh"]
                elif "context" in data:
                    ctx = data["context"]
                else:
                    return None
                return torch.from_numpy(ctx).float()
        except Exception as exc:
            self.console.debug("Context visualization skipped for %s: %s", sample_path, exc)
        return None

    @staticmethod
    def _compute_context_stride(prediction: torch.Tensor, context: torch.Tensor) -> int:
        def _shape(t: torch.Tensor) -> torch.Size:
            vol = t
            while vol.dim() > 3 and vol.size(0) == 1:
                vol = vol.squeeze(0)
            if vol.dim() > 3:
                vol = vol[0]
            return vol.shape[-3:]

        pred_shape = _shape(prediction)
        ctx_shape = _shape(context)
        ratios = []
        for ctx_dim, pred_dim in zip(ctx_shape, pred_shape, strict=False):
            if pred_dim > 0:
                ratios.append(max(1, round(ctx_dim / pred_dim)))
        if not ratios:
            return 1
        return max(1, round(sum(ratios) / len(ratios)))

    def _kl_beta(self, epoch: int) -> float:
        if self.options.kl_anneal_epochs <= 0:
            return self.options.kl_weight
        return min(1.0, (epoch + 1) / self.options.kl_anneal_epochs) * self.options.kl_weight

    def _sample_timesteps(self, batch: int) -> torch.Tensor:
        return torch.randint(
            0, self.diffusion.timesteps, (batch,), device=self.device, dtype=torch.long
        )

    def _trainable_parameters(self):
        for module in (self.vae, self.diff_unet):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    @staticmethod
    def _grad_norm_from_params(params: Sequence[torch.nn.Parameter]) -> float | None:
        total = 0.0
        has_grad = False
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                return float("nan")
            total += grad.pow(2).sum().item()
            has_grad = True
        if not has_grad:
            return None
        return math.sqrt(total)

    def load_checkpoint(self, state: dict[str, object]) -> int:
        epoch = int(state.get("epoch", -1))

        vae_state = state.get("vae")
        if vae_state:
            self.vae.load_state_dict(vae_state)  # type: ignore[arg-type]

        diff_state = state.get("diff_unet")
        if diff_state:
            self.diff_unet.load_state_dict(diff_state)  # type: ignore[arg-type]

        optimizer_state = state.get("optimizer")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)  # type: ignore[arg-type]

        scaler_state = state.get("scaler")
        if scaler_state and self.use_amp:
            self.scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]

        ema_state = state.get("ema")
        if self.ema is not None and isinstance(ema_state, dict):
            self.ema.shadow = {
                k: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
                for k, tensor in ema_state.items()
            }

        self.best_loss = float(state.get("best_loss", self.best_loss))
        best_metrics = state.get("best_metrics")
        if isinstance(best_metrics, dict):
            self.best_metrics = best_metrics

        self.console.info(
            "Restored checkpoint | epoch=%d | best_loss=%.6f",
            epoch,
            self.best_loss,
        )
        return max(epoch + 1, 0)

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "vae": self.vae.state_dict(),
            "diff_unet": self.diff_unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "best_loss": self.best_loss,
            "best_metrics": self.best_metrics,
            "ema": {k: v.detach().cpu() for k, v in self.ema.shadow.items()}
            if self.ema is not None
            else None,
        }
        ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(state, self.run_dir / "best.pt")


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _build_ldm_epoch_summary(
    epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float], mode: str
) -> str:
    train_loss = train_metrics.get("loss", 0.0)
    train_vae = train_metrics.get("vae_loss")
    train_diff = train_metrics.get("diff_loss")

    val_loss = val_metrics.get("loss", 0.0)
    val_vae = val_metrics.get("vae_loss")
    val_diff = val_metrics.get("diff_loss")

    parts = [f"{BOLD}Epoch {epoch}{RESET}", f"{CYAN}train_loss={train_loss:.4f}{RESET}"]
    if train_vae is not None:
        parts.append(f"vae={train_vae:.4f}")
    if train_diff is not None:
        parts.append(f"diff={train_diff:.4f}")
    parts.append(f"{MAGENTA}val_loss={val_loss:.4f}{RESET}")
    if val_vae is not None:
        parts.append(f"vae={val_vae:.4f}")
    if val_diff is not None:
        parts.append(f"diff={val_diff:.4f}")
    parts.append(f"mode={mode}")
    return " | ".join(parts)
