"""Run diagnostic mini-training for all model families and save visual outputs."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dataio.voxel_dataset import VoxelDataset
from metrics import seg3d
from models.ldm.diffusion_core import GaussianDiffusion
from models.ldm.diffusion_unet3d import DiffusionUNet3D
from models.ldm.vae3d import VAE3D
from models.unet.losses import BCEDiceLoss
from models.unet.unet3d import UNet3D
from models.unet.unet3d_cond import ConditionalUNet3D
from utils.args import add_hyphenated_aliases
from utils.device import select_device, set_seed
from utils.io import ensure_dir, save_json
from utils.visuals import (
    save_3d_context_visualization,
    save_dual_projection,
    save_max_projection,
)

INPUT_CHANNELS = 4
CONDITION_DIMENSION = 3
LATENT_CHANNELS = 2
SEGMENTATION_BASE_CHANNELS = 4
VAE_BASE_CHANNELS = 4
DIFFUSION_BASE_CHANNELS = 4
DEFAULT_THRESHOLD = 0.5
MAX_3D_RENDER_SIZE = 8


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Store metrics and artifacts produced for one diagnostic model path."""

    model: str
    final_loss: float
    metrics: dict[str, float]
    prediction_npz: str
    projection_png: str
    dual_projection_png: str
    context_3d_png: str


def parse_args() -> argparse.Namespace:
    """Parse the diagnostic smoke-run options."""
    parser = argparse.ArgumentParser(
        description=(
            "Mini-train all three model families on one generated sample and save diagnostics."
        )
    )
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--downsample_stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    add_hyphenated_aliases(parser)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or unexpectedly expensive smoke configurations."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.downsample_stride <= 0:
        raise ValueError("--downsample-stride must be positive")
    if args.sample_index < 0:
        raise ValueError("--sample-index must not be negative")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between zero and one")


def _resolve_device(preference: str) -> torch.device:
    """Resolve a reliable CPU default or the best available accelerator."""
    if preference == "cpu":
        return torch.device("cpu")
    return select_device().device


def _load_real_batch(
    data_root: Path,
    sample_index: int,
    downsample_stride: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Load one deterministic model batch and its neighboring-building context."""
    dataset = VoxelDataset(
        root_dir=str(data_root),
        split="train",
        seed=42,
        augment=False,
        downsample_stride=downsample_stride,
        cond_dim=CONDITION_DIMENSION,
        auto_cond_stats=True,
    )
    if sample_index >= len(dataset):
        raise IndexError(f"--sample-index {sample_index} exceeds train split size {len(dataset)}")

    sample = dataset[sample_index]
    batch = {
        "voxels": sample["voxels"].unsqueeze(0),
        "target": sample["target"].unsqueeze(0),
        "cond": sample["cond"].unsqueeze(0),
        "path": Path(sample["path"]),
    }
    with np.load(batch["path"], allow_pickle=False) as archive:
        context = torch.from_numpy(archive["Y_neigh"].astype(np.float32, copy=False))
    context = F.max_pool3d(
        context[None, None],
        kernel_size=downsample_stride,
        stride=downsample_stride,
    )
    return batch, context


def _fit_segmentation_model(
    model: torch.nn.Module,
    voxels: torch.Tensor,
    target: torch.Tensor,
    cond: torch.Tensor | None,
    steps: int,
) -> tuple[torch.Tensor, float]:
    """Optimize a compact segmentation model for a bounded number of steps."""
    loss_function = BCEDiceLoss(0.5, 0.5, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    final_loss = 0.0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(voxels, cond) if cond is not None else model(voxels)
        loss = loss_function(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        logits = model(voxels, cond) if cond is not None else model(voxels)
    return logits, final_loss


def _fit_vae(
    target: torch.Tensor,
    steps: int,
) -> tuple[VAE3D, torch.Tensor, torch.Tensor, float]:
    """Optimize the compact VAE and return reconstruction logits and latent mean."""
    vae = VAE3D(
        in_channels=1,
        base_channels=VAE_BASE_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        depth=3,
    ).to(target.device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    final_loss = 0.0
    vae.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mean, log_variance = vae(target)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, target)
        kl_loss = torch.mean(-0.5 * (1 + log_variance - mean.square() - log_variance.exp()))
        loss = reconstruction_loss + 1e-4 * kl_loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    vae.eval()
    with torch.no_grad():
        reconstruction, mean, _ = vae(target)
    return vae, reconstruction, mean, final_loss


def _fit_and_sample_diffusion(
    vae: VAE3D,
    latent: torch.Tensor,
    voxels: torch.Tensor,
    cond: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, float]:
    """Optimize a compact latent denoiser and decode one DDIM sample."""
    diffusion_model = DiffusionUNet3D(
        latent_channels=latent.shape[1],
        base_channels=DIFFUSION_BASE_CHANNELS,
        depth=2,
        time_dim=32,
        cond_dim=CONDITION_DIMENSION,
        context_channels=voxels.shape[1],
    ).to(latent.device)
    diffusion = GaussianDiffusion(timesteps=10, beta_schedule="linear")
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    final_loss = 0.0
    diffusion_model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        timestep = torch.tensor([(step % 9) + 1], device=latent.device, dtype=torch.long)
        loss = diffusion.training_loss(
            diffusion_model,
            latent.detach(),
            timestep,
            cond,
            context=voxels,
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    diffusion_model.eval()
    sampled_latent = diffusion.sample_ddim(
        diffusion_model,
        shape=tuple(latent.shape),
        cond=cond,
        cfg_scale=1.0,
        device=latent.device,
        ddim_steps=5,
        eta=0.0,
        context=voxels,
    )
    with torch.no_grad():
        sample_logits = vae.decode(sampled_latent)
    return sample_logits, final_loss


def _save_prediction(  # noqa: PLR0917
    out_dir: Path,
    model_name: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    threshold: float,
    final_loss: float,
) -> SmokeResult:
    """Persist arrays, metrics, and matching 2D/3D prediction views."""
    model_dir = ensure_dir(out_dir / model_name)
    probability = torch.sigmoid(logits).detach().cpu()
    target_cpu = target.detach().cpu()
    context_cpu = context.detach().cpu()
    metrics = seg3d.evaluate(logits.detach().cpu(), target_cpu, threshold)

    prediction_path = model_dir / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        probability=probability.numpy(),
        binary=(probability >= threshold).numpy().astype(np.uint8),
        target=target_cpu.numpy().astype(np.uint8),
    )
    projection_path = save_max_projection(model_dir, model_name, target_cpu, probability, threshold)
    dual_path = save_dual_projection(model_dir, model_name, probability, threshold)
    render_probability = _reduce_for_3d(probability, use_max=False)
    render_context = _reduce_for_3d(context_cpu, use_max=True)
    context_path = save_3d_context_visualization(
        model_dir,
        model_name,
        render_probability,
        context=render_context,
        threshold=threshold,
    )
    return SmokeResult(
        model=model_name,
        final_loss=final_loss,
        metrics=metrics,
        prediction_npz=str(prediction_path),
        projection_png=str(projection_path),
        dual_projection_png=str(dual_path),
        context_3d_png=str(context_path),
    )


def _reduce_for_3d(volume: torch.Tensor, *, use_max: bool) -> torch.Tensor:
    """Bound diagnostic 3D rendering cost while preserving full saved arrays."""
    largest_dimension = max(volume.shape[-3:])
    factor = max(1, largest_dimension // MAX_3D_RENDER_SIZE)
    if factor == 1:
        return volume
    pool = F.max_pool3d if use_max else F.avg_pool3d
    return pool(volume, kernel_size=factor, stride=factor)


def main() -> int:
    """Run all diagnostic model paths and save a machine-readable summary."""
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed)
    device = _resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    started_at = time.perf_counter()

    batch, context = _load_real_batch(
        args.data_root,
        args.sample_index,
        args.downsample_stride,
    )
    voxels = batch["voxels"].to(device)
    target = batch["target"].to(device)
    cond = batch["cond"].to(device)
    context = context.to(device)

    reference_dir = ensure_dir(out_dir / "reference")
    reference_dual = save_dual_projection(reference_dir, "target", target.cpu(), args.threshold)
    render_target = _reduce_for_3d(target.cpu(), use_max=True)
    render_context = _reduce_for_3d(context.cpu(), use_max=True)
    reference_3d = save_3d_context_visualization(
        reference_dir,
        "target",
        render_target,
        context=render_context,
        threshold=args.threshold,
    )

    unet = UNet3D(
        in_channels=voxels.shape[1],
        base_channels=SEGMENTATION_BASE_CHANNELS,
        depth=3,
    ).to(device)
    unet_logits, unet_loss = _fit_segmentation_model(unet, voxels, target, None, args.steps)

    conditional_unet = ConditionalUNet3D(
        in_channels=voxels.shape[1],
        base_channels=SEGMENTATION_BASE_CHANNELS,
        depth=3,
        cond_dim=cond.shape[1],
    ).to(device)
    cond_logits, cond_loss = _fit_segmentation_model(
        conditional_unet, voxels, target, cond, args.steps
    )

    vae, vae_logits, latent, vae_loss = _fit_vae(target, args.steps)
    ldm_logits, diffusion_loss = _fit_and_sample_diffusion(vae, latent, voxels, cond, args.steps)

    results = [
        _save_prediction(
            out_dir,
            "unet",
            unet_logits,
            target,
            context,
            args.threshold,
            unet_loss,
        ),
        _save_prediction(
            out_dir,
            "conditional_unet",
            cond_logits,
            target,
            context,
            args.threshold,
            cond_loss,
        ),
        _save_prediction(
            out_dir,
            "vae_reconstruction",
            vae_logits,
            target,
            context,
            args.threshold,
            vae_loss,
        ),
        _save_prediction(
            out_dir,
            "latent_diffusion",
            ldm_logits,
            target,
            context,
            args.threshold,
            diffusion_loss,
        ),
    ]
    summary = {
        "purpose": "diagnostic mini-training; results are not thesis model predictions",
        "sample": str(batch["path"]),
        "device": str(device),
        "steps_per_model": args.steps,
        "downsample_stride": args.downsample_stride,
        "grid_shape": list(target.shape[-3:]),
        "render_3d_grid_shape": list(render_target.shape[-3:]),
        "threshold": args.threshold,
        "duration_seconds": time.perf_counter() - started_at,
        "conditioning": cond.detach().cpu().flatten().tolist(),
        "reference": {
            "dual_projection_png": str(reference_dual),
            "context_3d_png": str(reference_3d),
        },
        "results": [asdict(result) for result in results],
    }
    save_json(out_dir / "summary.json", summary)
    print(f"Saved diagnostic outputs to {out_dir}")
    for result in results:
        print(f"- {result.model}: loss={result.final_loss:.6f}, IoU={result.metrics['iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
