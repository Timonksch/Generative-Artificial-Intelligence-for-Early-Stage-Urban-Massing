"""Unified inference script for UNet, conditional UNet, and LDM models."""

from __future__ import annotations

import argparse
import math
import os
import sys
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataio.cond_stats import COND_METRIC_KEYS, load_cond_stats  # noqa: E402
from dataio.voxel_dataset import VoxelDataset, voxel_collate  # noqa: E402
from metrics import seg3d as seg_metrics  # noqa: E402
from models.ldm.diffusion_core import GaussianDiffusion  # noqa: E402
from models.ldm.diffusion_unet3d import DiffusionUNet3D  # noqa: E402
from models.ldm.vae3d import VAE3D  # noqa: E402
from models.unet.unet3d import UNet3D  # noqa: E402
from models.unet.unet3d_cond import ConditionalUNet3D  # noqa: E402
from utils.args import add_hyphenated_aliases  # noqa: E402
from utils.device import DeviceConfig, select_device, set_seed, worker_init_fn  # noqa: E402
from utils.io import ensure_dir, load_json, save_json  # noqa: E402
from utils.visuals import save_3d_context_visualization, save_max_projection  # noqa: E402


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    candidates = [checkpoint_path.parent, checkpoint_path.parent.parent]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_run_config(checkpoint: Path, explicit: str | None) -> dict[str, object]:
    if explicit:
        return load_json(explicit)
    cfg_path = _resolve_run_dir(checkpoint) / "config.json"
    if cfg_path.exists():
        return load_json(cfg_path)
    return {}


def infer_target_channels(dataset: VoxelDataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer target channels.")
    sample = dataset[0]
    target = sample.get("target") if isinstance(sample, dict) else sample[1]
    if target is None:
        raise ValueError("Dataset sample does not contain 'target'.")
    target_tensor = target if torch.is_tensor(target) else torch.as_tensor(target)
    if target_tensor.dim() < 4:
        raise ValueError(
            f"Unexpected target shape {tuple(target_tensor.shape)}; expected (C,D,H,W)."
        )
    return int(target_tensor.shape[0])


def load_state_dict_strict(
    module: torch.nn.Module, state_dict: dict, component: str, checkpoint: Path
) -> None:
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Strict checkpoint load failed for '{component}' from '{checkpoint}': {exc}"
        ) from exc


def select_ldm_diffusion_weights(state: dict, checkpoint: Path) -> tuple[dict, str]:
    ema_state = state.get("ema")
    if isinstance(ema_state, dict) and ema_state:
        return ema_state, "ema"

    diff_state = state.get("diff_unet")
    if isinstance(diff_state, dict) and diff_state:
        return diff_state, "diff_unet"

    raise ValueError(
        f"Checkpoint '{checkpoint}' does not contain usable diffusion weights "
        "(expected non-empty 'ema' or 'diff_unet')."
    )


def resolve_threshold_from_artifacts(checkpoint_path: Path, default: float = 0.5) -> float:
    run_dir = _resolve_run_dir(checkpoint_path)
    candidates = [
        run_dir / "final_metrics.json",
        run_dir / "eval_val" / "metrics.json",
    ]
    for metrics_path in candidates:
        if not metrics_path.exists():
            continue
        metrics_doc = load_json(metrics_path)
        if isinstance(metrics_doc, dict) and "threshold" in metrics_doc:
            return float(metrics_doc["threshold"])
    return float(default)


def build_dataset(  # noqa: PLR0917
    split: str,
    cfg: dict[str, object],
    data_root: str,
    cond_dim: int,
    cond_stats_path: str | None,
    max_samples: int | None,
    split_manifest: str | None,
):
    stats_path = cond_stats_path
    if isinstance(stats_path, Path):
        stats_path = str(stats_path)
    if not stats_path:
        stats_path = None
    cond_stats = load_cond_stats(stats_path) if stats_path else None
    if cond_dim > 0 and cond_stats is None:
        stats_ds = VoxelDataset(
            root_dir=data_root,
            split="train",
            channels=cfg.get("channels", ("C0", "C1", "C2", "C3")),
            x_indices=cfg.get("x_indices"),
            add_coords=cfg.get("add_coords", False),
            downsample_stride=cfg.get("downsample_stride", 0),
            crop_size=cfg.get("crop_size", 0),
            cond_dim=cond_dim,
            cond_select=cfg.get("cond_select"),
            cond_stats=None,
            auto_cond_stats=True,
            augment=False,
            max_samples=None,
            split_manifest=split_manifest,
        )
        cond_stats = stats_ds.cond_stats
    dataset = VoxelDataset(
        root_dir=data_root,
        split=split,
        channels=cfg.get("channels", ("C0", "C1", "C2", "C3")),
        x_indices=cfg.get("x_indices"),
        add_coords=cfg.get("add_coords", False),
        downsample_stride=cfg.get("downsample_stride", 0),
        crop_size=cfg.get("crop_size", 0),
        cond_dim=cond_dim,
        cond_select=cfg.get("cond_select"),
        cond_stats=cond_stats,
        auto_cond_stats=False,
        augment=False,
        max_samples=max_samples,
        split_manifest=split_manifest,
    )
    return dataset


def predict_unet(  # noqa: PLR0917
    model, batch, cond_dim: int, device: torch.device, tta: bool, tta_mode: str = "rot90"
) -> torch.Tensor:
    vox = batch["voxels"].to(device)
    cond = batch["cond"].to(device) if cond_dim > 0 else None
    model.eval()

    with torch.no_grad():
        if not tta:
            return _forward(model, vox, cond, cond_dim)

        # Apply TTA based on mode (must match training configuration!)
        if tta_mode == "rot90":
            # Rotate in XY plane (dims 3,4) - same as training default
            preds = []
            for k in range(4):
                aug = vox if k == 0 else torch.rot90(vox, k, (3, 4))
                logits = _forward(model, aug, cond, cond_dim)
                if k != 0:
                    logits = torch.rot90(logits, -k, (3, 4))
                preds.append(logits)
            return torch.stack(preds, dim=0).mean(dim=0)
        else:  # flip mode
            # Flip along Z,Y axes (dims 2,3)
            flips = [(), (2,), (3,), (2, 3)]
            preds = []
            for dims in flips:
                aug = vox.flip(dims) if dims else vox
                logits = _forward(model, aug, cond, cond_dim)
                if dims:
                    logits = logits.flip(dims)
                preds.append(logits)
            return torch.stack(preds, dim=0).mean(dim=0)


def _forward(model, vox, cond, cond_dim):
    if cond_dim > 0:
        return model(vox, cond)
    return model(vox)


def _extract_context_volume(sample_path: str | Path) -> torch.Tensor | None:
    path = Path(sample_path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if "Y_neigh" in data:
                ctx = data["Y_neigh"]
            elif "context" in data:
                ctx = data["context"]
            else:
                return None
        return torch.from_numpy(ctx).float()
    except Exception:
        return None


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
    ratios: list[int] = []
    for ctx_dim, pred_dim in zip(ctx_shape, pred_shape, strict=False):
        if pred_dim > 0:
            ratios.append(max(1, round(ctx_dim / pred_dim)))
    if not ratios:
        return 1
    return max(1, round(sum(ratios) / len(ratios)))


def run_unet_inference(args, cfg: dict[str, object], device_cfg):
    cond_dim = resolve_cond_dim(cfg, args.model)
    use_tta = cfg.get("tta", False) if args.tta is None else args.tta
    tta_mode = args.tta_mode or cfg.get("tta_mode", "rot90")
    use_auto_thresh = (
        cfg.get("auto_thresh", False) if args.auto_thresh is None else args.auto_thresh
    )

    # For sample_ids mode, we need to load all data first, then select
    # For max_samples mode, we can limit the dataset directly
    max_samples_for_dataset = args.max_samples if args.sample_ids is None else None

    dataset = build_dataset(
        args.split,
        cfg,
        args.data_root,
        cond_dim,
        args.cond_stats_path or cfg.get("cond_stats_path"),
        max_samples_for_dataset,
        cfg.get("split_manifest"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device_cfg.device_type == "cuda",
        worker_init_fn=worker_init_fn,
        collate_fn=voxel_collate,
    )

    in_channels = len(dataset.x_indices) + (3 if dataset.add_coords else 0)
    if args.model == "unet":
        model = UNet3D(
            in_channels=in_channels, base_channels=cfg.get("base_ch", 16), depth=cfg.get("depth", 4)
        )
    else:
        model = ConditionalUNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
            cond_dim=cond_dim,
        )
    checkpoint_path = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        weights = state["model"]
    elif isinstance(state, dict) and any(k in state for k in ("vae", "diff_unet")):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' appears to be an LDM checkpoint. "
            "Use --model ldm for this checkpoint."
        )
    else:
        weights = state
    load_state_dict_strict(model, weights, "model", checkpoint_path)
    model.to(device_cfg.device)

    vis_dir = ensure_dir(Path(args.out_dir) / "vis")
    metrics_total: dict[str, float] = {
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "delta_vol": 0.0,
    }
    thresh_grid = args.thresh_grid or cfg.get("thresh_grid", [0.3, 0.5, 0.7])
    thr_default = 0.5
    thr_auto = thr_default
    logits_all, targets_all = [], []
    paths_all: list[str] = []
    count = 0

    for batch in dataloader:
        logits = predict_unet(model, batch, cond_dim, device_cfg.device, use_tta, tta_mode)
        target = batch["target"].to(device_cfg.device)
        logits_all.append(logits.cpu())
        targets_all.append(target.cpu())
        count += logits.size(0)
        batch_paths = batch.get("path")
        if isinstance(batch_paths, (list, tuple)):
            for p in batch_paths:
                if isinstance(p, bytes):
                    paths_all.append(p.decode())
                elif isinstance(p, (str, Path)):
                    paths_all.append(str(p))
        elif isinstance(batch_paths, (str, Path)):
            paths_all.append(str(batch_paths))
        elif isinstance(batch_paths, bytes):
            paths_all.append(batch_paths.decode())

    logits_tensor = torch.cat(logits_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)

    if args.threshold is not None:
        thr_auto = float(args.threshold)
        metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)
    elif use_auto_thresh:
        if args.split == "test":
            thr_auto = resolve_threshold_from_artifacts(Path(args.checkpoint), default=thr_default)
            metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)
        else:
            thr_auto, metrics_auto = seg_metrics.auto_threshold(
                logits_tensor, targets_tensor, thresh_grid
            )
    else:
        metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_default)

    metrics_main = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)

    metrics_total.update(metrics_main)

    # Determine which samples to visualize
    if args.sample_ids is not None:
        # Specific sample IDs requested
        sample_indices = [idx for idx in args.sample_ids if idx < logits_tensor.size(0)]
        if not sample_indices:
            print(f"Warning: No valid sample IDs found. Max index is {logits_tensor.size(0) - 1}")
            sample_indices = list(range(min(5, logits_tensor.size(0))))  # Fallback to first 5
        else:
            print(f"Selected sample IDs: {sample_indices}")
    else:
        # Visualize only a subset to keep inference runtime manageable.
        vis_limit = max(0, int(args.max_vis_samples))
        sample_indices = list(range(min(vis_limit, logits_tensor.size(0))))

    # Determine visualization mode
    vis_mode = args.vis_mode
    vis_threshold = args.render_threshold if args.render_threshold is not None else thr_auto
    vis_angles: Sequence[int] = tuple(args.render_angles) if args.render_angles else (30, 60, 120)

    if vis_mode != "none":
        print(f"Generating visualizations for {len(sample_indices)} samples (mode: {vis_mode})...")
    else:
        print("Skipping visualizations (mode: none)")

    for idx, i in enumerate(sample_indices):
        name = f"sample_{i:04d}"
        prob_tensor = torch.sigmoid(logits_tensor[i])
        target_tensor = targets_tensor[i]
        np.save(vis_dir / f"{name}_prob.npy", prob_tensor.detach().cpu().numpy())

        if vis_mode != "none":
            print(f"  [{idx + 1}/{len(sample_indices)}] Processing sample {i}...", end="")

        # 2D projection visualization
        if vis_mode in ("2d", "both"):
            save_max_projection(vis_dir, name, target_tensor, prob_tensor, thr_auto)
            print(" 2D ✓", end="")

        # 3D voxel visualization
        if vis_mode in ("3d", "both"):
            context_tensor = None
            sample_path = paths_all[i] if i < len(paths_all) else None
            if sample_path:
                context_tensor = _extract_context_volume(sample_path)
            if context_tensor is not None:
                stride = _compute_context_stride(prob_tensor, context_tensor)
                save_3d_context_visualization(
                    out_dir=vis_dir,
                    name=name,
                    prediction=prob_tensor,
                    context=context_tensor,
                    threshold=vis_threshold,
                    context_stride=stride,
                    angles=vis_angles,
                )
            else:
                save_3d_context_visualization(
                    out_dir=vis_dir,
                    name=name,
                    prediction=prob_tensor,
                    context=None,
                    threshold=vis_threshold,
                    angles=vis_angles,
                )
            print(" 3D ✓", end="")

        if vis_mode != "none":
            print()  # Newline after each sample

    results = {
        "threshold": thr_auto,
        "metrics": metrics_main,
        "auto_metrics": metrics_auto,
        "count": count,
    }
    save_json(Path(args.out_dir) / "metrics.json", results)


def resolve_cond_dim(cfg: dict[str, object], model_kind: str) -> int:
    if model_kind == "unet":
        return 0
    if cfg.get("cond_dim"):
        return int(cfg["cond_dim"])
    if cfg.get("cond_select"):
        return len(cfg["cond_select"])
    return len(COND_METRIC_KEYS)


def run_ldm_inference(args, cfg: dict[str, object], device_cfg):
    cond_dim = resolve_cond_dim(cfg, "cond_unet")
    use_auto_thresh = (
        cfg.get("auto_thresh", False) if args.auto_thresh is None else args.auto_thresh
    )
    cond_stats_path = args.cond_stats_path or cfg.get("cond_stats_path")
    sampler = args.sampler or cfg.get("sampler", "ddim")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 1.0)
    sample_timesteps = (
        args.sample_timesteps
        if args.sample_timesteps is not None
        else cfg.get("sample_timesteps", 50)
    )
    ddim_eta = args.ddim_eta if args.ddim_eta is not None else cfg.get("ddim_eta", 0.0)
    dataset = build_dataset(
        args.split,
        cfg,
        args.data_root,
        cond_dim,
        cond_stats_path,
        args.max_samples,
        cfg.get("split_manifest"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device_cfg.device_type == "cuda",
        worker_init_fn=worker_init_fn,
        collate_fn=voxel_collate,
    )

    in_channels = infer_target_channels(dataset)
    vae_depth = compute_vae_depth_from_cfg(cfg)
    vae = VAE3D(
        in_channels=in_channels,
        base_channels=cfg.get("vae_base_ch", 32),
        latent_channels=cfg.get("latent_channels", 4),
        depth=vae_depth,
    )
    diff_unet = DiffusionUNet3D(
        latent_channels=cfg.get("latent_channels", 4),
        base_channels=cfg.get("diff_base_ch", 64),
        depth=cfg.get("diff_depth", 3),
        time_dim=cfg.get("time_dim", 256),
        cond_dim=cond_dim,
        context_channels=len(dataset.x_indices) + (3 if dataset.add_coords else 0),
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or "vae" not in state or "diff_unet" not in state:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is not a valid LDM checkpoint "
            "(required keys: 'vae', 'diff_unet')."
        )
    load_state_dict_strict(vae, state["vae"], "vae", checkpoint_path)
    diff_weights, diff_source = select_ldm_diffusion_weights(state, checkpoint_path)
    load_state_dict_strict(diff_unet, diff_weights, f"diff_unet[{diff_source}]", checkpoint_path)
    print(f"Using diffusion weights from checkpoint field '{diff_source}'.")
    vae.to(device_cfg.device)
    diff_unet.to(device_cfg.device)
    vae.eval()
    diff_unet.eval()

    diffusion = GaussianDiffusion(
        cfg.get("train_timesteps", 1000), cfg.get("noise_schedule", "linear")
    )

    vis_dir = ensure_dir(Path(args.out_dir) / "vis")
    logits_all, targets_all = [], []

    for _idx, batch in enumerate(dataloader):
        cond = batch["cond"].to(device_cfg.device)
        context = batch["voxels"].to(device_cfg.device).float()
        with torch.no_grad():
            if sampler == "ddim":
                latents = diffusion.sample_ddim(
                    diff_unet,
                    (
                        cond.size(0),
                        cfg.get("latent_channels", 4),
                        cfg.get("latent_res", 16),
                        cfg.get("latent_res", 16),
                        cfg.get("latent_res", 16),
                    ),
                    cond,
                    cfg_scale,
                    device_cfg.device,
                    sample_timesteps,
                    ddim_eta,
                    context=context,
                )
            else:
                latents = diffusion.sample_ddpm(
                    diff_unet,
                    (
                        cond.size(0),
                        cfg.get("latent_channels", 4),
                        cfg.get("latent_res", 16),
                        cfg.get("latent_res", 16),
                        cfg.get("latent_res", 16),
                    ),
                    cond,
                    cfg_scale,
                    device_cfg.device,
                    context=context,
                )
            recon = vae.decode(latents)
            if recon.dim() == 5 and recon.size(1) > 1:
                recon = recon[:, :1]
            target_shape = batch["target"].shape[-3:]
            if recon.shape[-3:] != target_shape:
                recon = F.interpolate(
                    recon, size=target_shape, mode="trilinear", align_corners=False
                )
        logits_all.append(recon.cpu())
        targets_all.append(batch["target"])

    logits_tensor = torch.cat(logits_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)

    thr_grid = args.thresh_grid or cfg.get("thresh_grid", [0.3, 0.5, 0.7])
    thr_default = 0.5
    thr_auto = thr_default
    if args.threshold is not None:
        thr_auto = float(args.threshold)
        metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)
    elif use_auto_thresh:
        if args.split == "test":
            thr_auto = resolve_threshold_from_artifacts(Path(args.checkpoint), default=thr_default)
            metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)
        else:
            thr_auto, metrics_auto = seg_metrics.auto_threshold(
                logits_tensor, targets_tensor, thr_grid
            )
    else:
        metrics_auto = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_default)
    metrics_main = seg_metrics.evaluate(logits_tensor, targets_tensor, thr_auto)

    vis_limit = max(0, int(args.max_vis_samples))
    vis_count = min(vis_limit, logits_tensor.size(0))
    print(f"Generating visualizations for {vis_count} samples (2d)...")

    for i in range(vis_count):
        name = f"sample_{i:04d}"
        prob_tensor = torch.sigmoid(logits_tensor[i])
        np.save(vis_dir / f"{name}_prob.npy", prob_tensor.detach().cpu().numpy())
        save_max_projection(vis_dir, name, targets_tensor[i], prob_tensor, thr_auto)

    results = {
        "threshold": thr_auto,
        "metrics": metrics_main,
        "auto_metrics": metrics_auto,
        "count": logits_tensor.size(0),
    }
    save_json(Path(args.out_dir) / "metrics.json", results)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified inference runner.")
    parser.add_argument("--model", required=True, choices=("unet", "cond_unet", "ldm"))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument(
        "--config", type=str, default=None, help="Optional training config to reuse."
    )
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cond_stats_path", type=str, default=None)
    parser.add_argument(
        "--tta",
        action=BooleanOptionalAction,
        default=None,
        help="Enable/disable test-time augmentation.",
    )
    parser.add_argument(
        "--tta_mode",
        type=str,
        default=None,
        choices=("rot90", "flip"),
        help="TTA mode: rot90 (default, matches training) or flip",
    )
    parser.add_argument(
        "--auto-thresh",
        "--auto_thresh",
        dest="auto_thresh",
        action=BooleanOptionalAction,
        default=None,
        help="Enable/disable automatic threshold search.",
    )
    parser.add_argument("--thresh_grid", nargs="+", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--sampler", type=str, default=None, choices=("ddpm", "ddim"))
    parser.add_argument("--sample_timesteps", type=int, default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # Sample selection mode
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (random selection)",
    )
    sample_group.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific sample indices to process (e.g., 0 5 10)",
    )

    # Visualization options
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="both",
        choices=("2d", "3d", "both", "none"),
        help="Visualization mode: 2d (projections only), 3d (voxels only), both, or none",
    )
    parser.add_argument(
        "--max_vis_samples",
        type=int,
        default=10,
        help="Maximum number of samples to visualize (default: 10).",
    )
    parser.add_argument(
        "--render-threshold",
        dest="render_threshold",
        type=float,
        default=None,
        help="Override the probability threshold used for 3D renders.",
    )
    parser.add_argument(
        "--render-angles",
        dest="render_angles",
        nargs="+",
        type=int,
        default=None,
        help="Optional azimuth angles (in degrees) for the 3D renders, e.g. 30 60 120.",
    )
    add_hyphenated_aliases(parser)
    return parser.parse_args()


def compute_vae_depth_from_cfg(cfg: dict[str, object]) -> int:
    ratio = max(int(cfg.get("input_res", 64)) // max(int(cfg.get("latent_res", 16)), 1), 1)
    return max(int(math.log2(ratio)) + 1, 2)


def resolve_device(preference: str | None) -> DeviceConfig:
    cfg = select_device()
    if preference is None:
        return cfg

    request = preference.lower()
    if request in {cfg.device_type, str(cfg.device)}:
        return cfg

    if request == "cpu":
        return DeviceConfig(device=torch.device("cpu"), device_type="cpu", amp_dtype=None)

    if request in {"mps", "metal"}:
        if torch.backends.mps.is_available():
            return DeviceConfig(device=torch.device("mps"), device_type="mps", amp_dtype=None)
        raise RuntimeError("MPS requested via --device but is not available on this system.")

    if request in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            amp_dtype = torch.float16
            if os.getenv("PYTORCH_ENABLE_BF16", "0") == "1":
                amp_dtype = torch.bfloat16
            return DeviceConfig(
                device=torch.device("cuda"), device_type="cuda", amp_dtype=amp_dtype
            )
        raise RuntimeError("CUDA requested via --device but no CUDA device is available.")

    raise ValueError(f"Unsupported device option '{preference}'. Use cpu, mps, or cuda.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint_path, args.config)
    device_cfg = resolve_device(args.device)
    ensure_dir(Path(args.out_dir))

    if args.model in {"unet", "cond_unet"}:
        run_unet_inference(args, cfg, device_cfg)
    else:
        run_ldm_inference(args, cfg, device_cfg)


if __name__ == "__main__":
    main()
