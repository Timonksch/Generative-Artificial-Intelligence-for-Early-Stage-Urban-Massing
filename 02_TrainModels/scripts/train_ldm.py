"""Training script for latent diffusion models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from dataio.cond_stats import COND_METRIC_KEYS, load_cond_stats, save_cond_stats
from dataio.voxel_dataset import VoxelDataset, voxel_collate
from engine.trainer_ldm import LDMTrainer, LDMTrainOptions
from models.ldm.diffusion_core import GaussianDiffusion
from models.ldm.diffusion_unet3d import DiffusionUNet3D
from models.ldm.vae3d import VAE3D
from torch.utils.data import DataLoader
from utils import args as args_utils
from utils.device import select_device, set_seed, worker_init_fn
from utils.io import ensure_dir, load_json, save_json
from utils.logging import RunLogger, configure_root_logger
from utils.sched import build_scheduler


def resolve_cond_dim(parsed: argparse.Namespace) -> int:
    if parsed.cond_dim and parsed.cond_dim > 0:
        return parsed.cond_dim
    if parsed.cond_select:
        return len(parsed.cond_select)
    return len(COND_METRIC_KEYS)


def build_optimizer(params, name: str, lr: float, weight_decay: float):
    decay_params, no_decay_params = [], []
    for p in params:
        if not p.requires_grad:
            continue
        if p.dim() <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{name}'")


def compute_vae_depth(input_res: int, latent_res: int) -> int:
    ratio = max(input_res // latent_res, 1)
    return max(int(math.log2(ratio)) + 1, 2)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def infer_target_channels(dataset: VoxelDataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot infer target channels.")
    original_augment = getattr(dataset, "augment", False)
    dataset.augment = False
    try:
        sample = dataset[0]
    finally:
        dataset.augment = original_augment
    target = sample.get("target") if isinstance(sample, dict) else sample[1]
    if target is None:
        raise ValueError("Dataset sample does not contain 'target'.")
    target_tensor = target if torch.is_tensor(target) else torch.as_tensor(target)
    if target_tensor.dim() < 4:
        raise ValueError(
            f"Unexpected target shape {tuple(target_tensor.shape)}; expected (C,D,H,W)."
        )
    return int(target_tensor.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent diffusion model.")
    args_utils.add_common_args(parser)
    args_utils.add_data_args(parser)
    args_utils.add_eval_args(parser)
    args_utils.add_cond_args(parser)
    args_utils.add_ldm_args(parser)
    parser.set_defaults(cond_drop=0.05)
    args_utils.add_hyphenated_aliases(parser)
    parsed = parser.parse_args()

    if parsed.mode == "diffusion" and not parsed.vae_checkpoint:
        raise ValueError("--vae_checkpoint required when mode='diffusion'")

    set_seed(parsed.seed)
    device_cfg = select_device()
    root_logger = configure_root_logger()
    root_logger.info(
        "Launching LDM training | mode=%s | data_root=%s | out_dir=%s | "
        "epochs=%d | device=%s | vis_every=%d",
        parsed.mode,
        parsed.data_root,
        parsed.out_dir,
        parsed.epochs,
        device_cfg.device.type,
        parsed.vis_every,
    )
    run_dir = ensure_dir(Path(parsed.out_dir))
    config_path = run_dir / "config.json"
    config_mismatch = False
    if config_path.exists():
        try:
            stored_config = load_json(config_path)
            if stored_config != vars(parsed):
                config_mismatch = True
                root_logger.warning(
                    "Existing config differs from current arguments. "
                    "Proceeding may lead to mismatched checkpoints."
                )
        except Exception as exc:  # pragma: no cover - defensive
            root_logger.warning("Failed to inspect existing config: %s", exc)
    else:
        save_json(config_path, vars(parsed))

    cond_dim = resolve_cond_dim(parsed)
    cond_stats = load_cond_stats(parsed.cond_stats_path) if parsed.cond_stats_path else None

    dataset_kwargs = {
        "root_dir": parsed.data_root,
        "channels": parsed.channels,
        "x_indices": parsed.x_indices,
        "add_coords": parsed.add_coords,
        "downsample_stride": parsed.downsample_stride,
        "crop_size": parsed.crop_size,
        "split_manifest": parsed.split_manifest,
        "cond_dim": cond_dim,
        "cond_select": parsed.cond_select,
        "max_samples": parsed.max_samples,
    }
    if parsed.split_manifest:
        root_logger.info("Using split manifest: %s", parsed.split_manifest)
    if cond_dim > 0:
        if cond_stats is None:
            train_ds = VoxelDataset(
                split="train",
                **dataset_kwargs,
                cond_stats=None,
                auto_cond_stats=True,
            )
            cond_stats = train_ds.cond_stats
        else:
            train_ds = VoxelDataset(
                split="train",
                **dataset_kwargs,
                cond_stats=cond_stats,
                auto_cond_stats=False,
            )
        val_ds = VoxelDataset(
            split="val",
            **dataset_kwargs,
            cond_stats=cond_stats,
            auto_cond_stats=False,
            augment=False,
        )
    else:
        train_ds = VoxelDataset(
            split="train",
            **dataset_kwargs,
            cond_stats=None,
            auto_cond_stats=False,
        )
        val_ds = VoxelDataset(
            split="val",
            **dataset_kwargs,
            cond_stats=None,
            auto_cond_stats=False,
            augment=False,
        )

    if cond_stats is not None:
        cond_stats_path = run_dir / "cond_stats.json"
        save_cond_stats(cond_stats_path, cond_stats)
        root_logger.info("Saved conditioning stats to %s", cond_stats_path)
    root_logger.info(
        "Loaded datasets | train=%d | val=%d | cond_dim=%d | add_coords=%s",
        len(train_ds),
        len(val_ds),
        cond_dim,
        parsed.add_coords,
    )

    loader_kwargs = {
        "batch_size": parsed.batch_size,
        "num_workers": parsed.num_workers,
        "pin_memory": device_cfg.device_type == "cuda",
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=voxel_collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=voxel_collate, **loader_kwargs)

    target_channels = infer_target_channels(train_ds)
    context_channels = len(train_ds.x_indices) + (3 if train_ds.add_coords else 0)
    vae_depth = compute_vae_depth(parsed.input_res, parsed.latent_res)
    vae = VAE3D(
        in_channels=target_channels,
        base_channels=parsed.vae_base_ch,
        latent_channels=parsed.latent_channels,
        depth=vae_depth,
    )

    diff_unet = DiffusionUNet3D(
        latent_channels=parsed.latent_channels,
        base_channels=parsed.diff_base_ch,
        depth=parsed.diff_depth,
        time_dim=parsed.time_dim,
        cond_dim=cond_dim,
        context_channels=context_channels,
    )
    diffusion = GaussianDiffusion(parsed.train_timesteps, parsed.noise_schedule)

    if parsed.mode in {"diffusion", "joint"} and parsed.vae_checkpoint:
        ckpt_path = Path(parsed.vae_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"--vae_checkpoint '{parsed.vae_checkpoint}' not found. "
                "Run VAE pretraining first or adjust the path."
            )
        state = torch.load(ckpt_path, map_location="cpu")
        # Accept all checkpoint layouts emitted by the training commands.
        if "vae" in state:
            vae_state = state["vae"]  # Full LDM checkpoint format
            ckpt_format = "full LDM"
        elif "model" in state:
            vae_state = state["model"]  # Standalone VAE checkpoint format
            ckpt_format = "standalone VAE"
        else:
            vae_state = state  # Direct state_dict
            ckpt_format = "direct state_dict"

        try:
            vae.load_state_dict(vae_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Strict VAE checkpoint load failed for '{ckpt_path}'. "
                "Checkpoint architecture must exactly match current VAE configuration."
            ) from exc
        print(f"✓ Loaded VAE checkpoint from {ckpt_path}")
        print(f"  Checkpoint format: {ckpt_format}")

    if parsed.mode == "diffusion":
        for param in vae.parameters():
            param.requires_grad = False
        trainable_params = diff_unet.parameters()
    elif parsed.mode == "vae_pretrain":
        for param in diff_unet.parameters():
            param.requires_grad = False
        trainable_params = vae.parameters()
    else:
        trainable_params = list(vae.parameters()) + list(diff_unet.parameters())

    optimizer = build_optimizer(trainable_params, parsed.optimizer, parsed.lr, parsed.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        parsed.lr_schedule,
        parsed.epochs,
        plateau_mode="min",
    )

    options = LDMTrainOptions(
        epochs=parsed.epochs,
        log_every=parsed.log_every,
        save_every=parsed.save_every,
        accum_steps=parsed.accum_steps,
        grad_clip=parsed.grad_clip,
        mode=parsed.mode,
        kl_weight=parsed.kl_weight,
        kl_anneal_epochs=parsed.kl_anneal_epochs,
        cfg_scale=parsed.cfg_scale,
        sampler=parsed.sampler,
        sample_steps=parsed.sample_timesteps,
        ddim_eta=parsed.ddim_eta,
        ema_decay=parsed.ema_decay,
        vis_every=parsed.vis_every,
        cond_drop=parsed.cond_drop,
        sample_every=parsed.sample_every,
        num_sample_batches=parsed.num_sample_batches,
        num_sample_conds=parsed.num_sample_conds,
        ema_sample_start=parsed.ema_sample_start,
        latent_channels=parsed.latent_channels,
        latent_res=parsed.latent_res,
    )

    logger = RunLogger(run_dir / "metrics", name="ldm", console=False)

    trainer = LDMTrainer(
        vae=vae,
        diff_unet=diff_unet,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        device_cfg=device_cfg,
        run_dir=run_dir,
        logger=logger,
        options=options,
        cond_dim=cond_dim,
        ema_enabled=parsed.ema,
    )

    start_epoch = 0
    latest_ckpt = find_latest_checkpoint(run_dir)
    if latest_ckpt is not None:
        if config_mismatch and not parsed.allow_resume_mismatch:
            raise RuntimeError(
                "Found existing checkpoint with config mismatch. "
                "Use a fresh out_dir or pass --allow_resume_mismatch explicitly."
            )
        root_logger.info("Found existing checkpoint -> %s", latest_ckpt.name)
        checkpoint_state = torch.load(latest_ckpt, map_location="cpu")
        start_epoch = trainer.load_checkpoint(checkpoint_state)
        root_logger.info("Checkpoint restored; will resume from epoch %d", start_epoch)
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
