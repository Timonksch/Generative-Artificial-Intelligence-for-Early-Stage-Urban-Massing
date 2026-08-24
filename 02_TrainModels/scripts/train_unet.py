"""Training script for unconditional 3D UNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dataio.voxel_dataset import VoxelDataset, voxel_collate
from engine.trainer_unet import UNetTrainer, UNetTrainOptions
from models.unet.losses import BCEDiceLoss
from models.unet.unet3d import UNet3D
from torch.utils.data import DataLoader
from utils import args as args_utils
from utils.device import select_device, set_seed, worker_init_fn
from utils.io import ensure_dir, load_json, save_json
from utils.logging import RunLogger, configure_root_logger
from utils.sched import build_scheduler


def build_optimizer(model: torch.nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if optimizer_name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unconditional UNet3D.")
    args_utils.add_common_args(parser)
    args_utils.add_data_args(parser)
    args_utils.add_eval_args(parser)
    args_utils.add_unet_args(parser)
    args_utils.add_hyphenated_aliases(parser)
    parsed = parser.parse_args()

    set_seed(parsed.seed)
    device_cfg = select_device()
    root_logger = configure_root_logger()
    root_logger.info(
        "Launching UNet training | data_root=%s | out_dir=%s | epochs=%d | "
        "device=%s | vis_every=%d",
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

    dataset_kwargs = {
        "root_dir": parsed.data_root,
        "channels": parsed.channels,
        "x_indices": parsed.x_indices,
        "add_coords": parsed.add_coords,
        "downsample_stride": parsed.downsample_stride,
        "crop_size": parsed.crop_size,
        "split_manifest": parsed.split_manifest,
        "cond_dim": 0,
        "max_samples": parsed.max_samples,
    }
    if parsed.split_manifest:
        root_logger.info("Using split manifest: %s", parsed.split_manifest)
    train_ds = VoxelDataset(split="train", **dataset_kwargs)
    val_ds = VoxelDataset(split="val", **dataset_kwargs, augment=False)
    root_logger.info(
        "Loaded datasets | train=%d | val=%d | add_coords=%s",
        len(train_ds),
        len(val_ds),
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

    in_channels = len(train_ds.x_indices) + (3 if parsed.add_coords else 0)
    model = UNet3D(in_channels=in_channels, base_channels=parsed.base_ch, depth=parsed.depth)

    optimizer = build_optimizer(model, parsed.optimizer, parsed.lr, parsed.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        parsed.lr_schedule,
        parsed.epochs,
        plateau_mode="max",
    )

    loss_fn = BCEDiceLoss(parsed.bce_w, parsed.dice_w, parsed.surface_weight)

    logger = RunLogger(run_dir / "metrics", name="unet", console=False)
    options = UNetTrainOptions(
        epochs=parsed.epochs,
        log_every=parsed.log_every,
        save_every=parsed.save_every,
        accum_steps=parsed.accum_steps,
        grad_clip=parsed.grad_clip,
        tta=parsed.tta,
        tta_mode=parsed.tta_mode,
        auto_thresh=parsed.auto_thresh,
        thresh_grid=tuple(float(x) for x in parsed.thresh_grid),
        cond_drop=0.0,
        early_stop_patience=parsed.early_stop_patience,
        vis_every=parsed.vis_every,
        volume_reg_weight=parsed.volume_reg_weight,
    )

    trainer = UNetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device_cfg=device_cfg,
        run_dir=run_dir,
        logger=logger,
        options=options,
        cond_dim=0,
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
