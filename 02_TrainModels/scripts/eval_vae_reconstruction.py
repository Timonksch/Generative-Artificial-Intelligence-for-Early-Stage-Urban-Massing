"""Evaluate Stage-A VAE reconstruction quality and export table-ready metrics."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from dataio.voxel_dataset import VoxelDataset, voxel_collate
from metrics import seg3d as seg_metrics
from metrics.regulatory import compute_regulatory
from metrics.vae import vae_loss as compute_vae_loss
from models.ldm.vae3d import VAE3D
from torch.utils.data import DataLoader
from utils.args import add_hyphenated_aliases
from utils.device import select_device, set_seed, worker_init_fn
from utils.io import ensure_dir, load_json, save_json


def compute_vae_depth(input_res: int, latent_res: int) -> int:
    ratio = max(input_res // max(latent_res, 1), 1)
    return max(int(math.log2(ratio)) + 1, 2)


def infer_target_channels(dataset: VoxelDataset) -> int:
    sample = dataset[0]
    target = sample["target"]
    return int(target.shape[0])


def build_dataset(cfg: dict, split: str, max_samples: int | None = None) -> VoxelDataset:
    return VoxelDataset(
        root_dir=cfg["data_root"],
        split=split,
        channels=cfg.get("channels", ("C0", "C1", "C2", "C3")),
        x_indices=cfg.get("x_indices"),
        add_coords=cfg.get("add_coords", False),
        downsample_stride=cfg.get("downsample_stride", 0),
        crop_size=cfg.get("crop_size", 0),
        cond_dim=0,
        cond_select=cfg.get("cond_select"),
        cond_stats=None,
        auto_cond_stats=False,
        augment=False,
        max_samples=max_samples,
        split_manifest=cfg.get("split_manifest"),
    )


def stack_tensors(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise RuntimeError("No tensors collected for evaluation.")
    return torch.cat(chunks, dim=0)


def select_best_threshold(
    logits: torch.Tensor, targets: torch.Tensor, grid: Iterable[float]
) -> tuple[float, dict[str, float]]:
    return seg_metrics.auto_threshold(logits, targets, grid)


def _get_meta_metric(meta: dict, key: str, default: float = 0.0) -> float:
    if not isinstance(meta, dict):
        return default
    metrics = meta.get("metrics", {})
    if not isinstance(metrics, dict):
        return default
    return float(metrics.get(key, default))


def _get_voxel_m(meta: dict, default: float = 0.5) -> float:
    if not isinstance(meta, dict):
        return default
    grid = meta.get("grid", {})
    if not isinstance(grid, dict):
        return default
    return float(grid.get("voxel_m", default))


def evaluate_split(
    loader: DataLoader,
    vae: VAE3D,
    device: torch.device,
    threshold: float,
    kl_weight: float,
) -> dict[str, float]:
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    recon_losses: list[float] = []
    kl_losses: list[float] = []
    total_losses: list[float] = []

    pred_grz: list[float] = []
    pred_gfz: list[float] = []
    pred_h: list[float] = []
    tgt_grz: list[float] = []
    tgt_gfz: list[float] = []
    tgt_h: list[float] = []

    vae.eval()
    with torch.no_grad():
        for batch in loader:
            targets = batch["target"].to(device=device, dtype=torch.float32)
            logits, mu, logvar = vae(targets)
            loss_dict = compute_vae_loss(logits, targets, mu, logvar, beta=kl_weight)

            recon_losses.append(float(loss_dict["recon_loss"].item()))
            kl_losses.append(float(loss_dict["kl_loss"].item()))
            total_losses.append(float(loss_dict["loss"].item()))

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

            probs = torch.sigmoid(logits)
            bin_pred = (probs > threshold).float().cpu()
            cpu_targets = targets.detach().cpu()
            metas = batch["meta"]

            for i in range(bin_pred.shape[0]):
                meta = metas[i] if i < len(metas) else {}
                parcel_area_m2 = _get_meta_metric(meta, "parcel_area_m2", 0.0)
                voxel_m = _get_voxel_m(meta, 0.5)
                storey_h = 3.0

                pred_regs = compute_regulatory(
                    bin_pred[i], parcel_area_m2, voxel_m, storey_h, threshold=0.5
                )
                tgt_regs = compute_regulatory(
                    cpu_targets[i], parcel_area_m2, voxel_m, storey_h, threshold=0.5
                )

                pred_grz.append(pred_regs["grz"])
                pred_gfz.append(pred_regs["gfz"])
                pred_h.append(pred_regs["height_m"])
                tgt_grz.append(tgt_regs["grz"])
                tgt_gfz.append(tgt_regs["gfz"])
                tgt_h.append(tgt_regs["height_m"])

    logits_all = stack_tensors(all_logits)
    targets_all = stack_tensors(all_targets)
    seg = seg_metrics.evaluate(logits_all, targets_all, threshold)

    out = {
        "threshold": float(threshold),
        "iou": float(seg["iou"]),
        "dice": float(seg["dice"]),
        "precision": float(seg["precision"]),
        "recall": float(seg["recall"]),
        "delta_vol": float(seg["delta_vol"]),
        "volume_error": float(seg["volume_error"]),
        "z_profile_error": float(seg["z_profile_error"]),
        "recon_loss": float(np.mean(recon_losses)) if recon_losses else 0.0,
        "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
        "vae_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "grz_pred_mean": float(np.mean(pred_grz)) if pred_grz else 0.0,
        "grz_target_mean": float(np.mean(tgt_grz)) if tgt_grz else 0.0,
        "gfz_pred_mean": float(np.mean(pred_gfz)) if pred_gfz else 0.0,
        "gfz_target_mean": float(np.mean(tgt_gfz)) if tgt_gfz else 0.0,
        "height_pred_mean": float(np.mean(pred_h)) if pred_h else 0.0,
        "height_target_mean": float(np.mean(tgt_h)) if tgt_h else 0.0,
    }
    return out


def format_latex_row(run_name: str, metrics: dict[str, float]) -> str:
    return (
        f"{run_name} & "
        f"{metrics['iou']:.4f} & {metrics['dice']:.4f} & "
        f"{metrics['precision']:.4f} & {metrics['recall']:.4f} & "
        f"{metrics['delta_vol']:.4f} & {metrics['volume_error']:.2f} & "
        f"{metrics['z_profile_error']:.2f} & "
        f"{metrics['grz_pred_mean']:.4f} & {metrics['grz_target_mean']:.4f} & "
        f"{metrics['gfz_pred_mean']:.4f} & {metrics['gfz_target_mean']:.4f} & "
        f"{metrics['height_pred_mean']:.2f} & {metrics['height_target_mean']:.2f} & "
        f"{metrics['recon_loss']:.3e} & {metrics['kl_loss']:.3e} & {metrics['vae_loss']:.3e} \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction metrics for Stage-A checkpoint."
    )
    parser.add_argument(
        "--run_dir", required=True, help="Run directory containing config.json and best.pt"
    )
    parser.add_argument(
        "--checkpoint", default="best.pt", help="Checkpoint filename inside run_dir"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples", type=int, default=0, help="Optional cap per split (0 = all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=("Fixed binarization threshold. If omitted, the best validation threshold is used."),
    )
    parser.add_argument("--seed", type=int, default=42)
    add_hyphenated_aliases(parser)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = load_json(run_dir / "config.json")
    ckpt_path = run_dir / args.checkpoint

    set_seed(args.seed)
    device_cfg = select_device()

    max_samples = args.max_samples if args.max_samples > 0 else None
    val_ds = build_dataset(cfg, split="val", max_samples=max_samples)
    test_ds = build_dataset(cfg, split="test", max_samples=max_samples)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device_cfg.device_type == "cuda",
        "worker_init_fn": worker_init_fn,
        "collate_fn": voxel_collate,
    }
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    target_channels = infer_target_channels(val_ds)
    vae_depth = compute_vae_depth(int(cfg.get("input_res", 64)), int(cfg.get("latent_res", 16)))
    vae = VAE3D(
        in_channels=target_channels,
        base_channels=int(cfg.get("vae_base_ch", 32)),
        latent_channels=int(cfg.get("latent_channels", 4)),
        depth=vae_depth,
    ).to(device_cfg.device)

    state = torch.load(ckpt_path, map_location=device_cfg.device)
    vae_state = state.get("vae") if isinstance(state, dict) and "vae" in state else state
    vae.load_state_dict(vae_state, strict=True)

    thresh_grid = cfg.get("thresh_grid", [0.4, 0.45, 0.5, 0.55, 0.6])
    kl_weight = float(cfg.get("kl_weight", 0.0))

    if args.threshold is not None:
        best_thr = float(args.threshold)
        val_best = {"iou": float("nan")}
    else:
        # Validation pass for threshold selection
        all_val_logits: list[torch.Tensor] = []
        all_val_targets: list[torch.Tensor] = []
        vae.eval()
        with torch.no_grad():
            for batch in val_loader:
                y = batch["target"].to(device=device_cfg.device, dtype=torch.float32)
                logits, _, _ = vae(y)
                all_val_logits.append(logits.detach().cpu())
                all_val_targets.append(y.detach().cpu())

        val_logits = stack_tensors(all_val_logits)
        val_targets = stack_tensors(all_val_targets)
        best_thr, val_best = select_best_threshold(val_logits, val_targets, thresh_grid)

    val_metrics = evaluate_split(
        val_loader, vae, device_cfg.device, threshold=best_thr, kl_weight=kl_weight
    )
    test_metrics = evaluate_split(
        test_loader, vae, device_cfg.device, threshold=best_thr, kl_weight=kl_weight
    )

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "selected_threshold": float(best_thr),
        "val_best_iou": float(val_best.get("iou", 0.0)),
        "val": val_metrics,
        "test": test_metrics,
        "latex_row_test": format_latex_row(run_dir.name, test_metrics),
    }

    out_dir = ensure_dir(run_dir / "vae_recon_eval")
    save_json(out_dir / "metrics.json", report)

    (out_dir / "table_row_test.tex").write_text(report["latex_row_test"] + "\n", encoding="utf-8")

    print("Saved:", out_dir / "metrics.json")
    print("Saved:", out_dir / "table_row_test.tex")
    print("Selected threshold:", f"{best_thr:.2f}")
    print("Test IoU:", f"{test_metrics['iou']:.4f}")


if __name__ == "__main__":
    main()
