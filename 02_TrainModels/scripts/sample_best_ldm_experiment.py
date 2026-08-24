"""Generate variant samples from the best completed LDM run in an experiment.

This script does not modify training or inference code paths. It reuses the
existing LDM, dataset, and visualization modules to:

1. Select the best completed diffusion run in an experiment directory
   (lowest validation loss from ``final_metrics.json``).
2. Load the run's ``best.pt`` checkpoint.
3. Generate multiple stochastic variants for the same conditioning/context
   pairs from a chosen dataset split.
4. Save both 2D dual projections and 3D context renders.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dataio.cond_stats import COND_METRIC_KEYS, load_cond_stats, save_cond_stats
from dataio.voxel_dataset import VoxelDataset
from models.ldm.diffusion_core import GaussianDiffusion
from models.ldm.diffusion_unet3d import DiffusionUNet3D
from models.ldm.vae3d import VAE3D
from utils.args import add_hyphenated_aliases
from utils.device import DeviceConfig, select_device, set_seed
from utils.io import ensure_dir, load_json, save_json
from utils.visuals import save_3d_context_visualization, save_dual_projection


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _resolve_device(preference: str | None) -> DeviceConfig:
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
        raise RuntimeError("MPS requested via --device but is not available.")
    if request in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return DeviceConfig(
                device=torch.device("cuda"), device_type="cuda", amp_dtype=torch.float16
            )
        raise RuntimeError("CUDA requested via --device but no CUDA device is available.")
    raise ValueError(f"Unsupported device option '{preference}'. Use cpu, mps, or cuda.")


def _resolve_cond_dim(cfg: dict[str, object]) -> int:
    cond_dim = int(cfg.get("cond_dim", 0) or 0)
    if cond_dim > 0:
        return cond_dim
    cond_select = cfg.get("cond_select")
    if cond_select:
        return len(cond_select)
    return len(COND_METRIC_KEYS)


def _compute_vae_depth(cfg: dict[str, object]) -> int:
    input_res = int(cfg.get("input_res", 64))
    latent_res = max(int(cfg.get("latent_res", 16)), 1)
    ratio = max(input_res // latent_res, 1)
    return max(int(math.log2(ratio)) + 1, 2)


def _resolve_run_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    candidates = [checkpoint_path.parent, checkpoint_path.parent.parent]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


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


def _select_diffusion_weights(state: dict, checkpoint: Path) -> tuple[dict, str]:
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


def _infer_target_channels(dataset: VoxelDataset) -> int:
    sample = dataset[0]
    target = sample["target"]
    return int(target.shape[0])


def _load_dataset(run_cfg: dict[str, object], split: str, run_dir: Path) -> VoxelDataset:
    cond_dim = _resolve_cond_dim(run_cfg)
    cond_stats_path = run_dir / "cond_stats.json"
    cond_stats = load_cond_stats(cond_stats_path) if cond_stats_path.exists() else None

    dataset_kwargs = {
        "root_dir": str(run_cfg["data_root"]),
        "split": split,
        "channels": run_cfg.get("channels", ("C0", "C1", "C2", "C3")),
        "x_indices": run_cfg.get("x_indices"),
        "add_coords": bool(run_cfg.get("add_coords", False)),
        "downsample_stride": int(run_cfg.get("downsample_stride", 0) or 0),
        "crop_size": int(run_cfg.get("crop_size", 0) or 0),
        "cond_dim": cond_dim,
        "cond_select": run_cfg.get("cond_select"),
        "cond_stats": cond_stats,
        "auto_cond_stats": False,
        "augment": False,
        "max_samples": None,
        "split_manifest": run_cfg.get("split_manifest"),
    }

    if cond_dim > 0 and cond_stats is None:
        train_ds = VoxelDataset(
            split="train",
            cond_stats=None,
            auto_cond_stats=True,
            augment=False,
            **{
                k: v
                for k, v in dataset_kwargs.items()
                if k not in {"split", "cond_stats", "auto_cond_stats", "augment"}
            },
        )
        cond_stats = train_ds.cond_stats
        save_cond_stats(run_dir / "cond_stats.json", cond_stats)
        dataset_kwargs["cond_stats"] = cond_stats

    return VoxelDataset(**dataset_kwargs)


def _pick_sample_indices(
    dataset: VoxelDataset, requested: Sequence[int] | None, count: int
) -> list[int]:
    if requested:
        valid = [idx for idx in requested if 0 <= idx < len(dataset)]
        if valid:
            return valid
    candidate = [0, len(dataset) // 2, len(dataset) - 1]
    picked: list[int] = []
    seen = set()
    for idx in candidate:
        if idx in seen or idx < 0 or idx >= len(dataset):
            continue
        seen.add(idx)
        picked.append(idx)
        if len(picked) >= max(1, count):
            break
    return picked


def _read_sample_meta(dataset: VoxelDataset, sample_index: int) -> dict[str, object]:
    sample_path = dataset.files[sample_index]
    return dataset._read_meta(sample_path)  # type: ignore[attr-defined]


def _sample_label(sample_index: int, meta: dict[str, object]) -> str:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    grz = float(metrics.get("grz_target", 0.0))
    gfz = float(metrics.get("gfz_target", 0.0))
    height = float(metrics.get("target_height_m", 0.0))
    return f"sample{sample_index:04d}_grz{grz:.2f}_gfz{gfz:.2f}_h{height:.1f}"


def _find_best_completed_run(
    experiment_dir: Path,
) -> tuple[Path, dict[str, object], dict[str, float]]:
    candidates: list[tuple[float, Path, dict[str, object], dict[str, float]]] = []
    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir():
            continue
        config_path = child / "config.json"
        metrics_path = child / "final_metrics.json"
        best_path = child / "best.pt"
        if not (config_path.exists() and metrics_path.exists() and best_path.exists()):
            continue
        cfg = load_json(config_path)
        if cfg.get("mode") not in {"diffusion", "joint"}:
            continue
        metrics = load_json(metrics_path)
        if "loss" not in metrics:
            continue
        candidates.append((float(metrics["loss"]), child, cfg, metrics))

    if not candidates:
        raise FileNotFoundError(
            "No completed diffusion runs with config.json, final_metrics.json, "
            f"and best.pt found under {experiment_dir}"
        )

    candidates.sort(key=lambda item: item[0])
    _, run_dir, cfg, metrics = candidates[0]
    return run_dir, cfg, metrics


def _load_ldm_from_run(run_dir: Path, cfg: dict[str, object], device: torch.device):
    checkpoint_path = run_dir / "best.pt"
    state = torch.load(checkpoint_path, map_location="cpu")

    dataset = _load_dataset(cfg, split="test", run_dir=run_dir)
    in_channels = _infer_target_channels(dataset)
    cond_dim = _resolve_cond_dim(cfg)
    context_channels = len(dataset.x_indices) + (3 if dataset.add_coords else 0)

    vae = VAE3D(
        in_channels=in_channels,
        base_channels=int(cfg.get("vae_base_ch", 32)),
        latent_channels=int(cfg.get("latent_channels", 4)),
        depth=_compute_vae_depth(cfg),
    )
    diff_unet = DiffusionUNet3D(
        latent_channels=int(cfg.get("latent_channels", 4)),
        base_channels=int(cfg.get("diff_base_ch", 64)),
        depth=int(cfg.get("diff_depth", 3)),
        time_dim=int(cfg.get("time_dim", 256)),
        cond_dim=cond_dim,
        context_channels=context_channels,
    )

    if "vae" not in state:
        raise ValueError(f"Checkpoint '{checkpoint_path}' does not contain a VAE state.")
    vae.load_state_dict(state["vae"], strict=True)
    diff_weights, diff_source = _select_diffusion_weights(state, checkpoint_path)
    diff_unet.load_state_dict(diff_weights, strict=True)

    vae.to(device).eval()
    diff_unet.to(device).eval()
    diffusion = GaussianDiffusion(
        int(cfg.get("train_timesteps", 1000)), str(cfg.get("noise_schedule", "linear"))
    )
    return dataset, vae, diff_unet, diffusion, diff_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample variants from the best completed LDM run in an experiment."
    )
    parser.add_argument(
        "--experiment_dir",
        required=True,
        type=str,
        help="Experiment directory containing multiple run subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <best_run>/best_samples.",
    )
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific sample indices from the selected split.",
    )
    parser.add_argument(
        "--num_conditions",
        type=int,
        default=3,
        help="How many conditioning/context cases to sample if --sample_ids is not given.",
    )
    parser.add_argument(
        "--variants_per_condition",
        type=int,
        default=4,
        help="How many stochastic variants to create for each condition.",
    )
    parser.add_argument("--sampler", type=str, default=None, choices=("ddpm", "ddim"))
    parser.add_argument("--sample_timesteps", type=int, default=None, help="DDIM steps override.")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for binary 2D/3D rendering."
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=int,
        default=[30, 60, 120],
        help="Azimuth angles for 3D renders.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    add_hyphenated_aliases(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    experiment_dir = Path(args.experiment_dir).resolve()
    run_dir, cfg, final_metrics = _find_best_completed_run(experiment_dir)
    device_cfg = _resolve_device(args.device)

    dataset, vae, diff_unet, diffusion, diff_source = _load_ldm_from_run(
        run_dir, cfg, device_cfg.device
    )
    if args.split != "test":
        dataset = _load_dataset(cfg, split=args.split, run_dir=run_dir)

    cond_dim = _resolve_cond_dim(cfg)
    sample_indices = _pick_sample_indices(dataset, args.sample_ids, args.num_conditions)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else run_dir / "best_samples")

    sampler = args.sampler or str(cfg.get("sampler", "ddim"))
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 1.0))
    sample_timesteps = int(
        args.sample_timesteps
        if args.sample_timesteps is not None
        else cfg.get("sample_timesteps", 50)
    )
    ddim_eta = float(args.ddim_eta if args.ddim_eta is not None else cfg.get("ddim_eta", 0.0))
    threshold = float(args.threshold)

    manifest = {
        "experiment_dir": str(experiment_dir),
        "selected_run": run_dir.name,
        "selected_run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "selection_metric": float(final_metrics["loss"]),
        "diffusion_weight_source": diff_source,
        "split": args.split,
        "sample_indices": sample_indices,
        "variants_per_condition": int(args.variants_per_condition),
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "sample_timesteps": sample_timesteps,
        "ddim_eta": ddim_eta,
        "threshold": threshold,
    }
    save_json(out_dir / "manifest.json", manifest)

    print(f"Selected best run: {run_dir.name}")
    print(f"Validation loss: {float(final_metrics['loss']):.6f}")
    print(f"Using diffusion weights from: {diff_source}")
    print(f"Output directory: {out_dir}")

    with torch.no_grad():
        for sample_index in sample_indices:
            sample = dataset[sample_index]
            meta = _read_sample_meta(dataset, sample_index)
            sample_name = _sample_label(sample_index, meta)
            sample_dir = ensure_dir(out_dir / sample_name)

            cond = sample["cond"].to(device_cfg.device) if cond_dim > 0 else None
            context = sample["voxels"].to(device_cfg.device).float()
            model_context = context.unsqueeze(0).repeat(args.variants_per_condition, 1, 1, 1, 1)
            cond_batch = None
            if cond is not None:
                cond_batch = cond.unsqueeze(0).repeat(args.variants_per_condition, 1)

            latent_res = int(cfg.get("latent_res", 16))
            latent_channels = int(cfg.get("latent_channels", 4))
            latent_shape = (
                int(args.variants_per_condition),
                latent_channels,
                latent_res,
                latent_res,
                latent_res,
            )

            if sampler == "ddim":
                latents = diffusion.sample_ddim(
                    diff_unet,
                    latent_shape,
                    cond_batch,
                    cfg_scale,
                    device_cfg.device,
                    sample_timesteps,
                    ddim_eta,
                    context=model_context,
                )
            else:
                latents = diffusion.sample_ddpm(
                    diff_unet,
                    latent_shape,
                    cond_batch,
                    cfg_scale,
                    device_cfg.device,
                    context=model_context,
                )

            decoded = torch.sigmoid(vae.decode(latents))
            target_shape = sample["target"].shape[-3:]
            if decoded.shape[-3:] != target_shape:
                decoded = F.interpolate(
                    decoded, size=target_shape, mode="trilinear", align_corners=False
                )

            context_vis = _extract_context_volume(dataset.files[sample_index])
            sample_manifest = {
                "sample_index": int(sample_index),
                "sample_path": str(dataset.files[sample_index]),
                "sample_name": sample_name,
                "meta": _to_jsonable(meta),
            }
            save_json(sample_dir / "sample.json", sample_manifest)

            print(f"Sampling {sample_name} -> {args.variants_per_condition} variants")
            for variant_idx in range(args.variants_per_condition):
                pred = decoded[variant_idx].detach().cpu()
                variant_name = f"{sample_name}_variant{variant_idx:02d}"
                np.save(sample_dir / f"{variant_name}_prob.npy", pred.numpy())
                np.save(
                    sample_dir / f"{variant_name}_bin.npy",
                    (pred.numpy() > threshold).astype(np.uint8),
                )

                save_dual_projection(
                    out_dir=sample_dir,
                    name=variant_name,
                    volume=pred,
                    threshold=threshold,
                )

                if context_vis is not None:
                    stride = _compute_context_stride(pred, context_vis)
                    save_3d_context_visualization(
                        out_dir=sample_dir,
                        name=variant_name,
                        prediction=pred,
                        context=context_vis,
                        threshold=threshold,
                        angles=tuple(args.angles),
                        context_stride=stride,
                    )
                else:
                    save_3d_context_visualization(
                        out_dir=sample_dir,
                        name=variant_name,
                        prediction=pred,
                        context=None,
                        threshold=threshold,
                        angles=tuple(args.angles),
                    )


if __name__ == "__main__":
    main()
