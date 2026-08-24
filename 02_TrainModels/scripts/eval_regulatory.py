"""Evaluate regulatory accuracy (GRZ/GFZ/height) for trained models."""

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

from dataio.cond_stats import load_cond_stats, normalize_cond  # noqa: E402
from dataio.voxel_dataset import VoxelDataset, voxel_collate  # noqa: E402
from metrics import seg3d as seg_metrics  # noqa: E402
from metrics.regulatory import (  # noqa: E402
    compute_regulatory,
    relative_error,
    resolution_tolerance,
    summarize,
)
from models.ldm.diffusion_core import GaussianDiffusion  # noqa: E402
from models.ldm.diffusion_unet3d import DiffusionUNet3D  # noqa: E402
from models.ldm.vae3d import VAE3D  # noqa: E402
from models.unet.unet3d import UNet3D  # noqa: E402
from models.unet.unet3d_cond import ConditionalUNet3D  # noqa: E402
from utils.args import add_hyphenated_aliases  # noqa: E402
from utils.device import DeviceConfig, select_device, set_seed, worker_init_fn  # noqa: E402
from utils.io import ensure_dir, load_json, save_json  # noqa: E402

TARGET_SCALE_INDICES = {
    "grz": 0,
    "gfz": 1,
    "height": 2,
    "height_m": 2,
}
TARGET_SCALE_NAMES = ("grz", "gfz", "height_m")


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


def resolve_cond_dim(cfg: dict[str, object], model_kind: str) -> int:
    if model_kind == "unet":
        return 0
    if cfg.get("cond_dim"):
        return int(cfg["cond_dim"])
    if cfg.get("cond_select"):
        return len(cfg["cond_select"])
    return 3


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


def compute_vae_depth_from_cfg(cfg: dict[str, object]) -> int:
    ratio = max(int(cfg.get("input_res", 64)) // max(int(cfg.get("latent_res", 16)), 1), 1)
    return max(int(math.log2(ratio)) + 1, 2)


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
    cond_select: Sequence[int] | None,
    cond_stats,
    max_samples: int | None,
) -> VoxelDataset:
    return VoxelDataset(
        root_dir=data_root,
        split=split,
        channels=cfg.get("channels", ("C0", "C1", "C2", "C3")),
        x_indices=cfg.get("x_indices"),
        add_coords=cfg.get("add_coords", False),
        downsample_stride=cfg.get("downsample_stride", 0),
        crop_size=cfg.get("crop_size", 0),
        cond_dim=cond_dim,
        cond_select=cond_select,
        cond_stats=cond_stats,
        auto_cond_stats=False,
        augment=False,
        max_samples=max_samples,
        split_manifest=cfg.get("split_manifest"),
    )


def compute_cond_stats_from_split(
    split: str,
    cfg: dict[str, object],
    data_root: str,
    cond_dim: int,
    cond_select: Sequence[int] | None,
) -> object:
    dataset = VoxelDataset(
        root_dir=data_root,
        split=split,
        channels=cfg.get("channels", ("C0", "C1", "C2", "C3")),
        x_indices=cfg.get("x_indices"),
        add_coords=cfg.get("add_coords", False),
        downsample_stride=cfg.get("downsample_stride", 0),
        crop_size=cfg.get("crop_size", 0),
        cond_dim=cond_dim,
        cond_select=cond_select,
        cond_stats=None,
        auto_cond_stats=True,
        augment=False,
        max_samples=None,
        split_manifest=cfg.get("split_manifest"),
    )
    return dataset.cond_stats


def extract_targets(meta: dict) -> tuple[float, float, float, float, float]:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    grz = float(metrics.get("grz_target", 0.0))
    gfz = float(metrics.get("gfz_target", 0.0))
    height_m = float(metrics.get("target_height_m", 0.0))
    parcel_area_m2 = float(metrics.get("parcel_area_m2", 0.0))
    voxel_m = float(grid.get("voxel_m", 0.5))
    return grz, gfz, height_m, parcel_area_m2, voxel_m


def resolve_target_scales(scale: float | Sequence[float]) -> np.ndarray:
    if isinstance(scale, (int, float, np.floating)):
        return np.full(3, float(scale), dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scales.shape[0] != 3:
        raise ValueError(f"Expected three target scales for GRZ/GFZ/height, got {scales.shape[0]}.")
    return scales


def parse_control_targets(values: Sequence[str]) -> list[str]:
    if any(value == "all" for value in values):
        return list(TARGET_SCALE_NAMES)
    targets: list[str] = []
    for value in values:
        key = "height_m" if value == "height" else value
        if key not in TARGET_SCALE_INDICES:
            raise ValueError(f"Unsupported control target '{value}'.")
        if key not in targets:
            targets.append(key)
    return targets


def build_control_variants(
    pct: float,
    mode: str,
    control_targets: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    if pct <= 0:
        return []

    tag = round(pct * 100)
    minus = max(0.0, 1.0 - pct)
    plus = 1.0 + pct
    variants: list[tuple[str, np.ndarray]] = []

    if mode in {"coupled", "both"}:
        variants.append((f"minus{tag}", np.full(3, minus, dtype=np.float32)))
        variants.append((f"plus{tag}", np.full(3, plus, dtype=np.float32)))

    if mode in {"isolated", "both"}:
        for target in control_targets:
            idx = TARGET_SCALE_INDICES[target]
            minus_scales = np.ones(3, dtype=np.float32)
            plus_scales = np.ones(3, dtype=np.float32)
            minus_scales[idx] = minus
            plus_scales[idx] = plus
            variants.append((f"{target}_minus{tag}", minus_scales))
            variants.append((f"{target}_plus{tag}", plus_scales))

    return variants


def build_cond_vector(
    meta: dict,
    cond_select: Sequence[int] | None,
    cond_dim: int,
    cond_stats,
    scale: float | Sequence[float],
) -> np.ndarray:
    grz, gfz, height_m, _, _ = extract_targets(meta)
    raw = np.asarray([grz, gfz, height_m], dtype=np.float32)
    raw = np.maximum(raw * resolve_target_scales(scale), 0.0)
    if cond_select:
        raw = raw[list(cond_select)]
    if cond_dim:
        if raw.shape[0] > cond_dim:
            raw = raw[:cond_dim]
        elif raw.shape[0] < cond_dim:
            raw = np.pad(raw, (0, cond_dim - raw.shape[0]), constant_values=0.0)
    if cond_stats is not None:
        raw = normalize_cond(raw, cond_stats)
    return raw.astype(np.float32, copy=False)


def init_metric_bucket() -> dict[str, list[float]]:
    return {
        "pred": [],
        "target": [],
        "abs_err": [],
        "abs_err_adj": [],
        "rel_err": [],
        "rel_err_adj": [],
        "tol_abs": [],
    }


def init_variant_bucket() -> dict[str, dict[str, list[float]]]:
    return {
        "grz": init_metric_bucket(),
        "gfz": init_metric_bucket(),
        "height_m": init_metric_bucket(),
    }


def append_metric(
    bucket: dict[str, list[float]], pred: float, target: float, tol_abs: float
) -> None:
    errs = relative_error(pred, target, tol_abs)
    bucket["pred"].append(pred)
    bucket["target"].append(target)
    bucket["abs_err"].append(errs["abs_err"])
    bucket["abs_err_adj"].append(errs["abs_err_adj"])
    bucket["rel_err"].append(errs["rel_err"])
    bucket["rel_err_adj"].append(errs["rel_err_adj"])
    bucket["tol_abs"].append(tol_abs)


def finalize_variant(
    bucket: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        key: {
            "pred": summarize(values["pred"]),
            "target": summarize(values["target"]),
            "abs_err": summarize(values["abs_err"]),
            "abs_err_adj": summarize(values["abs_err_adj"]),
            "rel_err": summarize(values["rel_err"]),
            "rel_err_adj": summarize(values["rel_err_adj"]),
            "tol_abs": summarize(values["tol_abs"]),
        }
        for key, values in bucket.items()
    }


def compute_metrics_for_logits(  # noqa: PLR0917
    logits: torch.Tensor,
    metas: Sequence[dict],
    threshold: float,
    downsample_stride: int,
    storey_height_m: float,
    target_scale: float | Sequence[float],
) -> dict[str, dict[str, list[float]]]:
    bucket = init_variant_bucket()
    probs = torch.sigmoid(logits).cpu()
    target_scales = resolve_target_scales(target_scale)
    for idx in range(probs.size(0)):
        meta = metas[idx]
        grz_t, gfz_t, height_t, parcel_area_m2, voxel_m = extract_targets(meta)
        voxel_eff = float(voxel_m) * float(downsample_stride)
        pred_metrics = compute_regulatory(
            probs[idx], parcel_area_m2, voxel_eff, storey_height_m, threshold=threshold
        )
        tol_grz, tol_gfz, tol_height = resolution_tolerance(
            parcel_area_m2, voxel_eff, storey_height_m
        )
        grz_target = grz_t * float(target_scales[0])
        gfz_target = gfz_t * float(target_scales[1])
        height_target = height_t * float(target_scales[2])

        append_metric(bucket["grz"], pred_metrics["grz"], grz_target, tol_grz)
        append_metric(bucket["gfz"], pred_metrics["gfz"], gfz_target, tol_gfz)
        append_metric(bucket["height_m"], pred_metrics["height_m"], height_target, tol_height)
    return bucket


def predict_unet_logits(
    model,
    vox: torch.Tensor,
    cond: torch.Tensor | None,
    cond_dim: int,
) -> torch.Tensor:
    if cond_dim > 0 and cond is not None:
        return model(vox, cond)
    return model(vox)


def predict_ldm_logits(  # noqa: PLR0917
    vae: VAE3D,
    diff_unet: DiffusionUNet3D,
    diffusion: GaussianDiffusion,
    cond: torch.Tensor,
    context: torch.Tensor,
    cfg: dict[str, object],
    sampler: str,
    cfg_scale: float,
    sample_timesteps: int,
    ddim_eta: float,
    device: torch.device,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    latent_channels = int(cfg.get("latent_channels", 4))
    latent_res = int(cfg.get("latent_res", 16))
    if sampler == "ddim":
        latents = diffusion.sample_ddim(
            diff_unet,
            (cond.size(0), latent_channels, latent_res, latent_res, latent_res),
            cond,
            cfg_scale,
            device,
            sample_timesteps,
            ddim_eta,
            context=context,
        )
    else:
        latents = diffusion.sample_ddpm(
            diff_unet,
            (cond.size(0), latent_channels, latent_res, latent_res, latent_res),
            cond,
            cfg_scale,
            device,
            context=context,
        )
    recon = vae.decode(latents)
    if recon.dim() == 5 and recon.size(1) > 1:
        recon = recon[:, :1]
    if recon.shape[-3:] != target_shape:
        recon = F.interpolate(recon, size=target_shape, mode="trilinear", align_corners=False)
    return recon


def parse_args():
    parser = argparse.ArgumentParser(description="Regulatory evaluation for trained models.")
    parser.add_argument("--model", required=True, choices=("unet", "cond_unet", "ldm"))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cond_stats_path", type=str, default=None)
    parser.add_argument(
        "--cond_stats_split", type=str, default="train", choices=("train", "val", "test")
    )
    parser.add_argument("--off_target_pct", type=float, default=0.1)
    parser.add_argument(
        "--target_scale_mode", type=str, default="coupled", choices=("coupled", "isolated", "both")
    )
    parser.add_argument(
        "--control_targets",
        nargs="+",
        default=["all"],
        choices=("all", "grz", "gfz", "height", "height_m"),
        help="Targets to scale when --target_scale_mode is isolated or both.",
    )
    parser.add_argument("--storey_height_m", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--auto_thresh", action=BooleanOptionalAction, default=None)
    parser.add_argument("--thresh_grid", nargs="+", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sampler", type=str, default=None, choices=("ddpm", "ddim"))
    parser.add_argument("--sample_timesteps", type=int, default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    add_hyphenated_aliases(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint_path, args.config)
    data_root = args.data_root or cfg.get("data_root")
    if not data_root:
        raise ValueError("data_root missing (set --data_root or include in config).")
    device_cfg = resolve_device(args.device)
    ensure_dir(Path(args.out_dir))

    cond_select = cfg.get("cond_select")
    cond_dim = resolve_cond_dim(cfg, args.model)
    if cond_select is not None:
        cond_select = list(cond_select)
    downsample_stride = int(cfg.get("downsample_stride", 0) or 1)
    if downsample_stride <= 0:
        downsample_stride = 1

    cond_stats = None
    if cond_dim > 0:
        cond_stats_path = args.cond_stats_path or cfg.get("cond_stats_path")
        if cond_stats_path:
            cond_stats = load_cond_stats(cond_stats_path)
        else:
            cond_stats = compute_cond_stats_from_split(
                args.cond_stats_split, cfg, data_root, cond_dim, cond_select
            )

    dataset = build_dataset(
        args.split,
        cfg,
        data_root,
        cond_dim,
        cond_select,
        cond_stats,
        args.max_samples,
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
    total_batches = len(dataloader)

    def _progress(phase: str, idx: int) -> None:
        if args.log_every <= 0 or total_batches <= 0:
            return
        if idx % args.log_every == 0 or idx + 1 == total_batches:
            percent = (idx + 1) / total_batches * 100.0
            line = f"\r[{phase}] {idx + 1}/{total_batches} ({percent:5.1f}%)"
            if idx + 1 == total_batches:
                print(line)
            else:
                print(line, end="", flush=True)

    if args.model == "ldm":
        in_channels = infer_target_channels(dataset)
    else:
        in_channels = len(dataset.x_indices) + (3 if dataset.add_coords else 0)
    model = None
    vae = None
    diff_unet = None
    diffusion = None

    if args.model == "unet":
        model = UNet3D(
            in_channels=in_channels, base_channels=cfg.get("base_ch", 16), depth=cfg.get("depth", 4)
        )
    elif args.model == "cond_unet":
        model = ConditionalUNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
            cond_dim=cond_dim,
        )
    else:
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
        diffusion = GaussianDiffusion(
            cfg.get("train_timesteps", 1000), cfg.get("noise_schedule", "linear")
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    if args.model in {"unet", "cond_unet"}:
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
        model.eval()
    else:
        if not isinstance(state, dict) or "vae" not in state or "diff_unet" not in state:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is not a valid LDM checkpoint "
                "(required keys: 'vae', 'diff_unet')."
            )
        load_state_dict_strict(vae, state["vae"], "vae", checkpoint_path)
        diff_weights, diff_source = select_ldm_diffusion_weights(state, checkpoint_path)
        load_state_dict_strict(
            diff_unet, diff_weights, f"diff_unet[{diff_source}]", checkpoint_path
        )
        print(f"Using diffusion weights from checkpoint field '{diff_source}'.")
        vae.to(device_cfg.device)
        diff_unet.to(device_cfg.device)
        vae.eval()
        diff_unet.eval()

    use_auto_thresh = (
        cfg.get("auto_thresh", False) if args.auto_thresh is None else args.auto_thresh
    )
    thresh_grid = args.thresh_grid or cfg.get("thresh_grid", [0.3, 0.5, 0.7])
    threshold = args.threshold

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    metas_all: list[dict] = []

    sampler = args.sampler or cfg.get("sampler", "ddim")
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 1.0))
    sample_timesteps = int(
        args.sample_timesteps
        if args.sample_timesteps is not None
        else cfg.get("sample_timesteps", 50)
    )
    ddim_eta = float(args.ddim_eta if args.ddim_eta is not None else cfg.get("ddim_eta", 0.0))

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            _progress("eval", idx)
            vox = batch["voxels"].to(device_cfg.device)
            target = batch["target"].to(device_cfg.device)
            metas = batch["meta"]

            cond_batch = None
            if cond_dim > 0:
                cond_vecs = [
                    build_cond_vector(meta, cond_select, cond_dim, cond_stats, 1.0)
                    for meta in metas
                ]
                cond_batch = torch.from_numpy(np.stack(cond_vecs)).to(device_cfg.device)

            if args.model in {"unet", "cond_unet"}:
                logits = predict_unet_logits(model, vox, cond_batch, cond_dim)
            else:
                logits = predict_ldm_logits(
                    vae,
                    diff_unet,
                    diffusion,
                    cond_batch,
                    vox,
                    cfg,
                    sampler,
                    cfg_scale,
                    sample_timesteps,
                    ddim_eta,
                    device_cfg.device,
                    target.shape[-3:],
                )

            logits_all.append(logits.cpu())
            targets_all.append(target.cpu())
            metas_all.extend(metas)

    logits_tensor = torch.cat(logits_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)

    if threshold is None:
        if use_auto_thresh:
            if args.split == "test":
                # Avoid selecting thresholds directly on test predictions.
                threshold = resolve_threshold_from_artifacts(checkpoint_path, default=0.5)
                use_auto_thresh = False
            else:
                threshold, _ = seg_metrics.auto_threshold(
                    logits_tensor, targets_tensor, thresh_grid
                )
        else:
            threshold = 0.5

    control_targets = parse_control_targets(args.control_targets)
    variant_specs = []
    if cond_dim > 0 and args.off_target_pct and args.off_target_pct > 0:
        variant_specs = build_control_variants(
            float(args.off_target_pct),
            args.target_scale_mode,
            control_targets,
        )

    results: dict[str, object] = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "count": int(logits_tensor.size(0)),
        "threshold": float(threshold),
        "auto_thresh": bool(use_auto_thresh),
        "downsample_stride": int(downsample_stride),
        "storey_height_m": float(args.storey_height_m),
        "off_target_pct": float(args.off_target_pct),
        "target_scale_mode": args.target_scale_mode,
        "control_targets": control_targets,
        "variant_scales": {
            name: [float(value) for value in scales.tolist()] for name, scales in variant_specs
        },
        "variants": {},
    }

    baseline_bucket = compute_metrics_for_logits(
        logits_tensor, metas_all, threshold, downsample_stride, args.storey_height_m, 1.0
    )
    results["variants"]["gt"] = finalize_variant(baseline_bucket)

    if cond_dim > 0:
        for name, scales in variant_specs:
            variant_bucket = init_variant_bucket()
            if args.model == "ldm":
                # Keep diffusion noise sequence identical across offset variants.
                set_seed(args.seed)
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    _progress(f"eval-{name}", idx)
                    vox = batch["voxels"].to(device_cfg.device)
                    metas = batch["meta"]
                    cond_vecs = [
                        build_cond_vector(meta, cond_select, cond_dim, cond_stats, scales)
                        for meta in metas
                    ]
                    cond_batch = torch.from_numpy(np.stack(cond_vecs)).to(device_cfg.device)

                    if args.model == "cond_unet":
                        logits = predict_unet_logits(model, vox, cond_batch, cond_dim)
                    else:
                        logits = predict_ldm_logits(
                            vae,
                            diff_unet,
                            diffusion,
                            cond_batch,
                            vox,
                            cfg,
                            sampler,
                            cfg_scale,
                            sample_timesteps,
                            ddim_eta,
                            device_cfg.device,
                            batch["target"].shape[-3:],
                        )

                    batch_bucket = compute_metrics_for_logits(
                        logits.cpu(),
                        metas,
                        threshold,
                        downsample_stride,
                        args.storey_height_m,
                        scales,
                    )
                    for metric_key in variant_bucket:
                        for entry_key in variant_bucket[metric_key]:
                            variant_bucket[metric_key][entry_key].extend(
                                batch_bucket[metric_key][entry_key]
                            )
            results["variants"][name] = finalize_variant(variant_bucket)

    save_json(Path(args.out_dir) / "reg_metrics.json", results)


if __name__ == "__main__":
    main()
