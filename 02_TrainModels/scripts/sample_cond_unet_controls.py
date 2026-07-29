"""Generate control variants for selected conditional UNet samples.

This script renders multiple conditioning-scale variants for the same
context/sample, e.g. gt, -10%, +10%, etc., and stores each variant in its own
subdirectory below the sample directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from models.unet.unet3d_cond import ConditionalUNet3D
from utils.args import add_hyphenated_aliases
from utils.device import DeviceConfig, select_device, set_seed
from utils.io import ensure_dir, save_json
from utils.visuals import save_3d_context_visualization, save_dual_projection

from scripts.eval_regulatory import build_cond_vector
from scripts.infer import (
    _compute_context_stride,
    _extract_context_volume,
    build_dataset,
    load_run_config,
    load_state_dict_strict,
    predict_unet,
    resolve_threshold_from_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render conditioning control variants for a trained conditional UNet."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        required=True,
        help="Specific sample indices to render.",
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.8, 0.9, 1.1, 1.2])
    parser.add_argument("--cond_stats_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angles", nargs="+", type=int, default=[30, 60, 120])
    add_hyphenated_aliases(parser)
    return parser.parse_args()


def _resolve_label(scale: float) -> str:
    if abs(scale - 1.0) < 1e-8:
        return "gt"
    pct = round(abs(scale - 1.0) * 100)
    prefix = "plus" if scale > 1.0 else "minus"
    return f"{prefix}{pct:02d}"


def _sample_name(index: int, meta: dict) -> str:
    sample_id = meta.get("sample_id") if isinstance(meta, dict) else None
    if isinstance(sample_id, str) and sample_id:
        return f"sample_{index:04d}_{sample_id}"
    return f"sample_{index:04d}"


def _raw_targets(meta: dict) -> dict[str, float]:
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    return {
        "grz_target": float(metrics.get("grz_target", 0.0)),
        "gfz_target": float(metrics.get("gfz_target", 0.0)),
        "target_height_m": float(metrics.get("target_height_m", 0.0)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint_path, args.config)
    data_root = args.data_root or str(cfg.get("data_root", ""))
    if not data_root:
        raise ValueError("Missing data root. Pass --data_root or use a config that defines it.")

    cond_dim = int(cfg.get("cond_dim", 3))
    cond_select = cfg.get("cond_select")
    cond_stats_path = args.cond_stats_path or cfg.get("cond_stats_path") or None
    split_manifest = cfg.get("split_manifest")
    dataset = build_dataset(
        split=args.split,
        cfg=cfg,
        data_root=data_root,
        cond_dim=cond_dim,
        cond_stats_path=cond_stats_path,
        max_samples=None,
        split_manifest=split_manifest,
    )

    cond_stats = dataset.cond_stats
    if cond_dim > 0 and cond_stats is None:
        raise ValueError(
            "Conditioning stats are required for cond_unet rendering but could not be resolved."
        )

    device_cfg = select_device() if args.device is None else None
    if args.device is not None:
        request = args.device.lower()
        if request == "cpu":
            device_cfg = DeviceConfig(device=torch.device("cpu"), device_type="cpu", amp_dtype=None)
        elif request == "mps":
            device_cfg = DeviceConfig(device=torch.device("mps"), device_type="mps", amp_dtype=None)
        elif request == "cuda":
            device_cfg = DeviceConfig(
                device=torch.device("cuda"), device_type="cuda", amp_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported --device value: {args.device}")
    if device_cfg is None:
        raise RuntimeError("Device selection did not produce a usable device.")

    model = ConditionalUNet3D(
        in_channels=len(cfg.get("channels", ("C0", "C1", "C2", "C3")))
        + (3 if cfg.get("add_coords", False) else 0),
        base_channels=int(cfg.get("base_ch", 16)),
        depth=int(cfg.get("depth", 4)),
        cond_dim=cond_dim,
        dropout=float(cfg.get("cond_drop", 0.1)),
    ).to(device_cfg.device)

    state = torch.load(checkpoint_path, map_location=device_cfg.device)
    model_state = state.get("model", state)
    load_state_dict_strict(model, model_state, "model", checkpoint_path)
    model.eval()

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else resolve_threshold_from_artifacts(checkpoint_path, default=0.5)
    )
    use_tta = bool(cfg.get("tta", False))
    tta_mode = str(cfg.get("tta_mode", "rot90"))
    out_dir = ensure_dir(Path(args.out_dir))

    manifest = {
        "checkpoint": str(checkpoint_path),
        "config": str(Path(args.config).resolve()) if args.config else None,
        "data_root": str(Path(data_root).resolve()),
        "split": args.split,
        "sample_ids": [int(x) for x in args.sample_ids],
        "scales": [float(x) for x in args.scales],
        "threshold": threshold,
        "tta": use_tta,
        "tta_mode": tta_mode,
    }
    save_json(out_dir / "manifest.json", manifest)

    for sample_index in args.sample_ids:
        if sample_index < 0 or sample_index >= len(dataset):
            print(f"Skipping invalid sample index {sample_index} (dataset size: {len(dataset)})")
            continue

        sample = dataset[sample_index]
        meta = sample["meta"]
        sample_name = _sample_name(sample_index, meta)
        sample_dir = ensure_dir(out_dir / sample_name)
        context_vis = _extract_context_volume(sample["path"])

        save_json(
            sample_dir / "sample.json",
            {
                "sample_index": int(sample_index),
                "sample_path": str(sample["path"]),
                "sample_name": sample_name,
                "meta": meta,
            },
        )

        target_ref = sample["target"]
        target_name = f"{sample_name}_target"
        save_dual_projection(
            out_dir=sample_dir,
            name=target_name,
            volume=target_ref,
            threshold=0.5,
        )
        if context_vis is not None:
            target_stride = _compute_context_stride(target_ref, context_vis)
            save_3d_context_visualization(
                out_dir=sample_dir,
                name=target_name,
                prediction=target_ref,
                context=context_vis,
                threshold=0.5,
                angles=tuple(args.angles),
                context_stride=target_stride,
            )
        else:
            save_3d_context_visualization(
                out_dir=sample_dir,
                name=target_name,
                prediction=target_ref,
                context=None,
                threshold=0.5,
                angles=tuple(args.angles),
            )

        vox = sample["voxels"].unsqueeze(0)
        cond_rows: list[np.ndarray] = []
        labels: list[str] = []
        scales: list[float] = []
        for scale in args.scales:
            labels.append(_resolve_label(float(scale)))
            scales.append(float(scale))
            cond_rows.append(
                build_cond_vector(meta, cond_select, cond_dim, cond_stats, float(scale))
            )

        cond_tensor = torch.from_numpy(np.stack(cond_rows, axis=0))
        batch = {
            "voxels": vox.repeat(len(scales), 1, 1, 1, 1),
            "cond": cond_tensor,
        }

        logits = predict_unet(model, batch, cond_dim, device_cfg.device, use_tta, tta_mode)
        probs = torch.sigmoid(logits).detach().cpu()

        print(f"Rendering {sample_name} -> {', '.join(labels)}")
        for variant_idx, (label, scale) in enumerate(zip(labels, scales, strict=False)):
            variant_dir = ensure_dir(sample_dir / label)
            pred = probs[variant_idx]
            variant_name = f"{sample_name}_{label}"
            np.save(variant_dir / f"{variant_name}_prob.npy", pred.numpy())
            np.save(
                variant_dir / f"{variant_name}_bin.npy", (pred.numpy() > threshold).astype(np.uint8)
            )

            save_json(
                variant_dir / "variant.json",
                {
                    "label": label,
                    "scale": float(scale),
                    "threshold": threshold,
                    "target_raw": _raw_targets(meta),
                    "target_scaled": {
                        key: value * float(scale) for key, value in _raw_targets(meta).items()
                    },
                },
            )

            save_dual_projection(
                out_dir=variant_dir,
                name=variant_name,
                volume=pred,
                threshold=threshold,
            )

            if context_vis is not None:
                stride = _compute_context_stride(pred, context_vis)
                save_3d_context_visualization(
                    out_dir=variant_dir,
                    name=variant_name,
                    prediction=pred,
                    context=context_vis,
                    threshold=threshold,
                    angles=tuple(args.angles),
                    context_stride=stride,
                )
            else:
                save_3d_context_visualization(
                    out_dir=variant_dir,
                    name=variant_name,
                    prediction=pred,
                    context=None,
                    threshold=threshold,
                    angles=tuple(args.angles),
                )


if __name__ == "__main__":
    main()
