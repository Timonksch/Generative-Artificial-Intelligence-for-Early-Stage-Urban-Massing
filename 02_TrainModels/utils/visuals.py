"""Visualization helpers for 3D volumes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_DISPLAY_VOXELS = 120_000


def _prepare_volume(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 3D tensor (D,H,W) from various input shapes."""
    vol = tensor.detach().cpu()
    while vol.dim() > 3 and vol.size(0) == 1:
        vol = vol.squeeze(0)
    if vol.dim() > 3:
        vol = vol[0]
    if vol.dim() != 3:
        raise ValueError(f"Expected tensor reducible to 3 dims, got shape {vol.shape}")
    return vol


def save_max_projection(
    out_dir: Path | str,
    name: str,
    target: torch.Tensor,
    prob: torch.Tensor,
    threshold: float = 0.5,
) -> Path:
    """Save XY/XZ/YZ maximum projections for ground truth and predictions as heatmaps."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_vol = _prepare_volume(target.float())
    prob_vol = _prepare_volume(prob)

    projections: tuple[tuple[torch.Tensor, str], ...] = (
        (target_vol.max(dim=0).values, "XY (Top)"),
        (target_vol.max(dim=1).values, "XZ (Front)"),
        (target_vol.max(dim=2).values, "YZ (Side)"),
    )
    pred_projections: tuple[tuple[torch.Tensor, str], ...] = (
        (prob_vol.max(dim=0).values, "XY (Top)"),
        (prob_vol.max(dim=1).values, "XZ (Front)"),
        (prob_vol.max(dim=2).values, "YZ (Side)"),
    )

    fig, axes_plots = plt.subplots(2, 3, figsize=(15, 10))

    # Original/Target row
    for col, (gt_proj, tag) in enumerate(projections):
        axes_plots[0, col].imshow(gt_proj.numpy(), cmap="viridis", origin="lower")
        axes_plots[0, col].set_title(f"Original - {tag}")
        axes_plots[0, col].axis("off")

    # Prediction/Reconstruction row
    for col, (pred_proj, tag) in enumerate(pred_projections):
        axes_plots[1, col].imshow(pred_proj.numpy(), cmap="viridis", origin="lower")
        axes_plots[1, col].set_title(f"Prediction - {tag}")
        axes_plots[1, col].axis("off")

    fig.suptitle(name, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06, wspace=0.08, hspace=0.15)
    out_file = out_path / f"{name}_projections.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def save_dual_projection(
    out_dir: Path | str,
    name: str,
    volume: torch.Tensor,
    threshold: float = 0.5,
) -> Path:
    """Save XY/XZ/YZ maximum projections with dual view: heatmap + binary.

    Args:
        out_dir: Output directory path
        name: Base name for the output file
        volume: 3D volume tensor with continuous values (0-1)
        threshold: Threshold for binary visualization

    Returns:
        Path to saved PNG file

    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vol = _prepare_volume(volume)
    vol_binary = (vol > threshold).float()

    # Compute projections for both heatmap and binary
    projections_continuous = [
        vol.max(dim=0).values.numpy(),
        vol.max(dim=1).values.numpy(),
        vol.max(dim=2).values.numpy(),
    ]
    projections_binary = [
        vol_binary.max(dim=0).values.numpy(),
        vol_binary.max(dim=1).values.numpy(),
        vol_binary.max(dim=2).values.numpy(),
    ]

    titles = ["XY (Top)", "XZ (Front)", "YZ (Side)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Heatmap (continuous values)
    for col in range(3):
        axes[0, col].imshow(projections_continuous[col], cmap="viridis", origin="lower")
        axes[0, col].set_title(f"Heatmap - {titles[col]}")
        axes[0, col].axis("off")

    # Row 2: Binary (threshold applied)
    for col in range(3):
        axes[1, col].imshow(projections_binary[col], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"Binary (>{threshold}) - {titles[col]}")
        axes[1, col].axis("off")

    fig.suptitle(name, fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06, wspace=0.08, hspace=0.18)
    out_file = out_path / f"{name}_dual.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def save_3d_context_visualization(  # noqa: PLR0917
    out_dir: Path | str,
    name: str,
    prediction: torch.Tensor,
    context: torch.Tensor | None = None,
    threshold: float = 0.5,
    angles: Sequence[int] = (30, 60, 120),
    context_stride: int = 4,
) -> Path:
    """Save 3D voxel visualization showing prediction in context of surrounding buildings.

    Args:
        out_dir: Output directory path
        name: Base name for the output file
        prediction: 3D volume tensor of the predicted building (0-1)
        context: Optional 3D volume tensor of surrounding buildings (0-1)
        threshold: Threshold for binary voxelization
        angles: Camera azimuth angles for different views
        context_stride: Downsample stride for context to reduce visual clutter

    Returns:
        Path to saved PNG file

    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Prepare prediction voxels
    pred_vol = _prepare_volume(prediction)
    pred_binary = (pred_vol > threshold).cpu().numpy().astype(bool)
    aligned_context = None
    if context is not None:
        ctx_vol = _prepare_volume(context)
        ctx_binary = (ctx_vol > threshold).cpu().numpy().astype(bool)
        aligned_context = _resample_volume(ctx_binary, pred_binary.shape, context_stride)
    else:
        aligned_context = None

    combined_shape = pred_binary.shape
    total_cells = np.prod(combined_shape)
    stride = max(1, round((total_cells / max(1, MAX_DISPLAY_VOXELS)) ** (1 / 3)))
    stride = max(1, stride)

    pred_display = _downsample_stride(pred_binary, stride)
    context_display = (
        _downsample_stride(aligned_context, stride) if aligned_context is not None else None
    )

    pred_voxels = pred_display.transpose(2, 1, 0)
    pred_edges = _voxel_edges(pred_voxels.shape, stride)

    context_voxels = None
    context_edges = None
    if context_display is not None and context_display.any():
        context_voxels = context_display.transpose(2, 1, 0)
        context_edges = _voxel_edges(context_voxels.shape, stride)

    real_dims = np.array(pred_vol.shape[::-1], dtype=float)
    if aligned_context is not None:
        ctx_dims = np.array(aligned_context.shape[::-1], dtype=float) * stride
        real_dims = np.maximum(real_dims, ctx_dims)

    # Create visualization for each angle
    n_angles = len(angles)
    fig = plt.figure(figsize=(4.5 * n_angles, 4.5))
    fig.patch.set_facecolor("white")

    for idx, azim in enumerate(angles):
        ax = fig.add_subplot(1, n_angles, idx + 1, projection="3d")
        ax.set_facecolor("white")

        if pred_voxels.any():
            pred_colors = np.zeros((*pred_voxels.shape, 4), dtype=np.float32)
            pred_colors[..., 0] = 0.12
            pred_colors[..., 1] = 0.35
            pred_colors[..., 2] = 0.72
            pred_colors[..., 3] = 0.92
            ax.voxels(
                *pred_edges,
                pred_voxels,
                facecolors=pred_colors,
                edgecolor="none",
                linewidth=0.0,
                label="Prediction",
            )

        if context_voxels is not None:
            ctx_colors = np.zeros((*context_voxels.shape, 4), dtype=np.float32)
            ctx_colors[..., :3] = 0.7
            ctx_colors[..., 3] = 0.18
            ax.voxels(
                *context_edges,
                context_voxels,
                facecolors=ctx_colors,
                edgecolor="none",
                linewidth=0.0,
                label="Context",
            )

        ax.view_init(elev=20, azim=azim)
        ax.set_axis_off()

        norm_dims = real_dims / max(real_dims.max(), 1.0)
        ax.set_box_aspect(norm_dims)
        ax.set_xlim(0, real_dims[0])
        ax.set_ylim(0, real_dims[1])
        ax.set_zlim(0, real_dims[2])
        ax.text2D(0.05, 0.88, f"{azim}°", transform=ax.transAxes, fontsize=12)

        # legend intentionally omitted for cleaner view

    fig.suptitle(f"{name} - Prediction in Urban Context", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.05)

    out_file = out_path / f"{name}_3d_context.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)

    return out_file


def _resample_volume(
    volume: np.ndarray | None,
    target_shape: tuple[int, int, int],
    stride_hint: int,
) -> np.ndarray | None:
    if volume is None:
        return None
    src = np.array(volume.shape, dtype=int)
    tgt = np.array(target_shape, dtype=int)
    out = volume
    for axis in range(3):
        if src[axis] == tgt[axis]:
            continue
        if src[axis] > tgt[axis]:
            factor = src[axis] // tgt[axis]
            out = _downsample_axis(out, axis, factor)
        else:
            factor = tgt[axis] // src[axis]
            out = np.repeat(out, factor, axis=axis)
        src = np.array(out.shape, dtype=int)
    return out


def _downsample_axis(volume: np.ndarray, axis: int, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume
    length = volume.shape[axis]
    trimmed = length - (length % factor)
    slicer = [slice(None)] * volume.ndim
    slicer[axis] = slice(0, trimmed)
    vol = volume[tuple(slicer)]
    vol = np.moveaxis(vol, axis, 0)
    new_shape = (trimmed // factor, factor, *vol.shape[1:])
    vol = vol.reshape(new_shape)
    vol = vol.any(axis=1)
    return np.moveaxis(vol, 0, axis)


def _downsample_stride(volume: np.ndarray | None, stride: int) -> np.ndarray | None:
    if volume is None or stride <= 1:
        return volume
    return _block_downsample(volume, stride)


def _voxel_edges(
    shape: tuple[int, int, int], scale: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = max(1, scale)
    x = np.arange(shape[0] + 1, dtype=float) * scale
    y = np.arange(shape[1] + 1, dtype=float) * scale
    z = np.arange(shape[2] + 1, dtype=float) * scale
    return np.meshgrid(x, y, z, indexing="ij")


def _block_downsample(volume: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return volume
    dims = np.array(volume.shape)
    new_dims = dims - (dims % stride)
    slicer = tuple(slice(0, nd) for nd in new_dims)
    vol = volume[slicer]
    vol = vol.reshape(
        new_dims[0] // stride,
        stride,
        new_dims[1] // stride,
        stride,
        new_dims[2] // stride,
        stride,
    )
    return vol.any(axis=(1, 3, 5))
