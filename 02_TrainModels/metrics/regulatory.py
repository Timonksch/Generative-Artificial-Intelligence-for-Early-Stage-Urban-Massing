"""Regulatory metrics for voxelized massing."""

from __future__ import annotations

import numpy as np
import torch


def _squeeze_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4 and mask.size(0) == 1:
        return mask[0]
    return mask


def compute_regulatory(
    mask: torch.Tensor,
    parcel_area_m2: float,
    voxel_m: float,
    storey_height_m: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute GRZ, GFZ, and height from a binary voxel mask."""
    mask = _squeeze_mask(mask)
    if mask.numel() == 0:
        return {"grz": 0.0, "gfz": 0.0, "height_m": 0.0}

    occ = mask > float(threshold)
    if not torch.any(occ):
        return {"grz": 0.0, "gfz": 0.0, "height_m": 0.0}

    footprint = torch.any(occ, dim=0)
    foot_cells = float(footprint.sum().item())
    vol_cells = float(occ.sum().item())

    z_any = torch.any(occ, dim=(1, 2))
    z_idx = torch.nonzero(z_any, as_tuple=False).flatten()
    height_m = 0.0
    if z_idx.numel() > 0:
        height_m = float((z_idx.max() - z_idx.min() + 1).item()) * float(voxel_m)

    voxel_area = float(voxel_m) ** 2
    voxel_vol = float(voxel_m) ** 3
    footprint_area = foot_cells * voxel_area
    volume_m3 = vol_cells * voxel_vol

    if parcel_area_m2 > 0.0:
        grz = footprint_area / float(parcel_area_m2)
        gfz = (
            volume_m3 / (float(parcel_area_m2) * float(storey_height_m))
            if storey_height_m > 0.0
            else 0.0
        )
    else:
        grz = 0.0
        gfz = 0.0

    return {"grz": float(grz), "gfz": float(gfz), "height_m": float(height_m)}


def resolution_tolerance(
    parcel_area_m2: float,
    voxel_m: float,
    storey_height_m: float,
) -> tuple[float, float, float]:
    """Return absolute tolerances for GRZ, GFZ, and height from one-voxel resolution."""
    voxel_area = float(voxel_m) ** 2
    voxel_vol = float(voxel_m) ** 3
    tol_grz = voxel_area / float(parcel_area_m2) if parcel_area_m2 > 0.0 else 0.0
    tol_gfz = (
        voxel_vol / (float(parcel_area_m2) * float(storey_height_m))
        if parcel_area_m2 > 0.0 and storey_height_m > 0.0
        else 0.0
    )
    tol_height = float(voxel_m)
    return tol_grz, tol_gfz, tol_height


def relative_error(pred: float, target: float, tol_abs: float) -> dict[str, float]:
    """Return absolute and relative errors with a resolution tolerance."""
    abs_err = abs(pred - target)
    abs_err_adj = max(0.0, abs_err - tol_abs)
    denom = max(abs(target), 1e-6)
    rel_err = abs_err / denom
    rel_err_adj = abs_err_adj / denom
    return {
        "abs_err": float(abs_err),
        "abs_err_adj": float(abs_err_adj),
        "rel_err": float(rel_err),
        "rel_err_adj": float(rel_err_adj),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}
