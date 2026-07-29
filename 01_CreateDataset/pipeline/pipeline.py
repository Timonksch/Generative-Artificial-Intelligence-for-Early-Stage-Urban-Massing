#!/usr/bin/env python3
"""Orchestrate parcel selection, voxelization, filtering, and sample output."""

# RULE_VIOLATION: Uppercase C0-C3/X/Y names intentionally mirror the published array schema.
# ruff: noqa: N802, N803, N806

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Any

import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree

from .data_io import BuildingDatabase, CachedBuildingPart, create_building_database, load_parcels
from .geo_utils import (
    binary_erode,
    extract_polygons,
    extrude_mask,
    lines_from_polygons,
    lines_to_mask,
    polygon_to_mask,
    robust_clean,
    safe_intersection,
    safe_union,
    safe_within,
    touches_border,
)
from .streets_wfs import street_mask
from .viz import make_overview_png, save_voxels_hires

np.seterr(invalid="ignore")

MINIMUM_COMPARISON_NEIGHBORS = 2
HIGH_RESOLUTION_GRID = 256

# ---------------------------------------- Helper Functions


def analyze_z_window(
    parts: list[CachedBuildingPart], z_margin_m: float, min_z_range_m: float = 12.0
) -> tuple[float, float]:
    """Calculate Z-window with margin and minimum range constraint.

    Args:
        parts: List of building parts with z_min/z_max attributes.
        z_margin_m: Vertical margin to add (meters).
        min_z_range_m: Minimum Z-range (meters).

    Returns:
        Tuple[float, float]: (z_min, z_max) in world coordinates.

    """
    if not parts:
        return 0.0, min_z_range_m
    amin = min(p.z_min for p in parts)
    amax = max(p.z_max for p in parts)
    zmin = amin - z_margin_m
    zmax = max(amax + z_margin_m, zmin + min_z_range_m)
    return float(zmin), float(zmax)


# RULE_VIOLATION: Preserve the published channel-builder API used by downstream scripts.
def build_C1_neighbor_height(  # noqa: PLR0913, PLR0917
    neighbor_parts: list[CachedBuildingPart],
    origin_xy: tuple[float, float],
    voxel_m: float,
    W: int,
    H: int,
    D: int,
    raster_ss: int = 2,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build C1 channel: 2D neighbor height heatmap with global normalization.

    Normalization: height_norm = clamp(height_m / H_max, 0..1).

    Args:
        neighbor_parts: List of neighbor building parts.
        origin_xy: Grid origin (minx, miny).
        voxel_m: Voxel size in meters.
        W: Grid width.
        H: Grid height.
        D: Grid depth.
        raster_ss: Rasterization supersampling factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (C1 volume, height_grid 2D).

    """
    height_grid = np.zeros((H, W), dtype=np.float32)
    H_max = float(D) * float(voxel_m)
    if H_max <= 0:
        H_max = 1.0

    for p in neighbor_parts:
        h_m = max(0.0, float(p.z_max) - float(p.z_min))
        h_norm = 0.0 if H_max <= 0 else min(1.0, h_m / H_max)
        if h_norm <= 0:
            continue
        m2d = polygon_to_mask(
            robust_clean(p.footprint), W, H, origin_xy, voxel_m, ss=raster_ss, thresh=0.5
        )
        height_grid = np.maximum(height_grid, m2d.astype(np.float32) * h_norm)

    C1 = np.repeat(height_grid[None, :, :], D, axis=0).astype(np.float32)
    return C1, height_grid


def _iter_tree_query(res: Any) -> list[Any]:
    """Normalize STRtree.query result to Python list.

    Args:
        res: Result from STRtree.query (array, list, or single value).

    Returns:
        List: Normalized list of indices or geometries.

    """
    if res is None:
        return []
    if isinstance(res, np.ndarray):
        return res.tolist()
    if isinstance(res, (list, tuple)):
        return list(res)
    return [res]


def get_parcel_id(parcel_data: dict[str, Any], idx: int) -> str:
    """Extract unique parcel ID from properties or generate fallback.

    Args:
        parcel_data: Parcel dictionary with 'props' key.
        idx: Parcel index for fallback ID.

    Returns:
        str: Unique parcel identifier.

    """
    props = parcel_data.get("props", {})
    return props.get("uuid") or props.get("id") or f"parcel_{idx:06d}"


def sample_exists(out_dir: str, parcel_id: str) -> bool:
    """Check if sample already exists in output directory.

    Args:
        out_dir: Output directory path.
        parcel_id: Parcel identifier.

    Returns:
        bool: True if NPZ file exists.

    """
    sample_dir = os.path.join(out_dir, parcel_id.replace(os.sep, "_"))
    npz_path = os.path.join(sample_dir, f"{parcel_id}.npz")
    return os.path.exists(npz_path)


def list_successful_parcel_ids(dataset_dir: str) -> set[str]:
    """Scan dataset directory and return set of successfully generated parcel IDs.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Set[str]: Set of parcel IDs with completed samples.

    """
    successful_ids = set()
    if not os.path.isdir(dataset_dir):
        return successful_ids
    for name in os.listdir(dataset_dir):
        sdir = os.path.join(dataset_dir, name)
        if os.path.isdir(sdir) and os.path.exists(os.path.join(sdir, f"{name}.npz")):
            successful_ids.add(name)
    return successful_ids


def _list_samples_for_testset_creation(dataset_dir: str) -> list[tuple[str, str]]:
    """Find all samples (subfolders with <pid>.npz) for testset creation.

    Args:
        dataset_dir: Source dataset directory.

    Returns:
        List[Tuple[str, str]]: List of (parcel_id, sample_dir) tuples.

    """
    samples = []
    if not os.path.isdir(dataset_dir):
        return samples
    for name in os.listdir(dataset_dir):
        sdir = os.path.join(dataset_dir, name)
        if not os.path.isdir(sdir):
            continue
        npz = os.path.join(sdir, f"{name}.npz")
        if os.path.exists(npz):
            samples.append((name, sdir))
    return samples


def _link_or_copy(src: str, dst: str, mode: str = "symlink") -> None:
    """Create symlink, hardlink, or copy with fallback.

    Args:
        src: Source file path.
        dst: Destination file path.
        mode: Link mode ('symlink', 'hardlink', or 'copy').

    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def create_testset_from_existing_samples(
    dataset_dir: str, count: int, seed: int = 42, mode: str = "symlink", force: bool = False
) -> str:
    """Create test dataset from existing samples via symlink/hardlink/copy.

    Creates 'testset_<datasetname>' directory in parent of dataset_dir.

    Args:
        dataset_dir: Source dataset directory.
        count: Number of samples to include (0 = all).
        seed: Random seed for selection.
        mode: Link mode ('symlink', 'hardlink', 'copy').
        force: Overwrite existing testset directory.

    Returns:
        str: Path to created testset directory.

    Raises:
        RuntimeError: If testset dir exists and force=False, or no samples found.

    """
    dataset_dir = os.path.abspath(dataset_dir)
    parent = os.path.dirname(dataset_dir)
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    out_dir = os.path.join(parent, f"testset_{dataset_name}")

    if os.path.exists(out_dir):
        if force:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"Testset directory already exists: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    samples = _list_samples_for_testset_creation(dataset_dir)
    if not samples:
        raise RuntimeError(f"No samples found in: {dataset_dir}")

    # RULE_VIOLATION: Seeded pseudo-random selection is required for reproducibility.
    rnd = random.Random(seed)  # noqa: S311
    rnd.shuffle(samples)
    if count <= 0 or count > len(samples):
        count = len(samples)

    sel = samples[:count]
    print(f"Creating testset with {len(sel)} samples (mode={mode}) -> {out_dir}")

    for i, (pid, sdir) in enumerate(sel):
        if (i + 1) % 100 == 0 or i == len(sel) - 1:
            print(f"  Processing sample {i + 1}/{len(sel)}: {pid}")
        dst_sdir = os.path.join(out_dir, pid)
        os.makedirs(dst_sdir, exist_ok=True)
        for fname in os.listdir(sdir):
            src = os.path.join(sdir, fname)
            dst = os.path.join(dst_sdir, fname)
            if os.path.isfile(src):
                _link_or_copy(src, dst, mode=mode)

    return out_dir


# ---------------------------------------- Core Sample Builder


# RULE_VIOLATION: Preserve the published keyword API and linear scientific filter sequence.
def build_sample(  # noqa: D417, PLR0911, PLR0912, PLR0913, PLR0915, PLR0917
    seed_idx: int,
    building_db: BuildingDatabase,
    parcels_all: list[dict[str, Any]],
    parcels_tree: STRtree,
    wkb_to_idx: dict[bytes, int],
    out_dir: str,
    grid_m: float,
    voxel_m: float,
    raster_ss: int,
    parcel_erode_vox: int,
    parcel_mask_thresh: float,
    parcel_buffer_m: float,
    grz_gfz_tol: float,
    grz_gfz_abs_tol: float,
    storey_height_m: float,
    require_two_neighbors: bool,
    min_bgf_m2: float,
    max_bgf_m2: float,
    min_neighbor_buildings: int,
    min_overlap_m2: float,
    max_property_frac: float,
    max_target_frac: float,
    skip_if_touch_border: bool,
    border_margin_vox: int,
    buildings_require_full_inside: bool,
    context_inside_tol_m: float,
    z_margin_m: float,
    overlay_stride3d: int,
    png_dpi: int,
    with_streets: bool,
    street_mode: str,
    street_width_m: float,
    wfs_url: str,
    typename: str,
    edge_width_m: float,
    street_cache_dir: str | None,
    no_viz: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Build single voxelized sample from parcel seed.

    This is the core sample generation function. It:
    1. Defines spatial window around parcel centroid
    2. Queries buildings from database
    3. Separates target (on-parcel) vs neighbor buildings
    4. Generates input channels (C0=parcel mask, C1=neighbor heights, C2=edges, C3=streets)
    5. Generates target volume Y (on-parcel buildings)
    6. Applies filters (BGF, GRZ/GFZ, border touch, etc.)
    7. Saves NPZ + JSON + visualizations

    Args:
        seed_idx: Index of seed parcel in parcels_all.
        building_db: Spatial database of buildings.
        parcels_all: Complete list of parcels.
        parcels_tree: STRtree for parcel spatial queries.
        wkb_to_idx: WKB->index mapping for parcels.
        out_dir: Output directory for sample.
        grid_m: Grid size in meters.
        voxel_m: Voxel size in meters.
        [Additional parameters - see function signature]

    Returns:
        Dict[str, Any]: Result dictionary with 'status' key:
            - 'ok': Successful sample
            - 'skip:<reason>': Skipped with reason

    """
    # Extract seed parcel
    seed_geom = robust_clean(parcels_all[seed_idx]["geom"])
    parcel_id = get_parcel_id(parcels_all[seed_idx], seed_idx)

    # Define spatial window
    cx, cy = seed_geom.centroid.x, seed_geom.centroid.y
    half = 0.5 * grid_m
    minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
    window_poly = box(minx, miny, maxx, maxy)

    # Query buildings
    cached = building_db.query_bbox((minx, miny, maxx, maxy))
    parts_all = []
    for bp in cached:
        if buildings_require_full_inside and not safe_within(
            bp.footprint, window_poly, tol_m=context_inside_tol_m
        ):
            continue
        fp = safe_intersection(bp.footprint, window_poly)
        if fp.is_empty:
            continue
        parts_all.append(type("BP", (), {"footprint": fp, "z_min": bp.z_min, "z_max": bp.z_max}))

    if not parts_all:
        return {"status": "skip:no_buildings"}

    # Clip seed parcel to window
    property_union = safe_intersection(seed_geom, window_poly)
    if property_union.is_empty:
        return {"status": "skip:empty_property"}

    # Grid dimensions
    D = H = W = round(grid_m / voxel_m)
    origin_xy = (minx, miny)
    H_max = float(D) * float(voxel_m)

    # Fixed Z-window (global height normalization)
    zmin_world_raw, _ = analyze_z_window(parts_all, z_margin_m=z_margin_m, min_z_range_m=12.0)
    zmin_world = float(zmin_world_raw)
    zmax_world = zmin_world + H_max

    # Check if any building exceeds max height
    amax = max(float(p.z_max) for p in parts_all)
    if amax > zmax_world:
        return {"status": "skip:too_tall_for_global_height"}

    # Separate target vs neighbor buildings
    target_parts, neighbor_parts = [], []
    for p in parts_all:
        inter = safe_intersection(p.footprint, property_union)
        if not inter.is_empty and inter.area >= min_overlap_m2:
            target_parts.append(p)
        else:
            neighbor_parts.append(p)

    if not target_parts:
        return {"status": "skip:no_target"}
    if len(neighbor_parts) < int(min_neighbor_buildings):
        return {"status": f"skip:few_neighbors({len(neighbor_parts)})"}

    tgt_union = safe_union(
        [safe_intersection(robust_clean(p.footprint), property_union) for p in target_parts]
    )
    if tgt_union.is_empty:
        return {"status": "skip:empty_target_fp"}

    # ---------- Build Input Channels ----------

    # C0: Parcel build mask
    parcel_fp = property_union
    if abs(parcel_buffer_m) > 0.0:
        parcel_fp = safe_intersection(
            robust_clean(property_union.buffer(float(parcel_buffer_m))), window_poly
        )

    C0_2d = polygon_to_mask(
        parcel_fp, W, H, origin_xy, voxel_m, ss=raster_ss, thresh=float(parcel_mask_thresh)
    )

    if parcel_erode_vox is not None and parcel_erode_vox > 0:
        C0_2d = binary_erode(C0_2d, iterations=int(parcel_erode_vox))

    C0 = np.repeat(C0_2d[None, :, :], D, axis=0).astype(np.uint8)

    # Target footprint mask (for filters and overlay)
    mask_target_2d = polygon_to_mask(tgt_union, W, H, origin_xy, voxel_m, ss=raster_ss, thresh=0.5)

    # Pre-filters (area fractions, border touch)
    if float(mask_target_2d.mean()) > max_target_frac:
        return {"status": "skip:target_large"}
    if float(C0_2d.mean()) > max_property_frac:
        return {"status": "skip:property_large"}
    if skip_if_touch_border and (
        touches_border(mask_target_2d, border_margin_vox)
        or touches_border(C0_2d, border_margin_vox)
    ):
        return {"status": "skip:border_touch"}

    # Y: Target volume
    Y_vols, zlo_all, zhi_all = [], [], []
    for p in target_parts:
        m2d = polygon_to_mask(
            safe_intersection(robust_clean(p.footprint), property_union),
            W,
            H,
            origin_xy,
            voxel_m,
            ss=raster_ss,
            thresh=0.5,
        )
        gz_lo = (p.z_min - zmin_world) / voxel_m
        gz_hi = (p.z_max - zmin_world) / voxel_m
        z_lo, z_hi = math.floor(gz_lo), math.ceil(gz_hi)
        Y_vols.append(extrude_mask(m2d, z_lo, z_hi, D))
        zlo_all.append(z_lo)
        zhi_all.append(z_hi)
    Y = np.logical_or.reduce(Y_vols).astype(np.uint8) if Y_vols else np.zeros((D, H, W), np.uint8)
    if Y.sum() == 0:
        return {"status": "skip:empty_Y"}

    # Y_neigh: Neighbor volume
    YN = np.zeros((D, H, W), dtype=np.uint8)
    for p in neighbor_parts:
        m2d = polygon_to_mask(
            robust_clean(p.footprint), W, H, origin_xy, voxel_m, ss=raster_ss, thresh=0.5
        )
        gz_lo = (p.z_min - zmin_world) / voxel_m
        gz_hi = (p.z_max - zmin_world) / voxel_m
        YN |= extrude_mask(m2d, math.floor(gz_lo), math.ceil(gz_hi), D)

    # C1: Neighbor height heatmap (global normalized)
    C1, height_grid = build_C1_neighbor_height(
        neighbor_parts, origin_xy, voxel_m, W, H, D, raster_ss=raster_ss
    )

    # C2: Parcel edges
    parcel_lines = lines_from_polygons(extract_polygons(robust_clean(seed_geom)))
    cand_geoms = _iter_tree_query(parcels_tree.query(window_poly))
    neigh_lines: list = []
    for g in cand_geoms:
        idx = None
        if hasattr(g, "wkb"):
            idx = wkb_to_idx.get(g.wkb)
        elif isinstance(g, int):
            idx = g
        if idx is None or idx == seed_idx:
            continue
        g_full = robust_clean(parcels_all[idx]["geom"])
        g_in = safe_intersection(g_full, window_poly)
        if g_in.is_empty:
            continue
        neigh_lines.extend(lines_from_polygons(extract_polygons(g_in)))

    edge_px = max(1, round(float(edge_width_m) / float(voxel_m)))
    C2_edges = np.clip(
        lines_to_mask(parcel_lines, W, H, origin_xy, voxel_m, edge_px)
        + lines_to_mask(neigh_lines, W, H, origin_xy, voxel_m, edge_px),
        0,
        1,
    )
    C2 = np.repeat(C2_edges[None, :, :], D, axis=0).astype(np.uint8)

    # ---------- BGF Filter ----------
    parcel_area_m2 = float(property_union.area)
    target_area_m2 = float(tgt_union.area)
    voxel_vol_m3 = float(Y.sum()) * (float(voxel_m) ** 3)
    bgf_target_m2 = (voxel_vol_m3 / float(storey_height_m)) if storey_height_m > 0 else 0.0

    if (min_bgf_m2 is not None and bgf_target_m2 < float(min_bgf_m2)) or (
        max_bgf_m2 is not None and float(max_bgf_m2) > 0 and bgf_target_m2 > float(max_bgf_m2)
    ):
        return {"status": "skip:bgf_range", "bgf_m2": float(bgf_target_m2)}

    # ---------- GRZ/GFZ Consistency with Direct Neighbors ----------
    grz_target = float(target_area_m2 / parcel_area_m2) if parcel_area_m2 > 0 else 0.0
    gfz_target = (
        (voxel_vol_m3 / (parcel_area_m2 * float(storey_height_m)))
        if (parcel_area_m2 > 0 and storey_height_m > 0)
        else 0.0
    )

    def parcel_metrics_local(
        parcel_geom_clip: Any, parts_list: list[CachedBuildingPart]
    ) -> tuple[float, float, float]:
        """Compute GRZ/GFZ for a parcel region."""
        area_eff = float(parcel_geom_clip.area)
        if area_eff <= 0:
            return 0.0, 0.0, 0.0
        foot_area = 0.0
        vol_m3 = 0.0
        for p in parts_list:
            fp = safe_intersection(robust_clean(p.footprint), parcel_geom_clip)
            if fp.is_empty:
                continue
            a = float(fp.area)
            if a <= 0:
                continue
            h = max(0.0, float(p.z_max) - float(p.z_min))
            foot_area += a
            vol_m3 += a * h
        grz = foot_area / area_eff if area_eff > 0 else 0.0
        gfz = (
            (vol_m3 / (area_eff * float(storey_height_m)))
            if (area_eff > 0 and storey_height_m > 0)
            else 0.0
        )
        return grz, gfz, area_eff

    # Find direct neighbors (shared boundary)
    seed_boundary = robust_clean(seed_geom).boundary
    neighbor_infos = []
    cand_geoms = _iter_tree_query(parcels_tree.query(window_poly))
    for g in cand_geoms:
        idx = None
        if hasattr(g, "wkb"):
            idx = wkb_to_idx.get(g.wkb)
        elif isinstance(g, int):
            idx = g
        if idx is None or idx == seed_idx:
            continue
        g_full = robust_clean(parcels_all[idx]["geom"])
        shared_len = safe_intersection(seed_boundary, g_full.boundary).length
        if shared_len <= 0.0:
            continue
        g_clip = safe_intersection(g_full, window_poly)
        if g_clip.is_empty:
            continue
        neighbor_infos.append((shared_len, idx, g_full, g_clip))

    neighbor_infos.sort(key=lambda t: t[0], reverse=True)
    if require_two_neighbors and len(neighbor_infos) < MINIMUM_COMPARISON_NEIGHBORS:
        return {"status": "skip:grz_gfz_neighbors_missing"}

    top_neighbors = neighbor_infos[:MINIMUM_COMPARISON_NEIGHBORS]
    neighbor_grz, neighbor_gfz = [], []
    for _, _neighbor_index, _neighbor_full, n_clip in top_neighbors:
        grz_n, gfz_n, _ = parcel_metrics_local(n_clip, parts_all)
        neighbor_grz.append(grz_n)
        neighbor_gfz.append(gfz_n)

    def within_tol(val_t: float, val_n: float, tol: float, abs_tol: float) -> bool:
        """Check if target value is within tolerance of neighbor value."""
        eps = 1e-6
        if val_n < eps:
            return abs(val_t) <= abs_tol
        return abs(val_t - val_n) <= max(tol * max(val_n, eps), abs_tol)

    for i in range(len(top_neighbors)):
        ok_grz = within_tol(grz_target, neighbor_grz[i], float(grz_gfz_tol), float(grz_gfz_abs_tol))
        ok_gfz = within_tol(gfz_target, neighbor_gfz[i], float(grz_gfz_tol), float(grz_gfz_abs_tol))
        if not (ok_grz and ok_gfz):
            return {
                "status": "skip:grz_gfz_tol",
                "detail": {
                    "target": {"grz": float(grz_target), "gfz": float(gfz_target)},
                    "neighbor": {"grz": float(neighbor_grz[i]), "gfz": float(neighbor_gfz[i])},
                },
            }

    # ---------- C3: Street Mask (Optional) ----------
    C3 = None
    if with_streets:
        C3_2d = street_mask(
            bbox=(minx, miny, maxx, maxy),
            grid_res=W,
            grid_m=grid_m,
            origin_xy=origin_xy,
            voxel_m=voxel_m,
            mode=street_mode,
            street_width_m=street_width_m,
            wfs_url=wfs_url,
            typename=typename,
            verbose=False,
            cache_dir=street_cache_dir,
        )
        C3 = np.repeat(C3_2d[None, :, :], D, axis=0).astype(np.uint8)

    # X: Input stack
    if C3 is not None:
        X = np.stack(
            [
                C0.astype(np.float32),
                C1.astype(np.float32),
                C2.astype(np.float32),
                C3.astype(np.float32),
            ],
            axis=0,
        )
        channels = ["C0_build_mask", "C1_neighbor_height", "C2_parcel_edges", "C3_street_mask"]
    else:
        X = np.stack([C0.astype(np.float32), C1.astype(np.float32), C2.astype(np.float32)], axis=0)
        channels = ["C0_build_mask", "C1_neighbor_height", "C2_parcel_edges"]

    # ---------- Metadata ----------
    t_z0, t_z1 = int(min(zlo_all)), int(max(zhi_all))
    target_height_m = max(0.0, (t_z1 - t_z0) * float(voxel_m))

    avg_neighbor_height_m = (
        float(np.mean([max(0.0, float(p.z_max) - float(p.z_min)) for p in neighbor_parts]))
        if neighbor_parts
        else 0.0
    )

    street_coverage_frac = None
    if C3 is not None:
        td_C3 = C3.max(axis=0)
        street_coverage_frac = float(td_C3.mean())

    meta = {
        "parcel_id": parcel_id,
        "grid": {
            "grid_m": float(grid_m),
            "voxel_m": float(voxel_m),
            "D": int(D),
            "H": int(H),
            "W": int(W),
        },
        "world_bbox_xy": (float(minx), float(miny), float(maxx), float(maxy)),
        "z_world_range": (float(zmin_world), float(zmax_world)),
        "max_height_m": float(H_max),
        "target_z_grid": (int(min(zlo_all)), int(max(zhi_all))),
        "channels": channels,
        "metrics": {
            "c1_mean": float(height_grid.mean()),
            "c1_max": float(height_grid.max()),
            "target_voxels": int(Y.sum()),
            "neighbor_voxels": int(YN.sum()),
            "num_neighbors": len(neighbor_parts),
            "parcel_area_m2": float(parcel_area_m2),
            "target_footprint_area_m2": float(target_area_m2),
            "coverage_frac": (
                float(target_area_m2 / parcel_area_m2) if parcel_area_m2 > 0 else 0.0
            ),
            "target_height_m": float(target_height_m),
            "avg_neighbor_height_m": float(avg_neighbor_height_m),
            "street_coverage_frac": street_coverage_frac,
            "bgf_target_m2": float(bgf_target_m2),
            "grz_target": float(grz_target),
            "gfz_target": float(gfz_target),
            "grz_neighbors": [float(x) for x in neighbor_grz],
            "gfz_neighbors": [float(x) for x in neighbor_gfz],
        },
    }

    # ---------- Save Output ----------
    sample_dir = os.path.join(out_dir, parcel_id.replace(os.sep, "_"))
    os.makedirs(sample_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(sample_dir, f"{parcel_id}.npz"),
        X=X,
        Y=Y.astype(np.uint8),
        Y_neigh=YN.astype(np.uint8),
        meta=meta,
    )
    with open(os.path.join(sample_dir, f"{parcel_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not no_viz:
        make_overview_png(
            C0,
            C1,
            C2,
            C3,
            mask_target_2d,
            Y,
            YN,
            parcel_id,
            os.path.join(sample_dir, "overview.png"),
            stride3d=max(1, overlay_stride3d),
            dpi=max(200, png_dpi),
            max_height_m=H_max,
        )
        save_voxels_hires(
            Y,
            YN,
            out_path=os.path.join(sample_dir, "voxel_hires.png"),
            stride3d=max(1, overlay_stride3d),
            figsize=(12, 12),
            dpi=600,
        )

    if verbose:
        print(
            f"DONE {parcel_id} | Yvox={int(Y.sum())} | neigh={len(neighbor_parts)} | "
            f"C1mean={float(height_grid.mean()):.3f} | H_max={H_max:.2f}m | "
            f"BGF={bgf_target_m2:.1f}m²"
        )
    else:
        print(f"DONE {parcel_id}")

    return {"status": "ok", "dir": sample_dir}


# ---------------------------------------- Pipeline Orchestration


# RULE_VIOLATION: Keep the three established execution modes in one backward-compatible entry point.
def run_pipeline(config: dict[str, Any]) -> None:  # noqa: PLR0912, PLR0915
    """Execute dataset creation pipeline with given configuration.

    This is the main orchestration function. It handles three modes:
    1. Training dataset generation
    2. Test dataset generation from remaining parcels
    3. Test dataset creation from existing samples

    Args:
        config (Dict[str, Any]): Configuration dictionary with keys:
            - Mode 1 (Training): citygml, parcels, out_dir, grid_m, grid_res, ...
            - Mode 2 (Test from remaining): citygml, parcels, reference_dataset_dir,
              generate_testset_from_remaining, testset_output_dir (optional), ...
            - Mode 3 (Test from existing): out_dir, create_testset_from_existing,
              testset_mode, testset_seed (optional), testset_force (optional)

    Raises:
        SystemExit: On configuration errors or missing required parameters.

    """
    # Set defaults
    defaults = {
        "grid_m": 128.0,
        "grid_res": 256,
        "raster_ss": 2,
        "parcel_mask_thresh": 0.35,
        "parcel_buffer_m": 0.0,
        "min_area": 100.0,
        "max_area": 3000.0,
        "min_neighbor_buildings": 3,
        "min_overlap_m2": 2.0,
        "max_property_frac": 0.60,
        "max_target_frac": 0.30,
        "skip_if_touch_border": False,
        "buildings_require_full_inside": False,
        "context_inside_tol_m": 2.0,
        "edge_width_m": 0.5,
        "with_streets": False,
        "street_mode": "buffer",
        "street_width_m": 8.0,
        "wfs_url": "https://gdi.berlin.de/services/wfs/detailnetz",
        "typename": "detailnetz:c_strassenabschnitte",
        "z_margin_m": 5.0,
        "png_dpi": 300,
        "no_viz": False,
        "grz_gfz_tol": 0.20,
        "grz_gfz_abs_tol": 0.05,
        "storey_height_m": 3.0,
        "require_two_neighbors": False,
        "min_bgf_m2": 0.0,
        "max_bgf_m2": 0.0,
        "test": 0,
        "seed": 42,
        "num_ok": 0,
        "verbose": False,
        "skip_existing": False,
        "cache_dir": "cache",
        "testset_seed": 42,
        "testset_mode": "symlink",
        "testset_force": False,
        "testset_name_suffix": "_testset",
    }

    # Merge config with defaults
    cfg = {**defaults, **config}

    # Determine mode
    mode_testset_existing = cfg.get("create_testset_from_existing", 0) > 0
    mode_testset_remaining = cfg.get("generate_testset_from_remaining", 0) > 0

    # ---------- Mode 3: Test Dataset from Existing Samples ----------
    if mode_testset_existing:
        if not cfg.get("out_dir"):
            raise SystemExit("ERROR: out_dir required for create_testset_from_existing")
        if not os.path.isdir(cfg["out_dir"]):
            raise SystemExit(f"ERROR: Source directory not found: {cfg['out_dir']}")

        print("\n--- Creating Test Dataset from Existing Samples ---")
        out_ts = create_testset_from_existing_samples(
            dataset_dir=cfg["out_dir"],
            count=cfg["create_testset_from_existing"],
            seed=cfg["testset_seed"],
            mode=cfg["testset_mode"],
            force=cfg["testset_force"],
        )
        print(f"\nTest dataset created successfully: {out_ts}")
        return

    # ---------- Modes 1 & 2: Require CityGML + Parcels ----------
    if not cfg.get("citygml") or not cfg.get("parcels"):
        raise SystemExit("ERROR: citygml and parcels required")

    # Initialize output directory
    current_generation_out_dir: str | None = None
    parcels_all_raw: list[dict[str, Any]] | None = None
    candidates: list[int] = []

    # ---------- Mode 2: Test Dataset from Remaining Parcels ----------
    if mode_testset_remaining:
        if not cfg.get("reference_dataset_dir"):
            raise SystemExit("ERROR: reference_dataset_dir required")
        if not os.path.isdir(cfg["reference_dataset_dir"]):
            raise SystemExit(f"ERROR: Reference dataset not found: {cfg['reference_dataset_dir']}")

        if cfg.get("testset_output_dir"):
            current_generation_out_dir = cfg["testset_output_dir"]
        else:
            base_name = os.path.basename(cfg["reference_dataset_dir"].rstrip(os.sep))
            current_generation_out_dir = os.path.join(
                os.path.dirname(cfg["reference_dataset_dir"]),
                f"{base_name}{cfg['testset_name_suffix']}",
            )

        os.makedirs(current_generation_out_dir, exist_ok=True)
        print("\n--- Generating Test Dataset from Remaining Parcels ---")
        print(f"Reference dataset: {cfg['reference_dataset_dir']}")
        print(f"Test dataset output: {current_generation_out_dir}")
        print(f"Target sample count: {cfg['generate_testset_from_remaining']}")

        parcels_all_raw = load_parcels(cfg["parcels"])
        all_initial_candidates_idx = [
            i
            for i, p in enumerate(parcels_all_raw)
            if cfg["min_area"] <= p["geom"].area <= cfg["max_area"]
        ]

        successful_ref_pids = list_successful_parcel_ids(cfg["reference_dataset_dir"])
        print(f"Successful samples in reference dataset: {len(successful_ref_pids)}")

        remaining_candidates_idx = []
        for idx in all_initial_candidates_idx:
            pid = get_parcel_id(parcels_all_raw[idx], idx)
            if pid not in successful_ref_pids:
                remaining_candidates_idx.append(idx)

        print(f"Remaining candidates (not in reference): {len(remaining_candidates_idx)}")

        if not remaining_candidates_idx:
            raise SystemExit("ERROR: No remaining candidates for test dataset")

        # RULE_VIOLATION: Seeded pseudo-random selection is required for reproducibility.
        random.Random(cfg["seed"]).shuffle(remaining_candidates_idx)  # noqa: S311
        candidates = remaining_candidates_idx[: cfg["generate_testset_from_remaining"]]

        if not candidates:
            raise SystemExit("ERROR: No candidates after selection")

        print(f"Generating {len(candidates)} test samples...")

    # ---------- Mode 1: Training Dataset Generation ----------
    else:
        if not cfg.get("out_dir"):
            raise SystemExit("ERROR: out_dir required")

        current_generation_out_dir = cfg["out_dir"]
        os.makedirs(current_generation_out_dir, exist_ok=True)

        print("\n--- Generating Training Dataset ---")
        parcels_all_raw = load_parcels(cfg["parcels"])
        candidates = [
            i
            for i, p in enumerate(parcels_all_raw)
            if cfg["min_area"] <= p["geom"].area <= cfg["max_area"]
        ]

        if not candidates:
            raise SystemExit("ERROR: No parcels after area filtering")

        if cfg["test"] > 0:
            # RULE_VIOLATION: Seeded pseudo-random selection is required for reproducibility.
            random.Random(cfg["seed"]).shuffle(candidates)  # noqa: S311
            candidates = candidates[: cfg["test"]]
            print(f"Test mode: Processing {len(candidates)} candidates")

    # ---------- Common Setup for Modes 1 & 2 ----------
    os.makedirs(cfg["cache_dir"], exist_ok=True)

    voxel_m = float(cfg["grid_m"]) / float(cfg["grid_res"])
    D = round(cfg["grid_m"] / voxel_m)

    # Auto-set missing parameters
    if cfg.get("parcel_erode_vox") is None:
        cfg["parcel_erode_vox"] = 2 if D >= HIGH_RESOLUTION_GRID else 1
    if cfg.get("border_margin_vox") is None:
        cfg["border_margin_vox"] = 2
    if cfg.get("overlay_stride3d") is None:
        cfg["overlay_stride3d"] = 4 if D >= HIGH_RESOLUTION_GRID else 2

    print("\n" + "-" * 70)
    print("Configuration Summary")
    print("-" * 70)
    print(f"CityGML: {os.path.basename(cfg['citygml'])}")
    print(f"Parcels: {os.path.basename(cfg['parcels'])}")
    print(f"Output: {current_generation_out_dir}")
    print(f"Cache: {cfg['cache_dir']}")
    print(f"Grid: {D}x{D}x{D} (voxel={voxel_m:.3f}m)")
    print(f"Streets: {'enabled' if cfg['with_streets'] else 'disabled'}")
    if cfg["with_streets"]:
        print(f"  Mode: {cfg['street_mode']}, Width: {cfg['street_width_m']}m")
    print(f"Visualization: {'disabled' if cfg['no_viz'] else 'enabled'}")
    if cfg["skip_existing"]:
        print("Skip existing: enabled")
    print("-" * 70 + "\n")

    # Load building database
    print("Loading building database...")
    building_db = create_building_database(
        gml_path=cfg["citygml"],
        target_crs="EPSG:25833",
        cache_file=os.path.join(cfg["cache_dir"], "building_database.json.gz"),
        verbose=True,
    )

    # Create parcel spatial index
    print("\nCreating parcel spatial index...")
    geoms = [robust_clean(p["geom"]) for p in parcels_all_raw]
    parcels_tree = STRtree(geoms)
    wkb_to_idx = {g.wkb: i for i, g in enumerate(geoms)}

    print(f"\nProcessing {len(candidates)} parcel candidates...\n")

    # ---------- Sample Generation Loop ----------
    ok = skip = 0
    t0 = time.time()
    need_exact = cfg["num_ok"] > 0

    for i, idx in enumerate(candidates):
        if need_exact and ok >= cfg["num_ok"]:
            break

        pid = get_parcel_id(parcels_all_raw[idx], idx)

        if cfg["skip_existing"] and sample_exists(current_generation_out_dir, pid):
            print(f"SKIP {i + 1}/{len(candidates)}: {pid} [exists]")
            skip += 1
            continue

        try:
            res = build_sample(
                seed_idx=idx,
                building_db=building_db,
                parcels_all=parcels_all_raw,
                parcels_tree=parcels_tree,
                wkb_to_idx=wkb_to_idx,
                out_dir=current_generation_out_dir,
                grid_m=cfg["grid_m"],
                voxel_m=voxel_m,
                raster_ss=cfg["raster_ss"],
                parcel_erode_vox=cfg["parcel_erode_vox"],
                parcel_mask_thresh=cfg["parcel_mask_thresh"],
                parcel_buffer_m=cfg["parcel_buffer_m"],
                grz_gfz_tol=cfg["grz_gfz_tol"],
                grz_gfz_abs_tol=cfg["grz_gfz_abs_tol"],
                storey_height_m=cfg["storey_height_m"],
                require_two_neighbors=cfg["require_two_neighbors"],
                min_bgf_m2=cfg["min_bgf_m2"],
                max_bgf_m2=cfg["max_bgf_m2"],
                min_neighbor_buildings=cfg["min_neighbor_buildings"],
                min_overlap_m2=cfg["min_overlap_m2"],
                max_property_frac=cfg["max_property_frac"],
                max_target_frac=cfg["max_target_frac"],
                skip_if_touch_border=cfg["skip_if_touch_border"],
                border_margin_vox=cfg["border_margin_vox"],
                buildings_require_full_inside=cfg["buildings_require_full_inside"],
                context_inside_tol_m=cfg["context_inside_tol_m"],
                z_margin_m=cfg["z_margin_m"],
                overlay_stride3d=cfg["overlay_stride3d"],
                png_dpi=cfg["png_dpi"],
                with_streets=cfg["with_streets"],
                street_mode=cfg["street_mode"],
                street_width_m=cfg["street_width_m"],
                wfs_url=cfg["wfs_url"],
                typename=cfg["typename"],
                edge_width_m=cfg["edge_width_m"],
                street_cache_dir=cfg.get("street_cache_dir"),
                no_viz=cfg["no_viz"],
                verbose=cfg["verbose"],
            )

            if res.get("status") == "ok":
                ok += 1
            else:
                skip += 1
                print(f"SKIP {i + 1}/{len(candidates)}: {pid} [{res.get('status')}]")

        # RULE_VIOLATION: One invalid third-party geometry must not abort a long batch run.
        except Exception as error:
            skip += 1
            print(f"ERROR {i + 1}/{len(candidates)}: {pid} [{type(error).__name__}: {error!s}]")

    dt = time.time() - t0
    total = ok + skip
    spd = total / dt if dt > 0 else 0.0

    print("\n" + "=" * 70)
    print("Dataset Generation Complete")
    print("=" * 70)
    print(f"Successful samples: {ok}")
    print(f"Skipped samples: {skip}")
    print(f"Total processing time: {dt:.1f}s")
    print(f"Processing speed: {spd:.1f} samples/s")
    print("=" * 70 + "\n")
