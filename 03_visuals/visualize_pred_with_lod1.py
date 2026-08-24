#!/usr/bin/env python3
# This specialized scientific renderer keeps explicit geometry dimensions and
# camera arguments visible. The complexity limits are intentionally waived for
# the orchestration boundary; reusable transformations remain separate helpers.
# ruff: noqa: E402, E501, N806, PLC0415, PLR0912, PLR0913, PLR0915, PLR2004
"""Render a model prediction together with its LOD1 surroundings.

The renderer operates in the source coordinate reference system and can create
an orbit sequence at a configurable angular step. See ``03_visuals/README.md``
for a complete invocation through the central CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box, shape
from shapely.ops import unary_union
from torch.utils.data import DataLoader

# Vorbereitungen für Paketimporte aus den Projektpfaden
ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "02_TrainModels"
DATASET_ROOT = ROOT / "01_CreateDataset"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.append(str(TRAINING_ROOT))
if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

import contextlib

from dataio.voxel_dataset import voxel_collate
from models.unet.unet3d import UNet3D  # type: ignore
from models.unet.unet3d_cond import ConditionalUNet3D  # type: ignore
from pipeline.data_io import iter_building_parts_in_bbox  # type: ignore
from pipeline.geo_utils import extract_polygons, robust_clean  # type: ignore
from pipeline.streets_wfs import _cache_path  # type: ignore
from scripts.infer import (  # type: ignore
    build_dataset,
    load_run_config,
    predict_unet,
    resolve_cond_dim,
)
from utils.device import select_device, set_seed  # type: ignore

# PyVista Offscreen-Rendering aktivieren
pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.smooth_shading = False


Bounds = tuple[float, float, float, float, float, float]
LOGGER = logging.getLogger(__name__)


def load_sample_metadata(sample_path: str | Path) -> dict:
    """Metadaten aus NPZ-Sample laden."""
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with np.load(sample_path, allow_pickle=True) as data:
        meta = data.get("meta")
        if meta is None:
            raise ValueError("Sample file missing metadata array `meta`")
        if isinstance(meta, np.ndarray):
            meta = meta.item() if meta.size == 1 else dict(meta)
        if not isinstance(meta, dict):
            raise ValueError("Sample metadata has unexpected format")

    required = ("world_bbox_xy", "grid", "parcel_id", "z_world_range")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"Sample metadata missing keys: {missing}")
    return meta


def load_sample_arrays(sample_path: str | Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Bestimmte Arrays aus dem Sample laden (falls vorhanden)."""
    arrays: dict[str, np.ndarray] = {}
    sample_path = Path(sample_path)
    if not sample_path.exists():
        return arrays
    with np.load(sample_path, allow_pickle=True) as data:
        for key in keys:
            if key in data:
                arrays[key] = np.array(data[key])
    return arrays


def ensure_3d_binary(volume: np.ndarray) -> np.ndarray:
    """Volumen in eine 3D-Bool-Array umwandeln."""
    vol = np.asarray(volume)
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = np.squeeze(vol, axis=0)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    return (vol > 0).astype(bool)


def compute_world_center_and_extent(
    mask: np.ndarray,
    *,
    meta: dict,
    stride: int,
    flip_y: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Bestimme Schwerpunkt und Ausdehnung eines binären Volumens in Weltkoordinaten."""
    if flip_y:
        mask = mask[:, ::-1, :]

    idx = np.argwhere(mask)
    if idx.size == 0:
        return None, None

    d_idx, h_idx, w_idx = idx.T
    voxel_m = float(meta["grid"]["voxel_m"])
    spacing = voxel_m * max(1, stride)

    minx, miny, _maxx, _maxy = map(float, meta["world_bbox_xy"])
    z0 = float(meta["z_world_range"][0])

    x_min = minx + w_idx.min() * spacing
    x_max = minx + (w_idx.max() + 1) * spacing
    y_min = miny + h_idx.min() * spacing
    y_max = miny + (h_idx.max() + 1) * spacing
    z_min = z0 + d_idx.min() * spacing
    z_max = z0 + (d_idx.max() + 1) * spacing

    center = np.array(
        [(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5],
        dtype=np.float32,
    )
    extent = np.array(
        [x_max - x_min, y_max - y_min, z_max - z_min],
        dtype=np.float32,
    )
    return center, extent


def load_prediction(pred_path: str | Path) -> np.ndarray:
    """Vorhersage aus .npy oder .npz laden."""
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    if pred_path.suffix == ".npy":
        pred = np.load(pred_path)
    elif pred_path.suffix == ".npz":
        with np.load(pred_path) as data:
            if "prediction" in data:
                pred = data["prediction"]
            elif "logits" in data:
                pred = data["logits"]
            elif "Y" in data:
                pred = data["Y"]
            else:
                raise ValueError(f"Missing expected keys in NPZ: {list(data.keys())}")
    else:
        raise ValueError(f"Unsupported prediction file type: {pred_path.suffix}")

    while pred.ndim > 3 and pred.shape[0] == 1:
        pred = np.squeeze(pred, axis=0)
    if pred.ndim != 3:
        raise ValueError(f"Prediction volume expected 3D, got shape {pred.shape}")
    return np.asarray(pred, dtype=np.float32)


def build_target_polygon(mask: np.ndarray, meta: dict) -> Polygon | None:
    """Erzeuge Polygon des Zielgrundstücks aus dem Voxel-Maskenwürfel."""
    binary = ensure_3d_binary(mask)
    footprint_mask = binary.any(axis=0)
    if not footprint_mask.any():
        return None

    H, _W = footprint_mask.shape
    voxel_m = float(meta["grid"]["voxel_m"])
    minx, miny, _maxx, _maxy = map(float, meta["world_bbox_xy"])

    cells = np.argwhere(footprint_mask)
    tiles: list[Polygon] = []
    for row, col in cells:
        x0 = minx + col * voxel_m
        x1 = x0 + voxel_m
        row_inv = (H - 1) - row
        y0 = miny + row_inv * voxel_m
        y1 = y0 + voxel_m
        tiles.append(box(x0, y0, x1, y1))

    if not tiles:
        return None
    try:
        merged = robust_clean(unary_union(tiles))
    except Exception:
        merged = unary_union(tiles)
    return merged if merged and not merged.is_empty else None


def infer_prediction_for_sample(sample_id: str, args, cfg: dict) -> np.ndarray:
    """Führe einmalige Modellinferenz für das gegebene Sample durch."""
    if args.infer_checkpoint is None:
        raise ValueError("Auto-Inferenz benötigt --infer_checkpoint")

    cond_dim = resolve_cond_dim(cfg, args.infer_model)
    cond_stats_path = args.infer_cond_stats or cfg.get("cond_stats_path")

    dataset = build_dataset(
        args.infer_split,
        cfg,
        str(args.infer_data_root),
        cond_dim,
        cond_stats_path,
        max_samples=None,
        split_manifest=cfg.get("split_manifest"),
    )

    matches = [idx for idx, path in enumerate(dataset.files) if Path(path).stem == sample_id]
    if not matches:
        raise ValueError(
            f"Sample-ID '{sample_id}' nicht im Split '{args.infer_split}' (root={args.infer_data_root}) gefunden"
        )

    target_idx = matches[0]
    dataset.files = [dataset.files[target_idx]]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=max(0, args.infer_num_workers),
        shuffle=False,
        collate_fn=voxel_collate,
    )

    in_channels = len(dataset.x_indices) + (3 if dataset.add_coords else 0)
    if args.infer_model == "unet":
        model = UNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
        )
    elif args.infer_model == "cond_unet":
        model = ConditionalUNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
            cond_dim=cond_dim,
        )
    else:
        raise ValueError(f"Automatische Inferenz unterstützt Modell '{args.infer_model}' nicht")

    state = torch.load(args.infer_checkpoint, map_location="cpu")
    weights = state.get("model", state)
    model.load_state_dict(weights, strict=False)

    device_cfg = select_device()
    set_seed(int(args.infer_seed))
    model.to(device_cfg.device)
    model.eval()

    batch = next(iter(dataloader))
    logits = predict_unet(
        model,
        batch,
        cond_dim,
        device_cfg.device,
        bool(args.infer_tta),
        args.infer_tta_mode,
    )
    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return prob.astype(np.float32, copy=False)


def prepare_prediction_voxels(
    volume: np.ndarray,
    *,
    world_bbox_xy: Sequence[float],
    z_world_range: Sequence[float],
    center_xy: tuple[float, float],
    voxel_size: float,
    stride: int,
    threshold: float,
) -> pv.UnstructuredGrid | None:
    """Bereitet Vorhersage als lokales Voxel-Gitter (UnstructuredGrid) auf."""
    binary = volume > threshold
    if not binary.any():
        return None

    # Raster-Ausrichtung korrigieren (Dataset-Y ist invertiert)
    binary = binary[:, ::-1, :]

    # In PyVista entspricht Achsenreihenfolge (x,y,z) -> (W,H,D)
    volume_xyz = np.transpose(binary, (2, 1, 0)).astype(np.uint8)

    nx, ny, nz = volume_xyz.shape
    minx, miny, _maxx, _maxy = map(float, world_bbox_xy)
    spacing = float(voxel_size) * max(1, stride)

    grid_img = pv.ImageData()
    grid_img.dimensions = (nx + 1, ny + 1, nz + 1)
    grid_img.origin = (minx - center_xy[0], miny - center_xy[1], 0.0)
    grid_img.spacing = (spacing, spacing, spacing)
    grid_img.cell_data["values"] = volume_xyz.ravel(order="F")

    voxels = grid_img.threshold(0.5, scalars="values")
    if voxels.n_cells == 0:
        return None
    return voxels.clean()


def polygon_to_prism(
    poly: Polygon,
    z_min: float,
    z_max: float,
    *,
    center_xy: tuple[float, float],
    z_origin: float,
) -> pv.PolyData | None:
    """Extrudiere ein Polygon zu einem Block (LOD1)."""
    if z_max <= z_min:
        return None

    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return None
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    cx, cy = center_xy
    base = [(x - cx, y - cy, z_min - z_origin) for x, y in coords]
    top = [(x - cx, y - cy, z_max - z_origin) for x, y in coords]

    points = np.array(base + top, dtype=np.float32)
    n = len(coords)
    if n < 3:
        return None

    faces: list[np.ndarray] = []
    faces.append(np.array([n, *list(range(n))[::-1]], dtype=np.int64))  # Boden
    faces.append(np.array([n] + [i + n for i in range(n)], dtype=np.int64))  # Dach
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([4, i, j, j + n, i + n], dtype=np.int64))

    mesh = pv.PolyData(points, np.concatenate(faces))
    return mesh.clean()


def collect_lod1_neighbor_mesh(
    citygml_path: str | Path,
    *,
    center_xy: tuple[float, float],
    radius: float,
    z_world_range: Sequence[float],
    target_polygon: Polygon | None = None,
) -> tuple[pv.PolyData | None, int]:
    """LOD1-Nachbarn extrudieren und zu einem Mesh kombinieren."""
    search_bbox = (
        center_xy[0] - radius,
        center_xy[1] - radius,
        center_xy[0] + radius,
        center_xy[1] + radius,
    )
    center_pt = Point(center_xy)
    z_origin = float(z_world_range[0])
    exclusion = None
    if target_polygon is not None and not target_polygon.is_empty:
        try:
            exclusion = robust_clean(target_polygon.buffer(0.01))
        except Exception:
            exclusion = target_polygon

    meshes: list[pv.PolyData] = []
    kept = 0
    for part in iter_building_parts_in_bbox(
        gml_src=str(citygml_path),
        bbox_xy=search_bbox,
        target_crs=None,
        verbose=False,
    ):
        footprint = robust_clean(part.footprint)
        if footprint.is_empty or footprint.area == 0:
            continue
        if center_pt.distance(footprint) > radius:
            continue
        if exclusion is not None:
            centroid = (
                footprint.centroid
                if not footprint.centroid.is_empty
                else footprint.representative_point()
            )
            if exclusion.contains(centroid):
                continue

        for poly in extract_polygons(footprint):
            prism = polygon_to_prism(
                poly,
                float(part.z_min),
                float(part.z_max),
                center_xy=center_xy,
                z_origin=z_origin,
            )
            if prism is None:
                continue
            meshes.append(prism)
            kept += 1

    if not meshes:
        return None, 0
    combined = pv.MultiBlock(meshes).combine()
    return combined.clean(), kept


def iter_tile_bboxes(
    center_xy: tuple[float, float],
    radius: float,
    tile_m: float,
) -> Iterable[tuple[float, float, float, float]]:
    """Tile-BBOXen ermitteln, die den Radius abdecken."""
    cx, cy = center_xy
    ix_min = math.floor((cx - radius) / tile_m)
    ix_max = math.floor((cx + radius) / tile_m)
    iy_min = math.floor((cy - radius) / tile_m)
    iy_max = math.floor((cy + radius) / tile_m)
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            yield (
                ix * tile_m,
                iy * tile_m,
                (ix + 1) * tile_m,
                (iy + 1) * tile_m,
            )


def lines_to_polydata(
    lines: Iterable[LineString],
    *,
    center_xy: tuple[float, float],
    z_offset: float,
) -> pv.PolyData | None:
    """Konvertiert Shapely-Linien in PolyData."""
    pts: list[tuple[float, float, float]] = []
    cells: list[list[int]] = []
    idx = 0
    cx, cy = center_xy

    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start_idx = idx
        for x, y in coords:
            pts.append((x - cx, y - cy, z_offset))
            idx += 1
        cells.append([len(coords), *list(range(start_idx, idx))])

    if not pts or not cells:
        return None

    points = np.array(pts, dtype=np.float32)
    connectivity = np.concatenate([np.array(cell, dtype=np.int64) for cell in cells])
    poly = pv.PolyData(points, lines=connectivity)
    return poly.clean()


def load_street_lines(
    streets_dir: str | Path,
    *,
    center_xy: tuple[float, float],
    radius: float,
    tile_m: float = 250.0,
    typename: str = "detailnetz:c_strassenabschnitte",
) -> list[LineString]:
    """Straßenstücke aus gecachten GeoJSON-Kacheln laden."""
    streets_dir = Path(streets_dir)
    window = box(
        center_xy[0] - radius,
        center_xy[1] - radius,
        center_xy[0] + radius,
        center_xy[1] + radius,
    )

    collected: list[LineString] = []
    for tile_bbox in iter_tile_bboxes(center_xy, radius, tile_m):
        cache_path = _cache_path(streets_dir, tile_bbox, typename, tile_m)
        if not cache_path.exists():
            continue
        with open(cache_path, encoding="utf-8") as f:
            gj = json.load(f)

        for ft in gj.get("features", []):
            geom_raw = ft.get("geometry")
            if not geom_raw:
                continue
            try:
                geom = robust_clean(shape(geom_raw))
            except Exception as error:
                LOGGER.debug("Skipping malformed street geometry: %s", error)
                continue
            if geom.is_empty:
                continue

            def _handle_lines(g):
                if isinstance(g, LineString):
                    clipped = robust_clean(g.intersection(window))
                    if isinstance(clipped, LineString) and not clipped.is_empty:
                        collected.append(clipped)
                    elif isinstance(clipped, MultiLineString):
                        collected.extend(
                            seg
                            for seg in clipped.geoms
                            if isinstance(seg, LineString) and not seg.is_empty
                        )
                elif isinstance(g, MultiLineString):
                    for seg in g.geoms:
                        _handle_lines(seg)

            _handle_lines(geom)

    if not collected:
        return []
    merged = robust_clean(unary_union(collected))
    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return [seg for seg in merged.geoms if isinstance(seg, LineString) and not seg.is_empty]
    return collected


def load_parcel_lines(
    parcels_path: str | Path,
    *,
    center_xy: tuple[float, float],
    radius: float,
) -> list[LineString]:
    """Flurstück-Umrisse als Linien extrahieren."""
    try:
        import geopandas as gpd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Geopandas (mit pyogrio/fiona) wird benötigt, um Flurstücke zu laden."
        ) from exc

    bbox = (
        center_xy[0] - radius,
        center_xy[1] - radius,
        center_xy[0] + radius,
        center_xy[1] + radius,
    )
    window = box(*bbox)

    gdf = gpd.read_file(parcels_path, bbox=bbox)
    lines: list[LineString] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        for poly in extract_polygons(geom):
            cleaned = robust_clean(poly)
            if cleaned.is_empty:
                continue
            ext_line = LineString(cleaned.exterior.coords)
            clipped = robust_clean(ext_line.intersection(window))
            if isinstance(clipped, LineString) and not clipped.is_empty:
                lines.append(clipped)
            elif isinstance(clipped, MultiLineString):
                lines.extend(
                    [
                        seg
                        for seg in clipped.geoms
                        if isinstance(seg, LineString) and not seg.is_empty
                    ]
                )
            for hole in cleaned.interiors:
                hole_line = LineString(hole.coords)
                clipped_hole = robust_clean(hole_line.intersection(window))
                if isinstance(clipped_hole, LineString) and not clipped_hole.is_empty:
                    lines.append(clipped_hole)
                elif isinstance(clipped_hole, MultiLineString):
                    lines.extend(
                        [
                            seg
                            for seg in clipped_hole.geoms
                            if isinstance(seg, LineString) and not seg.is_empty
                        ]
                    )
    return lines


def aggregate_bounds(meshes: Sequence[pv.DataSet | None]) -> Bounds | None:
    """Gesamtbounds aller verfügbaren Meshes ermitteln."""
    bounds: list[float] | None = None
    for mesh in meshes:
        if mesh is None or mesh.n_points == 0:
            continue
        b = mesh.bounds  # type: ignore[attr-defined]
        if bounds is None:
            bounds = list(b)
        else:
            bounds[0] = min(bounds[0], b[0])
            bounds[1] = max(bounds[1], b[1])
            bounds[2] = min(bounds[2], b[2])
            bounds[3] = max(bounds[3], b[3])
            bounds[4] = min(bounds[4], b[4])
            bounds[5] = max(bounds[5], b[5])
    if bounds is None:
        return None
    return tuple(bounds)  # type: ignore[return-value]


def derive_output_name(prediction_path: Path, explicit: str | None) -> str:
    """Ableitung eines Basisnamens für Render-Ordner."""
    if explicit:
        return explicit
    stem = prediction_path.stem
    if stem.endswith("_prob"):
        stem = stem[:-5]
    return stem


def orbit_render(
    *,
    plotter: pv.Plotter,
    angles: Sequence[float],
    output_dir: Path,
    focus_z: float,
    radius: float,
    height: float,
    parallel_scale: float,
) -> None:
    """Kamera um die Szene drehen und Screenshots sichern."""
    output_dir.mkdir(parents=True, exist_ok=True)
    focus = (0.0, 0.0, focus_z)
    view_up = (0.0, 0.0, 1.0)

    if plotter.camera is not None:
        plotter.camera.parallel_projection = True

    for angle in angles:
        rad = math.radians(angle)
        cam_pos = (
            math.cos(rad) * radius,
            math.sin(rad) * radius,
            focus_z + height,
        )
        plotter.camera_position = (cam_pos, focus, view_up)
        if plotter.camera is not None:
            plotter.camera.parallel_scale = max(parallel_scale, 1e-3)
        plotter.render()
        out_path = output_dir / f"angle_{round(angle) % 360:03d}.png"
        plotter.screenshot(filename=str(out_path), return_img=False)


def pyvista_render_available() -> bool:
    """Return whether VTK has an explicitly available display backend."""
    if os.environ.get("URBAN_FORCE_PYVISTA") == "1":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def render_mesh_point_cloud(
    *,
    prediction_meshes: Sequence[pv.DataSet],
    context_meshes: Sequence[pv.DataSet | None],
    bounds: Bounds,
    angles: Sequence[float],
    output_directory: Path,
    window_size: Sequence[int],
) -> None:
    """Render a bounded Matplotlib fallback for headless environments."""
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    width_pixels, height_pixels = (int(value) for value in window_size)
    figure_size = (width_pixels / 100.0, height_pixels / 100.0)
    context_points = [
        _bounded_mesh_points(mesh) for mesh in context_meshes if mesh is not None and mesh.n_points
    ]
    prediction_points = [_bounded_mesh_points(mesh) for mesh in prediction_meshes if mesh.n_points]
    for angle in angles:
        figure = plt.figure(figsize=figure_size, dpi=100)
        axis = figure.add_subplot(111, projection="3d")
        for points in context_points:
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c="#B8BEC9",
                s=0.3,
                alpha=0.25,
                depthshade=False,
            )
        for points in prediction_points:
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c="#1B49FF",
                s=1.0,
                alpha=0.9,
                depthshade=False,
            )
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_zlim(bounds[4], bounds[5])
        axis.set_box_aspect(
            (
                max(bounds[1] - bounds[0], 1.0),
                max(bounds[3] - bounds[2], 1.0),
                max(bounds[5] - bounds[4], 1.0),
            )
        )
        axis.view_init(elev=28.0, azim=float(angle))
        axis.set_axis_off()
        output_path = output_directory / f"angle_{round(angle) % 360:03d}.png"
        figure.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def _bounded_mesh_points(mesh: pv.DataSet, maximum_points: int = 50_000) -> np.ndarray:
    """Return a deterministic, bounded point sample from a mesh."""
    points = np.asarray(mesh.points)
    if points.shape[0] <= maximum_points:
        return points
    stride = math.ceil(points.shape[0] / maximum_points)
    return points[::stride]


def main(argv: Sequence[str] | None = None) -> Path:
    """Parse renderer arguments and write the requested orbit frames."""
    parser = argparse.ArgumentParser(
        description="Visualisiere Prediction-Mesh mit LOD1-Kontext, Straßen und Flurstücken (PyVista)."
    )
    parser.add_argument("--sample_path", type=Path, help="Pfad zur Sample-NPZ-Datei")
    parser.add_argument(
        "--sample_id", type=str, help="Eindeutige Sample-ID (z. B. DEBE02YY200007LA)"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("00_Data/02_GeneratedDatasets/data"),
        help="Basisverzeichnis der NPZ-Samples",
    )
    parser.add_argument("--prediction_path", type=Path, help="Pfad zur Prediction (.npy/.npz)")
    parser.add_argument(
        "--citygml",
        required=True,
        type=Path,
        help="CityGML LOD1 Merge (z. B. berlin_lod1_merged.gml.gz)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Basisverzeichnis für Visualisierungen (z. B. .../vis)",
    )
    parser.add_argument(
        "--streets_dir",
        type=Path,
        default=Path("00_Data/01_InputData/cache/streets"),
        help="Cache-Verzeichnis der Straßenkacheln",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=Path("00_Data/01_InputData/input/flurstuecke.geojson"),
        help="GeoJSON der Flurstücke",
    )
    parser.add_argument("--radius", type=float, default=500.0, help="Kontext-Radius in Metern")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Schwellwert für die Prediction (Default 0.5)"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Optionaler Basisname für Output-Ordner"
    )
    parser.add_argument(
        "--window_size",
        nargs=2,
        type=int,
        default=[1920, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="Renderfenster (Pixel)",
    )
    parser.add_argument(
        "--angle_step", type=float, default=1.0, help="Schrittweite in Grad (Default 1°)"
    )
    parser.add_argument(
        "--angle_start", type=float, default=0.0, help="Startwinkel in Grad (inklusive)"
    )
    parser.add_argument(
        "--angle_stop", type=float, default=360.0, help="Endwinkel in Grad (exklusiv)"
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=1.0,
        help="Zoom-Mischfaktor (0 = ursprünglicher Abstand, 1 = starker Nahzoom)",
    )
    parser.add_argument(
        "--infer_model",
        type=str,
        default="unet",
        choices=("unet", "cond_unet"),
        help="Modelltyp für automatische Inferenz",
    )
    parser.add_argument(
        "--infer_checkpoint", type=Path, help="Pfad zum Modell-Checkpoint für Auto-Inferenz"
    )
    parser.add_argument(
        "--infer_config", type=Path, default=None, help="Optionales Trainings-Config JSON"
    )
    parser.add_argument(
        "--infer_split", type=str, default="val", help="Dataset-Split für Auto-Inferenz"
    )
    parser.add_argument(
        "--infer_data_root", type=Path, default=None, help="Root-Verzeichnis für Auto-Inferenz"
    )
    parser.add_argument(
        "--infer_cond_stats", type=Path, default=None, help="Pfad zu Konditionsstatistiken"
    )
    parser.add_argument(
        "--infer_num_workers", type=int, default=0, help="DataLoader-Worker (Auto-Inferenz)"
    )
    parser.add_argument("--infer_seed", type=int, default=42, help="Seed für Auto-Inferenz")
    parser.add_argument(
        "--infer_tta", action="store_true", help="Test-Time-Augmentation aktivieren"
    )
    parser.add_argument(
        "--infer_tta_mode",
        type=str,
        default="rot90",
        choices=("rot90", "flip"),
        help="TTA-Modus für Auto-Inferenz",
    )
    parser.add_argument(
        "--neighbor_render",
        type=str,
        default="mesh",
        choices=("mesh", "outline"),
        help="Darstellung der LOD1-Nachbarn (Flächen oder nur Kanten)",
    )

    args = parser.parse_args(argv)
    zoom_factor = float(np.clip(args.zoom_factor, 0.0, 1.0))
    args.zoom_factor = zoom_factor

    args.data_root = args.data_root.resolve()
    if args.infer_data_root is not None:
        args.infer_data_root = args.infer_data_root.resolve()

    if args.sample_path is not None:
        sample_path = args.sample_path.resolve()
    else:
        if not args.sample_id:
            raise ValueError("Bitte entweder --sample_path oder --sample_id angeben")
        candidate = args.data_root / args.sample_id / f"{args.sample_id}.npz"
        if not candidate.exists():
            alt = args.data_root / f"{args.sample_id}.npz"
            if alt.exists():
                candidate = alt
            else:
                raise FileNotFoundError(
                    f"Sample '{args.sample_id}' nicht unter {candidate.parent} oder {alt} gefunden"
                )
        sample_path = candidate.resolve()
    args.sample_path = sample_path
    sample_id = args.sample_id or sample_path.stem

    if args.prediction_path is not None:
        args.prediction_path = args.prediction_path.resolve()
        prediction_volume = load_prediction(args.prediction_path)
    else:
        if args.infer_checkpoint is None:
            raise ValueError(
                "Auto-Inferenz benötigt --infer_checkpoint, wenn kein --prediction_path gesetzt ist"
            )
        args.infer_checkpoint = args.infer_checkpoint.resolve()
        if args.infer_config is not None:
            args.infer_config = args.infer_config.resolve()
        if args.infer_cond_stats is not None:
            args.infer_cond_stats = args.infer_cond_stats.resolve()
        args.infer_data_root = (
            args.infer_data_root.resolve() if args.infer_data_root else args.data_root
        )
        print(f"-> Auto-Inferenz für Sample {sample_id}…")
        cfg = load_run_config(
            args.infer_checkpoint, str(args.infer_config) if args.infer_config else None
        )
        prediction_volume = infer_prediction_for_sample(sample_id, args, cfg)

    meta = load_sample_metadata(sample_path)

    meta["grid"]
    world_bbox_xy = meta["world_bbox_xy"]
    z_world_range = meta["z_world_range"]
    minx, miny, maxx, maxy = map(float, world_bbox_xy)
    center_xy = ((minx + maxx) * 0.5, (miny + maxy) * 0.5)

    print("-> Prediction in Mesh umwandeln…")
    prediction_voxels = None
    alignment_shift = np.zeros(3, dtype=np.float32)

    prediction_volume = np.asarray(prediction_volume)
    while prediction_volume.ndim > 3 and prediction_volume.shape[0] == 1:
        prediction_volume = np.squeeze(prediction_volume, axis=0)

    stride_ratio = max(1, round(meta["grid"]["D"] / prediction_volume.shape[0]))
    pred_mask = prediction_volume > args.threshold
    pred_center_world, _pred_extent = compute_world_center_and_extent(
        pred_mask,
        meta=meta,
        stride=stride_ratio,
        flip_y=True,
    )

    target_center_world = None
    target_polygon = None
    sample_arrays = load_sample_arrays(sample_path, ("Y",))
    if "Y" in sample_arrays:
        try:
            target_mask = ensure_3d_binary(sample_arrays["Y"])
            target_center_world, _target_extent = compute_world_center_and_extent(
                target_mask,
                meta=meta,
                stride=1,
                flip_y=True,
            )
            target_polygon = build_target_polygon(target_mask, meta)
        except ValueError:
            target_center_world = None
            target_polygon = None

    if pred_center_world is not None:
        print(f"   Downsample stride: {stride_ratio}")

    if pred_center_world is not None and target_center_world is not None:
        alignment_shift = (target_center_world - pred_center_world).astype(np.float32)
        print(
            f"   Alignment shift (m): "
            f"dx={alignment_shift[0]:+.2f}, dy={alignment_shift[1]:+.2f}, dz={alignment_shift[2]:+.2f}"
        )

    prediction_voxels = prepare_prediction_voxels(
        prediction_volume,
        world_bbox_xy=world_bbox_xy,
        z_world_range=z_world_range,
        center_xy=center_xy,
        voxel_size=meta["grid"]["voxel_m"],
        stride=stride_ratio,
        threshold=args.threshold,
    )
    if prediction_voxels is None:
        raise RuntimeError("Prediction enthält keine Voxel über Schwellwert.")

    if alignment_shift.any():
        prediction_voxels.translate(tuple(alignment_shift.tolist()), inplace=True)

    print("-> LOD1-Nachbarn extrahieren…")
    neighbor_mesh, neighbor_count = collect_lod1_neighbor_mesh(
        args.citygml,
        center_xy=center_xy,
        radius=float(args.radius),
        z_world_range=z_world_range,
        target_polygon=target_polygon,
    )
    print(f"   Gefundene Gebäude: {neighbor_count}")

    print("-> Straßenlinien laden…")
    street_lines = load_street_lines(
        args.streets_dir,
        center_xy=center_xy,
        radius=float(args.radius),
    )
    streets_mesh = lines_to_polydata(
        street_lines,
        center_xy=center_xy,
        z_offset=0.3,
    )
    print(f"   Straßen-Segmente: {len(street_lines)}")

    print("-> Flurstück-Umrisse laden…")
    parcel_lines = load_parcel_lines(
        args.parcels,
        center_xy=center_xy,
        radius=float(args.radius),
    )
    parcels_mesh = lines_to_polydata(
        parcel_lines,
        center_xy=center_xy,
        z_offset=0.15,
    )
    print(f"   Flurstück-Linien: {len(parcel_lines)}")

    scene_bounds = aggregate_bounds([prediction_voxels, neighbor_mesh, streets_mesh, parcels_mesh])
    if scene_bounds is None:
        raise RuntimeError("Konnte keine Bounds bestimmen - mindestens ein Mesh erforderlich.")

    xy_extent = max(
        abs(scene_bounds[0]),
        abs(scene_bounds[1]),
        abs(scene_bounds[2]),
        abs(scene_bounds[3]),
        float(args.radius) * 0.65,
    )
    z_extent = max(scene_bounds[5] - scene_bounds[4], 1.0)

    pred_bounds = prediction_voxels.bounds if prediction_voxels is not None else None
    pred_xy_extent = None
    pred_z_extent = None
    if pred_bounds:
        pred_xy_extent = max(pred_bounds[1] - pred_bounds[0], pred_bounds[3] - pred_bounds[2])
        pred_z_extent = pred_bounds[5] - pred_bounds[4]

    if target_center_world is not None:
        focus_z = target_center_world[2] - float(z_world_range[0])
    elif pred_center_world is not None:
        focus_z = (pred_center_world[2] + alignment_shift[2]) - float(z_world_range[0])
    else:
        focus_z = (scene_bounds[4] + scene_bounds[5]) * 0.5

    # Legacy Zoom (weit, Kontext-orientiert)
    legacy_base_radius = xy_extent * 0.9
    if pred_xy_extent is not None:
        legacy_base_radius = max(legacy_base_radius, pred_xy_extent * 1.4)
    legacy_cam_radius = max(legacy_base_radius, float(args.radius) * 0.5)
    legacy_height_base = legacy_cam_radius * 1.4
    if pred_z_extent is not None:
        legacy_height_base = max(legacy_height_base, pred_z_extent * 2.0)
    legacy_cam_height = max(legacy_height_base, z_extent * 1.2)
    legacy_parallel = max(legacy_cam_radius, 1.0)

    # Nah-Zoom (prediction-orientiert)
    tight_cam_radius: float
    if pred_bounds:
        pred_x_range = pred_bounds[1] - pred_bounds[0]
        pred_y_range = pred_bounds[3] - pred_bounds[2]
        pred_diag = math.sqrt(max(pred_x_range, 1e-3) ** 2 + max(pred_y_range, 1e-3) ** 2)
        tight_cam_radius = max(pred_diag * 0.55, 5.0)
    else:
        tight_cam_radius = max(legacy_base_radius, 5.0)
    tight_height_base = tight_cam_radius * 0.9
    if pred_z_extent is not None:
        tight_height_base = max(tight_height_base, pred_z_extent * 1.6)
    tight_cam_height = max(tight_height_base, 5.0)
    tight_parallel = max(tight_cam_radius * 0.55, abs(tight_cam_height) * 0.75, 5.0)

    cam_radius = (1.0 - zoom_factor) * legacy_cam_radius + zoom_factor * tight_cam_radius
    cam_height = (1.0 - zoom_factor) * legacy_cam_height + zoom_factor * tight_cam_height
    parallel_scale = (1.0 - zoom_factor) * legacy_parallel + zoom_factor * tight_parallel

    angles = np.arange(args.angle_start, args.angle_stop, args.angle_step, dtype=float)
    if angles.size == 0:
        raise ValueError("Keine Winkel für Rendering erzeugt - prüfen Sie angle_* Parameter.")

    if args.prediction_path is not None:
        base_name = derive_output_name(args.prediction_path, args.name)
    else:
        base_name = args.name or sample_id
    render_root = args.output_dir / "renders" / base_name
    render_root.mkdir(parents=True, exist_ok=True)

    if not pyvista_render_available():
        print("-> Kein VTK-Display verfügbar; verwende Matplotlib-Offscreen-Fallback.")
        render_mesh_point_cloud(
            prediction_meshes=[prediction_voxels],
            context_meshes=[neighbor_mesh, streets_mesh, parcels_mesh],
            bounds=scene_bounds,
            angles=angles,
            output_directory=render_root,
            window_size=args.window_size,
        )
        print(f"Fertig! Frames gespeichert unter: {render_root}")
        return render_root

    plotter = pv.Plotter(off_screen=True, window_size=tuple(args.window_size))
    with contextlib.suppress(Exception):
        plotter.enable_depth_peeling()
    plotter.set_background("white")
    plotter.add_mesh(
        parcels_mesh,
        color=(0.0, 0.0, 0.0),
        line_width=1.2,
        opacity=1.0,
        render_lines_as_tubes=False,
    ) if parcels_mesh is not None else None
    if neighbor_mesh is not None:
        if args.neighbor_render == "mesh":
            plotter.add_mesh(
                neighbor_mesh,
                color=(0.8, 0.8, 0.8),
                opacity=0.3,
                smooth_shading=False,
            )
        else:
            neighbor_edges = neighbor_mesh.extract_all_edges()
            if neighbor_edges is not None and neighbor_edges.n_cells:
                plotter.add_mesh(
                    neighbor_edges,
                    color=(0.0, 0.0, 0.0),
                    opacity=0.9,
                    line_width=1.4,
                    render_lines_as_tubes=False,
                )
    plotter.add_mesh(
        streets_mesh,
        color=(0.8, 0.8, 0.8),
        line_width=12.0,
        opacity=1.0,
        render_lines_as_tubes=False,
    ) if streets_mesh is not None else None
    prediction_edges = (
        prediction_voxels.extract_all_edges() if prediction_voxels is not None else None
    )
    plotter.add_mesh(
        prediction_voxels,
        color=(0.12, 0.35, 0.92),
        opacity=1.0,
        specular=0.2,
        smooth_shading=False,
        style="surface",
    )
    if prediction_edges is not None and prediction_edges.n_cells:
        plotter.add_mesh(
            prediction_edges,
            color=(0.04, 0.1, 0.28),
            line_width=1.2,
            opacity=1.0,
            render_lines_as_tubes=False,
        )

    print("-> Rendern der Orbit-Sequenz…")
    orbit_render(
        plotter=plotter,
        angles=angles,
        output_dir=render_root,
        focus_z=focus_z,
        radius=cam_radius,
        height=cam_height,
        parallel_scale=parallel_scale,
    )
    plotter.close()

    print(f"Fertig! Frames gespeichert unter: {render_root}")
    return render_root


if __name__ == "__main__":
    main()
