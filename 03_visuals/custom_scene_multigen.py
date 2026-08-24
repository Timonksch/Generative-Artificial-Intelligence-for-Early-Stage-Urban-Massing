#!/usr/bin/env python3
# This specialized scientific renderer keeps explicit dimensions, arguments,
# and linear orchestration visible. Splitting those contracts would obscure the
# relationship between inference geometry and the rendered scene.
# ruff: noqa: E402, E501, N806, PLC0415, PLR0915, PLR2004
"""Generate and render a multi-building prediction scene.

The command selects parcels from WGS84 coordinates, builds temporary samples,
runs inference, and combines the predictions with LOD1 context. Usage and the
full parameter reference are documented in ``03_visuals/README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Subset

# Projektpfade injizieren (bestehenden Code nicht anfassen)
ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "02_TrainModels"
DATASET_ROOT = ROOT / "01_CreateDataset"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.append(str(TRAINING_ROOT))
if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

import contextlib

from dataio.voxel_dataset import voxel_collate  # type: ignore
from pipeline.data_io import create_building_database, load_parcels  # type: ignore
from pipeline.geo_utils import robust_clean  # type: ignore
from pipeline.pipeline import build_sample  # type: ignore
from scripts.infer import (  # type: ignore
    build_dataset,
    load_run_config,
    predict_unet,
    resolve_cond_dim,
)
from utils.device import select_device, set_seed  # type: ignore
from visualize_pred_with_lod1 import (  # type: ignore
    aggregate_bounds,
    build_target_polygon,
    collect_lod1_neighbor_mesh,
    compute_world_center_and_extent,
    lines_to_polydata,
    load_parcel_lines,
    load_sample_arrays,
    load_sample_metadata,
    load_street_lines,
    orbit_render,
    prepare_prediction_voxels,
    pyvista_render_available,
    render_mesh_point_cloud,
)

# PyVista Offscreen-Rendering erzwingen
pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.smooth_shading = False


@dataclass
class ScenePrediction:
    """One generated sample and its world-space rendering metadata."""

    sample_id: str
    prob: np.ndarray
    meta: dict
    target_polygon: Polygon | None
    voxels_mesh: pv.UnstructuredGrid
    center_world: np.ndarray | None
    stride_ratio: int


def parse_dms_token(token: str) -> float:
    r"""Parst DMS (52d30'36.5\"N) oder Dezimalzahlen in Grad."""
    token = token.strip().replace(",", "")
    # Dezimal direkt
    try:
        val = float(token)
        return val
    except ValueError:
        pass

    m = re.match(
        r"(?P<deg>-?\d+(?:\.\d+)?)"
        r"[d:]?\s*"
        r"(?P<min>\d+(?:\.\d+)?)?'?\s*"
        r"(?P<sec>\d+(?:\.\d+)?)?\"?\s*"
        r"(?P<hem>[NnSsEeWw])?",
        token,
    )
    if not m:
        raise ValueError(f"Koordinate nicht lesbar: {token}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min") or 0.0)
    seconds = float(m.group("sec") or 0.0)
    hem = (m.group("hem") or "").upper()
    dec = abs(deg) + minutes / 60.0 + seconds / 3600.0
    if hem in ("S", "W") or deg < 0:
        dec = -dec
    return dec


def parse_coord_pair(text: str) -> tuple[float, float]:
    """Parst einen String mit Lat/Lon in Grad (DMS oder Dezimal)."""
    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError(f"Erwarte 'lat lon', bekam: {text}")
    lat = parse_dms_token(parts[0])
    lon = parse_dms_token(parts[1])
    return lat, lon


def coords_to_polygon_wgs84(coords: Sequence[str]) -> Polygon:
    """Baue WGS84-Polygon aus Text-Koordinaten."""
    pts = [parse_coord_pair(c) for c in coords]
    if len(pts) < 3:
        raise ValueError("Mindestens 3 Koordinaten fuer ein Polygon benoetigt.")
    poly = Polygon([(lon, lat) for lat, lon in pts])
    if poly.is_empty or not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("Polygon aus Koordinaten ist leer.")
    return poly


def coords_to_points_wgs84(coords: Sequence[str]) -> list[tuple[float, float]]:
    """Parst Koordinatenliste als einzelne Punkte (lon, lat)."""
    pts = [parse_coord_pair(c) for c in coords]
    return [(lon, lat) for lat, lon in pts]


def wgs84_to_utm(poly: Polygon, target_epsg: str = "EPSG:25833") -> Polygon:
    """Transformiert ein WGS84-Polygon nach Ziel-CRS."""
    try:
        from pyproj import Transformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyproj wird benoetigt, bitte installieren.") from exc

    transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
    proj_pts = [transformer.transform(x, y) for x, y in poly.exterior.coords]
    projected = Polygon(proj_pts)
    if projected.is_empty:
        raise ValueError("Projektion lieferte leeres Polygon.")
    return projected


def wgs84_points_to_utm(
    points: Sequence[tuple[float, float]],
    target_epsg: str = "EPSG:25833",
) -> list[tuple[float, float]]:
    """Transform individual longitude/latitude points to the target CRS."""
    try:
        from pyproj import Transformer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyproj wird benoetigt, bitte installieren.") from exc

    transformer = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
    return [transformer.transform(longitude, latitude) for longitude, latitude in points]


def select_parcels_in_polygon(
    parcels: list[dict], poly: Polygon, mode: str = "intersects"
) -> list[int]:
    """Waehlt Parcels anhand des Zielpolygons.

    mode:
      - centroid: nimmt Parcels, deren Schwerpunkt im Polygon liegt.
      - intersects: nimmt Parcels, deren Geometrie das Polygon schneidet/ueberlappt (default).
      - contains: nimmt Parcels, die das Polygonzentrum enthalten.
    """
    selected: list[int] = []
    for idx, parcel in enumerate(parcels):
        g = robust_clean(parcel["geom"])
        if g.is_empty:
            continue
        if mode == "centroid":
            if poly.contains(g.centroid):
                selected.append(idx)
        elif mode == "contains":
            if g.contains(poly.centroid):
                selected.append(idx)
        elif g.intersects(poly):
            selected.append(idx)
    if selected:
        return selected
    # Fallback: wenn nichts gefunden, versuche centroid-basierte Auswahl als Reserve
    if mode != "centroid":
        for idx, parcel in enumerate(parcels):
            g = robust_clean(parcel["geom"])
            if g.is_empty:
                continue
            if poly.contains(g.centroid):
                selected.append(idx)
    return selected


def select_parcels_by_points(
    parcels: list[dict], pts_utm: list[tuple[float, float]], mode: str = "contains"
) -> list[int]:
    """Waehlt Parcels anhand einzelner Punkte (UTM).

    mode:
      - contains: Parcel enthaelt den Punkt (default)
      - intersects: Punkt liegt auf/innen der Geometrie
    """
    selected: list[int] = []
    for idx, parcel in enumerate(parcels):
        g = robust_clean(parcel["geom"])
        if g.is_empty:
            continue
        for x, y in pts_utm:
            pt = Point(x, y)
            ok = g.contains(pt) or g.intersects(pt) if mode == "contains" else g.intersects(pt)
            if ok:
                selected.append(idx)
                break
    return selected


def select_parcels_in_radius(parcels: list[dict], center_xy, radius_m: float) -> list[int]:
    """Waehlt Parcels aus, deren Schwerpunkt innerhalb eines Kreisradius liegt."""
    cx, cy = center_xy
    selected: list[tuple[float, int]] = []
    r2 = radius_m * radius_m
    for idx, parcel in enumerate(parcels):
        g = robust_clean(parcel["geom"])
        if g.is_empty:
            continue
        pt = g.centroid
        dx = pt.x - cx
        dy = pt.y - cy
        if dx * dx + dy * dy <= r2:
            selected.append((dx * dx + dy * dy, idx))
    selected.sort(key=lambda t: t[0])
    return [idx for _, idx in selected]


def ensure_four_channel_sample(sample_path: Path) -> None:
    """Stellt sicher, dass X vier Kanaele enthaelt (fuellt Strassenmaske ggf. mit Nullen)."""
    with np.load(sample_path, allow_pickle=True) as data:
        X = data["X"]
        Y = data["Y"]
        meta = data["meta"].item()
        y_neigh = data.get("Y_neigh")
    if X.shape[0] == 4:
        return
    if X.shape[0] > 4:
        raise ValueError(f"{sample_path}: hat mehr als 4 Kanaele ({X.shape[0]})")
    pad = np.zeros_like(X[:1])
    X_padded = np.concatenate([X, pad], axis=0)
    meta["channels"] = [
        "C0_build_mask",
        "C1_neighbor_height",
        "C2_parcel_edges",
        "C3_street_mask",
    ]
    np.savez_compressed(
        sample_path,
        X=X_padded.astype(np.float32),
        Y=Y.astype(np.float32),
        Y_neigh=None if y_neigh is None else y_neigh.astype(np.float32),
        meta=meta,
    )


def write_manifest(sample_ids: Sequence[str], out_path: Path) -> None:
    """Schreibt eine Mini-Split-Manifest-Datei, damit build_dataset die neuen Samples laden kann."""
    manifest = {
        "splits": {"train": list(sample_ids), "val": list(sample_ids), "test": list(sample_ids)}
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_samples_for_polygon(
    args,
    parcels: list[dict],
    parcel_indices: list[int],
) -> list[Path]:
    """Erzeugt Samples fuer die gewaehlten Parcel-Indizes."""
    if not parcel_indices:
        if args.fallback_radius_m and args.fallback_radius_m > 0:
            print(
                f"Keine Parcels im Polygon - fallback auf Umkreis {args.fallback_radius_m} m um Schwerpunkt."
            )
            poly_wgs = coords_to_polygon_wgs84(args.coords)
            poly_utm = wgs84_to_utm(poly_wgs)
            cx, cy = poly_utm.centroid.x, poly_utm.centroid.y
            parcel_indices = select_parcels_in_radius(
                parcels, (cx, cy), float(args.fallback_radius_m)
            )
        if not parcel_indices:
            raise RuntimeError("Keine Parcels im Zielpolygon gefunden.")

    voxel_m = float(args.grid_m) / float(args.grid_res)
    D = round(args.grid_m / voxel_m)
    parcel_erode_vox = 2 if D >= 256 else 1
    border_margin_vox = 2
    overlay_stride3d = 4 if D >= 256 else 2

    building_db = create_building_database(
        gml_path=str(args.citygml),
        target_crs="EPSG:25833",
        cache_file=str(args.cache_dir / "building_database.pkl.gz"),
        verbose=bool(args.verbose),
    )

    geoms = [robust_clean(p["geom"]) for p in parcels]
    from shapely.strtree import STRtree  # type: ignore

    parcels_tree = STRtree(geoms)
    wkb_to_idx = {g.wkb: i for i, g in enumerate(geoms)}

    samples: list[Path] = []
    for idx in parcel_indices[: args.max_buildings]:
        res = build_sample(
            seed_idx=idx,
            building_db=building_db,
            parcels_all=parcels,
            parcels_tree=parcels_tree,
            wkb_to_idx=wkb_to_idx,
            out_dir=str(args.samples_dir),
            grid_m=float(args.grid_m),
            voxel_m=voxel_m,
            raster_ss=int(args.raster_ss),
            parcel_erode_vox=parcel_erode_vox,
            parcel_mask_thresh=float(args.parcel_mask_thresh),
            parcel_buffer_m=float(args.parcel_buffer_m),
            grz_gfz_tol=float(args.grz_gfz_tol),
            grz_gfz_abs_tol=float(args.grz_gfz_abs_tol),
            storey_height_m=3.0,
            require_two_neighbors=bool(args.require_two_neighbors),
            min_bgf_m2=0.0,
            max_bgf_m2=0.0,
            min_neighbor_buildings=int(args.min_neighbor_buildings),
            min_overlap_m2=float(args.min_overlap_m2),
            max_property_frac=float(args.max_property_frac),
            max_target_frac=float(args.max_target_frac),
            skip_if_touch_border=bool(args.skip_if_touch_border),
            border_margin_vox=border_margin_vox,
            buildings_require_full_inside=False,
            context_inside_tol_m=2.0,
            z_margin_m=5.0,
            overlay_stride3d=overlay_stride3d,
            png_dpi=300,
            with_streets=not args.force_empty_streets,
            street_mode=args.street_mode,
            street_width_m=float(args.street_width_m),
            wfs_url="https://gdi.berlin.de/services/wfs/detailnetz",
            typename="detailnetz:c_strassenabschnitte",
            edge_width_m=0.5,
            street_cache_dir=str(args.street_cache_dir) if args.street_cache_dir else None,
            no_viz=True,
            verbose=bool(args.verbose),
        )
        if res.get("status") != "ok":
            print(f"SKIP {idx}: {res.get('status')}")
            continue
        pid = None
        if res.get("dir"):
            pid = Path(res["dir"]).name
        if not pid:
            pid = res.get("parcel_id") or res.get("id") or "sample"
        sample_path = args.samples_dir / str(pid) / f"{pid}.npz"
        if not sample_path.exists():
            raise FileNotFoundError(f"Erzeugtes Sample fehlt: {sample_path}")
        ensure_four_channel_sample(sample_path)
        samples.append(sample_path)
        print(f"Sample erzeugt: {sample_path}")
    return samples


def run_inference_for_samples(
    args,
    sample_paths: Sequence[Path],
    manifest_path: Path,
    scene_center_xy: tuple[float, float],
) -> list[ScenePrediction]:
    """Fuehrt nacheinander Inferenz fuer alle Samples aus."""
    cfg = load_run_config(args.checkpoint, str(args.config) if args.config else None)
    cond_dim = resolve_cond_dim(cfg, args.model)
    cond_stats_path = str(args.cond_stats) if args.cond_stats else cfg.get("cond_stats_path")

    dataset = build_dataset(
        split="val",
        cfg=cfg,
        data_root=str(args.samples_dir),
        cond_dim=cond_dim,
        cond_stats_path=cond_stats_path,
        max_samples=None,
        split_manifest=str(manifest_path),
    )

    id_to_index: dict[str, int] = {Path(p).stem: i for i, p in enumerate(dataset.files)}

    in_channels = len(dataset.x_indices) + (3 if dataset.add_coords else 0)
    if args.model == "unet":
        from models.unet.unet3d import UNet3D  # type: ignore

        model = UNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
        )
    else:
        from models.unet.unet3d_cond import ConditionalUNet3D  # type: ignore

        model = ConditionalUNet3D(
            in_channels=in_channels,
            base_channels=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 4),
            cond_dim=cond_dim,
        )

    state = torch.load(args.checkpoint, map_location="cpu")
    weights = state.get("model", state)
    model.load_state_dict(weights, strict=False)

    device_cfg = select_device()
    set_seed(int(args.seed))
    model.to(device_cfg.device)
    model.eval()

    preds: list[ScenePrediction] = []
    target_polys: list[Polygon] = []

    for sample_path in sample_paths:
        sample_id = sample_path.stem
        idx = id_to_index.get(sample_id)
        if idx is None:
            raise ValueError(f"Sample {sample_id} nicht im Manifest.")
        loader = DataLoader(
            Subset(dataset, [idx]),
            batch_size=1,
            num_workers=max(0, args.num_workers),
            shuffle=False,
            collate_fn=voxel_collate,
        )
        batch = next(iter(loader))
        logits = predict_unet(
            model,
            batch,
            cond_dim,
            device_cfg.device,
            bool(args.tta),
            args.tta_mode,
        )
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        meta = load_sample_metadata(sample_path)

        arrays = load_sample_arrays(sample_path, ("Y",))
        target_polygon = None
        if "Y" in arrays:
            try:
                target_polygon = build_target_polygon(arrays["Y"], meta)
                target_polys.append(target_polygon)
            except Exception:
                target_polygon = None

        grid = meta["grid"]
        world_bbox_xy = meta["world_bbox_xy"]
        stride_ratio = max(1, round(grid["D"] / prob.shape[0]))
        pred_mask = prob > float(args.threshold)
        pred_center_world, _ = compute_world_center_and_extent(
            pred_mask, meta=meta, stride=stride_ratio, flip_y=True
        )
        vox_mesh = prepare_prediction_voxels(
            prob,
            world_bbox_xy=world_bbox_xy,
            z_world_range=meta["z_world_range"],
            center_xy=scene_center_xy,
            voxel_size=grid["voxel_m"],
            stride=stride_ratio,
            threshold=float(args.threshold),
        )
        if vox_mesh is None:
            max_prob = float(np.max(prob))
            msg = (
                f"{sample_id}: Prediction enthaelt keine Voxel ueber Threshold "
                f"({args.threshold}); max_prob={max_prob:.4f}"
            )
            if args.allow_empty_pred:
                print(f"SKIP {sample_id}: {msg}")
                continue
            raise RuntimeError(msg)

        preds.append(
            ScenePrediction(
                sample_id=sample_id,
                prob=prob,
                meta=meta,
                target_polygon=target_polygon,
                voxels_mesh=vox_mesh,
                center_world=pred_center_world,
                stride_ratio=stride_ratio,
            )
        )

        out_npy = args.scene_dir / "predictions" / f"{sample_id}_prob.npy"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, prob.astype(np.float32))
        print(f"Inferenz gespeichert: {out_npy}")

    return preds


def render_scene(
    args,
    predictions: Sequence[ScenePrediction],
    scene_center_xy: tuple[float, float],
    exclusion: Polygon | None,
) -> Path:
    """Baut die kombinierte Szene und rendert ein oder mehrere Bilder."""
    if not predictions:
        raise RuntimeError("Keine Predictions zum Rendern.")

    # Gemeinsame Z-Range
    z_range_all: list[float] = []
    for pred in predictions:
        z_range_all.extend(list(pred.meta["z_world_range"]))
    if not z_range_all:
        z_range_all = [0.0, 50.0]
    z_world_range = (min(z_range_all), max(z_range_all))

    neighbor_mesh, neighbor_count = collect_lod1_neighbor_mesh(
        args.citygml,
        center_xy=scene_center_xy,
        radius=float(args.scene_radius),
        z_world_range=z_world_range,
        target_polygon=exclusion,
    )
    print(f"LOD1-Nachbarn: {neighbor_count}")

    street_lines = load_street_lines(
        args.streets_dir,
        center_xy=scene_center_xy,
        radius=float(args.scene_radius),
    )
    streets_mesh = lines_to_polydata(street_lines, center_xy=scene_center_xy, z_offset=0.3)

    parcel_lines = load_parcel_lines(
        args.parcels,
        center_xy=scene_center_xy,
        radius=float(args.scene_radius),
    )
    parcels_mesh = lines_to_polydata(parcel_lines, center_xy=scene_center_xy, z_offset=0.15)

    prediction_meshes = [p.voxels_mesh for p in predictions if p.voxels_mesh is not None]
    combined_pred = (
        pv.MultiBlock(prediction_meshes).combine().clean() if prediction_meshes else None
    )

    bounds = aggregate_bounds(
        [combined_pred, neighbor_mesh, streets_mesh, parcels_mesh, *prediction_meshes]
    )
    if bounds is None:
        raise RuntimeError("Konnte keine Bounds bestimmen.")

    xy_extent = max(
        abs(bounds[0]),
        abs(bounds[1]),
        abs(bounds[2]),
        abs(bounds[3]),
        float(args.scene_radius) * 0.65,
    )
    z_extent = max(bounds[5] - bounds[4], 1.0)

    legacy_cam_radius = max(xy_extent * 0.9, float(args.scene_radius) * 0.5)
    legacy_cam_height = max(legacy_cam_radius * 1.4, z_extent * 1.2)
    legacy_parallel = max(legacy_cam_radius, 1.0)
    tight_cam_radius = max(legacy_cam_radius * 0.5, 5.0)
    tight_cam_height = max(tight_cam_radius * 0.9, 5.0)
    tight_parallel = max(tight_cam_radius * 0.55, abs(tight_cam_height) * 0.75, 5.0)

    cam_radius = (1.0 - args.zoom_factor) * legacy_cam_radius + args.zoom_factor * tight_cam_radius
    cam_height = (1.0 - args.zoom_factor) * legacy_cam_height + args.zoom_factor * tight_cam_height
    parallel_scale = (1.0 - args.zoom_factor) * legacy_parallel + args.zoom_factor * tight_parallel

    focus_z = (bounds[4] + bounds[5]) * 0.5

    angles = [float(a) for a in args.angles]
    if not angles:
        angles = [40.0]

    render_dir = args.scene_dir / "renders"
    if not pyvista_render_available():
        print("Kein VTK-Display verfügbar; verwende Matplotlib-Offscreen-Fallback.")
        render_mesh_point_cloud(
            prediction_meshes=prediction_meshes,
            context_meshes=[neighbor_mesh, streets_mesh, parcels_mesh],
            bounds=bounds,
            angles=angles,
            output_directory=render_dir,
            window_size=args.window_size,
        )
        print(f"Renders gespeichert in {render_dir}")
        return render_dir

    plotter = pv.Plotter(off_screen=True, window_size=tuple(args.window_size))
    with contextlib.suppress(Exception):
        plotter.enable_depth_peeling()
    plotter.set_background("white")

    if parcels_mesh is not None:
        plotter.add_mesh(
            parcels_mesh,
            color=(0.0, 0.0, 0.0),
            line_width=1.2,
            opacity=1.0,
            render_lines_as_tubes=False,
        )
    if neighbor_mesh is not None:
        plotter.add_mesh(
            neighbor_mesh,
            color=(0.8, 0.8, 0.8),
            opacity=0.3,
            smooth_shading=False,
        )
    if streets_mesh is not None:
        plotter.add_mesh(
            streets_mesh,
            color=(0.8, 0.8, 0.8),
            line_width=10.0,
            opacity=1.0,
            render_lines_as_tubes=False,
        )
    for vox in prediction_meshes:
        edges = vox.extract_all_edges()
        plotter.add_mesh(
            vox,
            color=(0.12, 0.35, 0.92),
            opacity=1.0,
            specular=0.2,
            smooth_shading=False,
            style="surface",
        )
        if edges is not None and edges.n_cells:
            plotter.add_mesh(
                edges,
                color=(0.0, 0.0, 0.0),
                opacity=0.45,
                line_width=1.0,
                render_lines_as_tubes=False,
            )

    orbit_render(
        plotter=plotter,
        angles=angles,
        output_dir=render_dir,
        focus_z=focus_z,
        radius=cam_radius,
        height=cam_height,
        parallel_scale=parallel_scale,
    )
    plotter.close()
    print(f"Renders gespeichert in {render_dir}")
    return render_dir


def main(argv: Sequence[str] | None = None) -> Path:
    """Parse scene arguments and run generation, inference, and rendering."""
    parser = argparse.ArgumentParser(
        description="Mehrere KI-Gebaeude nacheinander generieren und als Szene rendern."
    )
    parser.add_argument("--scene_name", required=True, help="Name der Szene (Ordnername).")
    parser.add_argument(
        "--coords",
        nargs="+",
        required=True,
        help='Polygon-Koordinaten (WGS84, z.B. "52d...N 13d...E").',
    )
    parser.add_argument(
        "--citygml", type=Path, default=Path("00_Data/01_InputData/input/berlin_lod1_merged.gml.gz")
    )
    parser.add_argument(
        "--parcels", type=Path, default=Path("00_Data/01_InputData/input/flurstuecke.geojson")
    )
    parser.add_argument(
        "--streets_dir", type=Path, default=Path("00_Data/01_InputData/cache/streets")
    )
    parser.add_argument(
        "--output_root", type=Path, default=Path("03_visuals/outputs/custom_scenes")
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Pfad zum Modell-Checkpoint (.pt)."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optionale Trainings-Config JSON."
    )
    parser.add_argument(
        "--cond_stats", type=Path, default=None, help="Konditions-Stats (falls cond_unet)."
    )
    parser.add_argument("--model", type=str, default="cond_unet", choices=("unet", "cond_unet"))
    parser.add_argument("--grid_m", type=float, default=128.0)
    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--raster_ss", type=int, default=2)
    parser.add_argument("--parcel_mask_thresh", type=float, default=0.35)
    parser.add_argument("--parcel_buffer_m", type=float, default=0.0)
    parser.add_argument("--cache_dir", type=Path, default=Path("00_Data/01_InputData/cache"))
    parser.add_argument(
        "--street_cache_dir", type=Path, default=Path("00_Data/01_InputData/cache/streets")
    )
    parser.add_argument(
        "--street_mode", type=str, default="buffer", choices=("buffer", "centerline")
    )
    parser.add_argument("--street_width_m", type=float, default=8.0)
    parser.add_argument(
        "--force_empty_streets",
        action="store_true",
        help="Kein WFS; fuelle Strassenkanal mit Nullen.",
    )
    parser.add_argument(
        "--max_buildings", type=int, default=3, help="Maximale Anzahl neuer Gebaeude im Polygon."
    )
    parser.add_argument(
        "--scene_radius", type=float, default=500.0, help="Radius fuer LOD1-Kontext/Strassen (m)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Schwellwert fuer die Prediction."
    )
    parser.add_argument(
        "--allow_empty_pred",
        action="store_true",
        help="Wenn keine Voxel ueber Threshold liegen, Sample ueberspringen statt abbrechen.",
    )
    parser.add_argument(
        "--angles", nargs="+", type=float, default=[40.0], help="Winkel fuer Screenshots (Grad)."
    )
    parser.add_argument(
        "--window_size", nargs=2, type=int, default=[1920, 1080], metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument(
        "--zoom_factor", type=float, default=0.6, help="Kameramischung nah/weit (0..1)."
    )
    parser.add_argument(
        "--grz_gfz_tol", type=float, default=0.20, help="Relative Toleranz fuer GRZ/GFZ-Pruefung."
    )
    parser.add_argument(
        "--grz_gfz_abs_tol",
        type=float,
        default=0.05,
        help="Absolute Toleranz fuer GRZ/GFZ-Pruefung.",
    )
    parser.add_argument(
        "--min_neighbor_buildings", type=int, default=3, help="Mindestanzahl an Nachbargebaeuden."
    )
    parser.add_argument(
        "--min_overlap_m2",
        type=float,
        default=2.0,
        help="Mindestflaeche fuer Gebaeudeueberlappung (m^2).",
    )
    parser.add_argument(
        "--max_property_frac",
        type=float,
        default=0.60,
        help="Max. Anteil Kontextflaeche innerhalb Parcel.",
    )
    parser.add_argument(
        "--max_target_frac",
        type=float,
        default=0.30,
        help="Max. Anteil Zielgebaeude innerhalb Parcel.",
    )
    parser.add_argument(
        "--require_two_neighbors",
        action="store_true",
        help="Erzwingt mindestens zwei Nachbargebaeude.",
    )
    parser.add_argument(
        "--skip_if_touch_border", action="store_true", help="Skip, wenn Zielgebaeude Rand beruehrt."
    )
    parser.add_argument(
        "--fallback_radius_m",
        type=float,
        default=120.0,
        help="Falls kein Parcel im Polygon gefunden wird: Kreisradius um den Schwerpunkt, um naechstgelegene Parcels einzusammeln.",
    )
    parser.add_argument(
        "--parcel_select",
        type=str,
        default="intersects",
        choices=("centroid", "intersects", "contains"),
        help="Parcel-Auswahlmodus fuer Polygon: intersects=Polygon schneidet Parcel (default), centroid=Parcel-Schwerpunkt im Polygon, contains=Parcel enthaelt Polygon-Zentrum.",
    )
    parser.add_argument(
        "--coord_mode",
        type=str,
        default="polygon",
        choices=("polygon", "points"),
        help="polygon=Coords bilden ein Polygon, points=Coords werden als Einzelpunkte interpretiert und Parcels gesucht, die diese Punkte enthalten.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true", help="Test-Time-Augmentation einschalten.")
    parser.add_argument("--tta_mode", type=str, default="rot90", choices=("rot90", "flip"))
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    args.output_root = args.output_root.resolve()
    args.scene_dir = (args.output_root / args.scene_name).resolve()
    args.samples_dir = args.scene_dir / "samples"
    args.citygml = args.citygml.resolve()
    args.parcels = args.parcels.resolve()
    args.streets_dir = args.streets_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.config:
        args.config = args.config.resolve()
    if args.cond_stats:
        args.cond_stats = args.cond_stats.resolve()
    if args.street_cache_dir:
        args.street_cache_dir = args.street_cache_dir.resolve()

    args.samples_dir.mkdir(parents=True, exist_ok=True)

    # Polygon vorbereiten
    poly_wgs = None
    poly_utm = None
    pts_utm: list[tuple[float, float]] = []
    center_xy = None
    if args.coord_mode == "polygon":
        poly_wgs = coords_to_polygon_wgs84(args.coords)
        poly_utm = wgs84_to_utm(poly_wgs)
        poly_bbox = box(*poly_utm.bounds)
        center_xy = (poly_bbox.centroid.x, poly_bbox.centroid.y)
    else:
        pts_wgs = coords_to_points_wgs84(args.coords)
        pts_utm = wgs84_points_to_utm(pts_wgs)
        center_xy = (
            float(sum(x for x, y in pts_utm) / len(pts_utm)),
            float(sum(y for x, y in pts_utm) / len(pts_utm)),
        )

    print(f"Polygon-Zentrum (UTM): {center_xy}")
    print("Parcels laden...")
    parcels = load_parcels(str(args.parcels))
    if args.coord_mode == "points":
        parcel_indices = select_parcels_by_points(
            parcels, [(float(x), float(y)) for x, y in pts_utm]
        )
        print(f"Gefundene Parcels durch Punktabfrage: {len(parcel_indices)}")
    else:
        parcel_indices = select_parcels_in_polygon(parcels, poly_utm, mode=str(args.parcel_select))
        print(
            f"Gefundene Parcels im Ausschnitt: {len(parcel_indices)} (Modus: {args.parcel_select})"
        )

    print("Samples erzeugen...")
    sample_paths = build_samples_for_polygon(args, parcels, parcel_indices)
    if not sample_paths:
        raise SystemExit("Keine Samples erzeugt.")

    sample_ids = [p.stem for p in sample_paths]
    manifest_path = args.scene_dir / "manifest.json"
    write_manifest(sample_ids, manifest_path)

    print("Inferenz nacheinander...")
    predictions = run_inference_for_samples(args, sample_paths, manifest_path, center_xy)

    exclusion = None
    polys = [p for p in (pred.target_polygon for pred in predictions) if p is not None]
    if polys:
        try:
            exclusion = robust_clean(unary_union(polys))
        except Exception:
            exclusion = unary_union(polys)

    print("Rendern...")
    render_directory = render_scene(args, predictions, center_xy, exclusion)

    meta_out = args.scene_dir / "scene_info.json"
    meta_out.write_text(
        json.dumps(
            {
                "coords_wgs84": args.coords,
                "center_xy": center_xy,
                "samples": [str(p) for p in sample_paths],
                "manifest": str(manifest_path),
                "checkpoint": str(args.checkpoint),
                "config": str(args.config) if args.config else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Szenen-Metadaten gespeichert: {meta_out}")
    return render_directory


if __name__ == "__main__":
    main()
