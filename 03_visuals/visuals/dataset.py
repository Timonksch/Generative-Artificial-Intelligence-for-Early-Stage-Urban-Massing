"""Dataset-level figures generated directly from sample metadata."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from visuals.records import DatasetSample, load_dataset_samples
from visuals.style import ACCENT, DARK, GRID, LIGHT, PALETTE, PURPLE, RED, SPLIT_COLORS, save_figure

_METRICS = (
    ("grz_target", "GRZ", (0.0, 1.0)),
    ("gfz_target", "GFZ", None),
    ("target_height_m", "Target height (m)", None),
    ("parcel_area_m2", "Parcel area (m²)", None),
    ("target_footprint_area_m2", "Target footprint (m²)", None),
    ("num_neighbors", "Neighbour buildings", None),
)
_BUCKETS = (
    ("A", 100.0, 500.0),
    ("B", 500.0, 1500.0),
    ("C", 1500.0, 5000.0),
)
_REGULATORY_DISTRIBUTIONS = (
    ("grz_target", "GRZ", None, 1),
    ("gfz_target", "GFZ", 8.0, 2),
    ("target_height_m", "Height (m)", None, 1),
)
_PARCEL_NEIGHBOR_DISTRIBUTIONS = (
    ("parcel_area_m2", "Parcel area (m²)", 3000.0, 3),
    ("num_neighbors", "Neighbour buildings", None, 2),
)
_SPLIT_ORDER = ("all", "train", "val", "test")
_DENSITY_LABELS = {
    "all": "All samples",
    "train": "Training",
    "val": "Validation",
    "test": "Test",
}
_DENSITY_COLORS = {"all": PURPLE, "train": ACCENT, "val": RED, "test": PURPLE}
_DENSITY_GRID_BINS = 190
_DENSITY_SMOOTH_SIGMA = 3.5
_DENSITY_PAD_FRACTION = 0.09
_DENSITY_MIN_SPAN_M = 128.0
_DENSITY_VIEW_ELEVATION = 68
_DENSITY_VIEW_AZIMUTH = -90
_DENSITY_BASE_ALPHA = 0.50
_DENSITY_WIREFRAME_STRIDE = 7
_DENSITY_WIREFRAME_HEIGHT = 0.28
_TARGET_CRS = "EPSG:25833"


@dataclass(frozen=True)
class _DistributionPanelLayout:
    """Figure and spacing settings for one distribution panel."""

    figsize: tuple[float, float]
    left: float
    right: float
    bottom: float
    wspace: float


@dataclass(frozen=True)
class _DensityLayout:
    """Shared grids and optional outline geometry for density panels."""

    grids: tuple[np.ndarray, np.ndarray]
    x_edges: np.ndarray
    y_edges: np.ndarray
    center_x: float
    center_y: float
    outline: Any | None = None
    inside_mask: np.ndarray | None = None


@dataclass(frozen=True)
class _DensityPanelStyle:
    """Visual settings for one 3D density panel."""

    title: str
    color: str
    zoom: float
    outline_width: float


def generate_dataset_figures(
    dataset_directory: Path,
    output_directory: Path,
    *,
    limit: int | None = None,
    districts_path: Path | None = None,
) -> list[Path]:
    """Generate the canonical dataset figure set.

    Args:
        dataset_directory: Generated dataset containing sample metadata.
        output_directory: Destination for figures and the summary manifest.
        limit: Optional positive sample cap for smoke tests.
        districts_path: Optional district boundary GeoJSON for density surfaces.

    Returns:
        Every file written by the operation.

    """
    samples = load_dataset_samples(dataset_directory, limit=limit)
    output_directory.mkdir(parents=True, exist_ok=True)
    written = _plot_distributions(samples, output_directory)
    written.extend(_plot_buckets(samples, output_directory))
    written.extend(_plot_spatial_distribution(samples, output_directory))
    split_mapping = _load_split_mapping(dataset_directory)
    if split_mapping:
        written.extend(_plot_split_profile(samples, split_mapping, output_directory))
    outline = _load_density_outline(districts_path)
    written.extend(_plot_density_surfaces(samples, split_mapping, output_directory, outline))
    summary_path = output_directory / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(_build_summary(samples, dataset_directory), indent=2),
        encoding="utf-8",
    )
    written.append(summary_path)
    return written


def _metric_values(samples: list[DatasetSample], metric_name: str) -> np.ndarray:
    """Collect one finite metric as a NumPy array."""
    values = [sample.metrics[metric_name] for sample in samples if metric_name in sample.metrics]
    return np.asarray(values, dtype=float)


def _plot_distributions(samples: list[DatasetSample], output_directory: Path) -> list[Path]:
    """Plot the central dataset distributions in thesis-ready panels."""
    written = _plot_distribution_panel(
        samples,
        output_directory,
        _REGULATORY_DISTRIBUTIONS,
        "regulatory_distributions",
        _DistributionPanelLayout(
            figsize=(9.6, 3.1),
            left=0.08,
            right=0.99,
            bottom=0.18,
            wspace=0.26,
        ),
    )
    written.extend(
        _plot_distribution_panel(
            samples,
            output_directory,
            _PARCEL_NEIGHBOR_DISTRIBUTIONS,
            "parcel_neighbor_distributions",
            _DistributionPanelLayout(
                figsize=(7.1, 3.2),
                left=0.10,
                right=0.99,
                bottom=0.18,
                wspace=0.26,
            ),
        )
    )
    return written


def _plot_distribution_panel(
    samples: list[DatasetSample],
    output_directory: Path,
    specs: tuple[tuple[str, str, float | None, int], ...],
    stem: str,
    layout: _DistributionPanelLayout,
) -> list[Path]:
    """Plot one horizontal histogram panel."""
    figure, axes = plt.subplots(1, len(specs), figsize=layout.figsize, dpi=450)
    axis_array = np.atleast_1d(axes)
    for axis, (metric_name, label, maximum_x, coarsen) in zip(axis_array, specs, strict=True):
        _histogram(
            axis,
            _metric_values(samples, metric_name),
            xlabel=label,
            x_max=maximum_x,
            coarsen=coarsen,
        )
    figure.subplots_adjust(
        left=layout.left,
        right=layout.right,
        top=0.96,
        bottom=layout.bottom,
        wspace=layout.wspace,
    )
    return save_figure(figure, output_directory, stem)


def _histogram(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    xlabel: str,
    x_max: float | None = None,
    coarsen: int = 1,
) -> None:
    """Draw one consistently styled dataset histogram."""
    finite_values = values[np.isfinite(values)] if values.size else values
    if finite_values.size:
        bins = _histogram_bins(finite_values, x_max=x_max, coarsen=coarsen)
        axis.hist(
            finite_values,
            bins=bins,
            color=ACCENT,
            edgecolor="white",
            linewidth=0.65,
        )
    if x_max is not None:
        axis.set_xlim(0.0, x_max)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Samples")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, linewidth=0.55)
    axis.grid(axis="x", visible=False)
    axis.spines["left"].set_color(LIGHT)
    axis.spines["bottom"].set_color(LIGHT)
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)
    axis.tick_params(axis="both", length=3.0, width=0.75, direction="out")


def _histogram_bins(values: np.ndarray, *, x_max: float | None, coarsen: int) -> np.ndarray | int:
    """Return stable histogram bins with optional range clipping and coarsening."""
    base_bins = min(32, max(6, int(np.sqrt(values.size))))
    bins = max(4, base_bins // max(coarsen, 1))
    if x_max is None:
        return bins
    return np.linspace(0.0, x_max, bins + 1)


def _bucket_for_area(area: float) -> str | None:
    """Return the configured parcel-area bucket."""
    for label, lower_bound, upper_bound in _BUCKETS:
        if lower_bound <= area < upper_bound:
            return label
    return None


def _plot_buckets(samples: list[DatasetSample], output_directory: Path) -> list[Path]:
    """Plot parcel-area bucket counts."""
    counts = Counter(
        bucket
        for sample in samples
        if (area := sample.metrics.get("parcel_area_m2")) is not None
        if (bucket := _bucket_for_area(area)) is not None
    )
    labels = [bucket[0] for bucket in _BUCKETS]
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    bars = axis.bar(labels, values, color=PALETTE[: len(labels)])
    axis.bar_label(bars, padding=3)
    axis.set_xlabel("Parcel-area bucket")
    axis.set_ylabel("Samples")
    axis.set_title("A: 100-500 m² · B: 500-1,500 m² · C: 1,500-5,000 m²")
    figure.tight_layout()
    return save_figure(figure, output_directory, "dataset_bucket_distribution")


def _plot_spatial_distribution(
    samples: list[DatasetSample],
    output_directory: Path,
) -> list[Path]:
    """Plot sample centres in the dataset coordinate reference system."""
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    axis.scatter(
        [sample.center_x for sample in samples],
        [sample.center_y for sample in samples],
        c=ACCENT,
        edgecolors="none",
        alpha=0.72,
        s=14,
    )
    axis.set_xlabel("Easting")
    axis.set_ylabel("Northing")
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    return save_figure(figure, output_directory, "dataset_spatial_distribution")


def _plot_density_surfaces(
    samples: list[DatasetSample],
    split_mapping: dict[str, str],
    output_directory: Path,
    outline: Any | None,
) -> list[Path]:
    """Plot smoothed 3D density surfaces for all samples and known splits."""
    grouped_points = _group_density_points(samples, split_mapping)
    layout = _density_layout(grouped_points["all"], outline)
    panels = [split_name for split_name in _SPLIT_ORDER if grouped_points.get(split_name)]

    columns = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / columns))
    figure = plt.figure(figsize=(6.8 * columns, 4.9 * rows), dpi=300)
    for index, split_name in enumerate(panels, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        points = grouped_points[split_name]
        density = _surface_for_points(points, layout)
        panel_style = _density_panel_style(split_name, len(points), zoom=1.18, outline_width=2.0)
        if split_name == "all":
            _draw_density_line_panel(
                axis,
                layout,
                _line_density_groups(grouped_points, layout),
                panel_style,
            )
        else:
            _draw_density_panel(axis, layout, density, panel_style)
    figure.tight_layout(pad=1.1)

    written = save_figure(figure, output_directory, "dataset_density_surfaces_3d")
    for split_name in panels:
        single_figure = plt.figure(figsize=(8.4, 6.2), dpi=300)
        single_axis = single_figure.add_subplot(1, 1, 1, projection="3d")
        points = grouped_points[split_name]
        density = _surface_for_points(points, layout)
        panel_style = _density_panel_style(split_name, len(points), zoom=1.42, outline_width=2.4)
        if split_name == "all":
            _draw_density_line_panel(
                single_axis,
                layout,
                _line_density_groups(grouped_points, layout),
                panel_style,
            )
        else:
            _draw_density_panel(single_axis, layout, density, panel_style)
        single_figure.tight_layout(pad=0.6)
        written.extend(
            save_figure(
                single_figure,
                output_directory,
                f"dataset_density_surface_3d_{split_name}",
            )
        )
    return written


def _group_density_points(
    samples: list[DatasetSample],
    split_mapping: dict[str, str],
) -> dict[str, list[tuple[float, float]]]:
    """Group sample centres into all/split point clouds."""
    grouped_points: dict[str, list[tuple[float, float]]] = {
        split_name: [] for split_name in _SPLIT_ORDER
    }
    for sample in samples:
        point = (sample.center_x, sample.center_y)
        grouped_points["all"].append(point)
        split_name = split_mapping.get(sample.sample_id)
        if split_name in grouped_points:
            grouped_points[split_name].append(point)
    return grouped_points


def _density_layout(points: list[tuple[float, float]], outline: Any | None) -> _DensityLayout:
    """Build all spatial data needed for the 3D density panels."""
    x_edges, y_edges = _density_edges(points, outline)
    grid_x, grid_y = _density_grid(x_edges, y_edges)
    grid_x_km = _centered_kilometers(grid_x, x_edges)
    grid_y_km = _centered_kilometers(grid_y, y_edges)
    inside_mask = _outline_mask(outline, grid_x, grid_y) if outline is not None else None
    return _DensityLayout(
        grids=(grid_x_km, grid_y_km),
        x_edges=x_edges,
        y_edges=y_edges,
        center_x=(x_edges[0] + x_edges[-1]) / 2.0,
        center_y=(y_edges[0] + y_edges[-1]) / 2.0,
        outline=outline,
        inside_mask=inside_mask,
    )


def _density_edges(
    points: list[tuple[float, float]],
    outline: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded histogram edges from Berlin bounds or the point cloud."""
    if outline is not None:
        minimum_x, minimum_y, maximum_x, maximum_y = outline.bounds
    else:
        coordinates = np.asarray(points, dtype=float)
        minimum_x, minimum_y = np.min(coordinates, axis=0)
        maximum_x, maximum_y = np.max(coordinates, axis=0)
    x_min, x_max = _padded_bounds(float(minimum_x), float(maximum_x))
    y_min, y_max = _padded_bounds(float(minimum_y), float(maximum_y))
    return (
        np.linspace(x_min, x_max, _DENSITY_GRID_BINS + 1),
        np.linspace(y_min, y_max, _DENSITY_GRID_BINS + 1),
    )


def _outline_mask(outline: Any, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Return a mask for grid cells covered by the outline geometry."""
    from shapely.geometry import Point  # noqa: PLC0415
    from shapely.prepared import prep  # noqa: PLC0415

    prepared_outline = prep(outline)
    inside = [
        prepared_outline.covers(Point(x, y))
        for x, y in zip(grid_x.ravel(), grid_y.ravel(), strict=True)
    ]
    return np.asarray(inside, dtype=bool).reshape(grid_x.shape)


def _load_density_outline(path: Path | None) -> Any | None:
    """Load a district GeoJSON as one projected Berlin outline, if available."""
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return None
    from pyproj import Transformer  # noqa: PLC0415
    from shapely.geometry import shape  # noqa: PLC0415
    from shapely.ops import transform, unary_union  # noqa: PLC0415

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError(f"District GeoJSON must contain features: {resolved}")
    geometries = [
        shape(geometry)
        for feature in features
        if isinstance(feature, dict)
        if isinstance(geometry := feature.get("geometry"), dict)
    ]
    if not geometries:
        raise ValueError(f"District GeoJSON contains no geometries: {resolved}")
    outline = unary_union(geometries)
    source_crs = _geojson_crs(payload)
    if source_crs != _TARGET_CRS:
        transformer = Transformer.from_crs(source_crs, _TARGET_CRS, always_xy=True)
        outline = transform(transformer.transform, outline)
    return outline


def _geojson_crs(payload: dict[str, object]) -> str:
    """Return the GeoJSON CRS, defaulting to CRS84."""
    crs = payload.get("crs")
    if not isinstance(crs, dict):
        return "OGC:CRS84"
    properties = crs.get("properties")
    if not isinstance(properties, dict):
        return "OGC:CRS84"
    name = properties.get("name")
    if not isinstance(name, str):
        return "OGC:CRS84"
    if "25833" in name:
        return _TARGET_CRS
    if "4326" in name or "CRS84" in name:
        return "OGC:CRS84"
    return name


def _padded_bounds(minimum: float, maximum: float) -> tuple[float, float]:
    """Return a non-degenerate, padded coordinate interval."""
    span = max(maximum - minimum, _DENSITY_MIN_SPAN_M)
    center = (minimum + maximum) / 2.0
    half_span = span * (0.5 + _DENSITY_PAD_FRACTION)
    return center - half_span, center + half_span


def _density_grid(x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh grids located at histogram bin centres."""
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return np.meshgrid(x_centers, y_centers)


def _centered_kilometers(grid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert a grid axis from projected metres to centered kilometres."""
    return (grid - ((edges[0] + edges[-1]) / 2.0)) / 1000.0


def _surface_for_points(
    points: list[tuple[float, float]],
    layout: _DensityLayout,
) -> np.ndarray:
    """Convert points into a normalized, smoothed 3D density surface."""
    coordinates = np.asarray(points, dtype=float)
    density, _, _ = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=(layout.x_edges, layout.y_edges),
    )
    density = _smooth_density(density.T)
    if layout.inside_mask is not None:
        density[~layout.inside_mask] = np.nan
    finite_density = density[np.isfinite(density)]
    if not finite_density.size:
        return density
    peak = float(np.max(finite_density))
    if peak <= 0.0:
        return density
    return density / peak


def _smooth_density(density: np.ndarray) -> np.ndarray:
    """Apply a separable Gaussian-like blur using only NumPy."""
    kernel = _gaussian_kernel1d(_DENSITY_SMOOTH_SIGMA)
    smoothed = np.apply_along_axis(
        lambda values: np.convolve(values, kernel, mode="same"),
        0,
        density,
    )
    return np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 1, smoothed)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Return a normalized one-dimensional Gaussian kernel."""
    radius = max(3, round(sigma * 3))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _density_panel_style(
    split_name: str,
    sample_count: int,
    *,
    zoom: float,
    outline_width: float,
) -> _DensityPanelStyle:
    """Return the title and color for one density panel."""
    return _DensityPanelStyle(
        title=f"{_DENSITY_LABELS[split_name]} (n={sample_count:,})",
        color=_DENSITY_COLORS[split_name],
        zoom=zoom,
        outline_width=outline_width,
    )


def _line_density_groups(
    grouped_points: dict[str, list[tuple[float, float]]],
    layout: _DensityLayout,
) -> list[tuple[str, np.ndarray]]:
    """Return split densities for the line-only all panel."""
    split_groups = [
        (split_name, _surface_for_points(grouped_points[split_name], layout))
        for split_name in ("train", "val", "test")
        if grouped_points.get(split_name)
    ]
    if split_groups:
        return split_groups
    return [("all", _surface_for_points(grouped_points["all"], layout))]


def _draw_density_line_panel(
    axis: plt.Axes,
    layout: _DensityLayout,
    density_groups: list[tuple[str, np.ndarray]],
    panel_style: _DensityPanelStyle,
) -> None:
    """Draw the combined density panel as colored clipped wireframes only."""
    grid_x_km, grid_y_km = layout.grids
    for split_name, density in density_groups:
        finite_density = density[np.isfinite(density)]
        if not finite_density.size or float(np.max(finite_density)) <= 0.0:
            continue
        _plot_wireframe_lines(
            axis,
            grid_x_km,
            grid_y_km,
            density,
            color=_DENSITY_COLORS[split_name],
        )
    if layout.outline is not None:
        _plot_outline(
            axis,
            layout.outline,
            (layout.center_x, layout.center_y),
            linewidth=panel_style.outline_width,
        )
    _style_density_axis(axis, panel_style)


def _plot_wireframe_lines(
    axis: plt.Axes,
    grid_x_km: np.ndarray,
    grid_y_km: np.ndarray,
    density: np.ndarray,
    *,
    color: str,
) -> None:
    """Plot rows and columns of a masked density grid as 3D lines."""
    wire_density = density * _DENSITY_WIREFRAME_HEIGHT
    row_indices = range(0, density.shape[0], _DENSITY_WIREFRAME_STRIDE)
    column_indices = range(0, density.shape[1], _DENSITY_WIREFRAME_STRIDE)
    for row_index in row_indices:
        axis.plot(
            grid_x_km[row_index, :],
            grid_y_km[row_index, :],
            zs=wire_density[row_index, :],
            color=color,
            linewidth=0.72,
            alpha=0.88,
            zorder=4,
        )
    for column_index in column_indices:
        axis.plot(
            grid_x_km[:, column_index],
            grid_y_km[:, column_index],
            zs=wire_density[:, column_index],
            color=color,
            linewidth=0.72,
            alpha=0.88,
            zorder=4,
        )


def _draw_density_panel(
    axis: plt.Axes,
    layout: _DensityLayout,
    density: np.ndarray,
    panel_style: _DensityPanelStyle,
) -> None:
    """Draw one styled 3D density axis."""
    grid_x_km, grid_y_km = layout.grids
    if layout.outline is not None:
        _plot_base_fill(axis, layout.outline, (layout.center_x, layout.center_y), panel_style.color)
    else:
        axis.plot_surface(
            grid_x_km,
            grid_y_km,
            np.zeros_like(density),
            color=panel_style.color,
            alpha=0.16,
            linewidth=0,
            shade=False,
        )
    axis.plot_surface(
        grid_x_km,
        grid_y_km,
        density,
        rstride=1,
        cstride=1,
        facecolors=_density_tint(panel_style.color, density),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    if layout.outline is not None:
        _plot_outline(
            axis,
            layout.outline,
            (layout.center_x, layout.center_y),
            linewidth=panel_style.outline_width,
        )
    _style_density_axis(axis, panel_style)


def _style_density_axis(axis: plt.Axes, panel_style: _DensityPanelStyle) -> None:
    """Apply shared camera and axis styling to a 3D density panel."""
    axis.set_title(panel_style.title, pad=8)
    axis.set_zlim(0.0, 1.02)
    axis.view_init(elev=_DENSITY_VIEW_ELEVATION, azim=_DENSITY_VIEW_AZIMUTH)
    try:
        axis.set_box_aspect((1.55, 1.0, 0.36), zoom=panel_style.zoom)
    except TypeError:
        axis.set_box_aspect((1.55, 1.0, 0.36))
    axis.set_axis_off()
    axis.grid(False)


def _plot_outline(
    axis: plt.Axes,
    geometry: Any,
    center: tuple[float, float],
    *,
    z: float = 0.015,
    linewidth: float = 1.8,
) -> None:
    """Draw the exterior district outline on the 3D base plane."""
    center_x, center_y = center

    def draw_polygon(polygon: Any) -> None:
        exterior = np.asarray(polygon.exterior.coords)
        axis.plot(
            (exterior[:, 0] - center_x) / 1000.0,
            (exterior[:, 1] - center_y) / 1000.0,
            zs=z,
            color="#111111",
            linewidth=linewidth,
            zorder=10,
        )

    if geometry.geom_type == "Polygon":
        draw_polygon(geometry)
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            draw_polygon(polygon)


def _plot_base_fill(
    axis: plt.Axes,
    geometry: Any,
    center: tuple[float, float],
    color: str,
    *,
    z: float = 0.0,
) -> None:
    """Draw a translucent outline fill below one density surface."""
    center_x, center_y = center

    def polygon_vertices(polygon: Any) -> list[tuple[float, float, float]]:
        exterior = np.asarray(polygon.exterior.coords)
        return [((x - center_x) / 1000.0, (y - center_y) / 1000.0, z) for x, y in exterior]

    if geometry.geom_type == "Polygon":
        polygons = [polygon_vertices(geometry)]
    elif geometry.geom_type == "MultiPolygon":
        polygons = [polygon_vertices(polygon) for polygon in geometry.geoms]
    else:
        return
    axis.add_collection3d(
        Poly3DCollection(
            polygons,
            facecolors=color,
            edgecolors="none",
            alpha=_DENSITY_BASE_ALPHA,
            zorder=1,
        )
    )


def _density_tint(color: str, density: np.ndarray) -> np.ndarray:
    """Blend one palette color into a density-dependent RGBA surface."""
    rgb = np.asarray(to_rgb(color))
    finite_mask = np.isfinite(density)
    strength = np.nan_to_num(np.clip(density, 0.0, 1.0), nan=0.0)
    colors = np.ones((*density.shape, 4), dtype=float)
    color_weight = 0.42 + 0.58 * strength[..., None]
    colors[..., :3] = 1.0 - (1.0 - rgb) * color_weight
    colors[..., 3] = np.where(finite_mask, 0.98, 0.0)
    return colors


def _load_split_mapping(dataset_directory: Path) -> dict[str, str]:
    """Load sample-to-split assignments when a split report is available."""
    report_path = dataset_directory / "report_split.json"
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return {}
    mapping: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_payload = splits.get(split_name, {})
        if not isinstance(split_payload, dict):
            continue
        sample_ids = split_payload.get("sample_ids", [])
        if isinstance(sample_ids, list):
            mapping.update({str(sample_id): split_name for sample_id in sample_ids})
    return mapping


def _plot_split_profile(
    samples: list[DatasetSample],
    split_mapping: dict[str, str],
    output_directory: Path,
) -> list[Path]:
    """Plot parcel-area buckets for each dataset split."""
    split_names = ("train", "val", "test")
    bucket_labels = tuple(bucket[0] for bucket in _BUCKETS)
    counts = {split_name: Counter() for split_name in split_names}
    for sample in samples:
        split_name = split_mapping.get(sample.sample_id)
        area = sample.metrics.get("parcel_area_m2")
        if split_name not in counts or area is None:
            continue
        bucket = _bucket_for_area(area)
        if bucket is not None:
            counts[split_name][bucket] += 1
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    x_positions = np.arange(len(bucket_labels), dtype=float)
    width = 0.24
    for index, split_name in enumerate(split_names):
        offsets = x_positions + (index - 1) * width
        values = [counts[split_name][bucket] for bucket in bucket_labels]
        axis.bar(offsets, values, width, label=split_name, color=SPLIT_COLORS[split_name])
    axis.set_xticks(x_positions, bucket_labels)
    axis.set_xlabel("Parcel-area bucket")
    axis.set_ylabel("Samples")
    axis.legend(ncol=3)
    figure.tight_layout()
    return save_figure(figure, output_directory, "dataset_bucket_split_profile")


def _build_summary(samples: list[DatasetSample], dataset_directory: Path) -> dict[str, object]:
    """Build the machine-readable dataset figure manifest."""
    metrics: dict[str, dict[str, float]] = {}
    for metric_name, _label, _limits in _METRICS:
        values = _metric_values(samples, metric_name)
        if not values.size:
            continue
        metrics[metric_name] = {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "dataset": str(dataset_directory),
        "sample_count": len(samples),
        "metrics": metrics,
        "style": {"accent": ACCENT, "text": DARK},
    }
