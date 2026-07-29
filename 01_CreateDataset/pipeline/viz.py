"""Create standard overview panels and high-resolution voxel renders."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .geo_utils import boundary_voxels

plt.switch_backend("Agg")


@dataclass(frozen=True, slots=True)
class OverviewData:
    """Store all arrays and labels used by the standard overview."""

    channel_0: np.ndarray[Any, Any]
    channel_1: np.ndarray[Any, Any]
    channel_2: np.ndarray[Any, Any]
    channel_3: np.ndarray[Any, Any] | None
    target_footprint: np.ndarray[Any, Any]
    target: np.ndarray[Any, Any]
    neighbors: np.ndarray[Any, Any]
    parcel_id: str
    max_height_m: float | None


def _set_axes_white(axis: Any) -> None:
    """Set white tick and axis colors used by the established visual style.

    Args:
        axis: Matplotlib 2D or 3D axis.

    """
    axis.tick_params(colors="white", which="both")
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_color("white")
    if hasattr(axis, "zaxis"):
        axis.zaxis.label.set_color("white")
        axis.tick_params(axis="z", colors="white")


def _style_colorbar(colorbar: Any, label: str, max_height_m: float | None = None) -> None:
    """Apply compact colorbar styling and optional metric height ticks.

    Args:
        colorbar: Matplotlib colorbar.
        label: Axis label.
        max_height_m: Metric value corresponding to normalized value one.

    """
    colorbar.ax.tick_params(labelsize=7, length=2, colors="white")
    colorbar.set_label(label, fontsize=8, color="white")
    with suppress(AttributeError):
        colorbar.outline.set_edgecolor("white")
    if max_height_m is None or max_height_m <= 0:
        return
    ticks = np.linspace(0.0, 1.0, 5)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick * max_height_m:.0f} m" for tick in ticks])


def _add_colorbar(
    figure: Any,
    axis: Any,
    image: Any,
    label: str,
    max_height_m: float | None = None,
) -> None:
    """Attach an inset colorbar to one panel.

    Args:
        figure: Parent Matplotlib figure.
        axis: Panel axis.
        image: Scalar mappable returned by ``imshow``.
        label: Colorbar label.
        max_height_m: Optional normalized-to-metric scale.

    """
    colorbar_axis = inset_axes(axis, width="3%", height="90%", loc="center right", borderpad=1.8)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    _style_colorbar(colorbar, label, max_height_m)


def _finish_image_axis(axis: Any, title: str) -> None:
    """Apply the shared title/tick style to an image panel.

    Args:
        axis: Matplotlib image axis.
        title: Panel title.

    """
    axis.set_title(title, fontsize=9, fontweight="semibold")
    axis.set_xticks([])
    axis.set_yticks([])
    _set_axes_white(axis)


def _height_label(max_height_m: float | None) -> str:
    """Build the normalized or metric height label.

    Args:
        max_height_m: Metric height scale or ``None``.

    Returns:
        User-facing height label.

    """
    return f"Height (0-{max_height_m:.0f} m)" if max_height_m else "Normalized height"


def _plot_composite(figure: Any, axis: Any, data: OverviewData) -> None:
    """Draw the established composite input/target panel.

    Args:
        figure: Parent Matplotlib figure.
        axis: Target subplot.
        data: Overview arrays and labels.

    """
    top = [channel.max(axis=0) for channel in (data.channel_0, data.channel_1, data.channel_2)]
    image = axis.imshow(top[1], vmin=0, vmax=1, cmap="viridis", origin="upper")
    axis.imshow(np.where(top[0] > 0, 1.0, np.nan), cmap="Greens", alpha=0.35, origin="upper")
    axis.imshow(np.where(top[2] > 0, 1.0, np.nan), cmap="gray", alpha=0.85, origin="upper")
    title = "Composite: C1 + C0 (green) + C2 (white)"
    if data.channel_3 is not None:
        street_top = data.channel_3.max(axis=0)
        axis.imshow(np.where(street_top > 0, 1.0, np.nan), cmap="Reds", alpha=0.4)
        title += " + C3 (red)"
    axis.contour(data.target.max(axis=0), levels=[0.5], colors=["magenta"], linewidths=1)
    _finish_image_axis(axis, f"{title} + Y (magenta)")
    _add_colorbar(figure, axis, image, _height_label(data.max_height_m), data.max_height_m)


def _plot_channel_panels(figure: Any, grid: Any, data: OverviewData) -> None:
    """Draw scalar input and target panels.

    Args:
        figure: Parent Matplotlib figure.
        grid: Three-by-three subplot grid.
        data: Overview arrays and labels.

    """
    channel_0_axis = figure.add_subplot(grid[0, 1])
    channel_0_axis.imshow(data.channel_0.max(axis=0), cmap="gray", vmin=0, vmax=1)
    _finish_image_axis(channel_0_axis, "C0 - Parcel build mask (binary)")

    channel_1_axis = figure.add_subplot(grid[0, 2])
    channel_1_image = channel_1_axis.imshow(
        data.channel_1.max(axis=0), cmap="viridis", vmin=0, vmax=1
    )
    _finish_image_axis(channel_1_axis, "C1 - Neighbor height heatmap")
    _add_colorbar(
        figure,
        channel_1_axis,
        channel_1_image,
        _height_label(data.max_height_m),
        data.max_height_m,
    )

    auxiliary_axis = figure.add_subplot(grid[1, 0])
    auxiliary = data.channel_3 if data.channel_3 is not None else data.channel_2
    auxiliary_axis.imshow(auxiliary.max(axis=0), cmap="gray", vmin=0, vmax=1)
    title = "C3 - Street mask" if data.channel_3 is not None else "C2 - Parcel edges"
    _finish_image_axis(auxiliary_axis, title)

    footprint_axis = figure.add_subplot(grid[2, 0])
    footprint = data.channel_2.max(axis=0) if data.channel_3 is not None else data.target_footprint
    footprint_axis.imshow(footprint, cmap="gray", vmin=0, vmax=1)
    title = "C2 - Parcel edges" if data.channel_3 is not None else "Target footprint"
    _finish_image_axis(footprint_axis, title)


def _plot_target_panel(figure: Any, axis: Any, array: np.ndarray[Any, Any], title: str) -> None:
    """Draw one target projection/slice panel with occupancy colorbar.

    Args:
        figure: Parent Matplotlib figure.
        axis: Target subplot.
        array: Two-dimensional target array.
        title: Panel title.

    """
    image = axis.imshow(array, cmap="jet", vmin=0, vmax=1, origin="lower", aspect="auto")
    axis.set_title(title, fontsize=9, fontweight="semibold")
    _add_colorbar(figure, axis, image, "Occupancy (0..1)")
    _set_axes_white(axis)


def _voxel_surfaces(
    target: np.ndarray[Any, Any], neighbors: np.ndarray[Any, Any], stride: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build combined occupancy and RGBA surfaces for voxel plotting.

    Args:
        target: Target occupancy volume.
        neighbors: Neighbor occupancy volume.
        stride: Positive downsampling stride.

    Returns:
        Combined boundary occupancy and RGBA face colors.

    Raises:
        ValueError: If stride is not positive or shapes differ.

    """
    if stride <= 0:
        raise ValueError("3D rendering stride must be positive")
    if target.shape != neighbors.shape:
        raise ValueError("Target and neighbor volumes must have equal shapes")
    target_boundary = boundary_voxels(target.astype(bool)[::stride, ::stride, ::stride])
    neighbor_boundary = boundary_voxels(neighbors.astype(bool)[::stride, ::stride, ::stride])
    occupied = np.logical_or(neighbor_boundary, target_boundary)
    colors = np.zeros((*occupied.shape, 4), dtype=float)
    colors[neighbor_boundary] = (0.55, 0.55, 0.55, 0.25)
    colors[target_boundary] = (1.0, 0.0, 1.0, 0.95)
    return occupied, colors


def _plot_voxels(
    axis: Any, target: np.ndarray[Any, Any], neighbors: np.ndarray[Any, Any], stride: int
) -> None:
    """Draw target and neighbor boundary voxels on one axis.

    Args:
        axis: Matplotlib 3D axis.
        target: Target occupancy volume.
        neighbors: Neighbor occupancy volume.
        stride: Positive downsampling stride.

    """
    with suppress(AttributeError, TypeError):
        axis.set_proj_type("ortho")
    occupied, colors = _voxel_surfaces(target, neighbors, stride)
    depth, height, width = occupied.shape
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_zlim(0, depth)
    axis.set_box_aspect((width, height, depth))
    axis.voxels(
        occupied.transpose(2, 1, 0),
        facecolors=colors.transpose(2, 1, 0, 3),
        edgecolor=(0, 0, 0, 0.9),
        linewidth=0.6,
        antialiased=False,
    )
    axis.set_xlabel("x (grid)")
    axis.set_ylabel("y (grid)")
    axis.set_zlabel("z (grid)")
    axis.view_init(elev=20, azim=35)
    _set_axes_white(axis)


def _validate_overview_data(data: OverviewData) -> None:
    """Validate spatial shapes used by an overview.

    Args:
        data: Overview arrays.

    Raises:
        ValueError: If required spatial shapes differ.

    """
    volumes = [data.channel_0, data.channel_1, data.channel_2, data.target, data.neighbors]
    if data.channel_3 is not None:
        volumes.append(data.channel_3)
    if any(volume.shape != data.target.shape for volume in volumes):
        raise ValueError("All overview volumes must share one (D,H,W) shape")
    if data.target_footprint.shape != data.target.shape[1:]:
        raise ValueError("Target footprint must match target (H,W)")


# RULE_VIOLATION: Preserve the established positional visualization API used by the pipeline.
def make_overview_png(  # noqa: PLR0913, PLR0917
    channel_0: np.ndarray[Any, Any],
    channel_1: np.ndarray[Any, Any],
    channel_2: np.ndarray[Any, Any],
    channel_3: np.ndarray[Any, Any] | None,
    mask_target_2d: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    neighbors: np.ndarray[Any, Any],
    parcel_id: str,
    out_path: str,
    stride3d: int = 2,
    dpi: int = 300,
    max_height_m: float | None = None,
) -> None:
    """Create the established three-by-three sample overview panel.

    Args:
        channel_0: Parcel build mask volume.
        channel_1: Neighbor height volume.
        channel_2: Parcel edge volume.
        channel_3: Optional street mask volume.
        mask_target_2d: Target footprint mask.
        target: Target occupancy volume.
        neighbors: Neighbor occupancy volume.
        parcel_id: Sample label.
        out_path: Output PNG path.
        stride3d: Positive 3D rendering stride.
        dpi: Positive output resolution.
        max_height_m: Optional metric scale for channel one.

    Raises:
        ValueError: If shapes or rendering settings are invalid.

    """
    if dpi <= 0:
        raise ValueError("Overview DPI must be positive")
    data = OverviewData(
        channel_0,
        channel_1,
        channel_2,
        channel_3,
        mask_target_2d,
        target,
        neighbors,
        parcel_id,
        max_height_m,
    )
    _validate_overview_data(data)
    figure = plt.figure(figsize=(12, 12), dpi=dpi, facecolor="white")
    grid = figure.add_gridspec(3, 3, wspace=0.12, hspace=0.18)
    _plot_composite(figure, figure.add_subplot(grid[0, 0]), data)
    _plot_channel_panels(figure, grid, data)
    _plot_target_panel(figure, figure.add_subplot(grid[1, 1]), target.max(axis=0), "Y - Top-down")
    _plot_target_panel(
        figure, figure.add_subplot(grid[1, 2]), target[:, target.shape[1] // 2, :], "Y - Mid-Y"
    )
    _plot_target_panel(
        figure, figure.add_subplot(grid[2, 1]), target[:, :, target.shape[2] // 2], "Y - Mid-X"
    )
    voxel_axis = figure.add_subplot(grid[2, 2], projection="3d")
    _plot_voxels(voxel_axis, target, neighbors, stride3d)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.6)
    plt.close(figure)


# RULE_VIOLATION: Preserve the established keyword API for existing visualization callers.
def save_voxels_hires(  # noqa: PLR0913, PLR0917
    target: np.ndarray[Any, Any],
    neighbors: np.ndarray[Any, Any],
    out_path: str,
    stride3d: int = 2,
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 600,
) -> None:
    """Create a high-resolution target/neighbor boundary voxel render.

    Args:
        target: Target occupancy volume.
        neighbors: Neighbor occupancy volume.
        out_path: Output PNG path.
        stride3d: Positive downsampling stride.
        figsize: Positive figure width and height in inches.
        dpi: Positive output resolution.

    Raises:
        ValueError: If dimensions or rendering settings are invalid.

    """
    if dpi <= 0 or any(size <= 0 for size in figsize):
        raise ValueError("Figure size and DPI must be positive")
    figure = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    _plot_voxels(axis, target, neighbors, stride3d)
    axis.set_title(
        "3D voxels - neighbors (gray) / target (magenta)",
        fontsize=12,
        fontweight="semibold",
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.6)
    plt.close(figure)
