"""Provide robust geometry operations and vector-to-raster conversion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

MINIMUM_POLYGON_POINTS: Final = 3
MINIMUM_LINE_POINTS: Final = 2
EROSION_2D_NEIGHBORS: Final = 9
MAX_GEOMETRY_PARTS: Final = 100_000
MASK_DIMENSIONS: Final = 2
VOLUME_DIMENSIONS: Final = 3


@dataclass(frozen=True, slots=True)
class RasterTransform:
    """Store world-to-image conversion parameters."""

    origin_xy: tuple[float, float]
    voxel_m: float
    height: int
    scale: int


def robust_clean(geometry: Any) -> Any:
    """Repair a geometry while retaining the original as final fallback.

    Args:
        geometry: Shapely geometry or ``None``.

    Returns:
        Repaired geometry, the original fallback, or ``None``.

    """
    if geometry is None:
        return None
    try:
        cleaned = make_valid(geometry)
        if cleaned.is_empty or not cleaned.is_valid:
            cleaned = geometry.buffer(0)
        return cleaned
    except GEOSException:
        try:
            return geometry.buffer(0)
        except GEOSException:
            return geometry


def extract_polygons(geometry: Any) -> list[Polygon]:
    """Extract positive-area polygons from a potentially nested geometry.

    Args:
        geometry: Polygonal or collection geometry.

    Returns:
        Valid positive-area polygon components.

    Raises:
        ValueError: If a malformed collection exceeds the component limit.

    """
    cleaned = robust_clean(geometry)
    if cleaned is None or cleaned.is_empty:
        return []
    pending = [cleaned]
    polygons: list[Polygon] = []
    visited = 0
    while pending:
        visited += 1
        if visited > MAX_GEOMETRY_PARTS:
            raise ValueError(f"Geometry exceeds {MAX_GEOMETRY_PARTS} components")
        current = pending.pop()
        if isinstance(current, Polygon):
            if current.area > 0:
                polygons.append(current)
            continue
        if isinstance(current, (MultiPolygon, GeometryCollection)):
            pending.extend(current.geoms)
    return polygons


def safe_union(polygons: list[Polygon]) -> Any:
    """Union polygons through bounded repair fallbacks.

    Args:
        polygons: Candidate polygon components.

    Returns:
        Unioned polygonal geometry or an empty polygon.

    """
    valid = [polygon for polygon in polygons if not polygon.is_empty and polygon.area > 0]
    if not valid:
        return Polygon()
    try:
        return robust_clean(unary_union(valid))
    except GEOSException:
        repaired = [robust_clean(polygon) for polygon in valid]
    try:
        return robust_clean(unary_union(repaired))
    except GEOSException:
        pass
    try:
        merged_boundaries = unary_union([polygon.boundary for polygon in repaired])
        polygonized = list(polygonize(merged_boundaries))
        return robust_clean(unary_union(polygonized)) if polygonized else Polygon()
    except GEOSException:
        return max(repaired, key=lambda polygon: polygon.area)


def safe_buffer(geometry: Any, distance: float) -> Any:
    """Buffer a geometry through one repair fallback.

    Args:
        geometry: Shapely geometry or ``None``.
        distance: Signed buffer distance.

    Returns:
        Buffered geometry or the original fallback.

    """
    if geometry is None or geometry.is_empty or distance == 0:
        return geometry
    try:
        return robust_clean(geometry.buffer(distance))
    except GEOSException:
        cleaned = robust_clean(geometry)
        try:
            return robust_clean(cleaned.buffer(distance))
        except GEOSException:
            return geometry


def safe_intersection(first: Any, second: Any) -> Any:
    """Intersect two geometries through one repair fallback.

    Args:
        first: First Shapely geometry.
        second: Second Shapely geometry.

    Returns:
        Cleaned intersection or an empty polygon on failure.

    """
    if first is None or second is None or first.is_empty or second.is_empty:
        return Polygon()
    try:
        return robust_clean(first.intersection(second))
    except GEOSException:
        try:
            return robust_clean(robust_clean(first).intersection(robust_clean(second)))
        except GEOSException:
            return Polygon()


def safe_within(geometry: Any, window: Any, tolerance_m: float = 0.0) -> bool:
    """Test whether a repaired geometry lies within a window.

    Args:
        geometry: Candidate Shapely geometry.
        window: Containing geometry.
        tolerance_m: Optional inward buffer applied to the candidate.

    Returns:
        Whether the candidate is within the window.

    """
    if geometry is None or geometry.is_empty:
        return False
    try:
        candidate = robust_clean(geometry)
        if tolerance_m > 0:
            candidate = safe_buffer(candidate, -tolerance_m)
        return bool(not candidate.is_empty and candidate.within(window))
    except GEOSException:
        return False


def lines_from_polygons(polygons: list[Polygon]) -> list[LineString]:
    """Extract exterior and interior rings as line strings.

    Args:
        polygons: Polygon components.

    Returns:
        Boundary line strings.

    """
    lines: list[LineString] = []
    for polygon in polygons:
        cleaned = robust_clean(polygon)
        if cleaned.is_empty or cleaned.area <= 0:
            continue
        try:
            lines.append(LineString(cleaned.exterior.coords))
            lines.extend(LineString(hole.coords) for hole in cleaned.interiors)
        except (AttributeError, GEOSException, ValueError):
            continue
    return lines


def _validate_raster(width: int, height: int, voxel_m: float) -> None:
    """Validate common raster dimensions.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        voxel_m: Pixel edge length in meters.

    Raises:
        ValueError: If a dimension is not positive.

    """
    if width <= 0 or height <= 0 or voxel_m <= 0:
        raise ValueError("Raster width, height, and voxel size must be positive")


def _pixel_coordinates(
    coordinates: Iterable[tuple[float, float]],
    origin_xy: tuple[float, float],
    voxel_m: float,
    height: int,
    scale: int,
) -> list[tuple[int, int]]:
    """Transform world coordinates to rounded image coordinates.

    Args:
        coordinates: World-coordinate pairs.
        origin_xy: Lower-left raster origin.
        voxel_m: Pixel edge length in meters.
        height: Base raster height.
        scale: Supersampling scale.

    Returns:
        Image-coordinate pairs.

    """
    pixels = []
    for x, y, *_remainder in coordinates:
        pixel_x = (x - origin_xy[0]) / voxel_m
        pixel_y = height - 1 - (y - origin_xy[1]) / voxel_m
        pixels.append((round(pixel_x * scale), round(pixel_y * scale)))
    return pixels


def _draw_polygons(
    draw: ImageDraw.ImageDraw,
    polygons: list[Polygon],
    transform: RasterTransform,
    fill: int,
) -> None:
    """Draw polygon exteriors and holes on a Pillow image.

    Args:
        draw: Pillow drawing context.
        polygons: Polygon components.
        transform: World-to-image conversion parameters.
        fill: Exterior fill value.

    """
    for polygon in polygons:
        exterior = _pixel_coordinates(
            polygon.exterior.coords,
            transform.origin_xy,
            transform.voxel_m,
            transform.height,
            transform.scale,
        )
        if len(exterior) >= MINIMUM_POLYGON_POINTS:
            draw.polygon(exterior, outline=fill, fill=fill)
        for hole in polygon.interiors:
            points = _pixel_coordinates(
                hole.coords,
                transform.origin_xy,
                transform.voxel_m,
                transform.height,
                transform.scale,
            )
            if len(points) >= MINIMUM_POLYGON_POINTS:
                draw.polygon(points, outline=0, fill=0)


# RULE_VIOLATION: Preserve the established positional rasterization API.
def polygon_to_mask(  # noqa: PLR0913, PLR0917
    polygon: Polygon,
    width: int,
    height: int,
    origin_xy: tuple[float, float],
    voxel_m: float,
    ss: int = 2,
    thresh: float = 0.5,
) -> np.ndarray[Any, Any]:
    """Convert polygonal geometry to a supersampled binary mask.

    Args:
        polygon: Input polygonal geometry.
        width: Raster width in pixels.
        height: Raster height in pixels.
        origin_xy: Lower-left world-coordinate origin.
        voxel_m: Pixel edge length in meters.
        ss: Positive supersampling factor.
        thresh: Downsampling occupancy threshold in ``[0,1]``.

    Returns:
        Unsigned byte mask in ``(H,W)`` order.

    Raises:
        ValueError: If raster or supersampling options are invalid.

    """
    _validate_raster(width, height, voxel_m)
    if ss <= 0 or not 0 <= thresh <= 1:
        raise ValueError("Supersampling must be positive and threshold within [0,1]")
    cleaned = robust_clean(polygon)
    if cleaned is None or cleaned.is_empty:
        return np.zeros((height, width), dtype=np.uint8)
    polygons = extract_polygons(cleaned)
    image = Image.new("L", (width * ss, height * ss), 0)
    fill = 1 if ss == 1 else 255
    transform = RasterTransform(origin_xy, voxel_m, height, ss)
    _draw_polygons(ImageDraw.Draw(image), polygons, transform, fill)
    if ss == 1:
        return np.asarray(image, dtype=np.uint8)
    occupancy = np.asarray(image, dtype=np.float32) / 255.0
    occupancy = occupancy.reshape(height, ss, width, ss).mean(axis=(1, 3))
    return (occupancy >= thresh).astype(np.uint8)


# RULE_VIOLATION: Preserve the established positional line-rasterization API.
def lines_to_mask(  # noqa: PLR0913, PLR0917
    lines: list[LineString],
    width: int,
    height: int,
    origin_xy: tuple[float, float],
    voxel_m: float,
    width_px: int = 2,
) -> np.ndarray[Any, Any]:
    """Rasterize line strings to a binary mask.

    Args:
        lines: Line geometries.
        width: Raster width in pixels.
        height: Raster height in pixels.
        origin_xy: Lower-left world-coordinate origin.
        voxel_m: Pixel edge length in meters.
        width_px: Positive line width in pixels.

    Returns:
        Unsigned byte mask in ``(H,W)`` order.

    Raises:
        ValueError: If raster dimensions or line width are invalid.

    """
    _validate_raster(width, height, voxel_m)
    if width_px <= 0:
        raise ValueError("Line width must be positive")
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for line in lines:
        coordinates = _pixel_coordinates(line.coords, origin_xy, voxel_m, height, 1)
        if len(coordinates) >= MINIMUM_LINE_POINTS:
            draw.line(coordinates, fill=1, width=width_px)
    return np.asarray(image, dtype=np.uint8)


def extrude_mask(
    mask_2d: np.ndarray[Any, Any], z_start: int, z_stop: int, depth: int
) -> np.ndarray[Any, Any]:
    """Extrude a 2D mask over a clipped half-open Z range.

    Args:
        mask_2d: Two-dimensional binary mask.
        z_start: Inclusive start layer.
        z_stop: Exclusive stop layer.
        depth: Positive output depth.

    Returns:
        Unsigned byte volume in ``(D,H,W)`` order.

    Raises:
        ValueError: If mask or depth is invalid.

    """
    if mask_2d.ndim != MASK_DIMENSIONS or depth <= 0:
        raise ValueError("Extrusion requires a 2D mask and positive depth")
    start = max(0, min(depth - 1, int(z_start)))
    stop = max(start + 1, min(depth, int(z_stop)))
    volume = np.zeros((depth, *mask_2d.shape), dtype=np.uint8)
    volume[start:stop] = mask_2d[None, :, :]
    return volume


def binary_erode(mask: np.ndarray[Any, Any], iterations: int = 1) -> np.ndarray[Any, Any]:
    """Erode a binary 2D mask with a three-by-three kernel.

    Args:
        mask: Two-dimensional binary mask.
        iterations: Non-negative erosion count.

    Returns:
        Eroded unsigned byte mask.

    Raises:
        ValueError: If mask is not two-dimensional or iterations is negative.

    """
    if mask.ndim != MASK_DIMENSIONS or iterations < 0:
        raise ValueError("Erosion requires a 2D mask and non-negative iterations")
    eroded = mask.astype(bool)
    for _iteration in range(iterations):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        eroded = windows.all(axis=(-2, -1))
    return eroded.astype(np.uint8)


def touches_border(mask: np.ndarray[Any, Any], margin: int) -> bool:
    """Check whether a mask reaches a configured border band.

    Args:
        mask: Two-dimensional mask.
        margin: Positive border width; zero disables the check.

    Returns:
        Whether an occupied pixel occurs in the border band.

    Raises:
        ValueError: If mask is not two-dimensional or margin is negative.

    """
    if mask.ndim != MASK_DIMENSIONS or margin < 0:
        raise ValueError("Border checks require a 2D mask and non-negative margin")
    if margin == 0:
        return False
    height, width = mask.shape
    clipped = min(margin, height, width)
    return bool(
        mask[:clipped, :].any()
        or mask[-clipped:, :].any()
        or mask[:, :clipped].any()
        or mask[:, -clipped:].any()
    )


def boundary_voxels(volume: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Extract the outer shell of a three-dimensional volume.

    Args:
        volume: Three-dimensional occupancy array.

    Returns:
        Boolean boundary volume.

    Raises:
        ValueError: If volume is not three-dimensional.

    """
    if volume.ndim != VOLUME_DIMENSIONS:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    occupied = volume.astype(bool)
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    neighborhoods = np.lib.stride_tricks.sliding_window_view(padded, (3, 3, 3))
    interior = neighborhoods.all(axis=(-3, -2, -1))
    return occupied & ~interior
