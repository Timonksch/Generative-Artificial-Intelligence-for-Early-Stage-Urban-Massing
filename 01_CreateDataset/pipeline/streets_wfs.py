"""Fetch Berlin street geometries and rasterize a cached grid-aligned mask."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import requests
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiLineString, box, shape

from .geo_utils import lines_to_mask, polygon_to_mask, robust_clean

DEFAULT_WFS_URL: Final = "https://gdi.berlin.de/services/wfs/detailnetz"
DEFAULT_TYPENAME: Final = "detailnetz:c_strassenabschnitte"
HTTP_OK: Final = 200
WFS_MAX_FEATURES: Final = 1000
StreetMode = Literal["buffer", "centerline"]
Bbox = tuple[float, float, float, float]


class StreetDataError(RuntimeError):
    """Indicate that street data could not be loaded or validated."""


@dataclass(frozen=True, slots=True)
class WfsRequest:
    """Store bounded WFS request parameters."""

    base_url: str = DEFAULT_WFS_URL
    typename: str = DEFAULT_TYPENAME
    srs: str = "EPSG:25833"
    output_format: str = "application/json"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class StreetMaskConfig:
    """Store validated street-mask generation parameters."""

    bbox: Bbox
    grid_res: int
    grid_m: float
    origin_xy: tuple[float, float]
    voxel_m: float
    mode: StreetMode
    street_width_m: float
    min_edge_len_m: float
    request: WfsRequest
    verbose: bool
    cache_directory: Path | None
    tile_m: float


def _tile_key(bbox: Bbox, tile_m: float = 250.0) -> tuple[float, float, float, float, float]:
    """Snap a bounding box to a reusable tile grid.

    Args:
        bbox: Requested horizontal bounds.
        tile_m: Positive tile width in meters.

    Returns:
        Snapped bounds followed by tile width.

    Raises:
        ValueError: If bounds or tile width are invalid.

    """
    min_x, min_y, max_x, max_y = bbox
    if tile_m <= 0 or min_x >= max_x or min_y >= max_y:
        raise ValueError(f"Invalid street tile parameters: bbox={bbox}, tile_m={tile_m}")

    def snap(value: float) -> float:
        """Snap one coordinate down to the configured tile grid."""
        return tile_m * math.floor(value / tile_m)

    return snap(min_x), snap(min_y), snap(max_x) + tile_m, snap(max_y) + tile_m, tile_m


def _cache_path(cache_directory: Path, bbox: Bbox, typename: str, tile_m: float) -> Path:
    """Build a stable cache path for one snapped WFS tile.

    Args:
        cache_directory: Cache root.
        bbox: Requested horizontal bounds.
        typename: WFS feature type.
        tile_m: Tile width in meters.

    Returns:
        GeoJSON cache path.

    """
    digest_input = f"{_tile_key(bbox, tile_m)}_{typename}".encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:24]
    return cache_directory / f"streets_{digest}.geojson"


def _wfs_attempts(request: WfsRequest) -> list[tuple[str, str, str, str]]:
    """Build the bounded endpoint/version/format attempt list.

    Args:
        request: Preferred WFS request.

    Returns:
        Endpoint, typename, version, and format tuples.

    """
    endpoints = (
        (request.base_url or DEFAULT_WFS_URL, (request.typename or DEFAULT_TYPENAME,)),
        (
            "https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s_vms_detailnetz_spatial_gesamt",
            ("fis:s_vms_detailnetz_spatial_gesamt", "s_vms_detailnetz_spatial_gesamt"),
        ),
    )
    versions = (
        ("2.0.0", request.output_format),
        ("2.0.0", "json"),
        ("1.1.0", request.output_format),
        ("1.1.0", "json"),
    )
    return [
        (url, typename, version, output_format)
        for url, typenames in endpoints
        for typename in typenames
        for version, output_format in versions
    ]


def _request_parameters(
    request: WfsRequest, bbox: Bbox, typename: str, version: str, output: str
) -> dict[str, str]:
    """Build WFS GetFeature query parameters.

    Args:
        request: Shared request options.
        bbox: Query bounds.
        typename: Attempted feature type.
        version: Attempted WFS version.
        output: Attempted output format.

    Returns:
        URL query parameter mapping.

    """
    min_x, min_y, max_x, max_y = bbox
    type_key = "typeNames" if version.startswith("2") else "typeName"
    count_key = "count" if version.startswith("2") else "maxFeatures"
    return {
        "service": "WFS",
        "version": version,
        "request": "GetFeature",
        type_key: typename,
        "srsName": request.srs,
        "bbox": f"{min_x},{min_y},{max_x},{max_y},{request.srs}",
        "outputFormat": output,
        count_key: str(WFS_MAX_FEATURES),
    }


def _validate_feature_collection(payload: object) -> dict[str, Any]:
    """Validate a minimal GeoJSON FeatureCollection.

    Args:
        payload: Parsed JSON value.

    Returns:
        Validated feature collection.

    Raises:
        StreetDataError: If the payload has an invalid structure.

    """
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        raise StreetDataError("WFS response is not a GeoJSON FeatureCollection")
    return payload


def fetch_wfs_geojson(request: WfsRequest, bbox: Bbox, *, verbose: bool = False) -> dict[str, Any]:
    """Fetch a street tile through bounded WFS fallbacks.

    Args:
        request: Preferred endpoint and network limits.
        bbox: Snapped query bounds.
        verbose: Print failed attempt details.

    Returns:
        GeoJSON FeatureCollection.

    Raises:
        ValueError: If timeout is not positive.
        StreetDataError: If every bounded attempt fails.

    """
    if request.timeout_seconds <= 0:
        raise ValueError("WFS timeout must be positive")
    errors: list[str] = []
    for url, typename, version, output_format in _wfs_attempts(request):
        parameters = _request_parameters(request, bbox, typename, version, output_format)
        try:
            response = requests.get(url, params=parameters, timeout=request.timeout_seconds)
            if response.status_code != HTTP_OK:
                errors.append(f"{version}/{typename}: HTTP {response.status_code}")
                continue
            return _validate_feature_collection(response.json())
        except (
            requests.RequestException,
            requests.exceptions.JSONDecodeError,
            StreetDataError,
        ) as error:
            errors.append(f"{version}/{typename}: {error}")
            if verbose:
                print(f"[streets] WFS attempt failed: {errors[-1]}")
    preview = "; ".join(errors[-3:])
    raise StreetDataError(f"All {len(errors)} WFS attempts failed. Last errors: {preview}")


def _read_cache(path: Path) -> dict[str, Any] | None:
    """Read and validate a cached feature collection.

    Args:
        path: Cache file path.

    Returns:
        Valid collection or ``None`` for absent/corrupt cache data.

    """
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return _validate_feature_collection(json.load(handle))
    except (OSError, json.JSONDecodeError, StreetDataError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write one cached feature collection atomically.

    Args:
        path: Cache destination.
        payload: Validated feature collection.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".geojson.part")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    temporary_path.replace(path)


def _load_features(config: StreetMaskConfig) -> dict[str, Any]:
    """Load a tile from cache or WFS.

    Args:
        config: Street-mask generation options.

    Returns:
        Validated feature collection.

    """
    cache_path = None
    if config.cache_directory is not None:
        cache_path = _cache_path(
            config.cache_directory, config.bbox, config.request.typename, config.tile_m
        )
        cached = _read_cache(cache_path)
        if cached is not None:
            if config.verbose:
                print(f"[streets] cache hit: {cache_path.name}")
            return cached
    if config.verbose:
        print("[streets] cache miss; fetching WFS")
    tile_key = _tile_key(config.bbox, config.tile_m)
    payload = fetch_wfs_geojson(config.request, tile_key[:4], verbose=config.verbose)
    if cache_path is not None:
        _write_cache(cache_path, payload)
        if config.verbose:
            print(f"[streets] cache write: {cache_path.name}")
    return payload


def _parse_lines(
    payload: dict[str, Any], minimum_length: float, *, verbose: bool
) -> list[LineString]:
    """Extract valid line strings from GeoJSON features.

    Args:
        payload: FeatureCollection object.
        minimum_length: Minimum accepted segment length.
        verbose: Print rejected-geometry messages.

    Returns:
        Valid line segments.

    """
    lines: list[LineString] = []
    for feature in payload["features"]:
        try:
            geometry = shape(feature["geometry"])
        except (KeyError, TypeError, ValueError, GEOSException) as error:
            if verbose:
                print(f"[streets] ignored invalid feature: {error}")
            continue
        if isinstance(geometry, LineString) and geometry.length >= minimum_length:
            lines.append(geometry)
        elif isinstance(geometry, MultiLineString):
            lines.extend(
                line
                for line in geometry.geoms
                if not line.is_empty and line.length >= minimum_length
            )
    return lines


def _clip_lines(lines: list[LineString], bbox: Bbox, *, verbose: bool) -> list[LineString]:
    """Clip line strings to the requested output window.

    Args:
        lines: Source line strings.
        bbox: Output bounds.
        verbose: Print rejected-intersection messages.

    Returns:
        Non-empty clipped line strings.

    """
    window = box(*bbox)
    clipped: list[LineString] = []
    for line in lines:
        try:
            intersection = robust_clean(line.intersection(window))
        except GEOSException as error:
            if verbose:
                print(f"[streets] ignored invalid intersection: {error}")
            continue
        if isinstance(intersection, LineString) and not intersection.is_empty:
            clipped.append(intersection)
        elif isinstance(intersection, MultiLineString):
            clipped.extend(part for part in intersection.geoms if not part.is_empty)
    return clipped


def _validate_config(config: StreetMaskConfig) -> None:
    """Validate physical and raster parameters.

    Args:
        config: Street-mask options.

    Raises:
        ValueError: If an option is inconsistent or out of bounds.

    """
    if config.grid_res <= 0 or config.voxel_m <= 0 or config.grid_m <= 0:
        raise ValueError("Street grid dimensions and voxel size must be positive")
    if not math.isclose(config.grid_res * config.voxel_m, config.grid_m, abs_tol=1e-6):
        raise ValueError("grid_res * voxel_m must equal grid_m")
    if config.mode not in {"buffer", "centerline"}:
        raise ValueError(f"Unsupported street mask mode: {config.mode}")
    if config.street_width_m <= 0 or config.min_edge_len_m < 0:
        raise ValueError("Street width must be positive and minimum length non-negative")


def _render_mask(config: StreetMaskConfig, lines: list[LineString]) -> np.ndarray[Any, Any]:
    """Rasterize clipped lines according to the selected mode.

    Args:
        config: Street-mask options.
        lines: Clipped street segments.

    Returns:
        Unsigned byte mask in ``(H,W)`` order.

    """
    if config.mode == "centerline":
        return lines_to_mask(
            lines,
            config.grid_res,
            config.grid_res,
            config.origin_xy,
            config.voxel_m,
            width_px=1,
        )
    if not lines:
        return np.zeros((config.grid_res, config.grid_res), dtype=np.uint8)
    buffered = MultiLineString(lines).buffer(config.street_width_m * 0.5, cap_style=2, join_style=2)
    return polygon_to_mask(
        buffered,
        config.grid_res,
        config.grid_res,
        config.origin_xy,
        config.voxel_m,
        ss=2,
        thresh=0.25,
    )


# RULE_VIOLATION: Preserve the established keyword API used by pipeline and external scripts.
def street_mask(  # noqa: PLR0913, PLR0917
    bbox: Bbox,
    grid_res: int,
    grid_m: float,
    origin_xy: tuple[float, float],
    voxel_m: float,
    mode: StreetMode = "buffer",
    street_width_m: float = 8.0,
    min_edge_len_m: float = 5.0,
    wfs_url: str = DEFAULT_WFS_URL,
    typename: str = DEFAULT_TYPENAME,
    verbose: bool = False,
    cache_dir: str | None = None,
    tile_m: float = 250.0,
) -> np.ndarray[Any, Any]:
    """Generate a street mask from cached or live WFS geometries.

    Args:
        bbox: Output bounds in EPSG:25833.
        grid_res: Raster width and height.
        grid_m: Physical raster width and height in meters.
        origin_xy: Raster origin in projected coordinates.
        voxel_m: Pixel edge length in meters.
        mode: Buffered surface or one-pixel centerline mode.
        street_width_m: Buffer width used in surface mode.
        min_edge_len_m: Minimum source segment length.
        wfs_url: Preferred WFS endpoint.
        typename: Preferred WFS feature type.
        verbose: Print cache and geometry diagnostics.
        cache_dir: Optional persistent tile-cache directory.
        tile_m: WFS tile width in meters.

    Returns:
        Unsigned byte street mask.

    Raises:
        StreetDataError: If neither cache nor WFS provides usable data.
        ValueError: If parameters are invalid.

    """
    config = StreetMaskConfig(
        bbox=bbox,
        grid_res=grid_res,
        grid_m=grid_m,
        origin_xy=origin_xy,
        voxel_m=voxel_m,
        mode=mode,
        street_width_m=street_width_m,
        min_edge_len_m=min_edge_len_m,
        request=WfsRequest(wfs_url, typename),
        verbose=verbose,
        cache_directory=Path(cache_dir).expanduser() if cache_dir else None,
        tile_m=tile_m,
    )
    _validate_config(config)
    source_lines = _parse_lines(
        _load_features(config), config.min_edge_len_m, verbose=config.verbose
    )
    clipped_lines = _clip_lines(source_lines, config.bbox, verbose=config.verbose)
    if config.verbose:
        print(f"[streets] segments: tile={len(source_lines)}, window={len(clipped_lines)}")
    return _render_mask(config, clipped_lines)
