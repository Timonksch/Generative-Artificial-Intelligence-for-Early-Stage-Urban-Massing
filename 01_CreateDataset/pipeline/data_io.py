#!/usr/bin/env python3
"""Stream CityGML buildings, index footprints, and load parcel GeoJSON."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from lxml import etree
from pyproj import Transformer
from pyproj.exceptions import ProjError
from shapely import wkb
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from shapely.validation import make_valid

# Namespaces for CityGML 1.0 and 2.0
NS = {
    "gml": "http://www.opengis.net/gml",
    "gml3": "http://www.opengis.net/gml/3.2",
    "core": "http://www.opengis.net/citygml/2.0",
    "core1": "http://www.opengis.net/citygml/1.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "bldg1": "http://www.opengis.net/citygml/building/1.0",
}
COORDINATE_DIMENSIONS_2D = 2
COORDINATE_DIMENSIONS_3D = 3
MINIMUM_RING_POINTS = 3
UNBOUNDED_ENVELOPE = (-1e15, -1e15, 1e15, 1e15)


@dataclass(frozen=True, slots=True)
class BuildingPart:
    """Basic building component with footprint and height range."""

    footprint: Polygon
    z_min: float
    z_max: float


@dataclass(frozen=True, slots=True)
class CachedBuildingPart:
    """Optimized BuildingPart with pre-computed properties for fast queries."""

    footprint: Polygon
    z_min: float
    z_max: float
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    area: float
    height: float
    centroid_xy: tuple[float, float]
    wkb_hash: int  # for LRU cache keys

    @classmethod
    def from_building_part(cls, bp: BuildingPart) -> CachedBuildingPart:
        """Convert BuildingPart to optimized CachedBuildingPart."""
        fp = make_valid(bp.footprint) if not bp.footprint.is_valid else bp.footprint
        if fp.is_empty or fp.area <= 0:
            fp = Polygon()

        bounds = fp.bounds if not fp.is_empty else (0, 0, 0, 0)
        area = fp.area if not fp.is_empty else 0.0
        height = max(0.0, bp.z_max - bp.z_min)

        if not fp.is_empty:
            cx, cy = fp.centroid.x, fp.centroid.y
        else:
            cx = cy = 0.0

        wkb_hash = hash(fp.wkb) if not fp.is_empty else 0

        return cls(
            footprint=fp,
            z_min=bp.z_min,
            z_max=bp.z_max,
            bounds=bounds,
            area=area,
            height=height,
            centroid_xy=(cx, cy),
            wkb_hash=wkb_hash,
        )


# -----------------------------------------------------------------------------
# CityGML Parsing Functions
# -----------------------------------------------------------------------------


def _clean(geom: Any) -> Any:
    """Robust geometry cleaning with fallbacks."""
    try:
        g = make_valid(geom)
        if g.is_empty:
            g = geom.buffer(0)
        if not g.is_valid:
            g = g.buffer(0)
        return g
    except GEOSException:
        try:
            return geom.buffer(0)
        except GEOSException:
            return geom


def _open_any(path: str) -> BinaryIO:
    """Open regular or gzipped files."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _parse_poslist_any(text: str) -> list[tuple[float, float, float]]:
    """Parse CityGML posList with 2D or 3D coordinates."""
    vals = [float(v) for v in (text or "").strip().split()]
    n = len(vals)
    if n < COORDINATE_DIMENSIONS_2D:
        return []
    if n % COORDINATE_DIMENSIONS_3D == 0:
        return [
            (vals[index], vals[index + 1], vals[index + 2])
            for index in range(0, n, COORDINATE_DIMENSIONS_3D)
        ]
    if n % COORDINATE_DIMENSIONS_2D == 0:
        return [
            (vals[index], vals[index + 1], 0.0) for index in range(0, n, COORDINATE_DIMENSIONS_2D)
        ]
    coordinates = [
        (vals[index], vals[index + 1], vals[index + 2])
        for index in range(0, n - 2, COORDINATE_DIMENSIONS_3D)
    ]
    if coordinates:
        return coordinates
    return [
        (vals[index], vals[index + 1], 0.0) for index in range(0, n - 1, COORDINATE_DIMENSIONS_2D)
    ]


def _parse_pos(text: str) -> tuple[float, float, float] | None:
    """Parse single CityGML pos element."""
    vals = [float(v) for v in (text or "").strip().split()]
    if len(vals) >= COORDINATE_DIMENSIONS_3D:
        return (vals[0], vals[1], vals[2])
    if len(vals) == COORDINATE_DIMENSIONS_2D:
        return (vals[0], vals[1], 0.0)
    return None


def _parse_coordinate_token(token: str) -> tuple[float, float, float] | None:
    """Parse one legacy comma-separated GML coordinate token."""
    parts = token.split(",")
    try:
        if len(parts) >= COORDINATE_DIMENSIONS_3D:
            return float(parts[0]), float(parts[1]), float(parts[2])
        if len(parts) == COORDINATE_DIMENSIONS_2D:
            return float(parts[0]), float(parts[1]), 0.0
    except ValueError:
        return None
    return None


def _parse_coordinates(text: str) -> list[tuple[float, float, float]]:
    """Parse legacy GML coordinates format."""
    parsed = [_parse_coordinate_token(token) for token in (text or "").strip().split()]
    return [coordinate for coordinate in parsed if coordinate is not None]


def _footprint_and_z(
    polys_xyz: list[list[tuple[float, float, float]]],
) -> tuple[Polygon, float, float]:
    """Create a 2D footprint from 3D polygons using a convex hull.

    Fast and robust against sliver/overlapping polygons.
    """
    pts2d = []
    zs = []
    for ring in polys_xyz:
        if not ring or len(ring) < MINIMUM_RING_POINTS:
            continue
        for x, y, z in ring:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            pts2d.append((round(x, 3), round(y, 3)))
            if math.isfinite(z):
                zs.append(z)

    if len(pts2d) < MINIMUM_RING_POINTS:
        return Polygon(), 0.0, 0.0

    try:
        hull = MultiPoint(pts2d).convex_hull
    except GEOSException:
        return Polygon(), 0.0, 0.0

    # Reduce to polygon
    if isinstance(hull, MultiPolygon):
        polys = [p for p in hull.geoms if isinstance(p, Polygon) and not p.is_empty]
        hull = max(polys, key=lambda g: g.area) if polys else Polygon()
    elif isinstance(hull, GeometryCollection):
        polys = [g for g in hull.geoms if isinstance(g, Polygon) and not g.is_empty]
        hull = max(polys, key=lambda g: g.area) if polys else Polygon()

    if hull.is_empty or hull.area == 0:
        return Polygon(), 0.0, 0.0

    zmin = float(min(zs)) if zs else 0.0
    zmax = float(max(zs)) if zs else 0.0
    return hull, zmin, zmax


# CRS Detection and Transformation
_SRS_EPSG_RE = re.compile(r"EPSG[:/ ]+(\d+)", re.IGNORECASE)


def _normalize_adv_crs(s: str) -> str | None:
    """Normalize various CRS string formats to EPSG:XXXX."""
    su = s.upper()
    m = _SRS_EPSG_RE.search(su)
    if m:
        return f"EPSG:{m.group(1)}"
    for zone, epsg in (("32", "25832"), ("33", "25833"), ("34", "25834")):
        compact_match = "ETRS89" in su and f"UTM{zone}" in su
        verbose_match = f"UTM ZONE {zone}" in su and ("ETRS89" in su or "ETRF" in su)
        if compact_match or verbose_match:
            return f"EPSG:{epsg}"
    return None


def detect_file_crs(path: str, verbose: bool = False) -> str | None:
    """Detect CRS from CityGML file srsName attributes."""
    try:
        with _open_any(path) as f:
            blob = f.read(32_000_000)
        for m in re.finditer(rb'srsName="([^"]+)"', blob):
            s = m.group(1).decode("utf-8", "ignore")
            crs = _normalize_adv_crs(s)
            if crs:
                if verbose:
                    print(f"  CRS detected in {os.path.basename(path)}: {crs}  ({s})")
                return crs
    except OSError as error:
        if verbose:
            print(f"  Error reading CRS in {os.path.basename(path)}: {error}")
    return None


def make_xy_transform(src_crs: str | None, dst_crs: str | None) -> Callable[[Any], Any] | None:
    """Create coordinate transformation function if needed."""
    if not src_crs or not dst_crs or src_crs == dst_crs:
        return None
    try:
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        def fn(geom: Any) -> Any:
            """Transform one Shapely geometry."""

            def _f(x: Any, y: Any, z: Any = None) -> Any:
                """Transform coordinate arrays while preserving Z values."""
                x2, y2 = transformer.transform(x, y)
                return (x2, y2) if z is None else (x2, y2, z)

            return shp_transform(_f, geom)

        return fn
    except ProjError as error:
        raise RuntimeError(f"CRS transformation {src_crs} to {dst_crs} failed: {error}") from error


def read_envelope_xy(path: str) -> tuple[float, float, float, float]:
    """Read bounding box from CityGML envelope element."""
    try:
        with _open_any(path) as f:
            it = etree.iterparse(
                f,
                events=("end",),
                tag=(f"{{{NS['gml']}}}Envelope", f"{{{NS['gml3']}}}Envelope"),
                no_network=True,
                resolve_entities=False,
            )
            for _event, element in it:
                namespace = etree.QName(element).namespace
                lower = element.find(f"{{{namespace}}}lowerCorner")
                upper = element.find(f"{{{namespace}}}upperCorner")
                if lower is not None and upper is not None and lower.text and upper.text:
                    low_values = [float(value) for value in lower.text.split()]
                    high_values = [float(value) for value in upper.text.split()]
                    return low_values[0], low_values[1], high_values[0], high_values[1]
        return UNBOUNDED_ENVELOPE
    except (OSError, ValueError, etree.XMLSyntaxError):
        return UNBOUNDED_ENVELOPE


# RULE_VIOLATION: The streaming parser keeps one explicit state machine to bound XML memory.
def iter_building_parts_in_file(  # noqa: PLR0912, PLR0915
    path: str,
    target_crs: str | None = None,
    verbose: bool = False,
    bbox_xy_for_prefilter: tuple[float, float, float, float] | None = None,
) -> Iterable[BuildingPart]:
    """Stream CityGML and yield building parts.

    Includes early BBOX pre-filtering for performance.
    """
    if verbose:
        print(f"  Processing file: {os.path.basename(path)}")
    src_crs = detect_file_crs(path, verbose=verbose)
    xform = make_xy_transform(src_crs, target_crs) if target_crs else None

    building_count = 0
    ring_count = 0
    yielded_count = 0

    with _open_any(path) as f:
        it = etree.iterparse(f, events=("start", "end"), no_network=True, resolve_entities=False)
        current_poslists: list[list[tuple[float, float, float]]] = []
        inside_building = 0

        # For fast pre-filtering: bounds of current building coordinates
        bx0 = by0 = float("inf")
        bx1 = by1 = float("-inf")

        def _acc_bounds(pts: list[tuple[float, float, float]]) -> None:
            """Expand the current building bounds with ring coordinates."""
            nonlocal bx0, by0, bx1, by1
            for x, y, _ in pts:
                bx0 = min(bx0, x)
                by0 = min(by0, y)
                bx1 = max(bx1, x)
                by1 = max(by1, y)

        for ev, el in it:
            tag = el.tag
            if ev == "start":
                local = tag.split("}")[-1]
                if local in ("Building", "BuildingPart"):
                    inside_building += 1
                    if inside_building == 1:
                        building_count += 1
                        bx0 = by0 = float("inf")
                        bx1 = by1 = float("-inf")
                        current_poslists = []

            if inside_building > 0 and ev == "end":
                t_end = tag.split("}")[-1]
                if t_end == "posList" and el.text:
                    pts = _parse_poslist_any(el.text)
                    if len(pts) >= MINIMUM_RING_POINTS:
                        current_poslists.append(pts)
                        ring_count += 1
                        _acc_bounds(pts)
                elif t_end == "pos" and el.text:
                    parent = el.getparent()
                    if parent is not None and str(parent.tag).endswith("LinearRing"):
                        ring = []
                        for pos_el in parent.findall(".//{*}pos"):
                            pt = _parse_pos(pos_el.text or "")
                            if pt:
                                ring.append(pt)
                        if len(ring) >= MINIMUM_RING_POINTS:
                            current_poslists.append(ring)
                            ring_count += 1
                            _acc_bounds(ring)
                elif t_end == "coordinates" and el.text:
                    pts = _parse_coordinates(el.text)
                    if len(pts) >= MINIMUM_RING_POINTS:
                        current_poslists.append(pts)
                        ring_count += 1
                        _acc_bounds(pts)

            if ev == "end":
                local = tag.split("}")[-1]
                if local in ("Building", "BuildingPart"):
                    # Early pre-filter against BBOX (only when CRS matches/no transform needed)
                    if bbox_xy_for_prefilter and xform is None:
                        minx, miny, maxx, maxy = bbox_xy_for_prefilter
                        outside = bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy
                        if current_poslists and outside:
                            current_poslists = []
                            inside_building -= 1
                            el.clear()
                            continue

                    if current_poslists:
                        fp, zmin, zmax = _footprint_and_z(current_poslists)
                        if not fp.is_empty and fp.area > 0 and zmax > zmin:
                            if xform:
                                fp = xform(fp)
                            yielded_count += 1
                            yield BuildingPart(fp, zmin, zmax)
                    current_poslists = []
                    inside_building -= 1
                    el.clear()
                else:
                    el.clear()

    if verbose:
        print(f"    -> {building_count} Buildings, {ring_count} Rings, {yielded_count} Parts")


# RULE_VIOLATION: File/directory streaming share this public generator for compatibility.
def iter_building_parts_in_bbox(  # noqa: PLR0912
    gml_src: str,
    bbox_xy: tuple[float, float, float, float],
    target_crs: str | None = None,
    verbose: bool = False,
) -> Iterable[BuildingPart]:
    """Iterate building parts within a bounding box from file or directory."""
    minx, miny, maxx, maxy = bbox_xy
    if verbose:
        print(f"Searching buildings in BBOX: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
        print(f"Target CRS: {target_crs or 'None (Source CRS)'}")

    if os.path.isfile(gml_src):
        if verbose:
            print(f"Processing single file: {gml_src}")
        for part in iter_building_parts_in_file(
            gml_src, target_crs=target_crs, verbose=verbose, bbox_xy_for_prefilter=bbox_xy
        ):
            if part.footprint.is_empty:
                continue
            bx0, by0, bx1, by1 = part.footprint.bounds
            if bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy:
                continue
            yield part
        return

    if verbose:
        print(f"Scanning directory: {gml_src}")
    file_paths: list[str] = []
    for root, _directories, filenames in os.walk(gml_src):
        file_paths.extend(
            os.path.join(root, filename)
            for filename in filenames
            if filename.lower().endswith((".gml", ".xml", ".gml.gz", ".xml.gz"))
        )
    if verbose:
        print(f"Found CityGML files: {len(file_paths)}")

    for fpath in file_paths:
        if verbose:
            print(f"\nFile: {os.path.relpath(fpath, gml_src)}")
        src_crs = detect_file_crs(fpath, verbose=verbose)
        use_envelope_prefilter = True
        if target_crs and (src_crs is None or src_crs != target_crs):
            use_envelope_prefilter = False
            if verbose:
                print(f"  Envelope pre-filter off (CRS mismatch: {src_crs} vs {target_crs})")
        if use_envelope_prefilter:
            ex_minx, ex_miny, ex_maxx, ex_maxy = read_envelope_xy(fpath)
            if (ex_maxx < minx) or (ex_minx > maxx) or (ex_maxy < miny) or (ex_miny > maxy):
                if verbose:
                    print("  -> Skipped (Envelope outside BBOX)")
                continue

        found_file = 0
        for part in iter_building_parts_in_file(
            fpath, target_crs=target_crs, verbose=False, bbox_xy_for_prefilter=bbox_xy
        ):
            if part.footprint.is_empty:
                continue
            bx0, by0, bx1, by1 = part.footprint.bounds
            if bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy:
                continue
            found_file += 1
            yield part
        if verbose:
            print(f"  -> Hits in file: {found_file}")


# -----------------------------------------------------------------------------
# BuildingDatabase Class
# -----------------------------------------------------------------------------


class BuildingDatabase:
    """High-performance spatial database for CityGML buildings.

    Features:
    - One-time loading of complete merge file
    - STRtree spatial index for O(log n) BBOX queries
    - Pre-computed geometry properties
    - Optional disk caching for persistence
    """

    def __init__(
        self,
        gml_path: str,
        target_crs: str | None = None,
        cache_file: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize and load a spatial building database.

        Args:
            gml_path: Path to the merged CityGML file.
            target_crs: Target CRS for coordinate transformation.
            cache_file: Optional JSON or JSON.GZ cache file.
            verbose: Print progress details.

        """
        self.gml_path = gml_path
        self.target_crs = target_crs
        self.cache_file = cache_file
        self.verbose = verbose

        # Core data structures
        self._buildings: list[CachedBuildingPart] = []
        self._spatial_index: STRtree | None = None
        self._loaded = False

        # Metadata
        self._total_count = 0
        self._load_time = 0.0
        self._index_time = 0.0
        self._bbox_global = (0, 0, 0, 0)

        # Auto-load
        self._load_or_cache()

    def _load_or_cache(self) -> None:
        """Smart loading: cache file if available, else fresh load."""
        if self.cache_file and os.path.exists(self.cache_file):
            if self.verbose:
                print(f"Loading building database from cache: {self.cache_file}")
            self._load_from_cache()
        else:
            if self.verbose:
                print(f"Creating building database from: {os.path.basename(self.gml_path)}")
            self._load_from_gml()
            if self.cache_file:
                self._save_to_cache()

    def _load_from_gml(self) -> None:
        """One-time loading of complete GML file."""
        if self.verbose:
            print("   Starting full parsing of merge file...")

        t0 = time.time()
        buildings_raw = [
            building
            for building in iter_building_parts_in_file(
                self.gml_path, target_crs=self.target_crs, verbose=False
            )
            if not building.footprint.is_empty and building.footprint.area > 0
        ]

        t1 = time.time()
        self._load_time = t1 - t0

        if self.verbose:
            print(f"   Loaded {len(buildings_raw)} buildings in {self._load_time:.1f}s")
            print("   Converting to CachedBuildingParts...")

        # Convert to optimized format
        t1 = time.time()
        self._buildings = []
        bounds_all = []

        for bp in buildings_raw:
            cached = CachedBuildingPart.from_building_part(bp)
            if cached.area > 0:
                self._buildings.append(cached)
                bounds_all.append(cached.bounds)

        t2 = time.time()

        # Global bounding box
        if bounds_all:
            minx = min(b[0] for b in bounds_all)
            miny = min(b[1] for b in bounds_all)
            maxx = max(b[2] for b in bounds_all)
            maxy = max(b[3] for b in bounds_all)
            self._bbox_global = (minx, miny, maxx, maxy)

        if self.verbose:
            print(f"   Converted {len(self._buildings)} valid buildings in {t2 - t1:.1f}s")
            print(f"   Global BBOX: {self._bbox_global}")

        # Build spatial index
        self._build_spatial_index()

        self._total_count = len(self._buildings)
        self._loaded = True

    def _build_spatial_index(self) -> None:
        """Create STRtree spatial index for O(log n) queries."""
        if not self._buildings:
            self._spatial_index = STRtree([])
            return

        if self.verbose:
            print(f"   Creating STRtree index for {len(self._buildings)} buildings...")

        t0 = time.time()

        geometries = [bp.footprint for bp in self._buildings]
        self._spatial_index = STRtree(geometries)

        t1 = time.time()
        self._index_time = t1 - t0

        if self.verbose:
            print(f"   Spatial index created in {self._index_time:.1f}s")

    def _save_to_cache(self) -> None:
        """Save database to cache file for future use."""
        if not self.cache_file or not self._loaded:
            return

        if self.verbose:
            print("Saving building database to cache...")

        cache_data = {
            "format_version": 1,
            "buildings": [
                {
                    "footprint_wkb_hex": wkb.dumps(building.footprint, hex=True),
                    "z_min": building.z_min,
                    "z_max": building.z_max,
                }
                for building in self._buildings
            ],
            "total_count": self._total_count,
            "bbox_global": self._bbox_global,
            "gml_path": self.gml_path,
            "target_crs": self.target_crs,
            "load_time": self._load_time,
            "index_time": self._index_time,
            "created_at": time.time(),
        }

        cache_path = Path(self.cache_file)
        temporary_path = cache_path.with_name(f"{cache_path.name}.part")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.suffix == ".gz":
                with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
                    json.dump(cache_data, handle, separators=(",", ":"))
            else:
                with temporary_path.open("w", encoding="utf-8") as handle:
                    json.dump(cache_data, handle, separators=(",", ":"))
            temporary_path.replace(cache_path)
            if self.verbose:
                cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"   Cache saved: {cache_size_mb:.1f} MB")
        except (OSError, TypeError, ValueError) as error:
            temporary_path.unlink(missing_ok=True)
            if self.verbose:
                print(f"   Cache save failed: {error}")

    def _load_from_cache(self) -> None:
        """Load database from cache file."""
        try:
            cache_path = Path(self.cache_file)
            if cache_path.suffix == ".gz":
                with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                    cache_data = json.load(handle)
            else:
                with cache_path.open(encoding="utf-8") as handle:
                    cache_data = json.load(handle)
            if cache_data.get("format_version") != 1:
                raise ValueError("Unsupported building cache format")

            # Restore data
            self._buildings = []
            for item in cache_data["buildings"]:
                building = BuildingPart(
                    footprint=wkb.loads(item["footprint_wkb_hex"], hex=True),
                    z_min=float(item["z_min"]),
                    z_max=float(item["z_max"]),
                )
                self._buildings.append(CachedBuildingPart.from_building_part(building))
            self._total_count = cache_data["total_count"]
            self._bbox_global = cache_data["bbox_global"]
            self._load_time = cache_data.get("load_time", 0.0)
            self._index_time = cache_data.get("index_time", 0.0)

            # Rebuild spatial index (not cacheable)
            self._build_spatial_index()
            self._loaded = True

            if self.verbose:
                age_hours = (time.time() - cache_data.get("created_at", 0)) / 3600
                print(f"   Loaded {self._total_count} buildings from cache")
                print(f"   Cache age: {age_hours:.1f} hours")

        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            GEOSException,
        ) as error:
            if self.verbose:
                print(f"   Cache load failed: {error}")
                print("   Fallback to GML parsing...")
            self._load_from_gml()

    def query_bbox(
        self, bbox_xy: tuple[float, float, float, float], buffer_m: float = 0.0
    ) -> list[CachedBuildingPart]:
        """Fast BBOX query with STRtree.

        Args:
            bbox_xy: (minx, miny, maxx, maxy)
            buffer_m: Additional buffer around BBOX

        Returns:
            List of buildings in the BBOX

        Performance: O(log n) instead of O(n) - up to 100x faster!

        """
        if not self._loaded or not self._spatial_index:
            return []

        minx, miny, maxx, maxy = bbox_xy
        if buffer_m > 0:
            minx -= buffer_m
            miny -= buffer_m
            maxx += buffer_m
            maxy += buffer_m

        # Fast spatial query
        query_box = box(minx, miny, maxx, maxy)

        try:
            # STRtree returns indices
            candidate_indices = self._spatial_index.query(query_box)

            # Convert indices to buildings
            result = []
            for idx in candidate_indices:
                if 0 <= idx < len(self._buildings):
                    bp = self._buildings[idx]
                    # Double-check bounds (STRtree uses envelopes)
                    bx0, by0, bx1, by1 = bp.bounds
                    if not (bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy):
                        result.append(bp)

            return result

        except Exception as e:
            # Fallback to linear search
            if self.verbose:
                print(f"STRtree query failed, fallback: {e}")
            return self._query_bbox_linear(bbox_xy, buffer_m)

    def _query_bbox_linear(
        self, bbox_xy: tuple[float, float, float, float], buffer_m: float = 0.0
    ) -> list[CachedBuildingPart]:
        """Fallback: linear search."""
        minx, miny, maxx, maxy = bbox_xy
        if buffer_m > 0:
            minx -= buffer_m
            miny -= buffer_m
            maxx += buffer_m
            maxy += buffer_m

        result = []
        for bp in self._buildings:
            bx0, by0, bx1, by1 = bp.bounds
            if not (bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy):
                result.append(bp)
        return result

    def get_stats(self) -> dict:
        """Database statistics."""
        return {
            "total_buildings": self._total_count,
            "loaded": self._loaded,
            "global_bbox": self._bbox_global,
            "load_time_s": self._load_time,
            "index_time_s": self._index_time,
            "gml_source": os.path.basename(self.gml_path),
            "target_crs": self.target_crs,
            "cache_file": self.cache_file,
        }

    def clear_cache(self):
        """Delete cache file."""
        if self.cache_file and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            if self.verbose:
                print(f"Cache deleted: {self.cache_file}")


def create_building_database(
    gml_path: str,
    target_crs: str = "EPSG:25833",
    cache_file: str | None = None,
    cache_dir: str | None = None,
    verbose: bool = True,
) -> BuildingDatabase:
    """Create and load a ``BuildingDatabase``.

    Args:
        gml_path: Path to CityGML merge file
        target_crs: Target CRS for coordinates
        cache_file: Direct path to cache file (optional)
        cache_dir: Directory for cache files (optional, only if cache_file not set)
        verbose: Progress output

    Returns:
        Ready-to-use BuildingDatabase

    """
    final_cache_file = cache_file
    if not final_cache_file and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(gml_path))[0]
        crs_suffix = target_crs.replace(":", "_").replace("/", "_")
        final_cache_file = os.path.join(cache_dir, f"{basename}_{crs_suffix}.json.gz")

    return BuildingDatabase(
        gml_path=gml_path, target_crs=target_crs, cache_file=final_cache_file, verbose=verbose
    )


# -----------------------------------------------------------------------------
# Parcel IO
# -----------------------------------------------------------------------------


def load_parcels(geojson_path: str) -> list[dict[str, Any]]:
    """Load parcels from GeoJSON and extract valid geometries."""
    with open(geojson_path, encoding="utf-8") as handle:
        geojson = json.load(handle)
    if not isinstance(geojson, dict) or not isinstance(geojson.get("features"), list):
        raise TypeError(f"Expected a GeoJSON FeatureCollection: {geojson_path}")
    parcels = []
    for feature in geojson["features"]:
        try:
            geom = _clean(shape(feature["geometry"]))
            if (
                isinstance(geom, (Polygon, MultiPolygon, GeometryCollection))
                and not geom.is_empty
                and geom.area > 0
            ):
                parcels.append({"geom": geom, "props": feature.get("properties", {})})
        except (KeyError, TypeError, ValueError, GEOSException):
            continue
    return parcels
