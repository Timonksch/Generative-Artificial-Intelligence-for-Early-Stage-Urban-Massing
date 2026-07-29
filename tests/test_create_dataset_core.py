"""Test CityGML parsing, spatial caching, and core raster operations."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from pipeline.data_io import BuildingDatabase, iter_building_parts_in_file, read_envelope_xy
from pipeline.geo_utils import boundary_voxels, extrude_mask, polygon_to_mask
from shapely.geometry import box

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "citygml_parser" / "building.gml"
EXPECTED_BUILDING_AREA_M2 = 16.0
EXPECTED_BUILDING_HEIGHT_M = 10.0


def test_citygml_parser_extracts_footprint_height_and_envelope() -> None:
    """Extract the projected bounds and one valid building part."""
    parts = list(iter_building_parts_in_file(str(FIXTURE_PATH), target_crs="EPSG:25833"))

    assert read_envelope_xy(str(FIXTURE_PATH)) == (0.0, 0.0, 4.0, 4.0)
    assert len(parts) == 1
    assert parts[0].footprint.area == EXPECTED_BUILDING_AREA_M2
    assert parts[0].z_min == 0.0
    assert parts[0].z_max == EXPECTED_BUILDING_HEIGHT_M


def test_building_database_uses_safe_json_cache(tmp_path: Path) -> None:
    """Persist and reload the building index without pickle deserialization."""
    cache_path = tmp_path / "buildings.json.gz"
    first = BuildingDatabase(
        str(FIXTURE_PATH), target_crs="EPSG:25833", cache_file=str(cache_path), verbose=False
    )
    second = BuildingDatabase(
        str(FIXTURE_PATH), target_crs="EPSG:25833", cache_file=str(cache_path), verbose=False
    )

    assert len(first.query_bbox((-1.0, -1.0, 5.0, 5.0))) == 1
    assert len(second.query_bbox((-1.0, -1.0, 5.0, 5.0))) == 1
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["format_version"] == 1
    assert len(payload["buildings"]) == 1


def test_rasterization_extrusion_and_boundary_shell() -> None:
    """Rasterize one parcel, extrude it, and extract a non-empty shell."""
    mask = polygon_to_mask(box(0, 0, 4, 4), 4, 4, (0.0, 0.0), 1.0, ss=1)
    volume = extrude_mask(mask, 0, 4, 4)
    shell = boundary_voxels(volume)

    assert mask.shape == (4, 4)
    assert np.count_nonzero(mask) > 0
    assert volume.shape == (4, 4, 4)
    assert 0 < np.count_nonzero(shell) <= np.count_nonzero(volume)
