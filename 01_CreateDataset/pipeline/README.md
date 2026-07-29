# Dataset pipeline internals

This package contains the importable implementation behind the numbered
command-line scripts. Most users should use `../dataset_cli.py`; these modules
are documented here for maintenance and reproducibility review.

## Module responsibilities

- `data_io.py` streams CityGML, transforms coordinates, creates a spatial
  building index, writes a JSON/WKB cache, and loads parcel GeoJSON.
- `geo_utils.py` repairs geometries and converts polygons/lines into 2D or 3D
  masks.
- `streets_wfs.py` fetches bounded WFS tiles, validates GeoJSON, and writes
  deterministic SHA-256-addressed cache files.
- `pipeline.py` applies parcel filters, assembles channels and targets, computes
  metrics, and writes complete samples.
- `viz.py` creates the standard 3x3 overview and 3D boundary-voxel render.
- `data_inspector.py` validates and summarizes one generated NPZ sample.

## Sample contract

Each successful sample is stored in its own directory:

```text
<dataset>/<parcel_id>/
├── <parcel_id>.npz
├── <parcel_id>.json
├── overview.png       # unless no_viz=true
└── voxel_hires.png    # unless no_viz=true
```

The NPZ archive contains:

- `X`: float32 input tensor in `(C,D,H,W)` order;
- `Y`: uint8 target occupancy in `(D,H,W)` order;
- `Y_neigh`: uint8 neighboring-building occupancy in `(D,H,W)` order; and
- `meta`: the same metadata dictionary stored in the JSON sidecar.

Input channels are ordered as C0 parcel build mask, C1 normalized neighbor
height, C2 parcel/neighbor edges, and optional C3 street mask. The JSON sidecar
is the human- and tool-readable source of grid, coordinate, channel, and metric
metadata.

## Coordinate and cache contracts

- All published Berlin inputs and generated samples use EPSG:25833.
- The CityGML input is one merged `.gml`, `.xml`, or `.gml.gz` file.
- Building caches are gzip-compressed JSON with WKB-encoded footprints; they do
  not deserialize Python pickle objects.
- Street cache entries are validated GeoJSON FeatureCollections. Network
  attempts have finite timeouts and a bounded fallback list.
- Cache files are derived artifacts and may be deleted and rebuilt. Dataset
  samples and provenance manifests are not caches.

## Maintainer checks

Run from the repository root:

```bash
ruff check 01_CreateDataset tests
pytest tests/test_create_dataset_cli.py tests/test_create_dataset_core.py
```

The tests cover central command dispatch, downloader dry-runs, true streaming
CityGML merging, XML envelope/building parsing, safe cache reload, raster
operations, validator exit codes, and duplicate-free NPZ metadata updates.
