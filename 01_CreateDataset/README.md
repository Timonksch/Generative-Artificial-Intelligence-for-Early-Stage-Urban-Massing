# Dataset creation

This component converts official Berlin LoD1 buildings, ALKIS parcels, and the
Detailnetz street layer into parcel-centered voxel samples. All commands below
are run from the repository root.

## Central command

Use one entry point for the complete workflow:

```bash
python 01_CreateDataset/dataset_cli.py --help
```

```text
download  Download official input data
merge     Merge extracted CityGML tiles
create    Create voxel samples interactively or from JSON
analyze   Analyze a dataset and create deterministic splits
split     Run the guided analysis/split dialog
validate  Validate NPZ arrays, metadata, and a split manifest
metrics   Recompute voxel-derived GRZ/GFZ/BGF/height metadata
inspect   Summarize one sample and optionally export images
status    Show whether required local inputs are present
```

Every subcommand preserves the corresponding numbered script's options. Use
`COMMAND --help` for its complete interface.

## Installation

The dataset component does not require the training stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dataset.txt
```

Python 3.11 is the reference version. The merger needs additional temporary
disk space for extraction and its final atomically written output, but its XML
processing is streaming and does not retain all CityGML objects in memory.

## Fast path: source data to smoke dataset

Preview the official downloads without network access:

```bash
python 01_CreateDataset/dataset_cli.py download \
  --datasets lod1 parcels \
  --prepare-lod1 \
  --dry-run
```

Download, safely extract, and merge the two required local inputs:

```bash
python 01_CreateDataset/dataset_cli.py download \
  --datasets lod1 parcels \
  --prepare-lod1
```

Then run the ten-sample smoke configuration:

```bash
python 01_CreateDataset/dataset_cli.py create \
  --config 01_CreateDataset/configs/example_smoke.json
```

The street channel is queried from the official Berlin WFS in bounded spatial
tiles. Its persistent cache prevents repeated network calls. A full local
street snapshot is optional and is not consumed by the current mask builder.

## Guided and reproducible creation

For a first manual run:

```bash
python 01_CreateDataset/dataset_cli.py create \
  --interactive \
  --save-config 01_CreateDataset/configs/my_dataset.json
```

Repeat the exact configuration later with:

```bash
python 01_CreateDataset/dataset_cli.py create \
  --config 01_CreateDataset/configs/my_dataset.json
```

Configuration paths are interpreted relative to the repository root when the
central CLI is used. The smoke example and configuration provenance rules are
documented in [`configs/README.md`](configs/README.md).

## Processing stages

1. `00_download_input_data.py` downloads sources with timeouts, bounded retries,
   paging, safe ZIP extraction, SHA-256 checksums, and a provenance manifest.
2. `01_merge_citygml.py` makes IDs tile-unique, unions source envelopes, and
   writes one CityGML 1.0/2.0 document as a constant-memory XML stream.
3. `02_create_dataset.py` validates interactive/JSON input and starts the
   scientific pipeline.
4. `03_analyze_dataset.py` computes descriptive/quality statistics and can
   create a seeded 70/15/15 split. Selection is deterministic random splitting;
   the bucket table reports its result but does not claim stratified sampling.
5. `04_validate_dataset.py` checks array shapes/dtypes, JSON consistency,
   finite values on request, duplicate IDs, and manifest membership.
6. `05_recompute_voxel_metrics.py` updates native/coarse metrics and atomically
   replaces the single embedded `meta.npy` entry without archive duplication.

Internal responsibilities and the exact NPZ/JSON schema are documented in
[`pipeline/README.md`](pipeline/README.md).

## Input contract

- CityGML: one `.gml`, `.xml`, or `.gml.gz` CityModel in EPSG:25833.
- Parcels: a GeoJSON FeatureCollection containing positive-area polygonal
  geometries in EPSG:25833.
- Streets: WFS FeatureCollection from `detailnetz:c_strassenabschnitte`, queried
  automatically with a finite timeout and bounded fallbacks.

Official URLs, licensing, verified feature counts, output paths, and storage
notes are maintained in
[`00_Data/01_InputData/README.md`](../00_Data/01_InputData/README.md).

## Analyze, validate, and inspect

Create one report, plots, and a fixed split manifest payload:

```bash
python 01_CreateDataset/dataset_cli.py analyze \
  --dataset-dir 00_Data/02_GeneratedDatasets/my_dataset \
  --mode split_analyze \
  --seed 42 \
  --plot \
  --report-dir 00_Data/02_GeneratedDatasets/my_dataset_report
```

Validate the generated samples and save a machine-readable report:

```bash
python 01_CreateDataset/dataset_cli.py validate \
  --root 00_Data/02_GeneratedDatasets/my_dataset \
  --check-nan \
  --report 00_Data/02_GeneratedDatasets/my_dataset_report/validation.json
```

Inspect one sample without initializing plotting libraries:

```bash
python 01_CreateDataset/dataset_cli.py inspect \
  00_Data/02_GeneratedDatasets/my_dataset/PARCEL_ID/PARCEL_ID.npz
```

Add `--overview overview.png` and/or `--save-3d voxels.png` only when images are
needed.

## Exit behavior and safe reruns

- Help, status, successful commands, and valid datasets return exit code 0.
- Input/configuration failures return 1 where a script exposes an integer code.
- The validator returns 2 when validation completes but finds sample errors.
- Downloads and reports use temporary siblings and replace final files only
  after successful writes.
- `skip_existing` supports resuming long creation runs. It does not verify an
  existing sample; run `validate` after interrupted or migrated runs.
- Keep the source `download_manifest.json`, creation config, analysis/split
  report, and validation report with every archived dataset release.

## Development checks

Install `requirements-dev.txt`, then run:

```bash
ruff check 01_CreateDataset tests
pytest tests/test_create_dataset_cli.py tests/test_create_dataset_core.py
```

The repository CI installs only `requirements-dataset.txt` plus development
tools for these checks, avoiding the unrelated GPU training stack.
