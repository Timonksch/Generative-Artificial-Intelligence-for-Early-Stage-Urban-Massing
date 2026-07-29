# Official Berlin input data

The voxel-dataset pipeline uses three official datasets published by the State
of Berlin. All inputs are processed in **EPSG:25833**.

## One-command setup

Run commands from the repository root. To download the two required local
inputs, extract the LoD1 archive, and create the merged CityGML file:

```bash
python 01_CreateDataset/dataset_cli.py download \
  --datasets lod1 parcels \
  --prepare-lod1
```

This produces:

```text
00_Data/01_InputData/
├── download_manifest.json
└── input/
    ├── LoD1.zip
    ├── lod1_tiles/
    ├── berlin_lod1_merged.gml.gz
    └── flurstuecke.geojson
```

The manifest records source URLs, retrieval time, file sizes, feature counts,
and SHA-256 checksums. Existing files are preserved unless `--force` is passed.

Preview the complete operation without downloading data:

```bash
python 01_CreateDataset/dataset_cli.py download \
  --datasets lod1 parcels \
  --prepare-lod1 \
  --dry-run
```

## 1. Berlin LoD1 building model

- **Dataset:** 3D building models, Level of Detail 1
- **Provider:** Senate Department for Urban Development, Building and Housing
- **Format:** zipped CityGML/XML
- **CRS:** EPSG:25833
- **Official ATOM feed:** <https://gdi.berlin.de/data/a_lod1/atom/0.atom>
- **Archive:** <https://gdi.berlin.de/data/a_lod1/atom/LoD1.zip>
- **Archive size verified 2026-06-21:** 297,502,934 bytes
- **Archive content verified 2026-06-21:** 1,006 XML tiles

The downloader can safely extract the official archive and invoke the existing
`01_merge_citygml.py` workflow. The merged output is the file consumed by
`02_create_dataset.py`.

## 2. ALKIS Berlin parcels

- **Dataset:** ALKIS Berlin parcels
- **Provider:** Senate Department for Urban Development, Building and Housing
- **Format used here:** GeoJSON from WFS 2.0
- **CRS:** EPSG:25833
- **Berlin Open Data record:**
  <https://daten.berlin.de/datensaetze/alkis-berlin-flurstucke-wfs-1bc014d7>
- **WFS endpoint:**
  <https://gdi.berlin.de/services/wfs/alkis_flurstuecke>
- **Feature type:** `alkis_flurstuecke:flurstuecke`
- **Feature count verified 2026-06-21:** 403,524

The downloader requests this layer in bounded pages and combines the pages into
`input/flurstuecke.geojson`. This avoids one fragile request containing the
entire Berlin parcel dataset.

## 3. Berlin detailed street network

- **Dataset:** Detailnetz Berlin
- **Provider named by the service:** Senate Department for Urban Development,
  Building and Housing Berlin
- **Format used here:** GeoJSON from WFS 2.0
- **CRS:** EPSG:25833
- **WFS endpoint:** <https://gdi.berlin.de/services/wfs/detailnetz>
- **Feature type:** `detailnetz:c_strassenabschnitte`
- **Feature count verified 2026-06-21:** 43,508

The existing dataset pipeline requests street segments for small spatial tiles
and stores them in its local cache. Therefore no full street download is needed
for normal dataset construction. A complete reproducibility snapshot remains
available when required:

```bash
python 01_CreateDataset/dataset_cli.py download --datasets streets
```

This optional command creates `input/strassenabschnitte.geojson`; the current
pipeline continues to use the online WFS and its tile cache.

## Source license

The service metadata for all three sources specifies **Data licence Germany -
Zero - Version 2.0**:

<https://www.govdata.de/dl-de/zero-2-0>

The source services are updated by their providers. Re-running the downloader
at a later date may therefore produce a different checksum, feature count, or
dataset revision. Keep `download_manifest.json` with every reproducibility
archive.

## Continue with dataset generation

After downloading and preparing the inputs:

```bash
python 01_CreateDataset/dataset_cli.py create --interactive
```

Use these paths when prompted:

```text
CityGML: 00_Data/01_InputData/input/berlin_lod1_merged.gml.gz
Parcels: 00_Data/01_InputData/input/flurstuecke.geojson
Cache:   00_Data/01_InputData/cache
Streets: enabled, with cache at 00_Data/01_InputData/cache/streets
```

## Storage and failure handling

- Downloads are first written to `.part` files and renamed only after success.
- Network calls use explicit timeouts and bounded retries.
- WFS pages are bounded to limit memory consumption.
- LoD1 extraction rejects path traversal, symbolic links, and unexpectedly large
  expanded archives.
- Merging all 1,006 LoD1 tiles is CPU-, memory-, and storage-intensive and can
  take substantially longer than the download itself.
- Use `--force` only when an existing local snapshot should be replaced.
