# Data

This directory is the local data workspace for dataset construction and model
training. Large inputs, generated voxel samples, caches, and checkpoints are
excluded from Git.

- [`01_InputData/`](01_InputData/README.md) documents and downloads the official
  Berlin source data.
- [`02_GeneratedDatasets/`](02_GeneratedDatasets/README.md) documents the sample
  schema location and required release artifacts.

External dataset artifacts, restore locations, file counts, and public R2
records are specified in [`ARTIFACTS.md`](../ARTIFACTS.md). Large dataset
directories such as `00_Data/02_GeneratedDatasets/dataset_thesis/` exist only
after the external artifacts have been restored.

`build_public_file_indexes.py` is a maintainer utility for regenerating the
public Cloudflare R2 file indexes referenced by `artifacts_manifest.json`. It
requires R2 S3 credentials and is not needed for normal artifact downloads.

The complete creation, analysis, validation, and inspection interface is
documented in [`01_CreateDataset/README.md`](../01_CreateDataset/README.md).

The repository's MIT License applies to source code only. See
[`DATA_LICENSE.md`](../DATA_LICENSE.md) for the separation between code and data
licensing.
