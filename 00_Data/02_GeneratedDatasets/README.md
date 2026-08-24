# Generated datasets

Full generated datasets are intentionally excluded from Git. The repository
contains `example_smoke/` with 30 curated samples for validation and model
smoke tests. A restored or newly generated dataset should use this adjacent
structure:

The public artifact location, restore path, file count, and byte total are
recorded in [`ARTIFACTS.md`](../../ARTIFACTS.md) and
[`artifacts_manifest.json`](../../artifacts_manifest.json).

```text
00_Data/02_GeneratedDatasets/
├── <dataset>/
│   ├── <parcel_id>/
│   │   ├── <parcel_id>.npz
│   │   └── <parcel_id>.json
│   ├── report.json
│   ├── report_split.json
│   └── validation.json
└── <dataset>_report/       # optional plots and extended local analysis
    └── plots/
```

The NPZ/JSON sample schema is defined in
[`01_CreateDataset/pipeline/README.md`](../../01_CreateDataset/pipeline/README.md).

Before publishing a dataset, include these reproducibility artifacts in the
external archive:

- the exact creation configuration;
- the source `download_manifest.json` and its SHA-256 checksums;
- the analysis report containing fixed split IDs and seed;
- the validator report;
- a dataset-level checksum manifest; and
- the applicable source-data license notice.

Create reports with the central CLI documented in
[`01_CreateDataset/README.md`](../../01_CreateDataset/README.md). The final
large dataset is restored from the external artifact record and must not be
committed to Git.

Validate all committed smoke samples from the repository root with:

```bash
python 01_CreateDataset/dataset_cli.py validate \
  --root 00_Data/02_GeneratedDatasets/example_smoke \
  --check-nan
```

The three model families consume the same samples in
`tests/test_model_data_smoke.py` after deterministic downsampling to 32³.
