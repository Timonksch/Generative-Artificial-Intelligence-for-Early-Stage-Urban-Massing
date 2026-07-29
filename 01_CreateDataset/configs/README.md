# Dataset configurations

`example_smoke.json` is a small end-to-end configuration for checking a local
installation. Run it from the repository root after preparing the source data:

```bash
python 01_CreateDataset/dataset_cli.py create \
  --config 01_CreateDataset/configs/example_smoke.json
```

It requests ten successful Bucket-A-sized samples, disables expensive
per-sample images, enables deterministic selection, and reuses both building
and street caches. It is an operational example, not the final thesis dataset
provenance record.

Every released dataset must keep its exact creation JSON, source download
manifest, analysis report, split manifest, and checksums together. Do not infer
the final thesis settings from defaults in source code.
