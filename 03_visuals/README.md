# Visualizations

`03_visuals` provides one command-line interface for reproducible figures from
generated datasets and model outputs. All maintained commands use the same
color palette, typography, output conventions, and input validation.

Run commands from the repository root. Generated files are written below
`03_visuals/outputs/` by default and are intentionally ignored by Git.

For the thesis layout, all generated figures were post-processed in Figma.
The SVG exports were used as the primary editable source whenever possible;
PNG exports serve mainly as quick previews and raster fallbacks.

## Quick start

Show all commands:

```bash
python 03_visuals/visuals_cli.py --help
```

Generate bounded dataset figures using the included example dataset:

```bash
python 03_visuals/visuals_cli.py dataset \
  --dataset 00_Data/02_GeneratedDatasets/example_smoke \
  --output 03_visuals/outputs/smoke/dataset \
  --limit 10
```

## Commands

### Dataset figures

The command reads the JSON metadata stored next to every NPZ sample. It does
not require a separately generated analysis report.

```bash
python 03_visuals/visuals_cli.py dataset \
  --dataset 00_Data/02_GeneratedDatasets/dataset_thesis \
  --output 03_visuals/outputs/thesis/dataset
```

Outputs include regulatory and parcel/neighbor distributions, parcel-area
buckets, spatial sample distribution, smoothed 3D density surfaces, an optional
split profile, and a machine-readable summary.

The density-surface figure uses `03_visuals/bezirksgrenzen.geojson` as the
Berlin outline when the file is present. Pass `--districts path/to/file.geojson`
to use another boundary file.

For a bounded check, add `--limit 30`.

### Dataset sample assets

```bash
python 03_visuals/visuals_cli.py samples \
  --dataset 00_Data/02_GeneratedDatasets/example_smoke \
  --sample-id DEBE00YY11Y001PB \
  --output 03_visuals/outputs/samples
```

The command loads one sample at a time and exports channel, target, and
neighbor projections. Selecting samples explicitly avoids accidental
full-dataset memory use.

### Experiment figures

The experiment command recursively discovers `metrics.csv` files. It therefore
supports a single run, a smoke-test directory, or a complete architecture
phase without hard-coded experiment names.

Restore the released model outputs described in
[`ARTIFACTS.md`](../ARTIFACTS.md), or provide another compatible local run
directory, before using this command.

```bash
python 03_visuals/visuals_cli.py experiments \
  --runs-root 02_TrainModels/outputs/thesis_runs/architecture_phase1 \
  --output 03_visuals/outputs/thesis/phase1
```

It generates:

- comparable training and validation curves;
- a ranking based on the best available validation metric;
- a qualitative panel from existing inference images;
- regulatory target/prediction comparisons when `reg_metrics.json` exists;
- a JSON inventory of all discovered runs.

Use the same command for `architecture_phase2` and
`architecture_phase3_ldm`. Add `--limit-runs N` for a bounded check.

### Prediction in LOD1 context

Arguments after `--` are forwarded to the specialized renderer:

```bash
python 03_visuals/visuals_cli.py render-prediction -- \
  --sample-id SAMPLE_ID \
  --data_root 00_Data/02_GeneratedDatasets/dataset_thesis \
  --prediction_path path/to/sample_probability.npy \
  --citygml 00_Data/01_InputData/input/berlin_lod1_merged.gml.gz \
  --output_dir 03_visuals/outputs/lod1 \
  --angle_step 60
```

Display its complete parameter reference with:

```bash
python 03_visuals/visuals_cli.py render-prediction -- --help
```

### Multi-building scene

The scene renderer builds samples for selected parcels, runs model inference,
and combines the predictions with the LOD1 surroundings. Because this command
can be expensive, paths and coordinates must be supplied explicitly.

The first run may need to load the complete parcel file and CityGML building
database. Keep the generated building database cache for subsequent runs and
use a small `--max_buildings` value while validating a configuration.

```bash
python 03_visuals/visuals_cli.py render-scene -- --help
```

### Animation

```bash
python 03_visuals/visuals_cli.py animate \
  --input 03_visuals/outputs/lod1/renders/SAMPLE_ID \
  --output 03_visuals/outputs/lod1/SAMPLE_ID.gif \
  --fps 12
```

The command accepts GIF output only and refuses to load more than 1,000 frames
unless `--maximum-frames` is changed explicitly.

## Package structure

```text
03_visuals/
├── visuals_cli.py                 central public entry point
├── visuals/
│   ├── animation.py              bounded GIF creation
│   ├── dataset.py                dataset figures
│   ├── experiments.py            run comparisons and result figures
│   ├── paths.py                  repository paths and validation
│   ├── records.py                typed dataset/run readers
│   ├── samples.py                NPZ sample projections
│   └── style.py                  shared thesis style
├── visualize_pred_with_lod1.py   specialized LOD1 renderer
└── custom_scene_multigen.py      specialized multi-building renderer
```

The two specialized renderers remain separate because they combine model
inference, geospatial processing, and PyVista scene construction. Their public
entry points are nevertheless exposed through `visuals_cli.py`.

## Dependencies

Install the repository requirements before rendering:

```bash
python -m pip install -r requirements.txt
```

Static dataset and experiment figures require Matplotlib, NumPy, and Pillow.
LOD1 and scene rendering additionally require the geospatial, PyTorch, and
PyVista dependencies listed in the repository requirements.

## Verification

```bash
ruff check 03_visuals
pytest tests/test_visuals_cli.py
```

Every public command returns a non-zero exit status when an input is missing or
invalid. This makes the CLI suitable for automated smoke tests and CI jobs.
