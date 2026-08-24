![GitHub Header](.github/statics/GitHub%20Header.png)

# Generative Artificial Intelligence for Early-Stage Urban Massing

<p align="left">
  <a href="LICENSE"><img src="https://shieldcn.dev/badge/license-AGPLv3-16A34A.svg" alt="AGPL-3.0-only source code license"></a>
  <a href=".python-version"><img src="https://shieldcn.dev/badge/python-3.11-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="Generative%20Artificial%20Intelligence%20for%20Early-Stage%20Urban%20Massing_TimoNikisch_MasterThesis_Public.pdf"><img src="https://shieldcn.dev/badge/thesis-PDF-B91C1C.svg?logo=adobeacrobatreader&logoColor=white" alt="Public thesis PDF"></a>
</p>

Research code accompanying a master's thesis on context-aware, voxel-based
generation of parcel-scale urban massing in Berlin. The repository covers the
complete experimental workflow: dataset construction, three model phases,
evaluation, and thesis visualizations.

Submission date: September 1, 2026.

## Research scope

The project investigates three successive model families:

1. an unconditional 3D U-Net driven by local urban context;
2. a conditional 3D U-Net controlled by GRZ, GFZ, and building height; and
3. a conditional latent diffusion model operating on compressed voxel grids.

All phases use the same fixed train/validation/test split. Model selection is
based on validation metrics; the test split is reserved for final reporting.

## Repository layout

```text
00_Data/            Data locations, examples, reports, and download guidance
01_CreateDataset/   CityGML preprocessing and voxel-dataset construction
02_TrainModels/     Models, training, inference, evaluation, and experiment configs
03_visuals/         Dataset, experiment, and qualitative visualizations
tests/              Repository, CLI, dataset, model, and visualization tests
```

Generated datasets, model checkpoints, and full training outputs are not stored
in Git because of their size and third-party licensing requirements.

## Requirements

- Python 3.11 is the reference version.
- A CUDA-capable GPU is strongly recommended for training.
- Dataset creation and geospatial visualizations require native geospatial
  libraries supported by GeoPandas, PyProj, and Pyogrio.

Create an isolated environment and install the runtime dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and validation, also install:

```bash
python -m pip install -r requirements-dev.txt
```

The dependency files form explicit layers:

| File | Purpose |
|---|---|
| `requirements-dataset.txt` | Dataset creation, validation, and static figures |
| `requirements-training.txt` | Dataset dependencies plus PyTorch training |
| `requirements.txt` | Complete runtime, including advanced geospatial and 3D visualization |
| `requirements-dev.txt` | Tests, linting, type checking, and pre-commit hooks |

Install only one runtime layer. Add `requirements-dev.txt` when modifying or
validating the repository.

## Quick start

All commands below are intended to be run from the repository root.

Inspect the central dataset-creation interface:

```bash
python 01_CreateDataset/dataset_cli.py --help
```

Download and prepare the required official Berlin input data:

```bash
python 01_CreateDataset/dataset_cli.py download \
  --datasets lod1 parcels \
  --prepare-lod1
```

See [`00_Data/01_InputData/README.md`](00_Data/01_InputData/README.md) for
source records, licenses, storage requirements, and a dry-run command.

Inspect a thesis experiment without starting training:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/thesis_runs/phase1/phase1_01_baseline.json \
  --out-parent 02_TrainModels/outputs/thesis_runs/architecture_phase1 \
  --dry-run
```

All training, inference, and evaluation commands are listed with:

```bash
python 02_TrainModels/train_cli.py --help
```

List the available thesis-figure targets:

```bash
python 03_visuals/visuals_cli.py --help
```

The dataset-creation workflow and its stable central CLI are documented in
[`01_CreateDataset/README.md`](01_CreateDataset/README.md). Use paths relative
to the repository root.

## External research artifacts

The public thesis manuscript is included as
[`Generative Artificial Intelligence for Early-Stage Urban Massing_TimoNikisch_MasterThesis_Public.pdf`](Generative%20Artificial%20Intelligence%20for%20Early-Stage%20Urban%20Massing_TimoNikisch_MasterThesis_Public.pdf).

Large datasets and model weights are distributed separately from the source
repository. [ARTIFACTS.md](ARTIFACTS.md) defines the release contents, expected
local restore paths, public R2 manifest, and restore workflow.

Restore the published artifacts from the public R2 bucket with:

```bash
python download_artifacts.py --public-download
```

The release package consists of:

- dataset reports and the fixed split manifest;
- small example samples for tests and demonstrations;
- all final thesis experiment configurations;
- compact metric summaries for every reported experiment;
- public manifest records and restore locations for the complete dataset and
  final checkpoints.

Large artifacts are published outside Git through the R2 artifact manifest and
can be moved to a long-term research-data archive later without changing the
repository paths.

## Development standards

Code changes must follow [CODING_STANDARDS.md](CODING_STANDARDS.md). New and
modified code is expected to meet these standards before it is committed.

Run the current repository checks with:

```bash
pytest
ruff check 01_CreateDataset 02_TrainModels 03_visuals tests
mypy tests
```

## Citation

Software citation metadata is provided in [CITATION.cff](CITATION.cff). The
associated public thesis PDF is available in
[`Generative Artificial Intelligence for Early-Stage Urban Massing_TimoNikisch_MasterThesis_Public.pdf`](Generative%20Artificial%20Intelligence%20for%20Early-Stage%20Urban%20Massing_TimoNikisch_MasterThesis_Public.pdf).
The external artifact location used for this submission is documented in
[ARTIFACTS.md](ARTIFACTS.md).

## License

This repository uses separate terms for separate material types:

- project source code, command-line tools, tests, and experiment configuration
  files are licensed under the [GNU Affero General Public License v3.0 only](LICENSE);
- the thesis manuscript, written documentation, rendered figures, diagrams, and
  presentation-style material are released for academic reading and citation
  under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
  International unless a file states otherwise; and
- third-party geospatial inputs, generated datasets, trained model weights,
  basemap tiles, fonts, restored artifacts, and public R2 downloads are not
  covered by the source-code license.

See [DATA_LICENSE.md](DATA_LICENSE.md) for the authoritative licensing
boundaries, provenance requirements, and artifact-use restrictions for this
thesis submission.
