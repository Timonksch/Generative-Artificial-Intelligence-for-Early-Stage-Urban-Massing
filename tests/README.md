# Tests

The test suite is part of the reproducibility package. It verifies that the
repository metadata, public command-line interfaces, dataset schema, model
families, experiment configs, and visualization workflows remain usable after
the large external artifacts have been removed from Git.

## Test groups

- `test_repository_foundation.py` checks required files, portable config paths,
  source syntax, documentation coverage, and download dry runs.
- `test_create_dataset_cli.py` and `test_create_dataset_core.py` verify dataset
  commands, validation, metric updates, CityGML handling, and spatial queries.
- `test_train_models_cli.py` verifies the central training dispatcher and
  experiment expansion without starting long training jobs.
- `test_model_data_smoke.py` runs bounded forward and backward passes for all
  model families when PyTorch and the committed smoke dataset are available.
- `test_visuals_cli.py` verifies the consolidated visualization interface and
  representative figure generation.

Shared repository paths, subprocess execution, and the committed example
dataset contract are centralized in `conftest.py`. The example dataset is
expected to contain 30 paired NPZ/JSON samples and remains separate from the
large external thesis dataset.

## Running tests

Install the relevant runtime and development dependencies, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
pytest
```

Dataset-only development can use the smaller environment:

```bash
python -m pip install -r requirements-dataset.txt
python -m pip install -r requirements-dev.txt
pytest tests/test_create_dataset_cli.py tests/test_create_dataset_core.py
```

Tests use temporary directories for generated outputs. They do not download
source data or run full thesis training.
