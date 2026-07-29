# Coding Standards

These standards keep the thesis code reviewable and reproducible.

## General

- Target Python 3.11.
- Prefer clear, explicit code over hidden state or notebook-only workflows.
- Keep paths repository-relative in committed configs and documentation.
- Do not commit generated datasets, model checkpoints, caches, local outputs,
  virtual environments, or private environment files.

## Python

- Format and lint with Ruff using the rules in `pyproject.toml`.
- Keep public command-line interfaces stable and route workflows through the
  central CLIs where available.
- Use typed helper functions for shared behavior, especially around file I/O,
  geospatial conversion, metrics, and model configuration.
- Avoid broad refactors when changing experiment code that is tied to reported
  thesis results.

## Reproducibility

- Store experiment inputs as versioned JSON configs.
- Preserve the fixed split manifest for reported comparisons.
- Record seeds, resolved configs, metrics, checkpoint-selection rules, restore
  paths, file counts, and byte totals with external releases.
- Use validation data for model selection and threshold calibration; reserve
  the test split for final reporting.

## Documentation

- Keep README examples runnable from the repository root.
- Document source-data requirements, licenses, and external artifact locations
  separately from source-code licensing.
- Add or update tests for changed behavior when the change affects a public
  CLI, data schema, model contract, or visualization output.
