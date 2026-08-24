# Contributing

This repository is a master's-thesis submission archive, not an open
contribution project. The code, configuration files, fixed data splits,
artifact manifest, and published R2 artifacts are intended to preserve one
reported research state.

## Contribution Status

External pull requests, feature requests, dataset changes, and model updates
are not expected for this archived thesis version. If this repository is reused
for later research, create a fork or a clearly named post-submission branch
rather than changing the submitted state.

## Maintainer-Only Changes

Changes after submission should be limited to maintenance tasks that preserve
reproducibility:

- fixing broken documentation links without changing scientific claims;
- rotating leaked credentials or removing accidental local files;
- updating public download indexes only when the R2 object set intentionally
  changes; and
- adding errata that explicitly state what changed and why.

Do not rewrite the fixed train/validation/test split, reported experiment
configs, model-selection records, generated dataset metadata, or R2 artifact
manifest without recording a new release state.

Do not broaden licensing terms without an explicit new release note. The
submitted state uses separate terms for source code, written thesis material,
figures, datasets, model weights, and external artifacts; maintainers should
preserve those boundaries as documented in `DATA_LICENSE.md`.

## Artifacts And Secrets

Generated datasets, model checkpoints, caches, and full training outputs remain
outside Git. Public downloads are described in `ARTIFACTS.md` and restored with:

```bash
python download_artifacts.py --public-download
```

Never commit Cloudflare, R2, AWS, API, SSH, or other credentials. If a token is
exposed, revoke it immediately and rebuild only the minimal access required for
maintenance.

## Validation

If a maintainer does change the archive, run the repository checks from the
repository root before committing:

```bash
pytest
ruff format --check 01_CreateDataset 02_TrainModels 03_visuals tests download_artifacts.py 00_Data/build_public_file_indexes.py
ruff check 01_CreateDataset 02_TrainModels 03_visuals tests download_artifacts.py 00_Data/build_public_file_indexes.py
ruff check . --preview --select PLR0917
mypy tests
```

Training-dependent smoke tests require `requirements-training.txt`.
