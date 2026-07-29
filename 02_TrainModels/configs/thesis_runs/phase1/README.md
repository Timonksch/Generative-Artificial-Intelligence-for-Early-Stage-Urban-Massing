# Phase I configurations: unconditional 3D U-Net

This directory contains the ordered Phase-I architecture and training
experiments. The sequence starts with a reference U-Net and varies capacity,
coordinate channels, surface weighting, combined capacity/coordinates, loss
balance, and threshold handling before the selected final rerun.

All files use the same dataset and split manifest. Each JSON contains shared
`global` parameters plus one or more named `overrides`. The `model` value must
be `unet`.

Run one configuration from the repository root:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/thesis_runs/phase1/phase1_01_baseline.json \
  --out-parent 02_TrainModels/outputs/thesis_runs/architecture_phase1
```

Use `phase1_final.json` for the selected long rerun. Test results must not be
used to choose among the preceding variants.
