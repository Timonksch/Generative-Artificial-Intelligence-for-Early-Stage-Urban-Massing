# Experiment configurations

JSON files are the reproducible input to `train_cli.py experiment`. A config
contains a human-readable `name` and `description`, shared `global` arguments,
and one or more `overrides`. Every override must specify `model` (`unet`,
`cond_unet`, or `ldm`) and a unique `run_name`.

Paths are repository-relative. Parameters are translated directly to the
corresponding training command, so keys must match that command's argument
names. Boolean `true` values become flags; false and null values are omitted.

## Collection

- `thesis_runs/phase1/`: eight unconditional U-Net experiments.
- `thesis_runs/phase2_cond/`: seven conditioned U-Net experiments.
- `thesis_runs/phase3_ldm/`: eight VAE/diffusion experiments.
- `example_smoke.json`: minimal dry-run and installation check.

Phase III is ordered: run `phase3_final_vae.json` first, then ensure the
`vae_checkpoint` in diffusion configs points to its archived `best.pt`.

Validate command expansion without allocating a model:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/example_smoke.json \
  --out-parent 02_TrainModels/outputs/smoke \
  --dry-run
```

Do not edit a config after producing reported results. Add a new file with a
new name so that command provenance remains unambiguous.
