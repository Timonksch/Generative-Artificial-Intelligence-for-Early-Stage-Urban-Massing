# Model training and evaluation

This component contains the three model phases used in the thesis: an
unconditional 3D U-Net, a condition-controlled 3D U-Net, and a VAE plus latent
diffusion model. Run every command from the repository root through the central
interface.

## Central command

```bash
python 02_TrainModels/train_cli.py --help
```

```text
experiment       Run trainings declared in a JSON experiment config
train-unet       Train the unconditional 3D U-Net directly
train-cond-unet  Train the condition-controlled 3D U-Net directly
train-ldm        Train the VAE or latent diffusion stage directly
infer            Run inference and export metrics/predictions
evaluate         Evaluate voxel and regulatory target metrics
evaluate-vae     Evaluate VAE reconstruction quality
sample-cond      Render conditional U-Net control variants
sample-ldm       Sample variants from the best LDM experiment run
smoke-models     Mini-train all models and save diagnostic outputs
configs          List the versioned thesis experiment configs
status           Check the default dataset, configs, and PyTorch
```

Use `COMMAND --help` for all command-specific options. `scripts/` is an
internal implementation directory; the central interface establishes the
module path and working directory consistently.

## Installation

Python 3.11 is the reference version. Install only the dataset and training
stack with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

A CUDA GPU is strongly recommended. CPU and Apple MPS are supported by device
selection where all required PyTorch operations are available, but full 3D
training is substantially slower.

## Reproduce one experiment

Inspect the available configurations:

```bash
python 02_TrainModels/train_cli.py configs
```

Resolve commands and output paths without training:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/example_smoke.json \
  --out-parent 02_TrainModels/outputs/smoke \
  --dry-run
```

Run a thesis experiment by replacing the config path, for example:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/thesis_runs/phase1/phase1_01_baseline.json \
  --out-parent 02_TrainModels/outputs/thesis_runs/architecture_phase1
```

The runner expands `global` parameters and each item in `overrides`, writes an
`experiment_summary.json`, and optionally evaluates a successful run. Automatic
test thresholds are calibrated on validation data before the test split is
evaluated.

## Model phases

1. Phase I predicts a parcel building mask from the four context channels with
   an unconditional 3D U-Net.
2. Phase II injects normalized GRZ, GFZ, and height controls through FiLM
   conditioning in a conditional 3D U-Net.
3. Phase III first trains the VAE (`mode=vae_pretrain`) and then freezes that
   VAE for latent diffusion (`mode=diffusion`). The diffusion config therefore
   requires the selected VAE checkpoint.

The full configuration collection and its sequencing rules are documented in
[`configs/README.md`](configs/README.md).

## Input contract

Training consumes NPZ samples produced by `01_CreateDataset` and a fixed split
manifest. Each sample provides:

- `X` with shape `(C, D, H, W)` or individual `C0` to `C3` arrays;
- `Y` with shape `(D, H, W)` or `(1, D, H, W)`;
- `meta.metrics`, including `grz_target`, `gfz_target`, and
  `target_height_m` for conditioning;
- the sample path, retained for traceable inference exports.

The default thesis paths are:

```text
00_Data/02_GeneratedDatasets/dataset_thesis/
00_Data/02_GeneratedDatasets/dataset_thesis/report_split.json
```

Use the same split manifest for all phases. Model selection and threshold
calibration use validation data; the test split is reserved for final reports.
See [`dataio/README.md`](dataio/README.md) for preprocessing details.

## Saved model smoke visualizations

Test every model family with a real committed sample and save arrays, metrics,
2D projections, and 3D context views:

```bash
python 02_TrainModels/train_cli.py smoke-models \
  --data-root 00_Data/02_GeneratedDatasets/example_smoke \
  --out-dir 02_TrainModels/outputs/smoke_models \
  --steps 2
```

This command uses compact models and bounded mini-training. Its outputs verify
the data/model/visualization integration; they are not predictions from the
trained thesis checkpoints. `summary.json` records the selected sample,
configuration, losses, metrics, runtime, and every generated artifact.

## Outputs and resume behavior

Each run directory contains the resolved `config.json`, local metric files,
checkpoints, and optional visual samples. Scalar metrics are written to CSV;
TensorBoard output is created only when enabled by the calling code. No network
service or account is required.

If checkpoint files exist, training resumes from the newest epoch checkpoint.
A changed configuration stops resume by default. Use a new output directory for
a distinct experiment; `--allow_resume_mismatch` is reserved for an intentional
manual recovery.

Important artifacts are:

```text
config.json                 resolved run arguments
metrics.csv                 epoch/batch scalar history
checkpoints/epoch_*.pt      periodic checkpoints
best.pt                     selected validation checkpoint
final_metrics.json          final best metric summary where produced
experiment_summary.json     multi-run experiment result
eval_<split>/metrics.json   inference/evaluation result
```

## Released checkpoints

Model checkpoints and full run outputs are intentionally excluded from Git.
The final artifact record contains the selected unconditional U-Net,
conditional U-Net, VAE, and latent diffusion checkpoints together with their
restore paths, resolved configs, conditioning statistics where applicable,
final metrics, and evaluation summaries.

Download records and expected restore paths are maintained in
[`ARTIFACTS.md`](../ARTIFACTS.md). Checkpoint paths referenced by the Phase III
configs become valid after the model outputs have been restored at the
repository root.

## Direct training and evaluation

Direct commands are useful for one-off runs:

```bash
python 02_TrainModels/train_cli.py train-unet \
  --data-root 00_Data/02_GeneratedDatasets/dataset_thesis \
  --split-manifest 00_Data/02_GeneratedDatasets/dataset_thesis/report_split.json \
  --out-dir 02_TrainModels/outputs/manual/unet \
  --epochs 1 --max-samples 8
```

```bash
python 02_TrainModels/train_cli.py infer \
  --model unet \
  --checkpoint 02_TrainModels/outputs/manual/unet/best.pt \
  --config 02_TrainModels/outputs/manual/unet/config.json \
  --data-root 00_Data/02_GeneratedDatasets/dataset_thesis \
  --out-dir 02_TrainModels/outputs/manual/unet/eval_test \
  --split test
```

Inference accepts `--vis-mode none`, `2d`, `3d`, or `both`. Use
`--render-threshold` and `--render-angles` to control 3D exports.

## Reproducibility checklist

- Archive the exact experiment JSON and generated `config.json`.
- Archive the split manifest and dataset report with the dataset version.
- Keep seed, library versions, hardware, and checkpoint selection rule.
- Calibrate automatic thresholds only on validation data.
- Do not compare runs that use different split manifests as one experiment.
- Publish large checkpoints and outputs through the external artifact record,
  not Git.

## Development checks

```bash
ruff check 02_TrainModels tests
pytest tests/test_train_models_cli.py tests/test_model_data_smoke.py
```

The model smoke test loads a real generated sample, downsamples it to 32³, and
runs a loss plus backward pass through all three model families. It is skipped
when the optional training dependencies are not installed.
