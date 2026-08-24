# Training engines

The engine package contains the stateful training loops. Model construction and
argument parsing remain in `scripts/`; mathematical layers remain in `models/`.

## `trainer_unet.py`

`UNetTrainer` is shared by unconditional and conditional U-Nets. It handles
mixed precision, gradient accumulation, gradient clipping, validation,
test-time augmentation, automatic threshold search, early stopping, metrics,
visual previews, and checkpoint resume. `UNetTrainOptions` is the immutable
runtime contract passed by both training commands.

The best U-Net checkpoint maximizes validation IoU. Automatic thresholds are
selected on validation predictions and must be reused unchanged for test data.

## `trainer_ldm.py`

`LDMTrainer` supports three explicit modes:

- `vae_pretrain`: optimize reconstruction and KL loss;
- `diffusion`: freeze the loaded VAE and optimize latent noise prediction;
- `joint`: optimize both components for controlled experiments.

It additionally maintains optional EMA diffusion weights and generates bounded
VAE/diffusion previews. The best latent checkpoint minimizes validation loss.

## Artifacts and resume

Trainers write scalar CSV data, epoch summaries, periodic checkpoints, and
`best.pt`. Checkpoints include model, optimizer, scheduler, epoch, best metric,
and EMA state where applicable. Training commands compare the stored
`config.json` before resuming; a mismatch requires an explicit override or a
new output directory.

Schedulers step once per completed epoch. Loggers are closed at the end of the
training loop so CSV and optional TensorBoard resources are flushed reliably.
