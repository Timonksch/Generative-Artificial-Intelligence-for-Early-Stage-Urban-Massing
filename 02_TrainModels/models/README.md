# Models

The model package contains only neural architectures and mathematical model
components. Dataset access, optimization, evaluation, and file output are kept
outside this layer.

```text
common/  shared 3D convolution and FiLM blocks
unet/    Phase-I and Phase-II segmentation models and losses
ldm/     Phase-III VAE, diffusion backbone, schedules, and samplers
```

All models consume channel-first 3D tensors `(B,C,D,H,W)`. Segmentation models
return one-channel logits; callers apply sigmoid only for probabilities and
visualization. Conditional models consume normalized vectors `(B,K)`.

Architecture dimensions are controlled by versioned JSON configs. Loading is
strict: checkpoint parameters must match base channels, depth, input/context
channels, conditioning dimension, and latent layout recorded in `config.json`.
This prevents silently evaluating a checkpoint with a different experiment.

See the README in each subpackage for its architecture and tensor contracts.
