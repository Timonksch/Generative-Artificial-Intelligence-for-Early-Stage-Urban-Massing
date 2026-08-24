# Metrics

All metric functions are side-effect free and operate on tensors or scalar
collections. The evaluation scripts serialize their results to JSON.

## Segmentation (`seg3d.py`)

`evaluate` applies a probability threshold and reports IoU, Dice/F1,
precision, recall, relative volume error, absolute voxel-volume error, and
vertical-profile error. `auto_threshold` evaluates a finite threshold grid and
selects the highest validation IoU.

## Regulatory quantities (`regulatory.py`)

`compute_regulatory` converts occupancy to GRZ, GFZ, and building height using
the configured voxel and storey dimensions. Resolution-aware tolerances and
relative errors prevent unstable division around zero. `summarize` produces
distribution statistics for report tables.

## VAE loss (`vae.py`)

`vae_loss` combines reconstruction-with-logits and KL divergence. The KL weight
and optional annealing schedule are experiment parameters and therefore belong
in the archived run configuration.

Thresholds and model selection must use validation data only. Test metrics are
computed once with the frozen checkpoint, preprocessing, and threshold.
