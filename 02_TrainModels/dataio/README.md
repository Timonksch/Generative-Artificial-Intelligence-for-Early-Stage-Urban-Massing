# Data loading

This package is the only data boundary used by training, inference, evaluation,
and sampling.

## Files

- `voxel_dataset.py`: discovers NPZ samples, applies split membership and
  preprocessing, and produces PyTorch tensors.
- `cond_stats.py`: extracts, normalizes, saves, and reloads conditioning
  statistics.

## NPZ contract

`VoxelDataset` accepts `X` with shape `(C,D,H,W)` or separate `C0` to `C3`
arrays. `Y` must be `(D,H,W)` or `(1,D,H,W)`. Returned targets always have one
channel. The canonical input order is parcel mask, normalized neighboring
height, edges, and street mask.

The collated batch contains:

```text
voxels  float32 (B,C,D,H,W)
target  float32 (B,1,D,H,W)
cond    float32 (B,K)
meta    per-sample metadata
path    source NPZ paths
```

Optional preprocessing includes deterministic stride downsampling, spatial
cropping, and normalized coordinate channels. Quarter rotations and flips are
enabled only for the training split.

## Conditioning

Condition vectors read `grz_target`, `gfz_target`, and `target_height_m` from
`meta.metrics`, in that order. Mean and standard deviation are estimated only
from training samples, written to `cond_stats.json`, and reused unchanged for
validation, test, inference, and sampling. `cond_select` selects a subset while
preserving this canonical ordering.

Use a fixed `report_split.json` for reported experiments. The seeded fallback
split exists for smoke tests, not final comparisons. Missing channels, invalid
shapes, empty splits, and unresolved manifest IDs fail before model execution.
