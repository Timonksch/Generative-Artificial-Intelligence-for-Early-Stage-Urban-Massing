# Shared model blocks

This package provides small layers reused across architectures.

- `DoubleConv3D`: two padded 3D convolutions with normalization and activation.
- `DownBlock3D`: feature extraction followed by stride-two spatial reduction.
- `UpBlock3D`: transposed convolution, skip concatenation, and feature fusion.
- `FiLM`: maps a conditioning vector to per-channel scale and bias values.

Blocks preserve channel-first `(B,C,D,H,W)` ordering. Up blocks pad small shape
differences before concatenating skips, which permits valid odd intermediate
sizes. FiLM requires one condition vector per batch item and broadcasts its
parameters across the three spatial axes.

These components contain no training state beyond their PyTorch parameters and
can be tested independently with small synthetic tensors.
