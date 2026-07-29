# 3D U-Net models

## Phase I

`UNet3D` predicts one occupancy-logit channel from the selected urban-context
channels. Encoder/decoder depth and base capacity come from the experiment
configuration. Skip connections retain parcel-scale spatial detail.

## Phase II

`ConditionalUNet3D` uses the same spatial task but injects normalized GRZ, GFZ,
and target-height controls through FiLM layers. `cond_dim` must match the saved
conditioning statistics and `cond_select` configuration.

## Losses

`BCEDiceLoss` combines dynamically weighted BCE-with-logits and soft Dice. An
optional surface weight emphasizes target boundaries. The trainer adds the
separately configured volume regularizer. Helper functions provide IoU,
precision/recall, relative volume error, and vertical-profile error.

Both networks accept `(B,C,D,H,W)` and return `(B,1,D,H,W)` logits. Never apply
sigmoid before passing outputs to `BCEDiceLoss`.
