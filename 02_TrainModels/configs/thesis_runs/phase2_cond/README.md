# Phase II configurations: conditional 3D U-Net

These experiments extend the U-Net with normalized GRZ, GFZ, and target-height
controls. The ordered studies cover coordinate references, channel ablation,
conditioning dropout, network capacity, loss balance, and evaluation
sensitivity before the selected final rerun.

The `model` value is `cond_unet`. `cond_select` indexes the canonical
`grz_target`, `gfz_target`, `target_height_m` vector. Conditioning statistics
must be estimated on the training split and reused for validation/test.

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/thesis_runs/phase2_cond/phase2_cond_final.json \
  --out-parent 02_TrainModels/outputs/thesis_runs/architecture_phase2
```

The final configuration records the selected validation protocol, including
test-time augmentation and threshold grid.
