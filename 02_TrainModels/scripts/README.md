# Command implementations

These files implement commands registered by `../train_cli.py`. They are kept
separate so each scientific workflow has one bounded responsibility.

| File | Central command | Responsibility |
|---|---|---|
| `experiment.py` | `experiment` | Expand JSON configs into one or more training runs and optional evaluation. |
| `train_unet.py` | `train-unet` | Construct and train the Phase-I model. |
| `train_unet_cond.py` | `train-cond-unet` | Construct and train the Phase-II model and conditioning statistics. |
| `train_ldm.py` | `train-ldm` | Train VAE, diffusion, or explicit joint mode. |
| `infer.py` | `infer` | Unified checkpoint inference and prediction exports. |
| `eval_regulatory.py` | `evaluate` | Segmentation and regulatory-control evaluation. |
| `eval_vae_reconstruction.py` | `evaluate-vae` | Stage-A reconstruction evaluation. |
| `sample_cond_unet_controls.py` | `sample-cond` | Controlled Phase-II variants for fixed samples. |
| `sample_best_ldm_experiment.py` | `sample-ldm` | Stochastic variants from the best completed LDM run. |
| `smoke_visualize_models.py` | `smoke-models` | Bounded integration test with NPZ, metrics, and visual outputs. |

Invoke scripts through the central CLI so repository-relative paths, working
directory, interpreter, and import path are consistent:

```bash
python 02_TrainModels/train_cli.py COMMAND --help
```

Direct script invocation is an internal development mechanism and is not part
of the documented user interface. All scripts return a nonzero process status
on argument, checkpoint, data, or evaluation failures.
