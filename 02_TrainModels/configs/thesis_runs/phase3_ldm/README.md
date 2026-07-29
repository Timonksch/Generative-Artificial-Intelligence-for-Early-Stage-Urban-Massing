# Phase III configurations: VAE and latent diffusion

Phase III separates compression from generation. Compression/capacity sweeps
first determine the VAE layout. Diffusion experiments then vary schedule,
capacity, context/coordinates, and conditioning dropout while reusing a frozen
VAE.

Run the final stages in order:

```bash
python 02_TrainModels/train_cli.py experiment \
  --config 02_TrainModels/configs/thesis_runs/phase3_ldm/phase3_final_vae.json \
  --out-parent 02_TrainModels/outputs/thesis_runs/architecture_phase3_ldm
```

Verify that `vae_checkpoint` in `phase3_final_diff.json` resolves to the
resulting VAE `best.pt`, then run the diffusion config with the same output
parent. Both configs use `model=ldm`; `mode` distinguishes `vae_pretrain` from
`diffusion`.

Sampling parameters (`sampler`, `sample_timesteps`, `ddim_eta`, `cfg_scale`)
are part of the experiment contract and must be archived with reported output.
