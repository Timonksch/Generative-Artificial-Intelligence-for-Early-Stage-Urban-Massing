# Latent diffusion models

Phase III is a two-stage pipeline.

## Stage A: VAE

`VAE3D` encodes one-channel building occupancy into spatial mean and log-variance
tensors, samples with the reparameterization trick, and reconstructs occupancy
logits. Compression is determined by input resolution, latent resolution,
latent channels, base capacity, and depth.

## Stage B: diffusion

`DiffusionUNet3D` predicts noise in the VAE latent space. It combines sinusoidal
timestep embeddings, optional regulatory FiLM conditioning, and resampled
urban-context channels. `GaussianDiffusion` supplies linear/cosine schedules,
the forward noising process, training loss, classifier-free guidance, and DDPM
or DDIM sampling.

The reproducible final workflow is:

1. train `mode=vae_pretrain` and select the best validation checkpoint;
2. reference that checkpoint in the diffusion config;
3. run `mode=diffusion`, which freezes the VAE;
4. sample with the recorded sampler, steps, eta, and guidance scale.

Latent resolution, latent channels, VAE capacity, and checkpoint must agree
between stages. Strict loading rejects a mismatch.
