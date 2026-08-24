"""Run all three model families on real generated smoke-dataset samples."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from conftest import (
    EXAMPLE_DATASET_DIRECTORY,
    EXPECTED_EXAMPLE_SAMPLE_COUNT,
    TRAIN_MODELS_DIRECTORY,
    paired_sample_paths,
)

torch = pytest.importorskip("torch", reason="model smoke tests require requirements-training.txt")

EXPECTED_INPUT_CHANNELS = 4
CONDITION_DIMENSION = 3
SMOKE_GRID_SIZE = 32

train_root_text = str(TRAIN_MODELS_DIRECTORY)
if train_root_text not in sys.path:
    sys.path.insert(0, train_root_text)

VoxelDataset = importlib.import_module("dataio.voxel_dataset").VoxelDataset
UNet3D = importlib.import_module("models.unet.unet3d").UNet3D
ConditionalUNet3D = importlib.import_module("models.unet.unet3d_cond").ConditionalUNet3D
BCEDiceLoss = importlib.import_module("models.unet.losses").BCEDiceLoss
VAE3D = importlib.import_module("models.ldm.vae3d").VAE3D
DiffusionUNet3D = importlib.import_module("models.ldm.diffusion_unet3d").DiffusionUNet3D
GaussianDiffusion = importlib.import_module("models.ldm.diffusion_core").GaussianDiffusion


@pytest.fixture(scope="module")
def real_batch() -> dict[str, Any]:
    """Load one deterministic 32³ batch from the generated example dataset."""
    npz_paths, json_paths = paired_sample_paths(EXAMPLE_DATASET_DIRECTORY)
    if (
        len(npz_paths) != EXPECTED_EXAMPLE_SAMPLE_COUNT
        or len(json_paths) != EXPECTED_EXAMPLE_SAMPLE_COUNT
    ):
        pytest.skip(f"generated example dataset is unavailable: {EXAMPLE_DATASET_DIRECTORY}")

    dataset = VoxelDataset(
        root_dir=str(EXAMPLE_DATASET_DIRECTORY),
        split="train",
        seed=42,
        augment=False,
        downsample_stride=8,
        cond_dim=CONDITION_DIMENSION,
        auto_cond_stats=True,
        max_samples=6,
    )
    sample = dataset[0]
    return {
        "voxels": sample["voxels"].unsqueeze(0),
        "target": sample["target"].unsqueeze(0),
        "cond": sample["cond"].unsqueeze(0),
        "path": sample["path"],
    }


def _assert_finite_gradients(model: Any) -> None:
    """Require at least one finite gradient after a smoke optimization step."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_generated_sample_contract(real_batch: dict[str, Any]) -> None:
    """Verify that generated arrays satisfy the shared model input contract."""
    voxels = real_batch["voxels"]
    target = real_batch["target"]
    cond = real_batch["cond"]

    assert voxels.shape == (1, EXPECTED_INPUT_CHANNELS, *(SMOKE_GRID_SIZE,) * 3)
    assert target.shape == (1, 1, *(SMOKE_GRID_SIZE,) * 3)
    assert cond.shape == (1, CONDITION_DIMENSION)
    assert torch.isfinite(voxels).all()
    assert torch.isfinite(target).all()
    assert torch.isfinite(cond).all()
    assert set(torch.unique(target).tolist()).issubset({0.0, 1.0})


def test_unconditional_unet_training_step(real_batch: dict[str, Any]) -> None:
    """Run one loss and backward pass through the Phase-I model."""
    model = UNet3D(in_channels=EXPECTED_INPUT_CHANNELS, base_channels=4, depth=3)
    logits = model(real_batch["voxels"])
    loss = BCEDiceLoss(0.5, 0.5, 1.0)(logits, real_batch["target"])

    assert logits.shape == real_batch["target"].shape
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_gradients(model)


def test_conditional_unet_training_step(real_batch: dict[str, Any]) -> None:
    """Run one conditioned loss and backward pass through the Phase-II model."""
    model = ConditionalUNet3D(
        in_channels=EXPECTED_INPUT_CHANNELS,
        base_channels=4,
        depth=3,
        cond_dim=CONDITION_DIMENSION,
    )
    logits = model(real_batch["voxels"], real_batch["cond"])
    loss = BCEDiceLoss(0.5, 0.5, 1.0)(logits, real_batch["target"])

    assert logits.shape == real_batch["target"].shape
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_gradients(model)


def test_latent_diffusion_training_step(real_batch: dict[str, Any]) -> None:
    """Run VAE reconstruction and diffusion losses through the Phase-III path."""
    vae = VAE3D(in_channels=1, base_channels=4, latent_channels=2, depth=3)
    reconstruction, mean, log_variance = vae(real_batch["target"])
    vae_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reconstruction, real_batch["target"]
    ) + 1e-4 * torch.mean(-0.5 * (1 + log_variance - mean.square() - log_variance.exp()))

    assert reconstruction.shape == real_batch["target"].shape
    assert torch.isfinite(vae_loss)
    vae_loss.backward()
    _assert_finite_gradients(vae)

    latent = mean.detach()
    diffusion_model = DiffusionUNet3D(
        latent_channels=latent.shape[1],
        base_channels=4,
        depth=2,
        time_dim=32,
        cond_dim=CONDITION_DIMENSION,
        context_channels=EXPECTED_INPUT_CHANNELS,
    )
    diffusion = GaussianDiffusion(timesteps=10, beta_schedule="linear")
    timesteps = torch.tensor([5], dtype=torch.long)
    diffusion_loss = diffusion.training_loss(
        diffusion_model,
        latent,
        timesteps,
        real_batch["cond"],
        context=real_batch["voxels"],
    )

    assert torch.isfinite(diffusion_loss)
    diffusion_loss.backward()
    _assert_finite_gradients(diffusion_model)
