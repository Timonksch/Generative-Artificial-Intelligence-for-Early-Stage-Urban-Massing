"""Device and reproducibility utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Container describing the selected compute device."""

    device: torch.device
    device_type: str
    amp_dtype: torch.dtype | None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding helper."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device() -> DeviceConfig:
    """Return the best available compute device with sane defaults."""
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("medium")
        device = torch.device("mps")
        return DeviceConfig(device=device, device_type="mps", amp_dtype=None)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        amp_dtype = torch.float16
        if os.getenv("PYTORCH_ENABLE_BF16", "0") == "1":
            amp_dtype = torch.bfloat16
        return DeviceConfig(device=device, device_type="cuda", amp_dtype=amp_dtype)

    device = torch.device("cpu")
    return DeviceConfig(device=device, device_type="cpu", amp_dtype=None)


def configure_determinism(enable: bool) -> None:
    """Optionally enable deterministic kernels (CUDA/MPS)."""
    torch.use_deterministic_algorithms(enable)
    torch.backends.cudnn.deterministic = enable
    torch.backends.cudnn.benchmark = not enable
