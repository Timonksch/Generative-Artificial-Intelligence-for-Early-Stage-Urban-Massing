"""Learning-rate scheduler factory."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    schedule: str,
    epochs: int,
    min_lr: float = 0.0,
    plateau_mode: str = "min",
) -> _LRScheduler | None:
    """Instantiate common scheduler types."""
    schedule = schedule.lower()
    if schedule == "none":
        return None
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if schedule == "plateau":
        mode = plateau_mode.lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported plateau_mode '{plateau_mode}'")
        return ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=5)

    raise ValueError(f"Unsupported lr_schedule '{schedule}'")
