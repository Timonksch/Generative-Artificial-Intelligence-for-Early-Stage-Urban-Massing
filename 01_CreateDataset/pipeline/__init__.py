"""Implement the urban voxel-dataset creation pipeline."""

from typing import Any

__all__ = ["run_pipeline"]


def run_pipeline(config: dict[str, Any]) -> None:
    """Load and execute the heavyweight pipeline lazily.

    Args:
        config: Dataset pipeline configuration.

    """
    from .pipeline import run_pipeline as execute  # noqa: PLC0415

    execute(config)
