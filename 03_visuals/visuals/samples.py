"""Export compact, explainable views of generated voxel samples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visuals.style import ACCENT, RED, save_figure

_CHANNEL_LABELS = (
    "Building mask",
    "Neighbour height",
    "Parcel edges",
    "Street mask",
)
_INPUT_DIMENSIONS = 4
_TARGET_DIMENSIONS = 3


def export_sample_figures(
    dataset_directory: Path,
    output_directory: Path,
    *,
    sample_ids: list[str] | None = None,
    limit: int = 1,
) -> list[Path]:
    """Export top-down channel and target projections for selected samples.

    Args:
        dataset_directory: Generated dataset containing NPZ samples.
        output_directory: Destination for the sample assets.
        sample_ids: Explicit identifiers, or ``None`` to select sorted samples.
        limit: Positive selection cap when identifiers are omitted.

    Returns:
        Every figure and manifest written by the operation.

    Raises:
        ValueError: If no matching samples exist or the limit is invalid.

    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    sample_paths = _resolve_sample_paths(dataset_directory, sample_ids, limit)
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest: list[dict[str, object]] = []
    for sample_path in sample_paths:
        sample_output = output_directory / sample_path.stem
        sample_output.mkdir(parents=True, exist_ok=True)
        written.extend(_export_one_sample(sample_path, sample_output))
        manifest.append({"sample_id": sample_path.stem, "source": str(sample_path)})
    manifest_path = output_directory / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def _resolve_sample_paths(
    dataset_directory: Path,
    sample_ids: list[str] | None,
    limit: int,
) -> list[Path]:
    """Resolve explicit or automatically selected sample NPZ files."""
    if sample_ids:
        paths = [dataset_directory / sample_id / f"{sample_id}.npz" for sample_id in sample_ids]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Sample does not exist: {missing[0]}")
        return paths
    paths = sorted(dataset_directory.glob("*/*.npz"))[:limit]
    if not paths:
        raise ValueError(f"No NPZ samples found under {dataset_directory}")
    return paths


def _export_one_sample(sample_path: Path, output_directory: Path) -> list[Path]:
    """Load one sample and render its channel projections."""
    with np.load(sample_path, allow_pickle=False) as archive:
        inputs = np.asarray(archive["X"], dtype=np.float32)
        target = np.asarray(archive["Y"], dtype=np.uint8)
        neighbors = np.asarray(archive["Y_neigh"], dtype=np.uint8)
    if inputs.ndim != _INPUT_DIMENSIONS or inputs.shape[0] != len(_CHANNEL_LABELS):
        raise ValueError(f"Expected X with shape (4, D, H, W), got {inputs.shape}")
    if target.ndim != _TARGET_DIMENSIONS or neighbors.shape != target.shape:
        raise ValueError(f"Invalid target shapes in {sample_path}")
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
    for channel_index, label in enumerate(_CHANNEL_LABELS):
        axis = axes.flat[channel_index]
        axis.imshow(np.max(inputs[channel_index], axis=0), cmap="gray")
        axis.set_title(label)
        axis.set_axis_off()
    axes.flat[4].imshow(np.max(target, axis=0), cmap="Blues")
    axes.flat[4].set_title("Target")
    axes.flat[4].set_axis_off()
    combined = np.zeros((*target.shape[1:], 3), dtype=np.float32)
    combined[np.max(neighbors, axis=0) > 0] = _hex_to_rgb(ACCENT)
    combined[np.max(target, axis=0) > 0] = _hex_to_rgb(RED)
    axes.flat[5].imshow(combined)
    axes.flat[5].set_title("Target and neighbors")
    axes.flat[5].set_axis_off()
    figure.tight_layout()
    return save_figure(figure, output_directory, "sample_overview")


def _hex_to_rgb(color: str) -> np.ndarray:
    """Convert a hexadecimal color to an RGB float triplet."""
    normalized = color.removeprefix("#")
    return np.asarray(
        [int(normalized[index : index + 2], 16) / 255.0 for index in (0, 2, 4)],
        dtype=np.float32,
    )
