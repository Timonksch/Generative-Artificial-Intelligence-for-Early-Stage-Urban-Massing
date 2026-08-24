"""Inspect one voxel sample and optionally export its standard visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_INPUT_DIMENSIONS = 4
EXPECTED_VOLUME_DIMENSIONS = 3
PARCEL_EDGE_CHANNEL_INDEX = 2
STREET_CHANNEL_INDEX = 3


@dataclass(frozen=True, slots=True)
class SampleData:
    """Store arrays and metadata loaded from one NPZ sample."""

    input_channels: np.ndarray[Any, Any]
    target: np.ndarray[Any, Any]
    neighbors: np.ndarray[Any, Any]
    metadata: dict[str, Any]


def _unpack_metadata(raw_metadata: np.ndarray[Any, Any], path: Path) -> dict[str, Any]:
    """Unpack an embedded object-array metadata value.

    Args:
        raw_metadata: NPZ ``meta`` array.
        path: Source path used in error messages.

    Returns:
        Embedded metadata dictionary.

    Raises:
        TypeError: If the embedded value is not a dictionary.
        ValueError: If the array cannot be unpacked as one object.

    """
    metadata = raw_metadata.item()
    if not isinstance(metadata, dict):
        raise TypeError(f"Embedded metadata in {path} must be a dictionary")
    return metadata


def load_sample(path: Path) -> SampleData:
    """Load and structurally validate one dataset sample.

    Args:
        path: NPZ sample path.

    Returns:
        Validated arrays and metadata.

    Raises:
        FileNotFoundError: If the NPZ path does not exist.
        KeyError: If a required array is absent.
        ValueError: If an array has an invalid dimensionality.
        TypeError: If embedded metadata is not a dictionary.

    """
    if not path.is_file():
        raise FileNotFoundError(f"Sample not found: {path}")
    # Embedded project metadata is intentionally stored as a trusted object array.
    with np.load(path, allow_pickle=True) as archive:
        required = {"X", "Y", "Y_neigh", "meta"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Sample is missing NPZ arrays: {missing}")
        inputs = archive["X"]
        target = archive["Y"]
        neighbors = archive["Y_neigh"]
        metadata = _unpack_metadata(archive["meta"], path)
    if inputs.ndim != EXPECTED_INPUT_DIMENSIONS:
        raise ValueError(f"X must have shape (C,D,H,W), got {inputs.shape}")
    for name, array in (("Y", target), ("Y_neigh", neighbors)):
        if array.ndim != EXPECTED_VOLUME_DIMENSIONS:
            raise ValueError(f"{name} must have shape (D,H,W), got {array.shape}")
    if inputs.shape[1:] != target.shape or neighbors.shape != target.shape:
        raise ValueError("X, Y, and Y_neigh spatial dimensions must match")
    return SampleData(inputs, target.astype(bool), neighbors.astype(bool), metadata)


def sample_summary(path: Path, sample: SampleData) -> dict[str, Any]:
    """Build a JSON-compatible structural summary.

    Args:
        path: Source sample path.
        sample: Loaded sample data.

    Returns:
        Summary object.

    """
    grid = sample.metadata.get("grid", {})
    metrics = sample.metadata.get("metrics", {})
    return {
        "path": str(path),
        "sample_id": sample.metadata.get("parcel_id", path.stem),
        "arrays": {
            "X": {
                "shape": list(sample.input_channels.shape),
                "dtype": str(sample.input_channels.dtype),
            },
            "Y": {
                "shape": list(sample.target.shape),
                "dtype": str(sample.target.dtype),
                "occupied_voxels": int(sample.target.sum()),
            },
            "Y_neigh": {
                "shape": list(sample.neighbors.shape),
                "dtype": str(sample.neighbors.dtype),
                "occupied_voxels": int(sample.neighbors.sum()),
            },
        },
        "grid": grid if isinstance(grid, dict) else {},
        "metrics": metrics if isinstance(metrics, dict) else {},
    }


def _channels(
    sample: SampleData,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any] | None,
]:
    """Return four visualization channels, padding absent optional channels.

    Args:
        sample: Loaded sample data.

    Returns:
        C0, C1, C2, and optional C3 arrays.

    """
    channel_count = sample.input_channels.shape[0]
    fallback = np.zeros_like(sample.target, dtype=np.float32)
    channel_0 = sample.input_channels[0]
    channel_1 = sample.input_channels[1] if channel_count > 1 else fallback
    channel_2 = (
        sample.input_channels[PARCEL_EDGE_CHANNEL_INDEX]
        if channel_count > PARCEL_EDGE_CHANNEL_INDEX
        else fallback
    )
    channel_3 = (
        sample.input_channels[STREET_CHANNEL_INDEX]
        if channel_count > STREET_CHANNEL_INDEX
        else None
    )
    return channel_0, channel_1, channel_2, channel_3


def export_visualizations(
    sample: SampleData,
    *,
    overview_path: Path | None,
    render_3d_path: Path | None,
    stride: int,
) -> None:
    """Export requested overview and high-resolution 3D images.

    Args:
        sample: Loaded sample data.
        overview_path: Optional overview image destination.
        render_3d_path: Optional 3D image destination.
        stride: Positive voxel rendering stride.

    Raises:
        ValueError: If stride is not positive.

    """
    if stride <= 0:
        raise ValueError("Visualization stride must be positive")
    if overview_path is None and render_3d_path is None:
        return
    # Delay Matplotlib initialization for summary-only inspection and CLI help.
    from .viz import make_overview_png, save_voxels_hires  # noqa: PLC0415

    sample_id = str(sample.metadata.get("parcel_id", "sample"))
    max_height = sample.metadata.get("max_height_m")
    max_height_m = float(max_height) if isinstance(max_height, (int, float)) else None
    if overview_path is not None:
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        channel_0, channel_1, channel_2, channel_3 = _channels(sample)
        make_overview_png(
            channel_0,
            channel_1,
            channel_2,
            channel_3,
            sample.target.max(axis=0),
            sample.target,
            sample.neighbors,
            sample_id,
            str(overview_path),
            stride3d=stride,
            max_height_m=max_height_m,
        )
        print(f"Overview saved: {overview_path}")
    if render_3d_path is not None:
        render_3d_path.parent.mkdir(parents=True, exist_ok=True)
        save_voxels_hires(
            sample.target,
            sample.neighbors,
            str(render_3d_path),
            stride3d=stride,
        )
        print(f"3D render saved: {render_3d_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the sample-inspector argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to one <sample_id>.npz file.")
    parser.add_argument("--overview", type=Path, help="Optional overview PNG output.")
    parser.add_argument(
        "--save-3d", "--save3d", dest="save_3d", type=Path, help="Optional 3D PNG output."
    )
    parser.add_argument("--stride", type=int, default=2, help="Positive 3D rendering stride.")
    parser.add_argument(
        "--json", action="store_true", help="Print compact JSON instead of text JSON."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect one sample from command-line arguments.

    Args:
        argv: Optional arguments without the program name.

    Returns:
        Zero on success or one for invalid input.

    """
    arguments = _build_parser().parse_args(argv)
    path = arguments.npz_path.expanduser().resolve()
    try:
        sample = load_sample(path)
        summary = sample_summary(path, sample)
        export_visualizations(
            sample,
            overview_path=arguments.overview,
            render_3d_path=arguments.save_3d,
            stride=arguments.stride,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=None if arguments.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
