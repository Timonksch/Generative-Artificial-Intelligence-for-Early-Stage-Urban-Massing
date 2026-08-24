#!/usr/bin/env python3
"""Recompute voxel-derived GRZ, GFZ, BGF, and building-height metadata."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

DOWNSAMPLE_THRESHOLD: Final = 0.5
ZIP_COPY_BUFFER_BYTES: Final = 1024 * 1024
VOXEL_DIMENSIONS: Final = 3
STACKED_TARGET_DIMENSIONS: Final = 4
MetricValues = dict[str, float | int]
SampleStatus = Literal["updated", "skipped"]


@dataclass(frozen=True, slots=True)
class SampleFiles:
    """Store the sidecar and array paths for one dataset sample."""

    sample_id: str
    json_path: Path
    npz_path: Path


@dataclass(frozen=True, slots=True)
class RecomputeOptions:
    """Store validated metric-recomputation parameters."""

    storey_height_m: float
    target_voxel_m: float
    dry_run: bool


def iter_samples(dataset_directory: Path) -> Iterator[SampleFiles]:
    """Yield complete samples from immediate child directories.

    Args:
        dataset_directory: Dataset root containing one directory per sample.

    Yields:
        Sample paths in stable identifier order.

    """
    for sample_directory in sorted(dataset_directory.iterdir()):
        if not sample_directory.is_dir():
            continue
        sample_id = sample_directory.name
        json_path = sample_directory / f"{sample_id}.json"
        npz_path = sample_directory / f"{sample_id}.npz"
        if json_path.is_file() and npz_path.is_file():
            yield SampleFiles(sample_id, json_path, npz_path)


def downsample_binary(mask: np.ndarray[Any, Any], stride: int) -> np.ndarray[Any, Any]:
    """Downsample a 3D binary mask by thresholded block averaging.

    Args:
        mask: Binary or numeric mask in ``(D,H,W)`` order.
        stride: Positive equal block width along every axis.

    Returns:
        Downsampled Boolean mask.

    Raises:
        ValueError: If dimensions or stride are invalid.

    """
    if mask.ndim != VOXEL_DIMENSIONS:
        raise ValueError(f"Expected a 3D mask, got {mask.shape}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride == 1:
        return mask.astype(bool, copy=False)

    depth, height, width = mask.shape
    if any(size % stride for size in mask.shape):
        raise ValueError(f"Mask shape {mask.shape} is not divisible by stride={stride}")
    pooled = (
        mask.astype(np.float32, copy=False)
        .reshape(
            depth // stride,
            stride,
            height // stride,
            stride,
            width // stride,
            stride,
        )
        .mean(axis=(1, 3, 5))
    )
    return pooled >= DOWNSAMPLE_THRESHOLD


def compute_voxel_metrics(
    mask: np.ndarray[Any, Any],
    parcel_area_m2: float,
    voxel_m: float,
    storey_height_m: float,
) -> MetricValues:
    """Compute regulatory metrics from one voxel occupancy mask.

    Args:
        mask: Building occupancy in ``(D,H,W)`` order.
        parcel_area_m2: Parcel plan area in square meters.
        voxel_m: Isotropic voxel edge length in meters.
        storey_height_m: Height assumed for BGF/GFZ conversion.

    Returns:
        Metric values and supporting voxel counts.

    Raises:
        ValueError: If a physical parameter is not positive.

    """
    if parcel_area_m2 < 0:
        raise ValueError("parcel_area_m2 cannot be negative")
    if voxel_m <= 0 or storey_height_m <= 0:
        raise ValueError("voxel_m and storey_height_m must be positive")
    occupied = mask.astype(bool, copy=False)
    volume_cells = int(occupied.sum())
    if volume_cells == 0:
        return {
            "voxel_m": voxel_m,
            "target_voxels": 0,
            "target_footprint_voxels": 0,
            "target_height_m": 0.0,
            "grz_target": 0.0,
            "gfz_target": 0.0,
            "bgf_target_m2": 0.0,
        }

    footprint_cells = int(np.any(occupied, axis=0).sum())
    occupied_layers = np.flatnonzero(np.any(occupied, axis=(1, 2)))
    layer_count = int(occupied_layers[-1] - occupied_layers[0] + 1)
    height_m = layer_count * voxel_m
    footprint_area_m2 = footprint_cells * voxel_m**2
    volume_m3 = volume_cells * voxel_m**3
    grz = footprint_area_m2 / parcel_area_m2 if parcel_area_m2 > 0 else 0.0
    bgf_m2 = volume_m3 / storey_height_m
    gfz = bgf_m2 / parcel_area_m2 if parcel_area_m2 > 0 else 0.0
    return {
        "voxel_m": voxel_m,
        "target_voxels": volume_cells,
        "target_footprint_voxels": footprint_cells,
        "target_height_m": height_m,
        "grz_target": grz,
        "gfz_target": gfz,
        "bgf_target_m2": bgf_m2,
    }


def _float_changed(old_value: object, new_value: float, epsilon: float = 1e-12) -> bool:
    """Compare a metadata scalar with a computed float.

    Args:
        old_value: Existing JSON-compatible value.
        new_value: Newly computed value.
        epsilon: Absolute comparison tolerance.

    Returns:
        Whether the values differ or the old value is not numeric.

    """
    try:
        old_float = float(old_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return not math.isclose(old_float, new_value, rel_tol=0.0, abs_tol=epsilon)


def _resolution_metrics(metrics: Mapping[str, float | int]) -> MetricValues:
    """Normalize one resolution's metric object.

    Args:
        metrics: Raw computed metric values.

    Returns:
        Metric object with explicit integer counts and float measurements.

    """
    return {
        "voxel_m": float(metrics["voxel_m"]),
        "target_voxels": int(metrics["target_voxels"]),
        "target_footprint_voxels": int(metrics["target_footprint_voxels"]),
        "target_height_m": float(metrics["target_height_m"]),
        "grz_target": float(metrics["grz_target"]),
        "gfz_target": float(metrics["gfz_target"]),
        "bgf_target_m2": float(metrics["bgf_target_m2"]),
    }


def _flat_updates(
    native: Mapping[str, float | int], coarse: Mapping[str, float | int]
) -> MetricValues:
    """Build legacy and explicit flat metric keys.

    Args:
        native: Native-resolution values.
        coarse: Coarse-resolution values.

    Returns:
        Flat metadata updates.

    """
    updates: MetricValues = {
        "target_voxels": int(native["target_voxels"]),
        "target_height_m": float(native["target_height_m"]),
        "bgf_target_m2": float(native["bgf_target_m2"]),
        "grz_target": float(native["grz_target"]),
        "gfz_target": float(native["gfz_target"]),
        "coverage_frac": float(native["grz_target"]),
    }
    for suffix, values in (("0p5m", native), ("2m", coarse)):
        updates[f"target_footprint_voxels_{suffix}"] = int(values["target_footprint_voxels"])
        updates[f"target_voxels_{suffix}"] = int(values["target_voxels"])
        for key in ("target_height_m", "grz_target", "gfz_target", "bgf_target_m2"):
            updates[f"{key}_{suffix}"] = float(values[key])
    return updates


def apply_metric_updates(
    metrics: dict[str, object],
    native: Mapping[str, float | int],
    coarse: Mapping[str, float | int],
) -> bool:
    """Apply native and coarse metric updates in place.

    Args:
        metrics: Mutable sample metadata metric object.
        native: Native-resolution computed values.
        coarse: Coarse-resolution computed values.

    Returns:
        Whether at least one value changed.

    """
    changed = False
    for key, value in _flat_updates(native, coarse).items():
        if _float_changed(metrics.get(key), float(value)):
            metrics[key] = value
            changed = True

    voxel_metrics = {"0p5m": _resolution_metrics(native), "2m": _resolution_metrics(coarse)}
    if metrics.get("voxel_metrics") != voxel_metrics:
        metrics["voxel_metrics"] = voxel_metrics
        changed = True
    return changed


def _copy_npz_with_meta(
    source_path: Path, destination_path: Path, metadata: dict[str, Any]
) -> None:
    """Copy an NPZ archive while replacing every previous ``meta.npy`` entry.

    Args:
        source_path: Existing NPZ archive.
        destination_path: New archive path.
        metadata: Metadata serialized as the sole ``meta.npy`` entry.

    Raises:
        zipfile.BadZipFile: If the source is not a valid NPZ/ZIP archive.
        OSError: If either archive cannot be read or written.

    """
    buffer = io.BytesIO()
    np.save(buffer, np.array(metadata, dtype=object), allow_pickle=True)
    with (
        zipfile.ZipFile(source_path, mode="r") as source,
        zipfile.ZipFile(destination_path, mode="w", allowZip64=True) as destination,
    ):
        for member in source.infolist():
            if member.filename == "meta.npy":
                continue
            with (
                source.open(member, "r") as input_stream,
                destination.open(member, "w", force_zip64=True) as output_stream,
            ):
                shutil.copyfileobj(input_stream, output_stream, ZIP_COPY_BUFFER_BYTES)
        destination.writestr("meta.npy", buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _write_updated_sample(sample: SampleFiles, metadata: dict[str, Any]) -> None:
    """Write updated NPZ and JSON files through temporary siblings.

    Args:
        sample: Sample files to replace.
        metadata: Updated metadata object.

    Raises:
        OSError: If temporary or final files cannot be written.
        zipfile.BadZipFile: If the source NPZ is invalid.

    """
    temporary_npz = sample.npz_path.with_suffix(".npz.part")
    temporary_json = sample.json_path.with_suffix(".json.part")
    try:
        _copy_npz_with_meta(sample.npz_path, temporary_npz, metadata)
        with temporary_json.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        temporary_npz.replace(sample.npz_path)
        temporary_json.replace(sample.json_path)
    except (OSError, zipfile.BadZipFile):
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
        raise


def _load_sample(
    sample: SampleFiles,
) -> tuple[dict[str, Any], dict[str, object], np.ndarray[Any, Any]]:
    """Load and validate metadata plus the target occupancy array.

    Args:
        sample: Sample files to read.

    Returns:
        Full metadata, mutable metric object, and 3D target array.

    Raises:
        TypeError: If metadata objects have invalid types.
        ValueError: If ``Y`` is absent or has an unsupported shape.

    """
    with sample.json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("Metadata root must be an object")
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("Metadata 'metrics' must be an object")

    with np.load(sample.npz_path, allow_pickle=False) as archive:
        if "Y" not in archive:
            raise ValueError("NPZ archive is missing target array 'Y'")
        target = archive["Y"]
    if target.ndim == STACKED_TARGET_DIMENSIONS and target.shape[0] == 1:
        target = target[0]
    if target.ndim != VOXEL_DIMENSIONS:
        raise ValueError(f"Unexpected Y shape {target.shape}")
    return metadata, metrics, target


def _target_stride(native_voxel_m: float, target_voxel_m: float) -> int:
    """Derive an exact integer downsampling stride.

    Args:
        native_voxel_m: Native voxel edge length.
        target_voxel_m: Requested coarse edge length.

    Returns:
        Positive integer stride.

    Raises:
        ValueError: If the target is not an integer multiple of the native size.

    """
    if native_voxel_m <= 0 or target_voxel_m <= 0:
        raise ValueError("Voxel sizes must be positive")
    stride = round(target_voxel_m / native_voxel_m)
    if stride <= 0 or not math.isclose(native_voxel_m * stride, target_voxel_m, abs_tol=1e-9):
        raise ValueError(
            f"Target voxel {target_voxel_m} is incompatible with native {native_voxel_m}"
        )
    return stride


def _recompute_sample(sample: SampleFiles, options: RecomputeOptions) -> SampleStatus:
    """Recompute and optionally persist metrics for one sample.

    Args:
        sample: Sample files to process.
        options: Physical and persistence options.

    Returns:
        ``updated`` or ``skipped``.

    Raises:
        KeyError: If required metadata fields are absent.
        TypeError: If required metadata fields have invalid types.
        ValueError: If dimensions or voxel sizes are incompatible.

    """
    metadata, metrics, target = _load_sample(sample)
    grid = metadata.get("grid")
    if not isinstance(grid, dict):
        raise TypeError("Metadata 'grid' must be an object")
    parcel_area_m2 = float(metrics["parcel_area_m2"])
    native_voxel_m = float(grid["voxel_m"])
    stride = _target_stride(native_voxel_m, options.target_voxel_m)
    native_mask = target > DOWNSAMPLE_THRESHOLD
    native = compute_voxel_metrics(
        native_mask, parcel_area_m2, native_voxel_m, options.storey_height_m
    )
    coarse_mask = downsample_binary(native_mask, stride)
    coarse = compute_voxel_metrics(
        coarse_mask,
        parcel_area_m2,
        options.target_voxel_m,
        options.storey_height_m,
    )
    if not apply_metric_updates(metrics, native, coarse):
        return "skipped"
    if not options.dry_run:
        _write_updated_sample(sample, metadata)
    return "updated"


def recompute_dataset(
    dataset_directory: Path,
    storey_height_m: float,
    target_voxel_m_2m: float,
    dry_run: bool = False,
    progress_every: int = 250,
) -> dict[str, int]:
    """Recompute metrics for every complete sample in a dataset.

    Args:
        dataset_directory: Dataset root.
        storey_height_m: Height used for BGF/GFZ conversion.
        target_voxel_m_2m: Requested coarse voxel edge length.
        dry_run: Compute changes without writing files.
        progress_every: Print progress every N samples; zero disables progress.

    Returns:
        Processing counts.

    Raises:
        ValueError: If a global option is invalid.

    """
    if storey_height_m <= 0 or target_voxel_m_2m <= 0 or progress_every < 0:
        raise ValueError("Heights/voxel sizes must be positive and progress_every non-negative")
    options = RecomputeOptions(storey_height_m, target_voxel_m_2m, dry_run)
    counts = {"samples_total": 0, "samples_updated": 0, "samples_skipped": 0, "samples_failed": 0}
    handled_errors = (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    )
    for index, sample in enumerate(iter_samples(dataset_directory), start=1):
        counts["samples_total"] += 1
        try:
            status = _recompute_sample(sample, options)
        except handled_errors as error:
            counts["samples_failed"] += 1
            print(f"[error] {sample.sample_id}: {error}", file=sys.stderr)
            continue
        counts[f"samples_{status}"] += 1
        if progress_every and index % progress_every == 0:
            print(f"[{index}] {status}: {sample.sample_id}")
    return counts


def _build_parser() -> argparse.ArgumentParser:
    """Create the metric-updater argument parser.

    Returns:
        Configured parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", "--dataset_dir", dest="dataset_dir", type=Path, required=True
    )
    parser.add_argument(
        "--storey-height-m",
        "--storey_height_m",
        dest="storey_height_m",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--target-voxel-m",
        "--target_voxel_m_2m",
        dest="target_voxel_m",
        type=float,
        default=2.0,
    )
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument(
        "--progress-every",
        "--progress_every",
        dest="progress_every",
        type=int,
        default=250,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run metric recomputation from command-line arguments.

    Args:
        argv: Optional arguments without the program name.

    Returns:
        Zero when every sample succeeds, otherwise two.

    """
    arguments = _build_parser().parse_args(argv)
    dataset_directory = arguments.dataset_dir.expanduser().resolve()
    if not dataset_directory.is_dir():
        print(f"Dataset directory not found: {dataset_directory}", file=sys.stderr)
        return 1
    try:
        counts = recompute_dataset(
            dataset_directory,
            arguments.storey_height_m,
            arguments.target_voxel_m,
            arguments.dry_run,
            arguments.progress_every,
        )
    except ValueError as error:
        print(f"Invalid option: {error}", file=sys.stderr)
        return 1
    print(
        "[done] total={samples_total} updated={samples_updated} "
        "skipped={samples_skipped} failed={samples_failed}".format(**counts)
    )
    return 0 if counts["samples_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
