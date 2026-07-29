#!/usr/bin/env python3
"""Validate URBAN NPZ samples, JSON metadata, and split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - the validator also works without progress UI
    tqdm = None

IssueLevel = Literal["error", "warning"]
JsonObject = dict[str, Any]
STACKED_INPUT_DIMENSIONS = 4
VOXEL_DIMENSIONS = 3


@dataclass(frozen=True, slots=True)
class SamplePaths:
    """Store all files belonging to one sample."""

    sample_id: str
    npz_path: Path
    json_path: Path | None


@dataclass(frozen=True, slots=True)
class SampleStats:
    """Store structural statistics extracted from one valid sample."""

    voxel_shape: tuple[int, int, int]
    channel_count: int
    voxel_dtype: str
    target_dtype: str


@dataclass(frozen=True, slots=True)
class Issue:
    """Describe one validation problem."""

    level: IssueLevel
    sample_id: str
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class VoxelLayout:
    """Describe validated input voxel arrays."""

    shape: tuple[int, int, int] | None
    channels: int | None
    dtype: str


@dataclass(frozen=True, slots=True)
class TargetLayout:
    """Describe a validated target array."""

    shape: tuple[int, int, int] | None
    dtype: str


def _load_json_object(path: Path) -> JsonObject:
    """Load a JSON document and require a top-level object.

    Args:
        path: JSON file to read.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the JSON root is not an object.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the document is invalid JSON.

    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def _issue(sample: SamplePaths, level: IssueLevel, message: str) -> Issue:
    """Create an issue referring to a sample's NPZ file.

    Args:
        sample: Sample being validated.
        level: Issue severity.
        message: Human-readable description.

    Returns:
        Populated issue value.

    """
    return Issue(level, sample.sample_id, message, str(sample.npz_path))


def find_samples(root: Path, max_samples: int | None) -> tuple[list[SamplePaths], list[Issue]]:
    """Discover unique NPZ samples below a dataset root.

    Args:
        root: Dataset directory searched recursively.
        max_samples: Optional positive discovery limit.

    Returns:
        Discovered samples and scan issues.

    Raises:
        ValueError: If ``max_samples`` is not positive.

    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    samples: list[SamplePaths] = []
    issues: list[Issue] = []
    seen: dict[str, Path] = {}
    for npz_path in sorted(root.rglob("*.npz")):
        sample_id = npz_path.stem
        if sample_id in seen:
            issues.append(
                Issue(
                    "error",
                    sample_id,
                    f"duplicate sample id (existing: {seen[sample_id]})",
                    str(npz_path),
                )
            )
            continue

        json_path = npz_path.with_suffix(".json")
        json_reference = json_path if json_path.is_file() else None
        if json_reference is None:
            issues.append(
                Issue("warning", sample_id, "missing sidecar JSON metadata", str(npz_path))
            )
        samples.append(SamplePaths(sample_id, npz_path, json_reference))
        seen[sample_id] = npz_path
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        issues.append(Issue("error", "*", "no NPZ files discovered under root", str(root)))
    return samples, issues


def load_metadata(json_path: Path | None) -> JsonObject | None:
    """Load optional sample metadata and report readable failures.

    Args:
        json_path: Sidecar JSON path or ``None``.

    Returns:
        Metadata object, or ``None`` if absent or unreadable.

    """
    if json_path is None:
        return None
    try:
        return _load_json_object(json_path)
    except (OSError, TypeError, json.JSONDecodeError) as error:
        print(f"[validator] failed to load JSON {json_path}: {error}", file=sys.stderr)
        return None


def read_manifest_ids(manifest_path: Path | None) -> dict[str, list[str]] | None:
    """Read split sample IDs from either supported manifest layout.

    Args:
        manifest_path: Manifest JSON path or ``None``.

    Returns:
        Mapping from split names to sample IDs, or ``None``.

    Raises:
        TypeError: If the manifest structure or an ID is invalid.
        ValueError: If a split has neither supported layout.

    """
    if manifest_path is None:
        return None
    manifest = _load_json_object(manifest_path)
    raw_splits = manifest.get("splits", manifest)
    if not isinstance(raw_splits, dict):
        raise TypeError("Manifest 'splits' value must be an object")

    split_ids: dict[str, list[str]] = {}
    for split_name, entry in raw_splits.items():
        raw_ids = entry.get("sample_ids") if isinstance(entry, dict) else entry
        if not isinstance(raw_ids, list):
            raise ValueError(
                f"Manifest split {split_name!r} must be a list or contain 'sample_ids'"
            )
        if not all(isinstance(sample_id, str) for sample_id in raw_ids):
            raise TypeError(f"Manifest split {split_name!r} contains a non-string sample ID")
        split_ids[str(split_name)] = raw_ids
    return split_ids


def _contains_non_finite(array: np.ndarray[Any, Any]) -> bool:
    """Check numeric arrays for NaN or infinity values.

    Args:
        array: NumPy array to inspect.

    Returns:
        ``True`` when a numeric array contains a non-finite value.

    """
    if not np.issubdtype(array.dtype, np.number):
        return False
    return bool(np.logical_not(np.isfinite(array)).any())


def _validate_stacked_input(
    sample: SamplePaths,
    array: np.ndarray[Any, Any],
    check_nan: bool,
) -> tuple[VoxelLayout, list[Issue]]:
    """Validate an input stored as one ``X`` array.

    Args:
        sample: Sample being validated.
        array: Input tensor in ``(C,D,H,W)`` order.
        check_nan: Enable full finite-value checks.

    Returns:
        Input layout and detected issues.

    """
    issues: list[Issue] = []
    if array.ndim != STACKED_INPUT_DIMENSIONS:
        issues.append(
            _issue(sample, "error", f"expected X with 4 dims (C,D,H,W), got {array.shape}")
        )
        return VoxelLayout(None, None, str(array.dtype)), issues

    shape = tuple(int(value) for value in array.shape[1:])
    if array.dtype not in (np.float16, np.float32, np.float64):
        issues.append(
            _issue(sample, "warning", f"X dtype {array.dtype} unexpected (prefer float32)")
        )
    if check_nan and _contains_non_finite(array):
        issues.append(_issue(sample, "error", "X contains NaN or Inf values"))
    return VoxelLayout(shape, int(array.shape[0]), str(array.dtype)), issues


def _validate_separate_inputs(
    sample: SamplePaths,
    archive: np.lib.npyio.NpzFile,
    channel_keys: Sequence[str],
    check_nan: bool,
) -> tuple[VoxelLayout, list[Issue]]:
    """Validate inputs stored in separate ``C*`` arrays.

    Args:
        sample: Sample being validated.
        archive: Open NPZ archive.
        channel_keys: Ordered channel array names.
        check_nan: Enable full finite-value checks.

    Returns:
        Input layout and detected issues.

    """
    issues: list[Issue] = []
    valid_arrays: list[np.ndarray[Any, Any]] = []
    shape: tuple[int, int, int] | None = None
    for key in channel_keys:
        array = archive[key]
        if array.ndim != VOXEL_DIMENSIONS:
            issues.append(
                _issue(sample, "error", f"channel {key} expected 3 dims, got {array.shape}")
            )
            continue
        current_shape = tuple(int(value) for value in array.shape)
        if shape is not None and current_shape != shape:
            issues.append(
                _issue(sample, "error", f"channel {key} shape {array.shape} differs from {shape}")
            )
            continue
        shape = current_shape
        valid_arrays.append(array)
        if check_nan and _contains_non_finite(array):
            issues.append(_issue(sample, "error", f"channel {key} contains NaN or Inf values"))

    if not valid_arrays:
        issues.append(
            _issue(sample, "error", "missing voxel data: neither X nor valid C* arrays found")
        )
        return VoxelLayout(shape, None, "unknown"), issues
    return VoxelLayout(shape, len(valid_arrays), str(valid_arrays[0].dtype)), issues


def _validate_target(
    sample: SamplePaths,
    archive: np.lib.npyio.NpzFile,
    check_nan: bool,
) -> tuple[TargetLayout, list[Issue]]:
    """Validate the required target array ``Y``.

    Args:
        sample: Sample being validated.
        archive: Open NPZ archive.
        check_nan: Enable full finite-value checks.

    Returns:
        Target layout and detected issues.

    """
    if "Y" not in archive.files:
        return TargetLayout(None, "unknown"), [_issue(sample, "error", "missing target mask 'Y'")]

    target = archive["Y"]
    issues: list[Issue] = []
    shape: tuple[int, int, int] | None = None
    if target.ndim == VOXEL_DIMENSIONS:
        shape = tuple(int(value) for value in target.shape)
    elif target.ndim == STACKED_INPUT_DIMENSIONS and target.shape[0] == 1:
        shape = tuple(int(value) for value in target.shape[1:])
    else:
        issues.append(_issue(sample, "error", f"unexpected Y shape {target.shape}"))

    dtype = str(target.dtype)
    if dtype not in {"uint8", "int16", "float32"}:
        issues.append(_issue(sample, "warning", f"Y dtype {dtype} is unusual"))
    if check_nan and _contains_non_finite(target):
        issues.append(_issue(sample, "error", "Y contains NaN or Inf values"))
    return TargetLayout(shape, dtype), issues


def _validate_optional_arrays(
    sample: SamplePaths,
    archive: np.lib.npyio.NpzFile,
    target_shape: tuple[int, int, int] | None,
) -> list[Issue]:
    """Validate optional neighbor and embedded-metadata arrays.

    Args:
        sample: Sample being validated.
        archive: Open NPZ archive.
        target_shape: Validated ``Y`` shape if available.

    Returns:
        Detected optional-array issues.

    """
    issues: list[Issue] = []
    if "Y_neigh" in archive.files and target_shape is not None:
        neighbor = archive["Y_neigh"]
        if neighbor.shape != target_shape:
            issues.append(
                _issue(sample, "warning", f"Y_neigh shape {neighbor.shape} differs from Y")
            )

    if "meta" not in archive.files:
        return issues
    try:
        metadata = archive["meta"].item()
    except (AttributeError, ValueError) as error:
        issues.append(_issue(sample, "warning", f"failed to unpack meta ({error})"))
        return issues
    if not isinstance(metadata, dict):
        issues.append(
            _issue(sample, "warning", f"meta is {type(metadata).__name__}, expected dict")
        )
    return issues


def validate_npz(sample: SamplePaths, check_nan: bool) -> tuple[SampleStats | None, list[Issue]]:
    """Validate one NPZ archive and derive its structural statistics.

    Args:
        sample: Sample archive and metadata paths.
        check_nan: Enable full finite-value checks.

    Returns:
        Optional statistics and all detected issues.

    """
    try:
        archive = np.load(sample.npz_path, allow_pickle=True, mmap_mode="r")
    except (OSError, ValueError, EOFError) as error:
        return None, [_issue(sample, "error", f"failed to load NPZ ({error})")]

    with archive:
        if "X" in archive.files:
            input_layout, issues = _validate_stacked_input(sample, archive["X"], check_nan)
        else:
            keys = sorted(key for key in archive.files if key.upper().startswith("C"))
            input_layout, issues = _validate_separate_inputs(sample, archive, keys, check_nan)
        target_layout, target_issues = _validate_target(sample, archive, check_nan)
        issues.extend(target_issues)
        issues.extend(_validate_optional_arrays(sample, archive, target_layout.shape))

    if input_layout.shape != target_layout.shape:
        issues.append(
            _issue(
                sample,
                "error",
                f"voxel shape {input_layout.shape} differs from target {target_layout.shape}",
            )
        )
    if input_layout.shape is None or input_layout.channels is None or target_layout.shape is None:
        return None, issues
    stats = SampleStats(
        input_layout.shape,
        input_layout.channels,
        input_layout.dtype,
        target_layout.dtype,
    )
    return stats, issues


def validate_metadata(
    sample: SamplePaths,
    stats: SampleStats | None,
    metadata: JsonObject | None,
) -> list[Issue]:
    """Validate sidecar metadata against file names and tensor dimensions.

    Args:
        sample: Sample being validated.
        stats: Validated NPZ statistics if available.
        metadata: Parsed sidecar metadata if available.

    Returns:
        Detected metadata issues.

    """
    if metadata is None:
        return []
    path = str(sample.json_path or "")
    issues: list[Issue] = []
    parcel_id = metadata.get("parcel_id")
    if isinstance(parcel_id, str) and parcel_id.casefold() != sample.sample_id.casefold():
        issues.append(
            Issue("error", sample.sample_id, f"parcel_id mismatch (json: {parcel_id})", path)
        )

    grid = metadata.get("grid")
    if stats is not None and isinstance(grid, dict):
        dimensions = tuple(grid.get(key) for key in ("D", "H", "W"))
        if any(value is None for value in dimensions):
            issues.append(Issue("warning", sample.sample_id, "grid missing D/H/W", path))
        elif not all(isinstance(value, (int, float)) for value in dimensions):
            issues.append(Issue("error", sample.sample_id, "grid D/H/W must be numeric", path))
        elif tuple(int(value) for value in dimensions) != stats.voxel_shape:
            issues.append(
                Issue("error", sample.sample_id, "grid dimensions differ from voxel shape", path)
            )

    channels = metadata.get("channels")
    if stats is not None and isinstance(channels, list) and len(channels) != stats.channel_count:
        message = f"metadata lists {len(channels)} channels, data has {stats.channel_count}"
        issues.append(Issue("warning", sample.sample_id, message, path))
    return issues


def summarize_issues(issues: Iterable[Issue]) -> tuple[list[Issue], list[Issue]]:
    """Separate errors and warnings.

    Args:
        issues: Issues to partition.

    Returns:
        Error list followed by warning list.

    """
    issue_list = list(issues)
    return (
        [issue for issue in issue_list if issue.level == "error"],
        [issue for issue in issue_list if issue.level == "warning"],
    )


def summarize_manifest(
    manifest_ids: dict[str, list[str]] | None,
    sample_ids: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    """Compare manifest IDs with discovered samples.

    Args:
        manifest_ids: Split IDs or ``None``.
        sample_ids: IDs discovered in the dataset.

    Returns:
        Missing and unassigned IDs grouped by split.

    """
    if manifest_ids is None:
        return {}
    sample_set = {sample_id.casefold() for sample_id in sample_ids}
    manifest_union: set[str] = set()
    summary: dict[str, dict[str, list[str]]] = {}
    for split, identifiers in manifest_ids.items():
        manifest_set = {sample_id.casefold() for sample_id in identifiers}
        manifest_union.update(manifest_set)
        summary[split] = {"missing": sorted(manifest_set - sample_set)}
    extra = sorted(sample_set - manifest_union)
    if extra:
        summary["__unassigned__"] = {"extra": extra}
    return summary


# RULE_VIOLATION: Six explicit report sections avoid a one-use container abstraction.
def make_report(  # noqa: PLR0913, PLR0917
    root: Path,
    discovered_count: int,
    stats_list: Sequence[SampleStats],
    errors: Sequence[Issue],
    warnings: Sequence[Issue],
    manifest_summary: dict[str, dict[str, list[str]]],
) -> JsonObject:
    """Build the machine-readable validation report.

    Args:
        root: Validated dataset root.
        discovered_count: Number of discovered samples.
        stats_list: Statistics for structurally valid samples.
        errors: Validation errors.
        warnings: Validation warnings.
        manifest_summary: Manifest comparison result.

    Returns:
        JSON-serializable report object.

    """
    voxel_shapes = Counter(stat.voxel_shape for stat in stats_list)
    channel_counts = Counter(stat.channel_count for stat in stats_list)
    return {
        "root": str(root),
        "samples_discovered": discovered_count,
        "samples_with_stats": len(stats_list),
        "voxel_shapes": {str(key): value for key, value in voxel_shapes.items()},
        "channel_counts": {str(key): value for key, value in channel_counts.items()},
        "voxel_dtypes": dict(Counter(stat.voxel_dtype for stat in stats_list)),
        "target_dtypes": dict(Counter(stat.target_dtype for stat in stats_list)),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "manifest_diffs": manifest_summary,
        "errors": [asdict(issue) for issue in errors],
        "warnings": [asdict(issue) for issue in warnings],
    }


def _print_manifest_differences(report: JsonObject) -> None:
    """Print a compact manifest-difference summary.

    Args:
        report: Validation report object.

    """
    differences = report["manifest_diffs"]
    if not isinstance(differences, dict):
        return
    for split, raw_difference in differences.items():
        if not isinstance(raw_difference, dict):
            continue
        for label in ("missing", "extra"):
            identifiers = raw_difference.get(label)
            if isinstance(identifiers, list) and identifiers:
                preview = identifiers[:3]
                print(
                    f"[validator] manifest {split!r} {label}: {len(identifiers)} (e.g. {preview})"
                )


def print_summary(report: JsonObject) -> None:
    """Print the key validation result fields.

    Args:
        report: Validation report object.

    """
    print(f"[validator] root: {report['root']}")
    print(f"[validator] samples discovered: {report['samples_discovered']}")
    print(f"[validator] samples with valid stats: {report['samples_with_stats']}")
    for key in ("voxel_shapes", "channel_counts", "voxel_dtypes", "target_dtypes"):
        if report[key]:
            print(f"[validator] {key.replace('_', ' ')}: {report[key]}")
    _print_manifest_differences(report)
    print(f"[validator] errors: {report['error_count']} | warnings: {report['warning_count']}")
    for level in ("errors", "warnings"):
        entries = report[level]
        if not isinstance(entries, list) or not entries:
            continue
        print(f"[validator] first 10 {level}:")
        for issue in entries[:10]:
            print(f"  - {issue['sample_id']}: {issue['message']} ({issue['path']})")


def _write_report(path: Path, report: JsonObject) -> None:
    """Write a validation report atomically.

    Args:
        path: Destination JSON file.
        report: JSON-serializable report.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def _validate_all(
    samples: Sequence[SamplePaths], check_nan: bool
) -> tuple[list[SampleStats], list[Issue]]:
    """Validate every discovered sample.

    Args:
        samples: Samples to validate.
        check_nan: Enable full finite-value checks.

    Returns:
        Valid sample statistics and all issues.

    """
    stats_list: list[SampleStats] = []
    issues: list[Issue] = []
    iterator: Iterable[SamplePaths] = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc="Validating samples", unit="sample")
    for sample in iterator:
        stats, npz_issues = validate_npz(sample, check_nan)
        issues.extend(npz_issues)
        issues.extend(validate_metadata(sample, stats, load_metadata(sample.json_path)))
        if stats is not None:
            stats_list.append(stats)
    return stats_list, issues


def _build_parser() -> argparse.ArgumentParser:
    """Create the validator argument parser.

    Returns:
        Configured parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("00_Data/02_GeneratedDatasets/data"), help="Dataset root."
    )
    parser.add_argument("--manifest", type=Path, help="Optional split manifest.")
    parser.add_argument("--max-samples", type=int, help="Positive sample limit.")
    parser.add_argument("--check-nan", action="store_true", help="Check all arrays for NaN/Inf.")
    parser.add_argument("--report", type=Path, help="Optional output report JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset validation.

    Args:
        argv: Optional arguments without the program name.

    Returns:
        Zero if valid, one for invocation/input errors, or two for validation errors.

    """
    arguments = _build_parser().parse_args(argv)
    root = arguments.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[validator] dataset root not found: {root}", file=sys.stderr)
        return 1
    manifest_path = arguments.manifest.expanduser().resolve() if arguments.manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        print(f"[validator] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        samples, scan_issues = find_samples(root, arguments.max_samples)
        stats_list, issues = _validate_all(samples, arguments.check_nan)
        issues.extend(scan_issues)
        manifest_ids = read_manifest_ids(manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[validator] invalid input: {error}", file=sys.stderr)
        return 1

    if manifest_ids and arguments.max_samples is not None:
        print("[validator] skipping manifest diff because --max-samples was provided.")
        manifest_summary: dict[str, dict[str, list[str]]] = {}
    else:
        manifest_summary = summarize_manifest(
            manifest_ids, [sample.sample_id for sample in samples]
        )
    errors, warnings = summarize_issues(issues)
    report = make_report(root, len(samples), stats_list, errors, warnings, manifest_summary)
    print_summary(report)
    if arguments.report is not None:
        report_path = arguments.report.expanduser().resolve()
        _write_report(report_path, report)
        print(f"[validator] report saved to {report_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
