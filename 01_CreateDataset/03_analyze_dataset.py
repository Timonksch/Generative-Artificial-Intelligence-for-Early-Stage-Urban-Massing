#!/usr/bin/env python3
"""Analyze voxel datasets and create deterministic train/validation/test splits."""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def _pyplot() -> Any:
    """Load and cache headless Matplotlib only when plots are requested."""
    from matplotlib import pyplot  # noqa: PLC0415

    pyplot.switch_backend("Agg")
    return pyplot


# ---------------------------------------- Bucket Definitions

BUCKET_DEFS: list[tuple[str, float, float]] = [
    ("A", 100.0, 500.0),
    ("B", 500.0, 1_500.0),
    ("C", 1_500.0, 5_000.0),
]

BUCKET_LABELS: dict[str, str] = {
    "A": "A (100-500 m²)",
    "B": "B (500-1 500 m²)",
    "C": "C (1 500-5 000 m²)",
    "other": "Other (outside A/B/C)",
}
VERY_SMALL_TARGET_VOXELS = 100
VERY_LARGE_TARGET_FRACTION = 0.5
MIN_EXPECTED_GRZ = 0.05
MAX_EXPECTED_GRZ = 0.8
MIN_EXPECTED_GFZ = 0.1
MAX_EXPECTED_GFZ = 3.0
MIN_EXPECTED_HEIGHT_M = 5.0
MAX_EXPECTED_HEIGHT_M = 100.0
MIN_EXPECTED_NEIGHBORS = 3
DEFAULT_SPLIT_RATIOS: list[tuple[str, float]] = [
    ("train", 0.70),
    ("val", 0.15),
    ("test", 0.15),
]


def classify_bucket(parcel_area_m2: float) -> str:
    """Return bucket label (A/B/C/other) for a parcel area in m²."""
    for name, lo, hi in BUCKET_DEFS:
        if lo <= parcel_area_m2 < hi:
            return name
    return "other"


# ---------------------------------------- Data Loading


def find_samples(dataset_dir: str) -> list[tuple[str, Path]]:
    """Find all valid samples in dataset directory.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        List of (parcel_id, sample_dir) tuples.

    """
    samples = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for item in dataset_path.iterdir():
        if not item.is_dir():
            continue

        sample_name = item.name
        npz_path = item / f"{sample_name}.npz"
        json_path = item / f"{sample_name}.json"

        if npz_path.exists() and json_path.exists():
            samples.append((sample_name, item))

    return sorted(samples)


def load_sample_metadata(sample_dir: Path, parcel_id: str) -> dict[str, Any]:
    """Load metadata JSON for a sample.

    Args:
        sample_dir: Path to sample directory.
        parcel_id: Sample identifier.

    Returns:
        Metadata dictionary.

    """
    json_path = sample_dir / f"{parcel_id}.json"
    with json_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"Metadata root must be an object: {json_path}")
    return metadata


def load_sample_npz(sample_dir: Path, parcel_id: str) -> dict[str, np.ndarray]:
    """Load NPZ data for a sample.

    Args:
        sample_dir: Path to sample directory.
        parcel_id: Sample identifier.

    Returns:
        Dictionary with X, Y, Y_neigh arrays.

    """
    npz_path = sample_dir / f"{parcel_id}.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("X", "Y", "Y_neigh")}


# ---------------------------------------- Analysis Functions


def analyze_dataset(
    dataset_dir: str,
    verbose: bool = True,
    samples: list[tuple[str, Path]] | None = None,
    dataset_label: str | None = None,
) -> dict[str, Any]:
    """Perform comprehensive dataset analysis.

    Args:
        dataset_dir: Path to dataset directory.
        verbose: Print progress messages.
        samples: Optional pre-selected samples to analyze.
        dataset_label: Optional label to use in logs/reports.

    Returns:
        Analysis results dictionary.

    """
    dataset_dir = str(dataset_dir)
    samples = find_samples(dataset_dir) if samples is None else list(samples)

    if dataset_label is None:
        dataset_label = dataset_dir

    if verbose:
        print(f"\n{'=' * 70}")
        print("Dataset Analysis")
        print(f"{'=' * 70}\n")
        print(f"Dataset: {dataset_label}")

    if not samples:
        raise ValueError(f"No valid samples found in {dataset_label}")

    if verbose:
        print(f"Found {len(samples)} samples\n")
        print("Loading metadata...")

    # Collect metadata
    all_metadata = []
    for parcel_id, sample_dir in samples:
        try:
            meta = load_sample_metadata(sample_dir, parcel_id)
            all_metadata.append(meta)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            if verbose:
                print(f"Warning: Failed to load {parcel_id}: {error}")

    if verbose:
        print(f"Successfully loaded {len(all_metadata)} metadata files\n")

    # Analyze
    results = {
        "dataset_path": dataset_dir,
        "dataset_label": dataset_label,
        "total_samples": len(samples),
        "successful_loads": len(all_metadata),
        "grid_config": analyze_grid_config(all_metadata),
        "channels": analyze_channels(all_metadata),
        "metrics": analyze_metrics(all_metadata),
        "distributions": compute_distributions(all_metadata),
        "quality_checks": perform_quality_checks(all_metadata),
        "buckets": analyze_buckets(all_metadata),
    }

    return results


def analyze_grid_config(metadata_list: list[dict]) -> dict[str, Any]:
    """Analyze grid configuration consistency.

    Args:
        metadata_list: List of metadata dictionaries.

    Returns:
        Grid configuration summary.

    """
    if not metadata_list:
        return {}

    # Check consistency
    grid_configs = [m["grid"] for m in metadata_list if "grid" in m]

    if not grid_configs:
        return {}

    first = grid_configs[0]

    # Compute grid_res (not stored in metadata)
    first_grid_res = round(first["grid_m"] / first["voxel_m"])

    # Check if all configs match
    consistent = all(
        g["grid_m"] == first["grid_m"]
        and g["voxel_m"] == first["voxel_m"]
        and g["D"] == first["D"]
        and g["H"] == first["H"]
        and g["W"] == first["W"]
        for g in grid_configs
    )

    return {
        "grid_m": first["grid_m"],
        "grid_res": first_grid_res,
        "voxel_m": first["voxel_m"],
        "dimensions": [first["D"], first["H"], first["W"]],
        "consistent": consistent,
        "unique_configs": len(
            {(g["grid_m"], g["voxel_m"], g["D"], g["H"], g["W"]) for g in grid_configs}
        ),
    }


def analyze_channels(metadata_list: list[dict]) -> dict[str, Any]:
    """Analyze channel configuration.

    Args:
        metadata_list: List of metadata dictionaries.

    Returns:
        Channel configuration summary.

    """
    if not metadata_list:
        return {}

    channel_counts = defaultdict(int)
    channel_names = defaultdict(set)

    for meta in metadata_list:
        if "channels" in meta:
            ch = meta["channels"]
            channel_counts[len(ch)] += 1
            channel_names[len(ch)].add(tuple(ch))

    # Most common config
    most_common_count = max(channel_counts.values()) if channel_counts else 0
    most_common_n = [k for k, v in channel_counts.items() if v == most_common_count]

    return {
        "channel_counts": dict(channel_counts),
        "most_common_n_channels": most_common_n[0] if most_common_n else None,
        "unique_configurations": sum(len(v) for v in channel_names.values()),
        "with_streets": any("C3_street_mask" in meta.get("channels", []) for meta in metadata_list),
    }


def analyze_metrics(metadata_list: list[dict]) -> dict[str, Any]:
    """Analyze metadata metrics.

    Args:
        metadata_list: List of metadata dictionaries.

    Returns:
        Metrics summary.

    """
    if not metadata_list:
        return {}

    # Extract all metrics
    metrics_data = defaultdict(list)

    for meta in metadata_list:
        if "metrics" not in meta:
            continue

        m = meta["metrics"]

        # Scalar metrics
        for key in [
            "c1_mean",
            "c1_max",
            "target_voxels",
            "neighbor_voxels",
            "num_neighbors",
            "parcel_area_m2",
            "target_footprint_area_m2",
            "coverage_frac",
            "target_height_m",
            "avg_neighbor_height_m",
            "bgf_target_m2",
            "grz_target",
            "gfz_target",
        ]:
            if key in m and m[key] is not None:
                metrics_data[key].append(float(m[key]))

        # Street coverage (optional)
        if "street_coverage_frac" in m and m["street_coverage_frac"] is not None:
            metrics_data["street_coverage_frac"].append(float(m["street_coverage_frac"]))

        # Neighbor metrics (lists)
        if m.get("grz_neighbors"):
            metrics_data["grz_neighbors"].extend([float(v) for v in m["grz_neighbors"]])
        if m.get("gfz_neighbors"):
            metrics_data["gfz_neighbors"].extend([float(v) for v in m["gfz_neighbors"]])

    # Compute statistics
    stats = {}
    for key, values in metrics_data.items():
        if not values:
            continue

        values_array = np.array(values)
        stats[key] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "count": len(values),
        }

    return stats


def compute_distributions(metadata_list: list[dict]) -> dict[str, Any]:
    """Compute distribution histograms for key metrics.

    Args:
        metadata_list: List of metadata dictionaries.

    Returns:
        Distribution data (bins and counts).

    """
    if not metadata_list:
        return {}

    # Extract key metrics
    metrics = {
        "target_height_m": [],
        "grz_target": [],
        "gfz_target": [],
        "parcel_area_m2": [],
        "coverage_frac": [],
        "num_neighbors": [],
        "target_voxels": [],
    }

    for meta in metadata_list:
        if "metrics" not in meta:
            continue
        m = meta["metrics"]

        for key, values in metrics.items():
            if key in m and m[key] is not None:
                values.append(float(m[key]))

    # Compute histograms
    distributions = {}

    for key, values in metrics.items():
        if not values:
            continue

        values_array = np.array(values)

        # Auto-select bins
        if key == "num_neighbors":
            bins = np.arange(int(np.min(values_array)), int(np.max(values_array)) + 2) - 0.5
        else:
            bins = "auto"

        counts, bin_edges = np.histogram(values_array, bins=bins)

        distributions[key] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "n_samples": len(values),
        }

    return distributions


def _number(value: object) -> float | None:
    """Convert a metadata scalar to float when possible."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _grid_resolution(metadata_list: list[dict]) -> int | None:
    """Find the first usable grid resolution."""
    for metadata in metadata_list:
        grid = metadata.get("grid")
        if not isinstance(grid, dict):
            continue
        direct = _number(grid.get("grid_res"))
        if direct is not None:
            return round(direct)
        grid_m = _number(grid.get("grid_m"))
        voxel_m = _number(grid.get("voxel_m"))
        if grid_m is not None and voxel_m:
            return round(grid_m / voxel_m)
        dimension = _number(grid.get("D") or grid.get("H") or grid.get("W"))
        if dimension is not None:
            return round(dimension)
    return None


def _quality_flags(metrics: dict[str, Any], grid_resolution: int | None) -> set[str]:
    """Return all quality flags triggered by one metric object."""
    flags: set[str] = set()
    target_voxels = _number(metrics.get("target_voxels")) or 0.0
    if target_voxels == 0:
        flags.add("empty_targets")
    elif target_voxels < VERY_SMALL_TARGET_VOXELS:
        flags.add("very_small_targets")
    if grid_resolution and target_voxels > VERY_LARGE_TARGET_FRACTION * grid_resolution**3:
        flags.add("very_large_targets")

    bounded_metrics = (
        ("grz_target", MIN_EXPECTED_GRZ, MAX_EXPECTED_GRZ, "extreme_grz"),
        ("gfz_target", MIN_EXPECTED_GFZ, MAX_EXPECTED_GFZ, "extreme_gfz"),
        ("target_height_m", MIN_EXPECTED_HEIGHT_M, MAX_EXPECTED_HEIGHT_M, "extreme_height"),
    )
    for key, minimum, maximum, flag in bounded_metrics:
        value = _number(metrics.get(key))
        if value is not None and not minimum <= value <= maximum:
            flags.add(flag)
    if (_number(metrics.get("num_neighbors")) or 0.0) < MIN_EXPECTED_NEIGHBORS:
        flags.add("few_neighbors")
    if (_number(metrics.get("c1_mean")) or 0.0) == 0:
        flags.add("missing_c1_mean")
    return flags


def perform_quality_checks(metadata_list: list[dict]) -> dict[str, Any]:
    """Count predefined quality flags across a dataset.

    Args:
        metadata_list: Sample metadata dictionaries.

    Returns:
        Counts and percentages for every quality flag.

    """
    if not metadata_list:
        return {}
    check_names = (
        "empty_targets",
        "very_small_targets",
        "very_large_targets",
        "extreme_grz",
        "extreme_gfz",
        "extreme_height",
        "few_neighbors",
        "missing_c1_mean",
    )
    checks = dict.fromkeys(check_names, 0)
    checks["total_samples"] = len(metadata_list)
    grid_resolution = _grid_resolution(metadata_list)
    for metadata in metadata_list:
        metrics = metadata.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for flag in _quality_flags(metrics, grid_resolution):
            checks[flag] += 1
    total = len(metadata_list)
    checks.update({f"{name}_pct": 100 * checks[name] / total for name in check_names})
    return checks


def analyze_buckets(metadata_list: list[dict]) -> dict[str, Any]:
    """Analyse dataset distribution across parcel size buckets A/B/C.

    Bucket A: 100-500 m²
    Bucket B: 500-1 500 m²
    Bucket C: 1 500-5 000 m²

    Args:
        metadata_list: List of metadata dictionaries.

    Returns:
        Dictionary keyed by bucket name with count, pct, and metric stats.

    """
    if not metadata_list:
        return {}

    bucket_meta: dict[str, list[dict]] = {name: [] for name, _, _ in BUCKET_DEFS}
    bucket_meta["other"] = []

    for meta in metadata_list:
        area: float | None = None
        if "metrics" in meta and meta["metrics"] and "parcel_area_m2" in meta["metrics"]:
            with suppress(TypeError, ValueError):
                area = float(meta["metrics"]["parcel_area_m2"])
        bucket_meta[classify_bucket(area) if area is not None else "other"].append(meta)

    total = len(metadata_list)
    result: dict[str, Any] = {}

    for name, lo, hi in BUCKET_DEFS:
        metas = bucket_meta[name]
        count = len(metas)
        pct = round(100.0 * count / total, 2) if total > 0 else 0.0
        result[name] = {
            "label": BUCKET_LABELS[name],
            "area_range_m2": [lo, hi],
            "count": count,
            "pct": pct,
            "metrics": analyze_metrics(metas) if metas else {},
        }

    other_count = len(bucket_meta["other"])
    if other_count > 0:
        result["other"] = {
            "label": BUCKET_LABELS["other"],
            "area_range_m2": None,
            "count": other_count,
            "pct": round(100.0 * other_count / total, 2) if total > 0 else 0.0,
            "metrics": {},
        }

    return result


# ---------------------------------------- Visualization


def plot_distributions(analysis: dict[str, Any], output_dir: str) -> None:
    """Generate distribution plots.

    Args:
        analysis: Analysis results dictionary.
        output_dir: Output directory for plots.

    """
    plt = _pyplot()
    distributions = analysis.get("distributions", {})

    if not distributions:
        print("No distribution data to plot")
        return

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Key metrics to plot
    plot_configs = {
        "target_height_m": ("Target Height (m)", "Height [m]"),
        "grz_target": ("GRZ Distribution", "GRZ"),
        "gfz_target": ("GFZ Distribution", "GFZ"),
        "parcel_area_m2": ("Parcel Area Distribution", "Area [m²]"),
        "coverage_frac": ("Coverage Fraction", "Coverage"),
        "num_neighbors": ("Number of Neighbors", "Count"),
        "target_voxels": ("Target Voxels", "Voxels"),
    }

    for metric, (title, xlabel) in plot_configs.items():
        if metric not in distributions:
            continue

        dist = distributions[metric]
        counts = np.array(dist["counts"])
        bin_edges = np.array(dist["bin_edges"])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        figure, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), edgecolor="black", alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{title} (N={dist['n_samples']})", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add statistics
        if metric in analysis["metrics"]:
            stats = analysis["metrics"][metric]
            textstr = (
                f"Mean: {stats['mean']:.2f}\nMedian: {stats['median']:.2f}\nStd: {stats['std']:.2f}"
            )
            ax.text(
                0.98,
                0.97,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        plt.tight_layout()

        output_path = output_directory / f"{metric}_distribution.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        print(f"Saved: {output_path}")


SUMMARY_PANELS = (
    ("target_height_m", "Target Height Distribution", "Height [m]", None, (0, 0)),
    ("grz_target", "GRZ Distribution", "GRZ", "orange", (0, 1)),
    ("gfz_target", "GFZ Distribution", "GFZ", "green", (0, 2)),
    ("parcel_area_m2", "Parcel Area Distribution", "Area [m²]", "red", (1, 0)),
    ("coverage_frac", "Coverage Fraction Distribution", "Coverage", "purple", (1, 1)),
    ("num_neighbors", "Number of Neighbors", "Count", "brown", (1, 2)),
    ("target_voxels", "Target Voxel Count Distribution", "Voxels", "cyan", (2, 0)),
)


def _plot_histogram_panel(
    axis: Any, distribution: dict[str, Any], title: str, label: str, color: str | None
) -> None:
    """Draw one histogram panel from precomputed counts."""
    counts = np.asarray(distribution["counts"])
    edges = np.asarray(distribution["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    axis.bar(centers, counts, width=np.diff(edges), edgecolor="black", alpha=0.7, color=color)
    axis.set_xlabel(label)
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)


def _summary_table(axis: Any, metrics: dict[str, Any]) -> None:
    """Draw the summary-statistics table."""
    rows = [["Metric", "Mean", "Median", "Std", "Min", "Max"]]
    for key, label in (
        ("target_height_m", "Height [m]"),
        ("grz_target", "GRZ"),
        ("gfz_target", "GFZ"),
        ("num_neighbors", "Neighbors"),
    ):
        if key not in metrics:
            continue
        values = metrics[key]
        rows.append(
            [label, *(f"{values[name]:.2f}" for name in ("mean", "median", "std", "min", "max"))]
        )
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for column in range(6):
        table[(0, column)].set_facecolor("#4CAF50")
        table[(0, column)].set_text_props(weight="bold", color="white")
    for row in range(2, len(rows), 2):
        for column in range(6):
            table[(row, column)].set_facecolor("#f0f0f0")


def plot_summary(analysis: dict[str, Any], output_path: str) -> None:
    """Generate the multi-panel summary visualization.

    Args:
        analysis: Analysis results dictionary.
        output_path: Output image path.

    """
    plt = _pyplot()
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    distributions = analysis.get("distributions", {})
    for key, title, label, color, position in SUMMARY_PANELS:
        if key not in distributions:
            continue
        axis = figure.add_subplot(grid[position])
        _plot_histogram_panel(axis, distributions[key], title, label, color)
    _summary_table(figure.add_subplot(grid[2, 1:]), analysis.get("metrics", {}))
    figure.suptitle(
        f"Dataset Analysis Summary (N={analysis['total_samples']} samples)",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def _bucket_tick_label(name: str, buckets: dict[str, Any]) -> str:
    """Format one bucket label with its numeric area range."""
    lower, upper = buckets[name]["area_range_m2"]
    return f"Bucket {name}\n{int(lower)}-{int(upper)} m²"


def plot_bucket_distribution(analysis: dict[str, Any], output_dir: str) -> None:
    """Generate bucket distribution bar chart (Bucket A / B / C).

    Args:
        analysis: Analysis results dictionary (must contain 'buckets' key).
        output_dir: Directory to save the plot.

    """
    plt = _pyplot()
    buckets = analysis.get("buckets", {})
    if not buckets:
        print("No bucket data available for bucket distribution plot.")
        return

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    plot_names = [n for n in ["A", "B", "C"] if n in buckets]
    counts = [buckets[n]["count"] for n in plot_names]
    pcts = [buckets[n]["pct"] for n in plot_names]
    tick_labels = [_bucket_tick_label(name, buckets) for name in plot_names]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    figure, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        tick_labels,
        counts,
        color=colors[: len(plot_names)],
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_xlabel("Parcel Size Bucket", fontsize=12)
    total = analysis.get("total_samples", sum(counts))
    ax.set_title(
        f"Bucket Distribution (N={total:,})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    max_count = max(counts) if counts else 1
    for bar, pct, count in zip(bars, pcts, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    output_path = output_directory / "bucket_distribution.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def plot_bucket_metrics(analysis: dict[str, Any], output_dir: str) -> None:
    """Generate per-bucket box-style summary plots for key regulatory metrics.

    Args:
        analysis: Analysis results dictionary.
        output_dir: Directory to save plots.

    """
    plt = _pyplot()
    buckets = analysis.get("buckets", {})
    plot_names = [n for n in ["A", "B", "C"] if n in buckets and buckets[n]["metrics"]]
    if not plot_names:
        return

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    key_metrics = [
        ("grz_target", "GRZ"),
        ("gfz_target", "GFZ"),
        ("target_height_m", "Height [m]"),
        ("parcel_area_m2", "Parcel Area [m²]"),
    ]
    colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}

    for metric_key, metric_label in key_metrics:
        figure, ax = plt.subplots(figsize=(8, 5))
        valid_names = [n for n in plot_names if metric_key in buckets[n]["metrics"]]
        if not valid_names:
            plt.close(figure)
            continue

        x = list(range(len(valid_names)))
        means = [buckets[n]["metrics"][metric_key]["mean"] for n in valid_names]
        stds = [buckets[n]["metrics"][metric_key]["std"] for n in valid_names]
        medians = [buckets[n]["metrics"][metric_key]["median"] for n in valid_names]
        bar_colors = [colors[n] for n in valid_names]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=bar_colors,
            edgecolor="black",
            alpha=0.8,
            capsize=5,
            label="Mean ± Std",
        )
        ax.scatter(x, medians, color="black", zorder=5, label="Median", marker="D", s=40)
        tick_labels = [_bucket_tick_label(name, buckets) for name in valid_names]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"{metric_label} per Bucket", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate with count
        for xi, name in zip(x, valid_names, strict=True):
            n_count = buckets[name]["count"]
            ax.text(
                xi,
                -0.08,
                f"n={n_count:,}",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                color="grey",
            )

        plt.tight_layout()
        output_path = output_directory / f"bucket_{metric_key}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {output_path}")


# ---------------------------------------- Reporting


def _print_section(title: str) -> None:
    """Print a standard report section heading."""
    print("-" * 70)
    print(title)
    print("-" * 70)


def _print_dataset_info(analysis: dict[str, Any]) -> None:
    """Print dataset identity, split, and sample counts."""
    dataset_label = analysis.get("dataset_label") or analysis.get("dataset_path", "Unknown dataset")
    dataset_path = analysis.get("dataset_path")
    print(f"Dataset: {dataset_label}")
    if dataset_path and dataset_path != dataset_label:
        print(f"Source: {dataset_path}")
    split_info = analysis.get("split_info")
    if isinstance(split_info, dict):
        details = [
            f"{key.removesuffix('_ratio')} {split_info[key] * 100:.2f}%"
            for key in ("target_ratio", "actual_ratio")
            if split_info.get(key) is not None
        ]
        name = split_info.get("name") or "split"
        print(f"Split Info: {name} ({', '.join(details)})")
    print(f"Total Samples: {analysis['total_samples']}")
    print(f"Successfully Loaded: {analysis['successful_loads']}\n")


def _print_grid_and_channels(analysis: dict[str, Any]) -> None:
    """Print grid and channel configuration sections."""
    _print_section("Grid Configuration")
    grid = analysis.get("grid_config", {})
    if grid:
        dimensions = "x".join(str(value) for value in grid["dimensions"])
        print(f"  Grid Size: {grid['grid_m']} m")
        print(f"  Grid Resolution: {grid['grid_res']}")
        print(f"  Voxel Size: {grid['voxel_m']} m")
        print(f"  Dimensions: {dimensions}")
        print(f"  Consistent: {grid['consistent']}")
        print(f"  Unique Configs: {grid['unique_configs']}\n")
    _print_section("Channels")
    channels = analysis.get("channels", {})
    if channels:
        print(f"  Most Common: {channels['most_common_n_channels']} channels")
        print(f"  Channel Count Distribution: {channels['channel_counts']}")
        print(f"  With Streets: {channels['with_streets']}\n")


def _print_metric_statistics(metrics: dict[str, Any]) -> None:
    """Print descriptive statistics for the primary metrics."""
    _print_section("Metric Statistics")
    labels = (
        ("target_height_m", "Target Height [m]"),
        ("grz_target", "GRZ"),
        ("gfz_target", "GFZ"),
        ("parcel_area_m2", "Parcel Area [m²]"),
        ("coverage_frac", "Coverage Fraction"),
        ("num_neighbors", "Number of Neighbors"),
        ("target_voxels", "Target Voxels"),
    )
    for key, label in labels:
        if key not in metrics:
            continue
        values = metrics[key]
        print(f"\n  {label}:")
        for name in ("mean", "median", "std", "min", "max", "q25", "q75"):
            print(f"    {name.capitalize():<7} {values[name]:>10.2f}")


def _print_quality_checks(checks: dict[str, Any]) -> None:
    """Print all quality-check counts and percentages."""
    _print_section("Quality Checks")
    labels = (
        ("empty_targets", "Empty Targets"),
        ("very_small_targets", "Very Small Targets (<100 voxels)"),
        ("very_large_targets", "Very Large Targets (>50% grid)"),
        ("extreme_grz", "Extreme GRZ"),
        ("extreme_gfz", "Extreme GFZ"),
        ("extreme_height", "Extreme Height"),
        ("few_neighbors", "Few Neighbors (<3)"),
        ("missing_c1_mean", "Missing C1 Mean"),
    )
    for key, label in labels:
        if key in checks:
            print(f"  {label}: {checks[key]} ({checks[f'{key}_pct']:.2f}%)")


def _print_bucket_table(analysis: dict[str, Any]) -> None:
    """Print the parcel-size bucket table."""
    _print_section("Bucket Distribution (Parcel Size)")
    buckets = analysis.get("buckets", {})
    if not buckets:
        return
    print(f"  {'Bucket':<22} {'Count':>8} {'Percent':>8}  {'GRZ':>9}  {'GFZ':>9}  {'Height':>12}")
    print(f"  {'-' * 80}")
    for name in ("A", "B", "C", "other"):
        if name not in buckets:
            continue
        bucket = buckets[name]
        metrics = bucket.get("metrics", {})
        grz = f"{metrics['grz_target']['mean']:.3f}" if "grz_target" in metrics else "N/A"
        gfz = f"{metrics['gfz_target']['mean']:.3f}" if "gfz_target" in metrics else "N/A"
        height = (
            f"{metrics['target_height_m']['mean']:.1f} m" if "target_height_m" in metrics else "N/A"
        )
        print(
            f"  {bucket.get('label', name):<22} {bucket['count']:>8,} {bucket['pct']:>7.1f}%"
            f"  {grz:>9}  {gfz:>9}  {height:>12}"
        )
    print(f"  {'-' * 80}")
    print(f"  {'Total':<22} {analysis['total_samples']:>8,} {'100.0%':>8}")


def print_report(analysis: dict[str, Any]) -> None:
    """Print the formatted analysis report.

    Args:
        analysis: Analysis results dictionary.

    """
    print(f"\n{'=' * 70}\nDATASET ANALYSIS REPORT\n{'=' * 70}\n")
    _print_dataset_info(analysis)
    _print_grid_and_channels(analysis)
    _print_metric_statistics(analysis.get("metrics", {}))
    print()
    _print_quality_checks(analysis.get("quality_checks", {}))
    print()
    _print_bucket_table(analysis)
    print(f"\n{'=' * 70}\n")


# ---------------------------------------- Split utilities


def compute_split_counts(total: int, ratios: list[tuple[str, float]]) -> dict[str, int]:
    """Compute split counts based on ratios while distributing remainders fairly."""
    if total <= 0:
        raise ValueError("Cannot split an empty dataset.")

    ratio_sum = sum(r for _, r in ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0 (got {ratio_sum})")

    raw_counts = [(name, total * ratio) for name, ratio in ratios]
    base_counts = [(name, math.floor(value)) for name, value in raw_counts]
    remainder = total - sum(count for _, count in base_counts)

    # Distribute remainder to splits with largest fractional parts
    fractional = sorted(
        ((name, value - math.floor(value)) for name, value in raw_counts),
        key=lambda x: x[1],
        reverse=True,
    )

    counts = dict(base_counts)
    for i in range(remainder):
        name, _ = fractional[i % len(fractional)]
        counts[name] += 1

    return counts


def split_samples(
    samples: list[tuple[str, Path]], ratios: list[tuple[str, float]], seed: int = 42
) -> dict[str, list[tuple[str, Path]]]:
    """Split samples into multiple subsets following the provided ratios."""
    counts = compute_split_counts(len(samples), ratios)

    shuffled = samples[:]
    # RULE_VIOLATION: A seeded non-cryptographic PRNG is required for reproducible splits.
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(shuffled)

    splits: dict[str, list[tuple[str, Path]]] = {}
    start = 0
    for name, _ in ratios:
        count = counts[name]
        subset = shuffled[start : start + count]
        splits[name] = subset
        start += count

    return splits


def save_json_report(analysis: dict[str, Any], output_path: str | Path) -> None:
    """Save analysis results atomically as JSON.

    Args:
        analysis: Analysis results dictionary.
        output_path: Output JSON file path.

    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(path)
    print(f"Saved JSON report: {path}")


# ---------------------------------------- Main


def _build_parser() -> argparse.ArgumentParser:
    """Create the analyzer argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", "--dataset_dir", dest="dataset_dir", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, help="Output analysis JSON.")
    parser.add_argument("--plot", action="store_true", help="Generate report plots.")
    parser.add_argument(
        "--plot-dir", "--plot_dir", dest="plot_dir", type=Path, default=Path("analysis_plots")
    )
    parser.add_argument("--summary-plot", "--summary_plot", dest="summary_plot", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Suppress console reports.")
    parser.add_argument("--mode", choices=("analyze", "split_analyze"), default="analyze")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--report-dir",
        "--report_dir",
        dest="report_dir",
        type=Path,
        default=Path("00_Data/02_GeneratedDatasets/report"),
    )
    return parser


def _resolve_report_paths(arguments: argparse.Namespace) -> None:
    """Apply report-directory defaults to parsed arguments."""
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    if arguments.output is None:
        filename = "report_split.json" if arguments.mode == "split_analyze" else "report.json"
        arguments.output = arguments.report_dir / filename
    if arguments.plot and arguments.plot_dir == Path("analysis_plots"):
        arguments.plot_dir = arguments.report_dir / "plots"


def _plot_analysis(analysis: dict[str, Any], plot_directory: Path, summary_path: Path) -> None:
    """Generate every plot for one analysis object."""
    plot_distributions(analysis, str(plot_directory))
    plot_summary(analysis, str(summary_path))
    plot_bucket_distribution(analysis, str(plot_directory))
    plot_bucket_metrics(analysis, str(plot_directory))


def _run_analysis_mode(arguments: argparse.Namespace) -> dict[str, Any]:
    """Analyze one complete dataset without splitting it."""
    analysis = analyze_dataset(str(arguments.dataset_dir), verbose=not arguments.quiet)
    if not arguments.quiet:
        print_report(analysis)
    if arguments.plot:
        summary = arguments.summary_plot or arguments.plot_dir / "summary.png"
        _plot_analysis(analysis, arguments.plot_dir, summary)
    return analysis


def _load_bucket_labels(samples: list[tuple[str, Path]]) -> dict[str, str]:
    """Classify samples by parcel-size bucket for reporting."""
    labels: dict[str, str] = {}
    for sample_id, sample_directory in samples:
        try:
            metadata = load_sample_metadata(sample_directory, sample_id)
            metrics = metadata.get("metrics", {})
            area = float(metrics.get("parcel_area_m2", -1))
            labels[sample_id] = classify_bucket(area) if area >= 0 else "other"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            labels[sample_id] = "other"
    return labels


def _bucket_split_counts(
    splits: dict[str, list[tuple[str, Path]]], labels: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Count split membership per parcel bucket."""
    bucket_names = [name for name, _minimum, _maximum in BUCKET_DEFS] + ["other"]
    counts = {bucket: dict.fromkeys(splits, 0) for bucket in bucket_names}
    for split, samples in splits.items():
        for sample_id, _directory in samples:
            counts[labels.get(sample_id, "other")][split] += 1
    return counts


def _print_split_overview(
    splits: dict[str, list[tuple[str, Path]]], labels: dict[str, str], seed: int
) -> None:
    """Print overall and per-bucket split counts."""
    total = sum(len(samples) for samples in splits.values())
    print(f"\n{'=' * 70}\nDataset Split\n{'=' * 70}")
    print(f"Total samples: {total}\nSplit seed: {seed}\n")
    for name, ratio in DEFAULT_SPLIT_RATIOS:
        count = len(splits[name])
        actual_percent = 100 * count / total
        target_percent = 100 * ratio
        print(
            f"  {name.capitalize():<5}: {count} "
            f"({actual_percent:.2f}%; target {target_percent:.2f}%)"
        )
    counts = _bucket_split_counts(splits, labels)
    print("\nPer-bucket split counts:")
    print(f"  {'Bucket':<22} {'Total':>10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for bucket, split_counts in counts.items():
        bucket_total = sum(split_counts.values())
        if bucket_total:
            print(
                f"  {BUCKET_LABELS[bucket]:<22} {bucket_total:>10,}"
                f" {split_counts['train']:>10,} {split_counts['val']:>10,}"
                f" {split_counts['test']:>10,}"
            )


def _analyze_split_subsets(
    dataset_directory: Path,
    splits: dict[str, list[tuple[str, Path]]],
    total: int,
    *,
    quiet: bool,
) -> dict[str, dict[str, Any]]:
    """Analyze each deterministic split independently."""
    results: dict[str, dict[str, Any]] = {}
    ratios = dict(DEFAULT_SPLIT_RATIOS)
    for name, samples in splits.items():
        analysis = analyze_dataset(
            str(dataset_directory),
            verbose=False,
            samples=samples,
            dataset_label=f"{name.capitalize()} split",
        )
        analysis["split_info"] = {
            "name": name,
            "target_ratio": ratios[name],
            "actual_ratio": len(samples) / total,
        }
        if not quiet:
            print_report(analysis)
        results[name] = {
            "analysis": analysis,
            "sample_ids": [sample_id for sample_id, _directory in samples],
        }
    return results


def _plot_split_results(
    full_analysis: dict[str, Any],
    split_results: dict[str, dict[str, Any]],
    plot_directory: Path,
    summary_path: Path,
) -> None:
    """Plot the full dataset and every split."""
    _plot_analysis(full_analysis, plot_directory, summary_path)
    for name, result in split_results.items():
        split_directory = plot_directory / name
        split_analysis = result["analysis"]
        plot_distributions(split_analysis, str(split_directory))
        plot_summary(split_analysis, str(split_directory / "summary.png"))
        plot_bucket_distribution(split_analysis, str(split_directory))


def _run_split_mode(arguments: argparse.Namespace) -> dict[str, Any]:
    """Create, analyze, and optionally plot deterministic dataset splits."""
    samples = find_samples(str(arguments.dataset_dir))
    if not samples:
        raise ValueError(f"No valid samples found in {arguments.dataset_dir}")
    splits = split_samples(samples, DEFAULT_SPLIT_RATIOS, seed=arguments.seed)
    labels = _load_bucket_labels(samples)
    if not arguments.quiet:
        _print_split_overview(splits, labels, arguments.seed)
    full_analysis = analyze_dataset(
        str(arguments.dataset_dir),
        verbose=not arguments.quiet,
        samples=samples,
        dataset_label="Full dataset",
    )
    if not arguments.quiet:
        print_report(full_analysis)
    split_results = _analyze_split_subsets(
        arguments.dataset_dir, splits, len(samples), quiet=arguments.quiet
    )
    if arguments.plot:
        summary = arguments.summary_plot or arguments.plot_dir / "summary.png"
        _plot_split_results(full_analysis, split_results, arguments.plot_dir, summary)
    return {
        "mode": "split_analyze",
        "dataset_dir": str(arguments.dataset_dir),
        "seed": arguments.seed,
        "split_ratios": dict(DEFAULT_SPLIT_RATIOS),
        "total_samples": len(samples),
        "full_analysis": full_analysis,
        "splits": split_results,
    }


def main(argv: list[str] | None = None) -> int:
    """Run dataset analysis from command-line arguments."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _resolve_report_paths(arguments)
    try:
        if arguments.mode == "analyze":
            result = _run_analysis_mode(arguments)
        else:
            result = _run_split_mode(arguments)
        save_json_report(result, arguments.output)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    print("Analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
