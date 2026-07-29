"""Typed readers for dataset metadata and model metric logs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

_BOUNDING_BOX_VALUE_COUNT = 4


@dataclass(frozen=True)
class DatasetSample:
    """Metadata required by the dataset figures."""

    sample_id: str
    metrics: dict[str, float]
    center_x: float
    center_y: float


@dataclass(frozen=True)
class MetricSeries:
    """Numeric values for one metric in one run."""

    steps: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class RunMetrics:
    """Metrics discovered for a single training run."""

    name: str
    path: Path
    series: dict[str, MetricSeries]

    def final(self, metric_name: str) -> float | None:
        """Return the last logged value of a metric, if present."""
        metric_series = self.series.get(metric_name)
        if metric_series is None or not metric_series.values:
            return None
        return metric_series.values[-1]


def _read_json(path: Path) -> object:
    """Read one JSON file using deterministic resource handling."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_samples(
    dataset_directory: Path,
    *,
    limit: int | None = None,
) -> list[DatasetSample]:
    """Load sample metadata files from a generated dataset.

    Args:
        dataset_directory: Directory containing one subdirectory per sample.
        limit: Optional positive cap used for quick checks.

    Returns:
        Samples sorted by identifier.

    Raises:
        ValueError: If ``limit`` is invalid or no samples are readable.

    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    metadata_paths = sorted(dataset_directory.glob("*/*.json"))
    if limit is not None:
        metadata_paths = metadata_paths[:limit]
    samples = [_parse_dataset_sample(path) for path in metadata_paths]
    if not samples:
        raise ValueError(f"No sample metadata found under {dataset_directory}")
    return samples


def _parse_dataset_sample(path: Path) -> DatasetSample:
    """Validate and convert one sample metadata document."""
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    metrics_raw = payload.get("metrics")
    bounds_raw = payload.get("world_bbox_xy")
    if not isinstance(metrics_raw, dict) or not isinstance(bounds_raw, list):
        raise ValueError(f"Missing metrics or world_bbox_xy in {path}")
    if len(bounds_raw) != _BOUNDING_BOX_VALUE_COUNT:
        raise ValueError(f"world_bbox_xy must contain four values in {path}")
    metrics = {
        str(key): float(value)
        for key, value in metrics_raw.items()
        if isinstance(value, (int, float))
    }
    minimum_x, minimum_y, maximum_x, maximum_y = (float(value) for value in bounds_raw)
    sample_id = str(payload.get("parcel_id") or path.stem)
    return DatasetSample(
        sample_id=sample_id,
        metrics=metrics,
        center_x=(minimum_x + maximum_x) / 2.0,
        center_y=(minimum_y + maximum_y) / 2.0,
    )


def discover_run_metrics(root: Path, *, limit: int | None = None) -> list[RunMetrics]:
    """Recursively discover and parse training metric logs."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    metric_paths = sorted(root.rglob("metrics.csv"))
    if limit is not None:
        metric_paths = metric_paths[:limit]
    runs = [_load_run_metrics(path, root) for path in metric_paths]
    if not runs:
        raise ValueError(f"No metrics.csv files found under {root}")
    return runs


def _load_run_metrics(metrics_path: Path, root: Path) -> RunMetrics:
    """Parse sparse train/validation rows from a metrics CSV file."""
    values: dict[str, list[tuple[int, float]]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = _parse_step(row.get("step"))
            if step is None:
                continue
            _collect_numeric_row(values, row, step)
    series = {
        name: MetricSeries(
            steps=tuple(item[0] for item in items),
            values=tuple(item[1] for item in items),
        )
        for name, items in values.items()
    }
    return RunMetrics(name=metrics_path.parent.name, path=metrics_path.parent, series=series)


def _parse_step(raw_value: str | None) -> int | None:
    """Convert a CSV step value, treating empty values as absent."""
    if raw_value in (None, ""):
        return None
    return int(float(raw_value))


def _collect_numeric_row(
    values: dict[str, list[tuple[int, float]]],
    row: dict[str, str | None],
    step: int,
) -> None:
    """Append all finite numeric cells from a CSV row."""
    for name, raw_value in row.items():
        if name == "step" or raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        values.setdefault(name, []).append((step, value))
