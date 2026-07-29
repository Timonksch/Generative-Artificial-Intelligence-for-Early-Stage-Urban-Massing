#!/usr/bin/env python3
"""Quantify spatial leakage between split windows.

The script checks how many samples from one split have a 2D 128 m context
window that overlaps at least one sample window from another split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOUNDING_BOX_VALUE_COUNT = 4


@dataclass(frozen=True)
class Window:
    """A dataset sample context window in projected coordinates."""

    sample_id: str
    bbox: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the centre point of the window."""
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure 2D window overlap between eval and train splits.",
    )
    parser.add_argument(
        "--manifest",
        default="00_Data/02_GeneratedDatasets/dataset_thesis/report_split.json",
        help="Path to split manifest JSON.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset root with one subdirectory per sample. Defaults to manifest parent.",
    )
    parser.add_argument("--train_split", default="train", help="Reference split name.")
    parser.add_argument("--eval_split", default="test", help="Evaluated split name.")
    parser.add_argument(
        "--window_m",
        type=float,
        default=None,
        help="Optional centroid window size in metres. Defaults to stored world_bbox_xy.",
    )
    parser.add_argument(
        "--min_overlap_area_m2",
        type=float,
        default=0.0,
        help="Minimum positive area threshold. Default counts any area greater than 0.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for the aggregate result.",
    )
    parser.add_argument(
        "--details_csv",
        default=None,
        help="Optional CSV output path with one row per eval sample.",
    )
    return parser.parse_args()


def split_sample_ids(manifest: dict[str, Any], split: str) -> list[str]:
    """Return sample identifiers for one split from a split manifest."""
    raw_splits = manifest.get("splits", manifest)
    if split not in raw_splits:
        raise KeyError(f"Split '{split}' not found in manifest")
    entry = raw_splits[split]
    if isinstance(entry, dict) and "sample_ids" in entry:
        return list(entry["sample_ids"])
    if isinstance(entry, list):
        return list(entry)
    raise ValueError(f"Split '{split}' must be a list or contain 'sample_ids'")


def load_window(dataset_dir: Path, sample_id: str, window_m: float | None) -> Window:
    """Load one sample context window from its metadata file."""
    meta_path = dataset_dir / sample_id / f"{sample_id}.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if "world_bbox_xy" not in meta:
        raise KeyError(f"{meta_path} does not contain world_bbox_xy")

    bbox = tuple(float(v) for v in meta["world_bbox_xy"])
    if len(bbox) != BOUNDING_BOX_VALUE_COUNT:
        raise ValueError(f"{meta_path} has invalid world_bbox_xy: {bbox}")

    if window_m is not None:
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        half = window_m * 0.5
        bbox = (cx - half, cy - half, cx + half, cy + half)

    return Window(sample_id=sample_id, bbox=bbox)


def overlap_area(a: Window, b: Window) -> float:
    """Return the 2D intersection area between two windows."""
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return width * height


def centroid_distance(a: Window, b: Window) -> float:
    """Return the Euclidean distance between two window centroids."""
    ax, ay = a.centroid
    bx, by = b.centroid
    return math.hypot(ax - bx, ay - by)


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Return a percentile from already sorted values."""
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float | None]:
    """Return a compact five-number summary for numeric values."""
    sorted_values = sorted(values)
    return {
        "min": percentile(sorted_values, 0.0),
        "q25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.5),
        "q75": percentile(sorted_values, 0.75),
        "max": percentile(sorted_values, 1.0),
    }


def analyze(
    train_windows: list[Window],
    eval_windows: list[Window],
    min_overlap_area_m2: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare evaluation windows against train windows."""
    detail_rows: list[dict[str, Any]] = []

    for eval_window in eval_windows:
        overlapping_train_windows = 0
        max_overlap = 0.0
        example_train_id = None
        nearest_train_id = None
        nearest_train_centroid_dist_m = float("inf")

        for train_window in train_windows:
            distance = centroid_distance(eval_window, train_window)
            if distance < nearest_train_centroid_dist_m:
                nearest_train_centroid_dist_m = distance
                nearest_train_id = train_window.sample_id

            area = overlap_area(eval_window, train_window)
            if area > min_overlap_area_m2:
                overlapping_train_windows += 1
                if example_train_id is None:
                    example_train_id = train_window.sample_id
                max_overlap = max(max_overlap, area)

        detail_rows.append(
            {
                "sample_id": eval_window.sample_id,
                "overlaps_train": overlapping_train_windows > 0,
                "overlapping_train_windows": overlapping_train_windows,
                "max_overlap_area_m2": max_overlap,
                "example_train_id": example_train_id,
                "nearest_train_id": nearest_train_id,
                "nearest_train_centroid_dist_m": nearest_train_centroid_dist_m,
            }
        )

    leaky_rows = [row for row in detail_rows if row["overlaps_train"]]
    overlap_counts = [float(row["overlapping_train_windows"]) for row in leaky_rows]
    max_areas = [float(row["max_overlap_area_m2"]) for row in leaky_rows]
    nearest_distances = [float(row["nearest_train_centroid_dist_m"]) for row in detail_rows]
    eval_total = len(eval_windows)
    overlap_count = len(leaky_rows)

    result = {
        "eval_total": eval_total,
        "overlap_count": overlap_count,
        "overlap_fraction": overlap_count / eval_total if eval_total else None,
        "overlap_percent": 100.0 * overlap_count / eval_total if eval_total else None,
        "overlap_free_count": eval_total - overlap_count,
        "overlap_free_fraction": (eval_total - overlap_count) / eval_total if eval_total else None,
        "overlap_free_percent": (
            100.0 * (eval_total - overlap_count) / eval_total if eval_total else None
        ),
        "overlap_count_per_overlapping_eval_window": summarize(overlap_counts),
        "max_overlap_area_m2_per_overlapping_eval_window": summarize(max_areas),
        "nearest_train_centroid_distance_m": summarize(nearest_distances),
    }
    return result, detail_rows


def write_details_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-sample leakage details as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "overlaps_train",
        "overlapping_train_windows",
        "max_overlap_area_m2",
        "example_train_id",
        "nearest_train_id",
        "nearest_train_centroid_dist_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the spatial leakage analysis command."""
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else manifest_path.parent
    )
    train_ids = split_sample_ids(manifest, args.train_split)
    eval_ids = split_sample_ids(manifest, args.eval_split)

    train_windows = [load_window(dataset_dir, sample_id, args.window_m) for sample_id in train_ids]
    eval_windows = [load_window(dataset_dir, sample_id, args.window_m) for sample_id in eval_ids]

    result, rows = analyze(train_windows, eval_windows, args.min_overlap_area_m2)
    result.update(
        {
            "manifest": str(manifest_path),
            "dataset_dir": str(dataset_dir),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "train_total": len(train_windows),
            "window_m": args.window_m,
            "min_overlap_area_m2": args.min_overlap_area_m2,
        }
    )

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.details_csv:
        write_details_csv(Path(args.details_csv).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
