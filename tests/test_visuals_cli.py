"""Contract tests for the consolidated visualization CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conftest import VISUALS_DIRECTORY, run_repository_command

CLI_PATH = VISUALS_DIRECTORY / "visuals_cli.py"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the visualization CLI in an isolated subprocess."""
    return run_repository_command(CLI_PATH, *arguments, timeout=30)


def _write_sample(dataset_directory: Path, sample_id: str) -> None:
    """Create minimal, valid dataset metadata for a CLI test."""
    sample_directory = dataset_directory / sample_id
    sample_directory.mkdir(parents=True)
    payload = {
        "parcel_id": sample_id,
        "world_bbox_xy": [100.0, 200.0, 228.0, 328.0],
        "metrics": {
            "grz_target": 0.35,
            "gfz_target": 1.1,
            "target_height_m": 12.0,
            "parcel_area_m2": 750.0,
            "target_footprint_area_m2": 250.0,
            "num_neighbors": 8,
        },
    }
    (sample_directory / f"{sample_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_districts(path: Path) -> None:
    """Create a minimal projected GeoJSON outline for density-surface tests."""
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:25833"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 100.0],
                            [320.0, 100.0],
                            [320.0, 420.0],
                            [0.0, 420.0],
                            [0.0, 100.0],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_help_lists_maintained_commands() -> None:
    """The public help must expose all maintained workflows."""
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "dataset" in result.stdout
    assert "experiments" in result.stdout
    assert "render-prediction" in result.stdout


def test_dataset_command_writes_canonical_outputs(tmp_path: Path) -> None:
    """Dataset figures must include split-aware density outputs when available."""
    dataset_directory = tmp_path / "dataset"
    output_directory = tmp_path / "figures"
    districts_path = tmp_path / "districts.geojson"
    _write_sample(dataset_directory, "sample_a")
    _write_sample(dataset_directory, "sample_b")
    _write_districts(districts_path)
    (dataset_directory / "report_split.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"sample_ids": ["sample_a"]},
                    "test": {"sample_ids": ["sample_b"]},
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "dataset",
        "--dataset",
        str(dataset_directory),
        "--output",
        str(output_directory),
        "--districts",
        str(districts_path),
    )

    assert result.returncode == 0, result.stderr
    assert (output_directory / "regulatory_distributions.png").is_file()
    assert (output_directory / "parcel_neighbor_distributions.svg").is_file()
    assert (output_directory / "dataset_bucket_distribution.svg").is_file()
    assert (output_directory / "dataset_bucket_split_profile.png").is_file()
    assert (output_directory / "dataset_density_surfaces_3d.png").is_file()
    assert (output_directory / "dataset_density_surface_3d_train.svg").is_file()
    assert (output_directory / "dataset_summary.json").is_file()


def test_experiments_command_discovers_nested_runs(tmp_path: Path) -> None:
    """Experiment discovery must support arbitrary nested output trees."""
    run_directory = tmp_path / "runs" / "experiment" / "run_a"
    run_directory.mkdir(parents=True)
    (run_directory / "metrics.csv").write_text(
        "step,train_loss,val_loss,val_iou\n0,1.0,,\n0,,0.8,0.2\n1,0.6,,\n1,,0.5,0.4\n",
        encoding="utf-8",
    )
    regulatory_directory = run_directory / "eval_regulatory_test"
    regulatory_directory.mkdir()
    (regulatory_directory / "reg_metrics.json").write_text(
        json.dumps(
            {
                "variants": {
                    "gt": {
                        "grz": {
                            "target": {"mean": 0.35},
                            "pred": {"mean": 0.33},
                        },
                        "gfz": {
                            "target": {"mean": 1.2},
                            "pred": {"mean": 1.1},
                        },
                        "height_m": {
                            "target": {"mean": 12.0},
                            "pred": {"mean": 11.0},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output_directory = tmp_path / "experiment_figures"

    result = _run_cli(
        "experiments",
        "--runs-root",
        str(tmp_path / "runs"),
        "--output",
        str(output_directory),
    )

    assert result.returncode == 0, result.stderr
    assert (output_directory / "training_overview.png").is_file()
    assert (output_directory / "run_ranking.svg").is_file()
    assert (output_directory / "final_regulatory_errors.svg").is_file()
    assert (output_directory / "regulatory_target_scatter.svg").is_file()
    assert (output_directory / "run_summary.json").is_file()


def test_missing_input_returns_nonzero_status(tmp_path: Path) -> None:
    """Invalid command inputs must fail visibly for CI callers."""
    result = _run_cli("dataset", "--dataset", str(tmp_path / "missing"))

    assert result.returncode != 0
    assert "does not exist" in result.stderr
