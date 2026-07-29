"""Test the central dataset CLI and the standalone sample validator."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from conftest import CREATE_DATASET_DIRECTORY, run_repository_command
from lxml import etree

CLI_PATH = CREATE_DATASET_DIRECTORY / "dataset_cli.py"
VALIDATION_ERROR_CODE = 2
GML_NAMESPACE = "http://www.opengis.net/gml"
EXPECTED_TARGET_VOXELS = 8
ANALYSIS_SAMPLE_COUNT = 10


def _run_cli(*arguments: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Run the central dataset CLI.

    Args:
        arguments: Command and command-specific arguments.
        timeout: Maximum subprocess runtime in seconds.

    Returns:
        Completed subprocess result.

    """
    return run_repository_command(CLI_PATH, *arguments, timeout=timeout)


def test_central_cli_lists_all_pipeline_stages() -> None:
    """Expose every numbered processing stage through one help screen."""
    result = _run_cli("--help")

    assert result.returncode == 0
    for command in ("download", "merge", "create", "analyze", "validate", "metrics"):
        assert command in result.stdout


def test_central_cli_forwards_command_help() -> None:
    """Forward command-specific help without loading the processing pipeline."""
    result = _run_cli("create", "--help")

    assert result.returncode == 0
    assert "--interactive" in result.stdout
    assert "--config" in result.stdout


def test_central_cli_rejects_unknown_command() -> None:
    """Return an invocation error and show available commands."""
    result = _run_cli("does-not-exist")

    assert result.returncode == VALIDATION_ERROR_CODE
    assert "Unknown command" in result.stderr
    assert "download" in result.stdout


def test_central_cli_runs_downloader_dry_run() -> None:
    """Forward downloader arguments unchanged and without network access."""
    result = _run_cli("download", "--datasets", "parcels", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "flurstuecke.geojson" in result.stdout


def test_citygml_merger_unions_envelopes_and_rewrites_ids(tmp_path: Path) -> None:
    """Merge multiple tiles without collisions in IDs or local references."""
    fixture_directory = Path(__file__).resolve().parent / "fixtures" / "citygml"
    output_path = tmp_path / "merged.gml"

    result = _run_cli("merge", "--src", str(fixture_directory), "--out", str(output_path))

    assert result.returncode == 0, result.stderr
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.parse(output_path, parser).getroot()
    identifiers = [element.get(f"{{{GML_NAMESPACE}}}id") for element in root.iter()]
    populated_identifiers = [identifier for identifier in identifiers if identifier]
    assert len(populated_identifiers) == len(set(populated_identifiers))
    lower_corner = root.find(f".//{{{GML_NAMESPACE}}}lowerCorner")
    upper_corner = root.find(f".//{{{GML_NAMESPACE}}}upperCorner")
    assert lower_corner is not None and lower_corner.text == "0.0 0.0"
    assert upper_corner is not None and upper_corner.text == "20.0 20.0"


def test_validator_accepts_consistent_sample(tmp_path: Path) -> None:
    """Accept a minimal structurally consistent NPZ/JSON pair."""
    sample_directory = tmp_path / "sample-1"
    sample_directory.mkdir()
    np.savez_compressed(
        sample_directory / "sample-1.npz",
        X=np.zeros((3, 4, 4, 4), dtype=np.float32),
        Y=np.zeros((4, 4, 4), dtype=np.uint8),
    )
    metadata = {
        "parcel_id": "sample-1",
        "grid": {"D": 4, "H": 4, "W": 4},
        "channels": ["buildings", "parcels", "streets"],
    }
    (sample_directory / "sample-1.json").write_text(json.dumps(metadata), encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = _run_cli("validate", "--root", str(tmp_path), "--report", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["samples_discovered"] == 1
    assert report["error_count"] == 0


def test_inspector_prints_machine_readable_sample_summary(tmp_path: Path) -> None:
    """Inspect a complete sample without loading visualization dependencies."""
    metadata = {"parcel_id": "inspect-me", "grid": {"voxel_m": 1.0}, "metrics": {}}
    sample_path = tmp_path / "inspect-me.npz"
    np.savez_compressed(
        sample_path,
        X=np.zeros((3, 2, 2, 2), dtype=np.float32),
        Y=np.zeros((2, 2, 2), dtype=np.uint8),
        Y_neigh=np.zeros((2, 2, 2), dtype=np.uint8),
        meta=np.array(metadata, dtype=object),
    )

    result = _run_cli("inspect", str(sample_path), "--json")

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["sample_id"] == "inspect-me"
    assert summary["arrays"]["X"]["shape"] == [3, 2, 2, 2]


def test_validator_rejects_mismatched_target_shape(tmp_path: Path) -> None:
    """Return the documented validation-error exit code for shape mismatch."""
    np.savez_compressed(
        tmp_path / "invalid.npz",
        X=np.zeros((2, 4, 4, 4), dtype=np.float32),
        Y=np.zeros((3, 3, 3), dtype=np.uint8),
    )

    result = _run_cli("validate", "--root", str(tmp_path))

    assert result.returncode == VALIDATION_ERROR_CODE
    assert "voxel shape" in result.stdout


def test_metric_update_replaces_embedded_metadata_without_duplicates(tmp_path: Path) -> None:
    """Keep exactly one embedded metadata entry across repeated updates."""
    sample_directory = tmp_path / "metric-sample"
    sample_directory.mkdir()
    metadata = {
        "parcel_id": "metric-sample",
        "grid": {"D": 4, "H": 4, "W": 4, "voxel_m": 1.0},
        "metrics": {"parcel_area_m2": 16.0},
    }
    json_path = sample_directory / "metric-sample.json"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = np.zeros((4, 4, 4), dtype=np.uint8)
    target[:2, :2, :2] = 1
    npz_path = sample_directory / "metric-sample.npz"
    np.savez_compressed(npz_path, Y=target, meta=np.array(metadata, dtype=object))

    arguments = (
        "metrics",
        "--dataset-dir",
        str(tmp_path),
        "--storey-height-m",
        "3",
        "--target-voxel-m",
        "2",
    )
    first_result = _run_cli(*arguments)
    second_result = _run_cli(*arguments)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    with zipfile.ZipFile(npz_path) as archive:
        assert archive.namelist().count("meta.npy") == 1
    updated = json.loads(json_path.read_text(encoding="utf-8"))
    assert updated["metrics"]["target_voxels"] == EXPECTED_TARGET_VOXELS
    assert "voxel_metrics" in updated["metrics"]


def test_analyzer_writes_complete_deterministic_split_report(tmp_path: Path) -> None:
    """Create a report containing every sample in exactly one fixed split."""
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()
    for index in range(ANALYSIS_SAMPLE_COUNT):
        sample_id = f"sample-{index:02d}"
        sample_directory = dataset_directory / sample_id
        sample_directory.mkdir()
        np.savez_compressed(sample_directory / f"{sample_id}.npz", Y=np.zeros((2, 2, 2)))
        metadata = {
            "parcel_id": sample_id,
            "grid": {"grid_m": 2.0, "voxel_m": 1.0, "D": 2, "H": 2, "W": 2},
            "channels": ["C0_build_mask", "C1_neighbor_height", "C2_parcel_edges"],
            "metrics": {
                "target_voxels": 1,
                "num_neighbors": 3,
                "parcel_area_m2": 100.0 + index,
                "coverage_frac": 0.2,
                "target_height_m": 10.0,
                "grz_target": 0.2,
                "gfz_target": 0.8,
                "c1_mean": 0.1,
            },
        }
        (sample_directory / f"{sample_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    report_path = tmp_path / "split_report.json"

    result = _run_cli(
        "analyze",
        "--dataset-dir",
        str(dataset_directory),
        "--mode",
        "split_analyze",
        "--seed",
        "42",
        "--quiet",
        "--output",
        str(report_path),
        "--report-dir",
        str(tmp_path / "report"),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_ids = [
        sample_id for split in report["splits"].values() for sample_id in split["sample_ids"]
    ]
    assert len(split_ids) == ANALYSIS_SAMPLE_COUNT
    assert len(set(split_ids)) == ANALYSIS_SAMPLE_COUNT
