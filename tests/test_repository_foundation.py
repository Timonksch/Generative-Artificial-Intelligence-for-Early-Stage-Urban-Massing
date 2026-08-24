"""Validate repository metadata and machine-readable thesis configurations.

These foundation tests intentionally use only the Python standard library so
they can run before heavyweight geospatial and machine-learning dependencies
are installed. Domain-specific tests will be added during module refactoring.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CREATE_DATASET_DIRECTORY,
    EXAMPLE_DATASET_DIRECTORY,
    EXPECTED_EXAMPLE_SAMPLE_COUNT,
    REPOSITORY_ROOT,
    TRAIN_MODELS_DIRECTORY,
    paired_sample_paths,
    run_repository_command,
)

REQUIRED_FILES = (
    ".editorconfig",
    ".gitignore",
    ".python-version",
    "ARTIFACTS.md",
    "artifacts_manifest.json",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATA_LICENSE.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements-dataset.txt",
    "requirements-dev.txt",
    "requirements-training.txt",
    "requirements.txt",
    "tests/README.md",
)
EXPECTED_THESIS_CONFIG_COUNT = 23
MINIMUM_README_CHARACTERS = 200
TRAIN_MODEL_DOCUMENTED_DIRECTORIES = (
    "02_TrainModels",
    "02_TrainModels/configs",
    "02_TrainModels/configs/thesis_runs",
    "02_TrainModels/configs/thesis_runs/phase1",
    "02_TrainModels/configs/thesis_runs/phase2_cond",
    "02_TrainModels/configs/thesis_runs/phase3_ldm",
    "02_TrainModels/dataio",
    "02_TrainModels/engine",
    "02_TrainModels/metrics",
    "02_TrainModels/models",
    "02_TrainModels/models/common",
    "02_TrainModels/models/ldm",
    "02_TrainModels/models/unet",
    "02_TrainModels/scripts",
    "02_TrainModels/utils",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require a top-level object.

    Args:
        path: JSON file to read.

    Returns:
        Parsed top-level JSON object.

    Raises:
        TypeError: If the document does not contain a top-level object.
        json.JSONDecodeError: If the document is not valid JSON.

    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def _assert_relative_path(config_path: Path, field_name: str, value: object) -> None:
    """Require a populated config path field to be repository-relative.

    Args:
        config_path: Configuration file containing the field.
        field_name: Name used to identify the field in assertion messages.
        value: Field value to validate. Empty optional values are accepted.

    Returns:
        None.

    Raises:
        AssertionError: If a populated value is not a string or is absolute.

    """
    if value in (None, ""):
        return

    assert isinstance(value, str), f"{config_path}: {field_name} must be a string"
    assert not Path(value).expanduser().is_absolute(), (
        f"{config_path}: {field_name} must be relative, got {value!r}"
    )


def test_required_repository_files_exist() -> None:
    """Verify that every repository-foundation document exists."""
    missing_files = [name for name in REQUIRED_FILES if not (REPOSITORY_ROOT / name).is_file()]
    assert not missing_files, f"Missing repository files: {missing_files}"


def test_committed_smoke_dataset_is_complete() -> None:
    """Require the curated dataset used by integration tests and examples."""
    npz_paths, json_paths = paired_sample_paths(EXAMPLE_DATASET_DIRECTORY)

    assert len(npz_paths) == EXPECTED_EXAMPLE_SAMPLE_COUNT
    assert len(json_paths) == EXPECTED_EXAMPLE_SAMPLE_COUNT
    assert {path.stem for path in npz_paths} == {path.stem for path in json_paths}


def test_python_sources_parse() -> None:
    """Parse all project Python files to detect syntax errors without importing them."""
    source_paths = sorted(REPOSITORY_ROOT.rglob("*.py"))
    assert source_paths, "Repository does not contain Python source files"

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(source_path))


def test_input_downloader_dry_run() -> None:
    """Verify that the official-data download plan needs no network access."""
    result = run_repository_command(
        CREATE_DATASET_DIRECTORY / "00_download_input_data.py",
        "--datasets",
        "lod1",
        "parcels",
        "--prepare-lod1",
        "--dry-run",
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "LoD1.zip" in result.stdout
    assert "flurstuecke.geojson" in result.stdout
    assert "berlin_lod1_merged.gml.gz" in result.stdout


def _thesis_config_paths() -> list[Path]:
    """Return every versioned thesis experiment configuration."""
    config_root = TRAIN_MODELS_DIRECTORY / "configs" / "thesis_runs"
    return sorted(config_root.rglob("*.json"))


@pytest.mark.parametrize("config_path", _thesis_config_paths())
def test_thesis_configs_are_valid_and_portable(config_path: Path) -> None:
    """Validate the common structure and path portability of a thesis config.

    Args:
        config_path: Experiment configuration selected by pytest.

    Returns:
        None.

    """
    payload = _load_json_object(config_path)

    assert isinstance(payload.get("name"), str) and payload["name"], (
        f"{config_path}: missing experiment name"
    )
    assert isinstance(payload.get("global"), dict), f"{config_path}: missing global object"
    assert isinstance(payload.get("overrides"), list), f"{config_path}: missing overrides list"
    assert payload["overrides"], f"{config_path}: overrides must not be empty"

    global_config = payload["global"]
    _assert_relative_path(config_path, "data_root", global_config.get("data_root"))
    _assert_relative_path(config_path, "split_manifest", global_config.get("split_manifest"))
    _assert_relative_path(config_path, "vae_checkpoint", global_config.get("vae_checkpoint"))


def test_thesis_config_collection_is_present() -> None:
    """Guard against accidentally publishing the repository without its configs."""
    config_paths = _thesis_config_paths()
    assert len(config_paths) == EXPECTED_THESIS_CONFIG_COUNT, (
        f"Expected {EXPECTED_THESIS_CONFIG_COUNT} thesis configs, found {len(config_paths)}"
    )


def test_train_model_documentation_is_visible_and_nonempty() -> None:
    """Require visible substantive documentation for every model-code area."""
    missing_or_short: list[str] = []
    for relative_directory in TRAIN_MODEL_DOCUMENTED_DIRECTORIES:
        readme_path = REPOSITORY_ROOT / relative_directory / "README.md"
        if (
            not readme_path.is_file()
            or len(readme_path.read_text(encoding="utf-8")) < MINIMUM_README_CHARACTERS
        ):
            missing_or_short.append(str(readme_path.relative_to(REPOSITORY_ROOT)))

    hidden_readmes = sorted(
        path.relative_to(REPOSITORY_ROOT)
        for path in (REPOSITORY_ROOT / "02_TrainModels").rglob(".README*")
    )
    assert not missing_or_short, f"Missing or insufficient READMEs: {missing_or_short}"
    assert not hidden_readmes, f"Hidden model documentation remains: {hidden_readmes}"


def test_training_example_config_is_populated() -> None:
    """Require a runnable, nonempty example for the experiment interface."""
    example_path = TRAIN_MODELS_DIRECTORY / "configs" / "example_smoke.json"
    payload = _load_json_object(example_path)

    assert payload.get("name") == "smoke_unet"
    assert isinstance(payload.get("global"), dict) and payload["global"]
    assert isinstance(payload.get("overrides"), list) and payload["overrides"]
