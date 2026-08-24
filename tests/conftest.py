"""Shared test configuration and repository-local helpers.

The tests exercise public command-line interfaces from subprocesses and import
selected internals from numbered source directories. This module centralizes
repository paths so every test uses the same reproducibility contract.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CREATE_DATASET_DIRECTORY = REPOSITORY_ROOT / "01_CreateDataset"
TRAIN_MODELS_DIRECTORY = REPOSITORY_ROOT / "02_TrainModels"
VISUALS_DIRECTORY = REPOSITORY_ROOT / "03_visuals"
GENERATED_DATASETS_DIRECTORY = REPOSITORY_ROOT / "00_Data" / "02_GeneratedDatasets"
EXAMPLE_DATASET_DIRECTORY = GENERATED_DATASETS_DIRECTORY / "example_smoke"
EXPECTED_EXAMPLE_SAMPLE_COUNT = 30


def _prepend_import_path(path: Path) -> None:
    """Make a repository source directory importable exactly once.

    Args:
        path: Source directory to add before third-party imports.

    Returns:
        None.

    """
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def run_repository_command(
    script_path: Path,
    *arguments: str,
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    """Run a repository-controlled Python script from the repository root.

    Args:
        script_path: Python script path below the repository root.
        arguments: Command-line arguments passed to the script.
        timeout: Maximum subprocess runtime in seconds.

    Returns:
        Completed subprocess result with captured text streams.

    Raises:
        ValueError: If the script is outside this repository.

    """
    resolved_script = script_path.resolve()
    if not resolved_script.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Script must be inside repository: {script_path}")

    command = [sys.executable, str(resolved_script), *arguments]
    return subprocess.run(  # noqa: S603 - command is built from repository-controlled scripts.
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def paired_sample_paths(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Return sorted NPZ and JSON sample files from a generated dataset root.

    Args:
        dataset_root: Directory containing one subdirectory per sample.

    Returns:
        Two sorted lists: NPZ sample files and adjacent JSON metadata files.

    """
    return sorted(dataset_root.glob("*/*.npz")), sorted(dataset_root.glob("*/*.json"))


_prepend_import_path(CREATE_DATASET_DIRECTORY)
