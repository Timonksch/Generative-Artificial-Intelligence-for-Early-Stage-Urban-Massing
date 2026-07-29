"""Test the central model CLI without initializing the training stack."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType

from conftest import (
    EXAMPLE_DATASET_DIRECTORY,
    REPOSITORY_ROOT,
    TRAIN_MODELS_DIRECTORY,
    paired_sample_paths,
    run_repository_command,
)

CLI_PATH = TRAIN_MODELS_DIRECTORY / "train_cli.py"
THESIS_CONFIG_ROOT = TRAIN_MODELS_DIRECTORY / "configs" / "thesis_runs"
USAGE_ERROR_EXIT_CODE = 2


def _run_cli(*arguments: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    """Run the central model CLI and capture its text output."""
    return run_repository_command(CLI_PATH, *arguments, timeout=timeout)


def _load_experiment_module() -> ModuleType:
    """Load the experiment runner without importing the training stack."""
    train_path = str(TRAIN_MODELS_DIRECTORY)
    if train_path not in sys.path:
        sys.path.insert(0, train_path)
    module_path = TRAIN_MODELS_DIRECTORY / "scripts" / "experiment.py"
    spec = importlib_util.spec_from_file_location("experiment_runner_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment module from {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_complete_public_interface() -> None:
    """Expose every supported workflow from one discoverable command."""
    result = _run_cli("--help")

    assert result.returncode == 0
    for command in (
        "experiment",
        "train-unet",
        "train-cond-unet",
        "train-ldm",
        "infer",
        "evaluate",
        "evaluate-vae",
        "sample-cond",
        "sample-ldm",
        "smoke-models",
        "configs",
        "status",
    ):
        assert command in result.stdout


def test_unknown_command_fails_with_usage() -> None:
    """Return argparse-compatible exit code 2 for unknown commands."""
    result = _run_cli("does-not-exist")

    assert result.returncode == USAGE_ERROR_EXIT_CODE
    assert "Unknown command" in result.stderr
    assert "Usage:" in result.stdout


def test_configs_lists_all_thesis_configs() -> None:
    """List exactly the versioned thesis experiment collection."""
    expected = sorted(THESIS_CONFIG_ROOT.rglob("*.json"))
    result = _run_cli("configs")

    assert result.returncode == 0
    assert f"{len(expected)} experiment configs" in result.stdout
    for config_path in expected:
        assert str(config_path.relative_to(REPOSITORY_ROOT)) in result.stdout


def test_status_is_read_only() -> None:
    """Report missing local assets as state rather than a command failure."""
    npz_paths, _json_paths = paired_sample_paths(EXAMPLE_DATASET_DIRECTORY)
    result = _run_cli("status")

    assert result.returncode == 0
    assert "default dataset" in result.stdout
    assert f"smoke dataset ({len(npz_paths)} samples)" in result.stdout
    assert "experiment configs" in result.stdout
    assert "PyTorch" in result.stdout


def test_experiment_dry_run_writes_resolved_summary(tmp_path: Path) -> None:
    """Expand the smoke experiment without importing PyTorch or training."""
    config_path = TRAIN_MODELS_DIRECTORY / "configs" / "example_smoke.json"
    result = _run_cli(
        "experiment",
        "--config",
        str(config_path),
        "--out-parent",
        str(tmp_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    summary_path = tmp_path / "smoke_unet" / "experiment_summary.json"
    assert summary_path.is_file()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["runs"][0]["status"] == "skipped"
    assert summary["runs"][0]["model"] == "unet"


def test_experiment_eval_command_respects_sample_limit() -> None:
    """Keep post-training evaluation bounded for smoke experiment configs."""
    experiment = _load_experiment_module()

    command = experiment.build_eval_command(
        python_exec=sys.executable,
        script=TRAIN_MODELS_DIRECTORY / "scripts" / "infer.py",
        model_kind="unet",
        checkpoint=Path("run") / "best.pt",
        config_path=Path("run") / "config.json",
        data_root="00_Data/02_GeneratedDatasets/dataset_thesis",
        out_dir=Path("run") / "eval_test",
        split="test",
        batch_size=1,
        num_workers=0,
        seed=42,
        max_samples=8,
    )

    assert "--max_samples" in command
    assert command[command.index("--max_samples") + 1] == "8"


def test_experiment_help_is_reachable_without_training_dependencies() -> None:
    """Forward command-specific help through the central dispatcher."""
    result = _run_cli("experiment", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
