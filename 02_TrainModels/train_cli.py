#!/usr/bin/env python3
"""Provide one stable command-line entry point for model workflows.

The scripts directory contains the scientific implementations. This module
exposes them through one discoverable interface while preserving every
command-specific argument contract.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

TRAIN_DIRECTORY: Final = Path(__file__).resolve().parent
REPOSITORY_ROOT: Final = TRAIN_DIRECTORY.parent
CONFIG_DIRECTORY: Final = TRAIN_DIRECTORY / "configs" / "thesis_runs"
DEFAULT_DATA_DIRECTORY: Final = (
    REPOSITORY_ROOT / "00_Data" / "02_GeneratedDatasets" / "dataset_thesis"
)
SMOKE_DATA_DIRECTORY: Final = REPOSITORY_ROOT / "00_Data" / "02_GeneratedDatasets" / "example_smoke"


@dataclass(frozen=True, slots=True)
class Command:
    """Describe one command exposed by the central training CLI.

    Attributes:
        script: Script path relative to ``02_TrainModels``.
        description: Short user-facing command description.

    """

    script: Path
    description: str


COMMANDS: Final[dict[str, Command]] = {
    "experiment": Command(
        Path("scripts/experiment.py"),
        "Run one or more trainings from a JSON experiment config.",
    ),
    "train-unet": Command(
        Path("scripts/train_unet.py"),
        "Train the unconditional 3D U-Net.",
    ),
    "train-cond-unet": Command(
        Path("scripts/train_unet_cond.py"),
        "Train the condition-controlled 3D U-Net.",
    ),
    "train-ldm": Command(
        Path("scripts/train_ldm.py"),
        "Train the VAE or latent diffusion model.",
    ),
    "infer": Command(
        Path("scripts/infer.py"),
        "Run unified inference for a trained model.",
    ),
    "evaluate": Command(
        Path("scripts/eval_regulatory.py"),
        "Evaluate voxel and regulatory target metrics.",
    ),
    "evaluate-vae": Command(
        Path("scripts/eval_vae_reconstruction.py"),
        "Evaluate VAE reconstruction quality.",
    ),
    "sample-cond": Command(
        Path("scripts/sample_cond_unet_controls.py"),
        "Render conditioning controls for a trained U-Net.",
    ),
    "sample-ldm": Command(
        Path("scripts/sample_best_ldm_experiment.py"),
        "Sample variants from the best LDM experiment run.",
    ),
    "smoke-models": Command(
        Path("scripts/smoke_visualize_models.py"),
        "Mini-train all model families and save diagnostic visualizations.",
    ),
}


def _print_help() -> None:
    """Print central CLI usage and all available commands."""
    print("Usage: python 02_TrainModels/train_cli.py COMMAND [OPTIONS]\n")
    print("Commands:")
    descriptions = {
        **{name: command.description for name, command in COMMANDS.items()},
        "configs": "List the versioned thesis experiment configs.",
        "status": "Check the local training environment and default dataset.",
    }
    width = max(len(name) for name in descriptions)
    for name, description in descriptions.items():
        print(f"  {name:<{width}}  {description}")
    print("\nRun COMMAND --help to see command-specific options.")


def _list_configs() -> int:
    """Print versioned experiment configurations relative to the repository."""
    configs = sorted(CONFIG_DIRECTORY.rglob("*.json"))
    if not configs:
        print(f"No configs found below {CONFIG_DIRECTORY.relative_to(REPOSITORY_ROOT)}")
        return 1

    for config in configs:
        print(config.relative_to(REPOSITORY_ROOT))
    print(f"\n{len(configs)} experiment configs")
    return 0


def _environment_status() -> int:
    """Print the state of required local training inputs and packages."""
    print(f"Repository: {REPOSITORY_ROOT}")
    smoke_sample_count = sum(1 for _ in SMOKE_DATA_DIRECTORY.rglob("*.npz"))
    checks = {
        "default dataset": DEFAULT_DATA_DIRECTORY.is_dir(),
        f"smoke dataset ({smoke_sample_count} samples)": smoke_sample_count > 0,
        "experiment configs": any(CONFIG_DIRECTORY.rglob("*.json")),
        "PyTorch": importlib.util.find_spec("torch") is not None,
    }
    for label, ready in checks.items():
        state = "ready" if ready else "missing"
        print(f"[{state:7}] {label}")
    print(f"\nDefault dataset: {DEFAULT_DATA_DIRECTORY.relative_to(REPOSITORY_ROOT)}")
    print(f"Smoke dataset: {SMOKE_DATA_DIRECTORY.relative_to(REPOSITORY_ROOT)}")
    print("Setup: python -m pip install -r requirements-training.txt")
    return 0


def _child_environment() -> dict[str, str]:
    """Return an environment in which internal training modules are importable."""
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{TRAIN_DIRECTORY}{os.pathsep}{existing_path}" if existing_path else str(TRAIN_DIRECTORY)
    )
    return environment


def _run_command(name: str, arguments: Sequence[str]) -> int:
    """Execute one internal script using the current Python interpreter."""
    script = TRAIN_DIRECTORY / COMMANDS[name].script
    if not script.is_file():
        raise RuntimeError(f"Registered command script does not exist: {script}")

    completed = subprocess.run(  # noqa: S603 - fixed executable and script, no shell
        [sys.executable, str(script), *arguments],
        cwd=REPOSITORY_ROOT,
        env=_child_environment(),
        check=False,
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one central model command and return its exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    name, *command_arguments = arguments
    if name == "configs":
        return _list_configs()
    if name == "status":
        return _environment_status()
    if name not in COMMANDS:
        print(f"Unknown command: {name!r}\n", file=sys.stderr)
        _print_help()
        return 2
    return _run_command(name, command_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
