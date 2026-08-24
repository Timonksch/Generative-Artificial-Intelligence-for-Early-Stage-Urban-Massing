#!/usr/bin/env python3
"""Provide one stable command-line entry point for dataset creation tools.

The numbered scripts remain executable and document the processing order. This
module adds a discoverable command layer without duplicating their scientific
implementation or changing their existing command-line contracts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SCRIPT_DIRECTORY: Final = Path(__file__).resolve().parent
REPOSITORY_ROOT: Final = SCRIPT_DIRECTORY.parent


@dataclass(frozen=True, slots=True)
class Command:
    """Describe a command exposed by the central dataset CLI.

    Attributes:
        script: Script path relative to ``01_CreateDataset``.
        description: Short user-facing command description.
        module: Optional module invocation used instead of a script path.

    """

    script: Path
    description: str
    module: str | None = None


COMMANDS: Final[dict[str, Command]] = {
    "download": Command(Path("00_download_input_data.py"), "Download official input data."),
    "merge": Command(Path("01_merge_citygml.py"), "Merge LoD1 CityGML source tiles."),
    "create": Command(Path("02_create_dataset.py"), "Create voxelized parcel samples."),
    "analyze": Command(Path("03_analyze_dataset.py"), "Analyze and split a dataset."),
    "split": Command(Path("03_split_and_analyze.py"), "Run the guided split workflow."),
    "validate": Command(Path("04_validate_dataset.py"), "Validate samples and metadata."),
    "metrics": Command(Path("05_recompute_voxel_metrics.py"), "Recompute voxel metrics."),
    "inspect": Command(
        Path("pipeline/data_inspector.py"),
        "Inspect one NPZ sample.",
        module="pipeline.data_inspector",
    ),
}


def _print_help() -> None:
    """Print central CLI usage and all available commands."""
    print("Usage: python 01_CreateDataset/dataset_cli.py COMMAND [OPTIONS]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, command in COMMANDS.items():
        print(f"  {name:<{width}}  {command.description}")
    print(f"  {'status':<{width}}  Show whether expected local inputs are available.")
    print("\nRun COMMAND --help to see command-specific options.")


def _input_status() -> int:
    """Print the availability of expected input files.

    Returns:
        Zero. Missing input data is reported as state, not as a CLI failure.

    """
    inputs = REPOSITORY_ROOT / "00_Data" / "01_InputData" / "input"
    expected = {
        "LoD1 CityGML": inputs / "berlin_lod1_merged.gml.gz",
        "parcels": inputs / "flurstuecke.geojson",
    }
    print(f"Repository: {REPOSITORY_ROOT}")
    for label, path in expected.items():
        state = "ready" if path.is_file() else "missing"
        print(f"[{state:7}] {label}: {path.relative_to(REPOSITORY_ROOT)}")
    print("\nSetup: dataset_cli.py download --datasets lod1 parcels --prepare-lod1")
    return 0


def _run_command(name: str, arguments: Sequence[str]) -> int:
    """Execute one numbered script with the current Python interpreter.

    Args:
        name: Command name from ``COMMANDS``.
        arguments: Unmodified command-specific arguments.

    Returns:
        Exit code returned by the selected script.

    Raises:
        RuntimeError: If the registered script is missing.

    """
    command = COMMANDS[name]
    script = SCRIPT_DIRECTORY / command.script
    if not script.is_file():
        raise RuntimeError(f"Registered command script does not exist: {script}")

    # The executable and script are fixed local paths; user arguments are never passed to a shell.
    target = ["-m", command.module] if command.module else [str(script)]
    environment = None
    if command.module:
        environment = os.environ.copy()
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{SCRIPT_DIRECTORY}{os.pathsep}{existing_path}"
            if existing_path
            else str(SCRIPT_DIRECTORY)
        )
    completed = subprocess.run(  # noqa: S603 - no shell and no executable lookup
        [sys.executable, *target, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one central dataset command.

    Args:
        argv: Optional arguments without the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        Zero for help/status or the selected command's exit code.

    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    name, *command_arguments = arguments
    if name == "status":
        return _input_status()
    if name not in COMMANDS:
        print(f"Unknown command: {name!r}\n", file=sys.stderr)
        _print_help()
        return 2
    return _run_command(name, command_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
