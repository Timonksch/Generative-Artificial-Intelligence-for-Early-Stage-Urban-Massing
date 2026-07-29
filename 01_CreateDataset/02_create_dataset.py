#!/usr/bin/env python3
"""Create voxel datasets from a JSON configuration or a guided dialog.

The scientific processing is implemented in :mod:`pipeline.pipeline`. This
script only validates configuration input and provides the user interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

MAX_PROMPT_ATTEMPTS: Final = 10
Config = dict[str, Any]


def _prompt(text: str, default: str | None = None) -> str:
    """Read one text value, applying an optional default.

    Args:
        text: Prompt shown to the user.
        default: Value returned for empty input.

    Returns:
        Entered or default text.

    Raises:
        ValueError: If the value is empty and no default is defined.

    """
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(f"A value is required for: {text}")


def _prompt_choice(text: str, choices: set[str], default: str | None = None) -> str:
    """Read a choice with a bounded number of attempts.

    Args:
        text: Prompt shown to the user.
        choices: Accepted normalized values.
        default: Optional default choice.

    Returns:
        A member of ``choices``.

    Raises:
        RuntimeError: If no valid answer is entered within the attempt limit.

    """
    for _attempt in range(MAX_PROMPT_ATTEMPTS):
        try:
            value = _prompt(text, default).lower()
        except ValueError as error:
            print(f"ERROR: {error}")
            continue
        if value in choices:
            return value
        print(f"ERROR: Choose one of: {', '.join(sorted(choices))}")
    raise RuntimeError(f"No valid answer after {MAX_PROMPT_ATTEMPTS} attempts")


def _prompt_bool(text: str, *, default: bool) -> bool:
    """Read a localized yes/no answer.

    Args:
        text: Prompt shown to the user.
        default: Boolean returned for empty input.

    Returns:
        Parsed Boolean answer.

    """
    marker = "Y/n" if default else "y/N"
    fallback = "yes" if default else "no"
    choices = {"y", "yes", "j", "ja", "n", "no", "nein"}
    answer = _prompt_choice(f"{text} [{marker}]", choices, fallback)
    return answer in {"y", "yes", "j", "ja"}


def _prompt_number(
    text: str,
    *,
    default: int | float,
    number_type: type[int] | type[float],
    minimum: float = 0.0,
) -> int | float:
    """Read and validate one numeric value.

    Args:
        text: Prompt shown to the user.
        default: Number returned for empty input.
        number_type: ``int`` or ``float`` conversion function.
        minimum: Inclusive lower bound.

    Returns:
        Parsed number of the requested type.

    Raises:
        RuntimeError: If no valid value is entered within the attempt limit.

    """
    for _attempt in range(MAX_PROMPT_ATTEMPTS):
        raw_value = input(f"{text} [{default}]: ").strip()
        try:
            value = number_type(raw_value) if raw_value else default
        except ValueError:
            print(f"ERROR: Expected a valid {number_type.__name__}")
            continue
        if value >= minimum:
            return value
        print(f"ERROR: Value must be at least {minimum}")
    raise RuntimeError(f"No valid number after {MAX_PROMPT_ATTEMPTS} attempts")


def _prompt_existing_path(text: str, *, directory: bool = False) -> Path:
    """Read an existing file or directory path.

    Args:
        text: Prompt shown to the user.
        directory: Require a directory instead of a file.

    Returns:
        Validated path without resolving it.

    Raises:
        RuntimeError: If no valid path is entered within the attempt limit.

    """
    for _attempt in range(MAX_PROMPT_ATTEMPTS):
        try:
            path = Path(_prompt(text)).expanduser()
        except ValueError as error:
            print(f"ERROR: {error}")
            continue
        valid = path.is_dir() if directory else path.is_file()
        if valid:
            return path
        kind = "directory" if directory else "file"
        print(f"ERROR: {kind.capitalize()} not found: {path}")
    raise RuntimeError(f"No valid path after {MAX_PROMPT_ATTEMPTS} attempts")


def _collect_input_config(mode: str) -> Config:
    """Collect input and output paths for one creation mode.

    Args:
        mode: Operation mode ``1``, ``2``, or ``3``.

    Returns:
        Partial pipeline configuration.

    """
    if mode == "3":
        source = _prompt_existing_path("Source dataset directory", directory=True)
        count = _prompt_number("Number of test samples", default=100, number_type=int, minimum=1)
        link_mode = _prompt_choice("Test dataset mode", {"symlink", "hardlink", "copy"}, "symlink")
        return {
            "out_dir": str(source),
            "create_testset_from_existing": count,
            "testset_mode": link_mode,
        }

    citygml = _prompt_existing_path("CityGML file (.gml, .xml, or .gml.gz)")
    parcels = _prompt_existing_path("Parcel GeoJSON file")
    config: Config = {"citygml": str(citygml), "parcels": str(parcels)}
    if mode == "1":
        config["out_dir"] = _prompt("Output dataset directory")
        return config

    reference = _prompt_existing_path("Reference training dataset directory", directory=True)
    config["reference_dataset_dir"] = str(reference)
    output = input("Test dataset output directory [automatic]: ").strip()
    if output:
        config["testset_output_dir"] = output
    config["generate_testset_from_remaining"] = _prompt_number(
        "Number of test samples", default=500, number_type=int, minimum=1
    )
    return config


def _collect_grid_config(mode: str) -> Config:
    """Collect grid, street, cache, and generation options.

    Args:
        mode: Operation mode ``1`` or ``2``.

    Returns:
        Partial pipeline configuration.

    """
    config: Config = {
        "grid_m": _prompt_number(
            "Grid extent in meters", default=128.0, number_type=float, minimum=1
        ),
        "grid_res": int(_prompt_choice("Grid resolution", {"128", "256"}, "256")),
    }
    config["with_streets"] = _prompt_bool("Include street masks?", default=True)
    if config["with_streets"]:
        config["street_mode"] = _prompt_choice("Street mode", {"buffer", "centerline"}, "buffer")
        config["street_width_m"] = _prompt_number(
            "Street width in meters", default=8.0, number_type=float, minimum=0.1
        )
        street_cache = input("Street cache directory [optional]: ").strip()
        if street_cache:
            config["street_cache_dir"] = street_cache

    if mode == "1":
        config["num_ok"] = _prompt_number(
            "Successful samples to generate", default=5000, number_type=int, minimum=1
        )
    if _prompt_bool("Limit candidates for a test run?", default=False):
        config["test"] = _prompt_number(
            "Candidates to test", default=100, number_type=int, minimum=1
        )
    config["cache_dir"] = _prompt("Cache directory", "cache")
    config["no_viz"] = _prompt_bool("Disable per-sample visualization?", default=False)
    config["verbose"] = _prompt_bool("Print detailed progress?", default=True)
    return config


def interactive_mode() -> Config:
    """Collect a complete pipeline configuration interactively.

    Returns:
        Validated pipeline configuration.

    """
    print("\nUrban Dataset Creator")
    print("1 = training dataset")
    print("2 = test dataset from remaining parcels")
    print("3 = test dataset from existing samples")
    mode = _prompt_choice("Mode", {"1", "2", "3"})

    config = _collect_input_config(mode)
    if mode in {"1", "2"}:
        config.update(_collect_grid_config(mode))
    config["seed"] = _prompt_number("Random seed", default=42, number_type=int)
    return config


def load_config_file(config_path: Path) -> Config:
    """Load and validate a JSON configuration object.

    Args:
        config_path: JSON file to read.

    Returns:
        Top-level configuration object.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the JSON root is not an object.
        json.JSONDecodeError: If the file is invalid JSON.

    """
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration root must be an object: {config_path}")
    return config


def save_config_file(config: Config, output_path: Path) -> None:
    """Write a configuration atomically as formatted JSON.

    Args:
        config: Pipeline configuration to serialize.
        output_path: Destination JSON file.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)
    print(f"Configuration saved to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--interactive", action="store_true", help="Run the guided dialog.")
    source.add_argument("--config", type=Path, help="Load a reusable JSON configuration.")
    parser.add_argument("--save-config", type=Path, help="Save an interactive configuration.")
    return parser


def _execute_pipeline(config: Config) -> None:
    """Load the heavyweight geospatial pipeline and execute it.

    Args:
        config: Validated pipeline configuration.

    """
    # Deferred so that `--help` and config errors do not initialize plotting/geospatial stacks.
    from pipeline.pipeline import run_pipeline  # noqa: PLC0415

    run_pipeline(config)


def main() -> None:
    """Load a configuration and execute the dataset pipeline."""
    parser = _build_parser()
    arguments = parser.parse_args()

    if arguments.interactive:
        config = interactive_mode()
        if arguments.save_config is not None:
            save_config_file(config, arguments.save_config)
    else:
        if arguments.save_config is not None:
            parser.error("--save-config can only be used with --interactive")
        config_path = arguments.config
        if config_path is None:  # Defensive guard for the argparse invariant.
            raise RuntimeError("Configuration path is missing")
        config = load_config_file(config_path)
        print(f"Loaded configuration from: {config_path}")

    print("Starting dataset creation pipeline")
    _execute_pipeline(config)


if __name__ == "__main__":
    main()
