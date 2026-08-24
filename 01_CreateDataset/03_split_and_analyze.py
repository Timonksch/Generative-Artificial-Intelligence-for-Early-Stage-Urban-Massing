#!/usr/bin/env python3
"""Provide a guided front-end for dataset analysis and deterministic splitting."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
ANALYZE_SCRIPT: Final = Path(__file__).resolve().with_name("03_analyze_dataset.py")
DEFAULT_DATASET_DIRECTORY: Final = REPOSITORY_ROOT / "00_Data" / "02_GeneratedDatasets"


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    """Store normalized options for one analysis run."""

    dataset_directory: Path
    mode: str
    output_path: Path
    plot_directory: Path | None
    summary_path: Path | None
    seed: int | None


def prompt(text: str, default: str | None = None) -> str:
    """Read one input value with an optional default.

    Args:
        text: Prompt shown to the user.
        default: Value returned for empty input.

    Returns:
        Entered or default text.

    """
    suffix = f" [{default}]" if default is not None else ""
    try:
        response = input(f"{text}{suffix}: ").strip()
    except EOFError:
        response = ""
    return response or default or ""


def prompt_bool(text: str, *, default: bool = True) -> bool:
    """Read one yes/no answer.

    Args:
        text: Prompt shown to the user.
        default: Value returned for empty input.

    Returns:
        Parsed Boolean answer.

    """
    suffix = "Y/n" if default else "y/N"
    answer = prompt(f"{text} [{suffix}]").casefold()
    if not answer:
        return default
    return answer in {"y", "yes", "j", "ja"}


def _parse_seed(raw_seed: str) -> int:
    """Parse a seed and fall back to the documented default.

    Args:
        raw_seed: User-provided seed text.

    Returns:
        Parsed seed or 42 for invalid input.

    """
    try:
        return int(raw_seed)
    except ValueError:
        print(f"Invalid seed {raw_seed!r}; using 42.")
        return 42


def _relative_to_dataset(value: str, dataset_directory: Path) -> Path:
    """Resolve a potentially relative output below the dataset directory.

    Args:
        value: Absolute or relative output path.
        dataset_directory: Base directory for relative values.

    Returns:
        Absolute normalized path.

    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (dataset_directory / path).resolve()


def collect_options() -> AnalysisOptions:
    """Collect and normalize all guided analysis options.

    Returns:
        Complete analysis options.

    """
    dataset_text = prompt("Dataset directory", str(DEFAULT_DATASET_DIRECTORY))
    dataset_directory = Path(dataset_text).expanduser().resolve()
    if not dataset_directory.is_dir():
        print(f"Warning: dataset directory does not exist: {dataset_directory}")

    mode_choice = prompt("Mode: [1] analyze, [2] split and analyze", "1")
    mode = "split_analyze" if mode_choice == "2" else "analyze"
    seed = _parse_seed(prompt("Random split seed", "42")) if mode == "split_analyze" else None
    generate_plots = prompt_bool("Generate plots?", default=True)
    default_report = "report_split.json" if mode == "split_analyze" else "report.json"
    output_path = _relative_to_dataset(prompt("Output JSON", default_report), dataset_directory)

    if not generate_plots:
        return AnalysisOptions(dataset_directory, mode, output_path, None, None, seed)
    plot_directory = _relative_to_dataset(
        prompt("Plot directory", "analysis_plots"), dataset_directory
    )
    summary_name = "summary_split.png" if mode == "split_analyze" else "summary.png"
    summary_path = _relative_to_dataset(prompt("Summary plot", summary_name), plot_directory)
    return AnalysisOptions(dataset_directory, mode, output_path, plot_directory, summary_path, seed)


def build_command(options: AnalysisOptions) -> list[str]:
    """Build the analyzer subprocess command.

    Args:
        options: Normalized analysis options.

    Returns:
        Subprocess arguments without shell syntax.

    """
    command = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--dataset_dir",
        str(options.dataset_directory),
        "--mode",
        options.mode,
        "--output",
        str(options.output_path),
    ]
    if options.seed is not None:
        command.extend(("--seed", str(options.seed)))
    if options.plot_directory is not None:
        command.extend(("--plot", "--plot_dir", str(options.plot_directory)))
    if options.summary_path is not None:
        command.extend(("--summary_plot", str(options.summary_path)))
    return command


def run_analysis(options: AnalysisOptions) -> int:
    """Run the analyzer with guided options.

    Args:
        options: Normalized analysis options.

    Returns:
        Analyzer exit code.

    Raises:
        FileNotFoundError: If the analyzer script is absent.

    """
    if not ANALYZE_SCRIPT.is_file():
        raise FileNotFoundError(f"Analysis script not found: {ANALYZE_SCRIPT}")
    command = build_command(options)
    print(f"\nLaunching: {shlex.join(command)}\n", flush=True)
    # Fixed interpreter/script paths and no shell prevent command injection.
    return subprocess.run(command, check=False).returncode  # noqa: S603


def main() -> int:
    """Run the guided analysis workflow.

    Returns:
        Zero when cancelled or the analyzer exit code.

    """
    try:
        return run_analysis(collect_options())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
