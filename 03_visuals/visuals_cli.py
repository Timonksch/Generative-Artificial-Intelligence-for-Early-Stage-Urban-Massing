#!/usr/bin/env python3
"""Central command-line interface for all maintained visualizations."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from visuals.animation import create_gif
from visuals.dataset import generate_dataset_figures
from visuals.experiments import generate_experiment_figures
from visuals.paths import (
    DEFAULT_DATASET,
    DEFAULT_DISTRICTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNS_ROOT,
    prepare_output_directory,
    require_directory,
)
from visuals.samples import export_sample_figures

CommandHandler = Callable[[argparse.Namespace], list[Path]]


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared dataset command arguments."""
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--districts",
        type=Path,
        default=DEFAULT_DISTRICTS,
        help="Optional district GeoJSON used as the 3D density outline.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate reproducible dataset and model visualizations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Generate dataset figures.")
    _add_dataset_arguments(dataset_parser)
    dataset_parser.set_defaults(handler=_handle_dataset)

    experiments_parser = subparsers.add_parser(
        "experiments",
        help="Generate training, ranking, qualitative, and regulatory figures.",
    )
    experiments_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    experiments_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "experiments",
    )
    experiments_parser.add_argument("--limit-runs", type=int, default=None)
    experiments_parser.set_defaults(handler=_handle_experiments)

    samples_parser = subparsers.add_parser(
        "samples",
        help="Export channel and mask projections from dataset samples.",
    )
    samples_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    samples_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "samples")
    samples_parser.add_argument("--sample-id", action="append", dest="sample_ids")
    samples_parser.add_argument("--limit", type=int, default=1)
    samples_parser.set_defaults(handler=_handle_samples)

    animation_parser = subparsers.add_parser("animate", help="Create a GIF from PNG frames.")
    animation_parser.add_argument("--input", required=True, type=Path)
    animation_parser.add_argument("--output", required=True, type=Path)
    animation_parser.add_argument("--fps", type=int, default=12)
    animation_parser.add_argument("--pattern", default="*.png")
    animation_parser.add_argument("--maximum-frames", type=int, default=1000)
    animation_parser.set_defaults(handler=_handle_animation)

    prediction_parser = subparsers.add_parser(
        "render-prediction",
        help="Render one prediction in its LOD1 context.",
    )
    prediction_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    prediction_parser.set_defaults(handler=_handle_prediction)

    scene_parser = subparsers.add_parser(
        "render-scene",
        help="Generate and render a multi-building scene.",
    )
    scene_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    scene_parser.set_defaults(handler=_handle_scene)

    all_parser = subparsers.add_parser("all", help="Generate dataset and experiment figures.")
    all_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    all_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    all_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "all")
    all_parser.add_argument("--limit-samples", type=int, default=None)
    all_parser.add_argument("--limit-runs", type=int, default=None)
    all_parser.add_argument(
        "--districts",
        type=Path,
        default=DEFAULT_DISTRICTS,
        help="Optional district GeoJSON used as the 3D density outline.",
    )
    all_parser.set_defaults(handler=_handle_all)
    return parser


def _handle_dataset(arguments: argparse.Namespace) -> list[Path]:
    """Validate inputs and generate dataset figures."""
    dataset = require_directory(arguments.dataset, label="Dataset")
    output = prepare_output_directory(arguments.output)
    return generate_dataset_figures(
        dataset,
        output,
        limit=arguments.limit,
        districts_path=arguments.districts,
    )


def _handle_experiments(arguments: argparse.Namespace) -> list[Path]:
    """Validate inputs and generate experiment figures."""
    runs_root = require_directory(arguments.runs_root, label="Runs root")
    output = prepare_output_directory(arguments.output)
    return generate_experiment_figures(runs_root, output, limit_runs=arguments.limit_runs)


def _handle_samples(arguments: argparse.Namespace) -> list[Path]:
    """Validate inputs and export dataset sample views."""
    dataset = require_directory(arguments.dataset, label="Dataset")
    output = prepare_output_directory(arguments.output)
    return export_sample_figures(
        dataset,
        output,
        sample_ids=arguments.sample_ids,
        limit=arguments.limit,
    )


def _handle_animation(arguments: argparse.Namespace) -> list[Path]:
    """Validate inputs and create one animation."""
    input_directory = require_directory(arguments.input, label="Frame directory")
    path = create_gif(
        input_directory,
        arguments.output.expanduser().resolve(),
        frames_per_second=arguments.fps,
        pattern=arguments.pattern,
        maximum_frames=arguments.maximum_frames,
    )
    return [path]


def _advanced_arguments(arguments: argparse.Namespace) -> list[str]:
    """Remove the optional separator from a forwarded argument list."""
    forwarded = list(arguments.arguments)
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


def _handle_prediction(arguments: argparse.Namespace) -> list[Path]:
    """Run the maintained single-prediction LOD1 renderer."""
    # Delay optional PyVista/PyTorch imports until this command is requested.
    from visualize_pred_with_lod1 import main as render_prediction  # noqa: PLC0415

    output_directory = render_prediction(_advanced_arguments(arguments))
    return [output_directory]


def _handle_scene(arguments: argparse.Namespace) -> list[Path]:
    """Run the maintained multi-building scene renderer."""
    # Delay optional PyVista/PyTorch imports until this command is requested.
    from custom_scene_multigen import main as render_scene  # noqa: PLC0415

    output_directory = render_scene(_advanced_arguments(arguments))
    return [output_directory]


def _handle_all(arguments: argparse.Namespace) -> list[Path]:
    """Run the maintained dataset, sample, and experiment workflows."""
    dataset = require_directory(arguments.dataset, label="Dataset")
    runs_root = require_directory(arguments.runs_root, label="Runs root")
    output = prepare_output_directory(arguments.output)
    dataset_output = output / "dataset"
    experiment_output = output / "experiments"
    sample_output = output / "samples"
    written = generate_dataset_figures(
        dataset,
        dataset_output,
        limit=arguments.limit_samples,
        districts_path=arguments.districts,
    )
    written.extend(
        generate_experiment_figures(
            runs_root,
            experiment_output,
            limit_runs=arguments.limit_runs,
        )
    )
    written.extend(export_sample_figures(dataset, sample_output, limit=1))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible status code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler: CommandHandler = arguments.handler
    try:
        written = handler(arguments)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Created {len(written)} artifact(s):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
