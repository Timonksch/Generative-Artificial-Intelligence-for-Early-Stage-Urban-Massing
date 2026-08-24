"""Static comparison figures for arbitrary training output trees."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visuals.records import RunMetrics, discover_run_metrics
from visuals.style import ACCENT, PALETTE, RED, save_figure

_CURVE_METRICS = (
    ("train_loss", "Training loss"),
    ("val_loss", "Validation loss"),
    ("val_iou", "Validation IoU"),
    ("val_dice", "Validation Dice"),
)
_MAX_OVERVIEW_LEGEND_RUNS = 8
_MAX_QUALITATIVE_RUNS = 6
_LOG_SCALE_RATIO = 100.0
_REGULATORY_METRICS = ("grz", "gfz", "height_m")
_MIN_CONTROL_POINTS = 2


@dataclass(frozen=True)
class RegulatoryRow:
    """Mean regulatory target and prediction for one evaluated variant."""

    run_name: str
    evaluation_name: str
    variant_name: str
    metric_name: str
    target: float
    prediction: float
    abs_error_adjusted: float | None
    relative_error_adjusted: float | None


@dataclass(frozen=True)
class ControlSeriesSpec:
    """Plot styling and variants for one control response line."""

    variants: Sequence[str]
    label: str
    color: str


def generate_experiment_figures(
    runs_root: Path,
    output_directory: Path,
    *,
    limit_runs: int | None = None,
) -> list[Path]:
    """Generate overview, ranking, qualitative, and regulatory figures.

    Args:
        runs_root: Directory recursively containing ``metrics.csv`` files.
        output_directory: Destination for generated artifacts.
        limit_runs: Optional positive run cap for smoke tests.

    Returns:
        Every file written by the operation.

    """
    runs = discover_run_metrics(runs_root, limit=limit_runs)
    output_directory.mkdir(parents=True, exist_ok=True)
    written = _plot_training_overview(runs, output_directory)
    written.extend(_plot_ranking(runs, output_directory))
    written.extend(_plot_qualitative_overview(runs, output_directory))
    written.extend(_plot_regulatory_results(runs_root, output_directory))
    summary_path = output_directory / "run_summary.json"
    summary_path.write_text(json.dumps(_run_summary(runs), indent=2), encoding="utf-8")
    written.append(summary_path)
    return written


def _plot_training_overview(
    runs: Sequence[RunMetrics],
    output_directory: Path,
) -> list[Path]:
    """Plot the central train and validation curves."""
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.5))
    for index, axis in enumerate(axes.flat):
        metric_name, label = _CURVE_METRICS[index]
        for run_index, run in enumerate(runs):
            series = run.series.get(metric_name)
            if series is None:
                continue
            axis.plot(
                series.steps,
                series.values,
                color=PALETTE[run_index % len(PALETTE)],
                label=run.name,
                linewidth=1.2,
            )
        axis.set_xlabel("Step / epoch")
        axis.set_ylabel(label)
        _apply_loss_scale(axis, runs, metric_name)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and len(runs) <= _MAX_OVERVIEW_LEGEND_RUNS:
        figure.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), fontsize=7)
        figure.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    else:
        figure.tight_layout()
    return save_figure(figure, output_directory, "training_overview")


def _apply_loss_scale(
    axis: plt.Axes,
    runs: Sequence[RunMetrics],
    metric_name: str,
) -> None:
    """Use a logarithmic axis when positive loss values span two orders."""
    if "loss" not in metric_name:
        return
    values = [
        value
        for run in runs
        if (series := run.series.get(metric_name)) is not None
        for value in series.values
        if value > 0
    ]
    if not values or max(values) / min(values) < _LOG_SCALE_RATIO:
        return
    axis.set_yscale("log")


def _ranking_metric(runs: Sequence[RunMetrics]) -> tuple[str, str, bool]:
    """Select the most informative metric shared by the available runs."""
    candidates = (
        ("val_iou", "Final validation IoU", True),
        ("val_dice", "Final validation Dice", True),
        ("val_loss", "Final validation loss", False),
        ("val_vae_loss", "Final VAE validation loss", False),
        ("val_diff_loss", "Final diffusion validation loss", False),
    )
    for metric_name, label, higher_is_better in candidates:
        if any(run.final(metric_name) is not None for run in runs):
            return metric_name, label, higher_is_better
    raise ValueError("No supported validation metric found in the discovered runs")


def _plot_ranking(runs: Sequence[RunMetrics], output_directory: Path) -> list[Path]:
    """Plot final values for the best available validation metric."""
    metric_name, label, higher_is_better = _ranking_metric(runs)
    ranked = [(run.name, value) for run in runs if (value := run.final(metric_name)) is not None]
    ranked.sort(key=lambda item: item[1], reverse=higher_is_better)
    figure_height = max(3.8, 0.33 * len(ranked) + 1.5)
    figure, axis = plt.subplots(figsize=(10.0, figure_height))
    positions = np.arange(len(ranked))
    colors = [ACCENT] + [RED] * max(0, len(ranked) - 1)
    axis.barh(positions, [item[1] for item in ranked], color=colors)
    axis.set_yticks(positions, [item[0] for item in ranked], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel(label)
    figure.tight_layout()
    return save_figure(figure, output_directory, "run_ranking")


def _find_visualization_directory(run: RunMetrics) -> Path | None:
    """Find the preferred inference visualization directory for one run."""
    candidates = (
        run.path / "infer_vis" / "vis",
        run.path / "eval_test" / "vis",
        run.path / "eval_val" / "vis",
    )
    return next((path for path in candidates if path.is_dir()), None)


def _plot_qualitative_overview(
    runs: Sequence[RunMetrics],
    output_directory: Path,
) -> list[Path]:
    """Combine the first available projection from every model run."""
    available: list[tuple[str, Path]] = []
    for run in runs:
        visual_directory = _find_visualization_directory(run)
        if visual_directory is None:
            continue
        projection = next(iter(sorted(visual_directory.glob("*_projections.png"))), None)
        if projection is not None:
            available.append((run.name, projection))
    if not available:
        return []
    selected = _select_evenly(available, maximum=_MAX_QUALITATIVE_RUNS)
    column_count = min(2, len(selected))
    row_count = int(np.ceil(len(selected) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12.0, 3.4 * row_count),
        squeeze=False,
    )
    axes_array = axes.flat
    for index, (run_name, image_path) in enumerate(selected):
        axis = axes_array[index]
        axis.imshow(plt.imread(image_path))
        axis.set_title(run_name, loc="left", fontsize=9)
        axis.set_axis_off()
    for index in range(len(selected), row_count * column_count):
        axes_array[index].set_axis_off()
    figure.tight_layout()
    return save_figure(figure, output_directory, "qualitative_overview")


def _select_evenly(
    items: list[tuple[str, Path]],
    *,
    maximum: int,
) -> list[tuple[str, Path]]:
    """Select a bounded set spanning an already sorted sequence."""
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=int)
    return [items[int(index)] for index in indices]


def _plot_regulatory_results(root: Path, output_directory: Path) -> list[Path]:
    """Plot mean target and prediction values from regulatory evaluations."""
    rows = _load_regulatory_rows(root)
    if not rows:
        return []
    written = []
    labels = [f"{row.run_name} · {row.variant_name} · {row.metric_name}" for row in rows]
    targets = [row.target for row in rows]
    predictions = [row.prediction for row in rows]
    positions = np.arange(len(rows), dtype=float)
    figure, axis = plt.subplots(figsize=(11.0, max(4.0, len(rows) * 0.24)))
    axis.scatter(targets, positions, color=ACCENT, label="Target", s=24)
    axis.scatter(predictions, positions, color=RED, label="Prediction", s=24)
    axis.set_yticks(positions, labels, fontsize=6)
    axis.set_xlabel("Mean regulatory value")
    axis.legend(ncol=2)
    axis.invert_yaxis()
    figure.tight_layout()
    written.extend(save_figure(figure, output_directory, "regulatory_overview"))
    written.extend(_plot_regulatory_target_scatter(rows, output_directory))
    written.extend(_plot_final_regulatory_errors(rows, output_directory))
    written.extend(_plot_phase2_control_response(rows, output_directory))
    return written


def _load_regulatory_rows(root: Path) -> list[RegulatoryRow]:
    """Load all mean regulatory comparisons below a run tree."""
    rows: list[RegulatoryRow] = []
    for result_path in sorted(root.rglob("reg_metrics.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        variants = payload.get("variants", {}) if isinstance(payload, dict) else {}
        if not isinstance(variants, dict):
            continue
        for variant_name, variant in variants.items():
            if isinstance(variant, dict):
                rows.extend(_regulatory_rows(result_path, str(variant_name), variant))
    return rows


def _plot_regulatory_target_scatter(
    rows: Sequence[RegulatoryRow],
    output_directory: Path,
) -> list[Path]:
    """Plot target versus predicted regulatory means for each metric."""
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for axis, metric_name in zip(axes, _REGULATORY_METRICS, strict=True):
        metric_rows = [row for row in rows if row.metric_name == metric_name]
        if not metric_rows:
            axis.set_axis_off()
            continue
        targets = np.asarray([row.target for row in metric_rows], dtype=float)
        predictions = np.asarray([row.prediction for row in metric_rows], dtype=float)
        colors = [PALETTE[index % len(PALETTE)] for index in range(len(metric_rows))]
        axis.scatter(targets, predictions, color=colors, edgecolor="white", linewidth=0.5, s=44)
        minimum = float(min(targets.min(), predictions.min()))
        maximum = float(max(targets.max(), predictions.max()))
        padding = max((maximum - minimum) * 0.08, 1e-6)
        axis.plot(
            [minimum - padding, maximum + padding],
            [minimum - padding, maximum + padding],
            color="#4B5563",
            linewidth=1.0,
            linestyle="--",
        )
        axis.set_xlim(minimum - padding, maximum + padding)
        axis.set_ylim(minimum - padding, maximum + padding)
        axis.set_xlabel("Target mean")
        axis.set_ylabel("Prediction mean")
        axis.set_title(_regulatory_metric_label(metric_name), loc="left", fontsize=10)
        axis.grid(True, linewidth=0.4, alpha=0.35)
    figure.tight_layout()
    return save_figure(figure, output_directory, "regulatory_target_scatter")


def _plot_final_regulatory_errors(
    rows: Sequence[RegulatoryRow],
    output_directory: Path,
) -> list[Path]:
    """Plot final-model regulatory errors without threshold sweep variants."""
    final_rows = [
        row
        for row in rows
        if row.variant_name == "gt" and row.evaluation_name in {"reg_eval", "eval_regulatory_test"}
    ]
    if not final_rows:
        return []
    run_names = list(dict.fromkeys(row.run_name for row in final_rows))
    x_positions = np.arange(len(run_names), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharex=True)
    for axis, metric_name in zip(axes, _REGULATORY_METRICS, strict=True):
        values = []
        for run_name in run_names:
            row = _find_regulatory_row(final_rows, run_name, metric_name)
            if row is None:
                values.append(np.nan)
            elif metric_name == "height_m":
                values.append(row.abs_error_adjusted or 0.0)
            else:
                values.append(100.0 * (row.relative_error_adjusted or 0.0))
        colors = [PALETTE[index % len(PALETTE)] for index in range(len(run_names))]
        axis.bar(x_positions, values, color=colors)
        axis.set_title(_regulatory_metric_label(metric_name), loc="left", fontsize=10)
        axis.set_xticks(
            x_positions,
            [_short_run_label(name) for name in run_names],
            rotation=25,
            ha="right",
        )
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.35)
        ylabel = (
            "Adjusted absolute error [m]"
            if metric_name == "height_m"
            else "Adjusted relative error [%]"
        )
        axis.set_ylabel(ylabel)
    figure.tight_layout()
    return save_figure(figure, output_directory, "final_regulatory_errors")


def _plot_phase2_control_response(
    rows: Sequence[RegulatoryRow],
    output_directory: Path,
) -> list[Path]:
    """Plot whether Phase-2 conditioning changes predictions in the requested direction."""
    phase2_rows = [
        row
        for row in rows
        if row.run_name == "phase2_final_drop_p00_tta_rot90"
        and row.evaluation_name in {"eval_regulatory_test", "eval_regulatory_test_isolated"}
    ]
    if not phase2_rows:
        return []
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for axis, metric_name in zip(axes, _REGULATORY_METRICS, strict=True):
        _plot_control_series(
            axis,
            phase2_rows,
            metric_name,
            ControlSeriesSpec(
                variants=("minus10", "gt", "plus10"),
                label="Coupled controls",
                color=ACCENT,
            ),
        )
        _plot_control_series(
            axis,
            phase2_rows,
            metric_name,
            ControlSeriesSpec(
                variants=(f"{metric_name}_minus10", "gt", f"{metric_name}_plus10"),
                label="Isolated control",
                color=RED,
            ),
        )
        axis.axhline(0.0, color="#94A3B8", linewidth=0.8)
        axis.axvline(0.0, color="#94A3B8", linewidth=0.8)
        axis.plot([-12.0, 12.0], [-12.0, 12.0], color="#4B5563", linestyle="--", linewidth=1.0)
        axis.set_xlim(-12.0, 12.0)
        axis.set_ylim(-12.0, 12.0)
        axis.set_title(_regulatory_metric_label(metric_name), loc="left", fontsize=10)
        axis.set_xlabel("Requested target change [%]")
        axis.grid(True, linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Prediction change [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        figure.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    else:
        figure.tight_layout()
    return save_figure(figure, output_directory, "phase2_control_response")


def _plot_control_series(
    axis: plt.Axes,
    rows: Sequence[RegulatoryRow],
    metric_name: str,
    spec: ControlSeriesSpec,
) -> None:
    """Plot one control response line for a single regulatory metric."""
    gt = _find_regulatory_variant(rows, "gt", metric_name)
    if gt is None or gt.target == 0.0 or gt.prediction == 0.0:
        return
    points: list[tuple[float, float]] = []
    for variant_name in spec.variants:
        row = _find_regulatory_variant(rows, variant_name, metric_name)
        if row is None:
            continue
        target_delta = 100.0 * (row.target - gt.target) / gt.target
        prediction_delta = 100.0 * (row.prediction - gt.prediction) / gt.prediction
        points.append((target_delta, prediction_delta))
    if len(points) < _MIN_CONTROL_POINTS:
        return
    points.sort(key=lambda item: item[0])
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        marker="o",
        color=spec.color,
        linewidth=1.4,
        label=spec.label,
    )


def _find_regulatory_row(
    rows: Sequence[RegulatoryRow],
    run_name: str,
    metric_name: str,
) -> RegulatoryRow | None:
    """Return the first row matching one run and metric."""
    return next(
        (row for row in rows if row.run_name == run_name and row.metric_name == metric_name),
        None,
    )


def _find_regulatory_variant(
    rows: Sequence[RegulatoryRow],
    variant_name: str,
    metric_name: str,
) -> RegulatoryRow | None:
    """Return the first row matching one variant and metric."""
    return next(
        (
            row
            for row in rows
            if row.variant_name == variant_name and row.metric_name == metric_name
        ),
        None,
    )


def _regulatory_metric_label(metric_name: str) -> str:
    """Return a compact plotting label for a regulatory metric."""
    labels = {"grz": "GRZ", "gfz": "GFZ", "height_m": "Height"}
    return labels.get(metric_name, metric_name)


def _short_run_label(run_name: str) -> str:
    """Return compact labels for the final comparison plot."""
    labels = {
        "phase1_final_capacity_down": "Phase 1 U-Net",
        "phase2_final_drop_p00_tta_rot90": "Phase 2 Cond. U-Net",
        "phase3_final_diff_linear_t1000_base64_depth3_drop002": "Phase 3 LDM",
    }
    return labels.get(run_name, run_name)


def _regulatory_rows(
    result_path: Path,
    variant_name: str,
    variant: dict[str, object],
) -> list[RegulatoryRow]:
    """Extract target/prediction means for one regulatory variant."""
    rows: list[RegulatoryRow] = []
    run_name = result_path.parents[1].name
    for metric_name in _REGULATORY_METRICS:
        metric = variant.get(metric_name)
        if not isinstance(metric, dict):
            continue
        target = metric.get("target")
        prediction = metric.get("pred")
        if not isinstance(target, dict) or not isinstance(prediction, dict):
            continue
        target_mean = target.get("mean")
        prediction_mean = prediction.get("mean")
        abs_error_adjusted = _optional_metric_mean(metric, "abs_err_adj")
        relative_error_adjusted = _optional_metric_mean(metric, "rel_err_adj")
        if isinstance(target_mean, (int, float)) and isinstance(prediction_mean, (int, float)):
            rows.append(
                RegulatoryRow(
                    run_name=run_name,
                    evaluation_name=result_path.parent.name,
                    variant_name=variant_name,
                    metric_name=metric_name,
                    target=float(target_mean),
                    prediction=float(prediction_mean),
                    abs_error_adjusted=abs_error_adjusted,
                    relative_error_adjusted=relative_error_adjusted,
                )
            )
    return rows


def _optional_metric_mean(metric: dict[str, object], key: str) -> float | None:
    """Return a nested metric mean when present."""
    bucket = metric.get(key)
    if not isinstance(bucket, dict):
        return None
    value = bucket.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _run_summary(runs: Sequence[RunMetrics]) -> list[dict[str, object]]:
    """Serialize the discovered run inventory."""
    return [
        {
            "name": run.name,
            "path": str(run.path),
            "metrics": sorted(run.series),
            "final": {metric_name: series.values[-1] for metric_name, series in run.series.items()},
        }
        for run in runs
    ]
