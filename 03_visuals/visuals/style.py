"""Single source of truth for the thesis figure style."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from visuals.paths import VISUALS_ROOT

ACCENT = "#1B49FF"
RED = "#FC4343"
GREEN = "#B3FC64"
TEAL = "#1FB6A6"
ORANGE = "#FFB000"
PURPLE = "#8E63FF"
DARK = "#182033"
MID = "#66708A"
LIGHT = "#D3D8E2"
GRID = "#D9DDE5"
BACKGROUND = "#FFFFFF"
PALETTE = (ACCENT, RED, GREEN, TEAL, ORANGE, PURPLE)
SPLIT_COLORS = {"train": ACCENT, "val": RED, "test": GREEN}
DPI = 300


def _register_project_font() -> str:
    """Register the bundled Inter font when it is available."""
    font_directory = VISUALS_ROOT / ".ressources" / "font" / "inter"
    for font_path in sorted(font_directory.glob("*.ttf")):
        font_manager.fontManager.addfont(str(font_path))
    regular_font = font_directory / "Inter_24pt-Regular.ttf"
    if not regular_font.exists():
        return "sans-serif"
    return font_manager.FontProperties(fname=str(regular_font)).get_name()


FONT_FAMILY = _register_project_font()

plt.rcParams.update(
    {
        "axes.edgecolor": LIGHT,
        "axes.facecolor": BACKGROUND,
        "axes.grid": True,
        "axes.labelcolor": DARK,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": DPI,
        "figure.facecolor": BACKGROUND,
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, "Arial", "sans-serif"],
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": DPI,
        "svg.fonttype": "none",
        "xtick.color": MID,
        "ytick.color": MID,
    }
)


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    *,
    formats: Sequence[str] = ("png", "svg"),
) -> list[Path]:
    """Save a figure in each requested format and close it.

    Args:
        figure: Matplotlib figure to persist.
        output_directory: Destination directory.
        stem: Output filename without an extension.
        formats: File extensions supported by Matplotlib.

    Returns:
        Paths written in the same order as ``formats``.

    Raises:
        ValueError: If no output format is requested.

    """
    if not formats:
        raise ValueError("At least one output format is required")
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for output_format in formats:
        path = output_directory / f"{stem}.{output_format}"
        figure.savefig(path, bbox_inches="tight", facecolor=BACKGROUND)
        written.append(path)
    plt.close(figure)
    return written
