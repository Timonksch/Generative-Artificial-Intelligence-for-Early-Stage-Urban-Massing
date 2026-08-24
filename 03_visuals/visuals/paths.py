"""Central filesystem locations and path validation."""

from __future__ import annotations

from pathlib import Path

VISUALS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = VISUALS_ROOT.parent
DEFAULT_OUTPUT_ROOT = VISUALS_ROOT / "outputs"
DEFAULT_DATASET = REPOSITORY_ROOT / "00_Data" / "02_GeneratedDatasets" / "example_smoke"
DEFAULT_RUNS_ROOT = REPOSITORY_ROOT / "02_TrainModels" / "outputs"
DEFAULT_DISTRICTS = VISUALS_ROOT / "bezirksgrenzen.geojson"


def require_directory(path: Path, *, label: str) -> Path:
    """Return a resolved directory or raise a descriptive error.

    Args:
        path: Directory supplied at a command boundary.
        label: Human-readable name used in the error message.

    Returns:
        The absolute, resolved directory path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.

    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def prepare_output_directory(path: Path) -> Path:
    """Create and return an absolute output directory."""
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
