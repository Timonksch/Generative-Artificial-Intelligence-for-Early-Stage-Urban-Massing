"""Reusable visualization package for dataset and model outputs."""

from __future__ import annotations

import os

from visuals.paths import DEFAULT_OUTPUT_ROOT, REPOSITORY_ROOT, VISUALS_ROOT

# Configure Matplotlib before any submodule can import it. The cache remains in
# the ignored output tree and therefore never changes tracked source files.
_MPL_CACHE = VISUALS_ROOT / "outputs" / ".mplconfig"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

__all__ = ["DEFAULT_OUTPUT_ROOT", "REPOSITORY_ROOT"]
