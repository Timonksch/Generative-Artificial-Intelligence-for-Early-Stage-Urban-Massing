"""Lightweight IO helpers for experiment management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist dictionary as formatted JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON document from disk."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)
