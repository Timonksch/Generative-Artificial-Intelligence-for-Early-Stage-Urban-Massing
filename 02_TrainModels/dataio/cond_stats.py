"""Conditioning statistics helper functions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COND_METRIC_KEYS = ("grz_target", "gfz_target", "target_height_m")


@dataclass
class CondStats:
    """Mean and std vectors for conditioning features."""

    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, Sequence[float]]:
        return {"cond_mean": self.mean.tolist(), "cond_std": self.std.tolist()}


def _extract_raw_cond(meta: dict) -> np.ndarray:
    metrics = meta.get("metrics") if isinstance(meta, dict) else {}
    if metrics is None:
        metrics = {}
    values = [
        float(metrics.get("grz_target", 0.0)),
        float(metrics.get("gfz_target", 0.0)),
        float(metrics.get("target_height_m", 0.0)),
    ]
    return np.asarray(values, dtype=np.float32)


def compute_cond_stats(metas: Iterable[dict]) -> CondStats:
    cond_vectors = [_extract_raw_cond(meta) for meta in metas]
    if not cond_vectors:
        dim = len(COND_METRIC_KEYS)
        return CondStats(mean=np.zeros(dim, dtype=np.float32), std=np.ones(dim, dtype=np.float32))
    stacked = np.stack(cond_vectors, axis=0)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std == 0.0, 1.0, std)
    return CondStats(mean=stacked.mean(axis=0).astype(np.float32), std=std)


def load_cond_stats(path: str | Path) -> CondStats:
    with Path(path).open("r", encoding="utf-8") as fp:
        doc = json.load(fp)

    def _resolve(keys: Sequence[str]) -> Sequence[float]:
        for key in keys:
            if key in doc:
                return doc[key]
        raise KeyError(f"{path}: missing keys {keys}")

    mean = np.asarray(
        _resolve(("cond_mean", "cond_means", "mean", "means")), dtype=np.float32
    ).reshape(-1)
    std = np.asarray(_resolve(("cond_std", "cond_stds", "std", "stds")), dtype=np.float32).reshape(
        -1
    )
    return CondStats(mean=mean, std=std)


def save_cond_stats(path: str | Path, stats: CondStats) -> None:
    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, indent=2, sort_keys=True)


def normalize_cond(raw: np.ndarray, stats: CondStats) -> np.ndarray:
    return (raw - stats.mean) / (stats.std + 1e-8)


def denormalize_cond(normed: np.ndarray, stats: CondStats) -> np.ndarray:
    return normed * stats.std + stats.mean
