"""Voxel dataset supporting conditioning and geometric preprocessing."""

from __future__ import annotations

import glob
import json
import random
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .cond_stats import (
    CondStats,
    compute_cond_stats,
    normalize_cond,
)

DEFAULT_CHANNEL_INDEX = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}


class VoxelDataset(Dataset):
    """Load NPZ voxel samples with optional conditioning and preprocessing."""

    def __init__(  # noqa: PLR0917
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        split_manifest: str | None = None,
        augment: bool = True,
        add_coords: bool = False,
        channels: Sequence[str] = ("C0", "C1", "C2", "C3"),
        x_indices: Sequence[int] | None = None,
        downsample_stride: int = 0,
        crop_size: int = 0,
        cond_dim: int = 0,
        cond_select: Sequence[int] | None = None,
        cond_stats: CondStats | None = None,
        auto_cond_stats: bool = False,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.channels = list(channels)
        self.x_indices = (
            list(x_indices)
            if x_indices is not None
            else [DEFAULT_CHANNEL_INDEX[ch] for ch in self.channels]
        )
        self.downsample_stride = downsample_stride
        self.crop_size = crop_size
        self.add_coords = add_coords
        self.augment = augment and split == "train"
        self.cond_dim = cond_dim
        self.cond_select = list(cond_select) if cond_select else None
        self.max_samples = max_samples
        self.split_manifest_path = Path(split_manifest).expanduser() if split_manifest else None

        if self.split_manifest_path:
            self.files = self._load_files_from_manifest(split)
        else:
            self.files = sorted(glob.glob(str(self.root_dir / "**" / "*.npz"), recursive=True))
            if not self.files:
                raise FileNotFoundError(f"No NPZ files found under {self.root_dir}")

            rng = random.Random(seed)  # noqa: S311 - deterministic dataset partitioning
            rng.shuffle(self.files)
            n_total = len(self.files)
            n_val = int(val_ratio * n_total)
            n_test = int(test_ratio * n_total)
            n_train = max(n_total - n_val - n_test, 1)
            splits = {
                "train": self.files[:n_train],
                "val": self.files[n_train : n_train + n_val],
                "test": self.files[n_train + n_val : n_train + n_val + n_test],
            }
            if split not in splits:
                raise ValueError(f"Unknown split '{split}'")
            self.files = splits[split]

        if self.max_samples is not None:
            self.files = self.files[: self.max_samples]
        if not self.files:
            raise ValueError(f"Split '{split}' is empty after partitioning.")

        stats = cond_stats
        if self.cond_dim and stats is None and auto_cond_stats:
            metas = [self._read_meta(path) for path in self.files]
            stats = compute_cond_stats(metas)
        self.cond_stats = self._prepare_cond_stats(stats)

    def __len__(self) -> int:
        return len(self.files)

    def _load_files_from_manifest(self, split: str) -> list[str]:
        if not self.split_manifest_path or not self.split_manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.split_manifest_path}")

        with self.split_manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        raw_splits = manifest.get("splits", manifest)
        if split not in raw_splits:
            raise ValueError(f"Split '{split}' not present in manifest {self.split_manifest_path}")

        entry = raw_splits[split]
        if isinstance(entry, dict) and "sample_ids" in entry:
            sample_ids = entry["sample_ids"]
        elif isinstance(entry, list):
            sample_ids = entry
        else:
            raise ValueError(
                f"Manifest entry for split '{split}' must contain 'sample_ids' or be a list."
            )

        files: list[str] = []
        missing: list[str] = []
        for sample_id in sample_ids:
            sample_dir = self.root_dir / sample_id
            candidate = sample_dir / f"{sample_id}.npz"
            if candidate.exists():
                files.append(str(candidate))
                continue
            flat_candidate = self.root_dir / f"{sample_id}.npz"
            if flat_candidate.exists():
                files.append(str(flat_candidate))
            else:
                missing.append(sample_id)

        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            warnings.warn(
                f"{len(missing)} samples listed in {self.split_manifest_path} "
                f"missing under {self.root_dir}: "
                f"{preview}{suffix}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not files:
            raise ValueError(f"Manifest split '{split}' contains no existing NPZ files.")

        return files

    def _read_meta(self, path: str) -> dict:
        with np.load(path, allow_pickle=True) as data:
            if "meta" in data:
                try:
                    return data["meta"].item()
                except Exception:
                    return {}
        return {}

    def _load_npz(self, path: str) -> tuple[np.ndarray, np.ndarray, dict]:
        with np.load(path, allow_pickle=True) as data:
            if "X" in data:
                vox = data["X"]
                if vox.ndim != 4:
                    raise ValueError(f"{path}: expected X of shape (C,D,H,W), got {vox.shape}")
                indices = self.x_indices
                if max(indices) >= vox.shape[0]:
                    raise IndexError(
                        f"{path}: channel index out of range for X with {vox.shape[0]} channels"
                    )
                vox = vox[indices]
            else:
                voxels: list[np.ndarray] = []
                for ch in self.channels:
                    if ch not in data:
                        raise KeyError(f"{path}: channel {ch} missing")
                    voxels.append(data[ch])
                vox = np.stack(voxels, axis=0)

            if "Y" not in data:
                raise KeyError(f"{path}: mask 'Y' missing")
            target = data["Y"]
            if target.ndim == 3:
                target = target[None, ...]
            if target.ndim != 4 or target.shape[0] != 1:
                raise ValueError(f"{path}: expected Y shape (1,D,H,W), got {target.shape}")

            meta = {}
            if "meta" in data:
                try:
                    meta = data["meta"].item()
                except Exception:
                    meta = {}

        return vox.astype(np.float32, copy=False), target.astype(np.float32, copy=False), meta

    def _apply_downsample(self, tensor: torch.Tensor, is_target: bool = False) -> torch.Tensor:
        if self.downsample_stride and self.downsample_stride > 1:
            stride = self.downsample_stride
            tensor = F.avg_pool3d(tensor, kernel_size=stride, stride=stride)
            if is_target:
                tensor = (tensor >= 0.5).to(tensor.dtype)
        return tensor

    def _compute_crop_slices(
        self, spatial_shape: tuple[int, int, int], size: int
    ) -> tuple[slice, slice, slice]:
        if size <= 0:
            return slice(None), slice(None), slice(None)
        depth, height, width = spatial_shape
        size = min(size, depth, height, width)
        max_d = max(depth - size, 0)
        max_h = max(height - size, 0)
        max_w = max(width - size, 0)

        if self.split == "train" and self.augment:
            d0 = int(torch.randint(0, max_d + 1, (1,)).item()) if max_d > 0 else 0
            h0 = int(torch.randint(0, max_h + 1, (1,)).item()) if max_h > 0 else 0
            w0 = int(torch.randint(0, max_w + 1, (1,)).item()) if max_w > 0 else 0
        else:
            d0 = max_d // 2
            h0 = max_h // 2
            w0 = max_w // 2
        return slice(d0, d0 + size), slice(h0, h0 + size), slice(w0, w0 + size)

    def _add_coord_channels(self, vox: torch.Tensor) -> torch.Tensor:
        depth, height, width = vox.shape[-3:]
        zs = (
            torch.linspace(-1.0, 1.0, depth, dtype=vox.dtype)
            .view(1, depth, 1, 1)
            .expand(1, depth, height, width)
        )
        ys = (
            torch.linspace(-1.0, 1.0, height, dtype=vox.dtype)
            .view(1, 1, height, 1)
            .expand(1, depth, height, width)
        )
        xs = (
            torch.linspace(-1.0, 1.0, width, dtype=vox.dtype)
            .view(1, 1, 1, width)
            .expand(1, depth, height, width)
        )
        coords = torch.cat([zs, ys, xs], dim=0)
        return torch.cat([vox, coords], dim=0)

    def _build_cond_vector(self, meta: dict) -> np.ndarray:
        if self.cond_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
        raw = np.asarray(
            [
                float(metrics.get("grz_target", 0.0)),
                float(metrics.get("gfz_target", 0.0)),
                float(metrics.get("target_height_m", 0.0)),
            ],
            dtype=np.float32,
        )
        if self.cond_select:
            raw = raw[self.cond_select]

        if raw.shape[0] != self.cond_dim:
            if raw.shape[0] > self.cond_dim:
                raw = raw[: self.cond_dim]
            else:
                raw = np.pad(raw, (0, self.cond_dim - raw.shape[0]), constant_values=0.0)

        if self.cond_stats:
            raw = normalize_cond(raw, self.cond_stats)

        return raw.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vox_np, target_np, meta = self._load_npz(self.files[idx])
        cond_np = self._build_cond_vector(meta)

        vox = torch.from_numpy(vox_np)
        target = torch.from_numpy(target_np)

        # Add temporary batch dim for spatial ops
        if vox.ndim == 4:
            vox = vox.unsqueeze(0)
        if target.ndim == 4:
            target = target.unsqueeze(0)

        vox = self._apply_downsample(vox, is_target=False)
        target = self._apply_downsample(target, is_target=True)

        if self.crop_size and self.crop_size > 0:
            d_slice, h_slice, w_slice = self._compute_crop_slices(
                tuple(target.shape[-3:]), self.crop_size
            )
            vox = vox[..., d_slice, h_slice, w_slice]
            target = target[..., d_slice, h_slice, w_slice]

        vox = vox.squeeze(0)
        target = target.squeeze(0)

        if self.add_coords:
            vox = self._add_coord_channels(vox)

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k:
                vox = torch.rot90(vox, k, (2, 3))
                target = torch.rot90(target, k, (2, 3))
            if torch.rand(1) > 0.5:
                vox = torch.flip(vox, dims=[2])
                target = torch.flip(target, dims=[2])
            if torch.rand(1) > 0.5:
                vox = torch.flip(vox, dims=[3])
                target = torch.flip(target, dims=[3])

        cond = torch.from_numpy(cond_np)
        return {
            "voxels": vox,
            "target": target,
            "cond": cond,
            "meta": meta,
            "path": self.files[idx],
        }

    def _prepare_cond_stats(self, stats: CondStats | None) -> CondStats | None:
        if stats is None:
            return None
        mean = np.asarray(stats.mean, dtype=np.float32).copy()
        std = np.asarray(stats.std, dtype=np.float32).copy()
        if self.cond_select:
            # Accept statistics that have already been reduced to selected metrics.
            # (e.g. train split computes stats, val split reuses them). Only
            # apply cond_select when the incoming stats still have the full
            # canonical conditioning dimensionality.
            max_index = max(self.cond_select)
            if mean.shape[0] > max_index:
                mean = mean[self.cond_select]
                std = std[self.cond_select]
        if self.cond_dim and mean.shape[0] != self.cond_dim:
            if mean.shape[0] > self.cond_dim:
                mean = mean[: self.cond_dim]
                std = std[: self.cond_dim]
            else:
                pad = self.cond_dim - mean.shape[0]
                mean = np.pad(mean, (0, pad))
                std = np.pad(std, (0, pad), constant_values=1.0)
        return CondStats(mean=mean, std=std)


def voxel_collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Voxel collate received an empty batch.")

    collated: dict[str, object] = {}
    first = batch[0]
    for key in first:
        values = [sample[key] for sample in batch]
        if key in {"meta", "path"}:
            collated[key] = values  # keep metadata/path per sample
        else:
            collated[key] = default_collate(values)
    return collated
