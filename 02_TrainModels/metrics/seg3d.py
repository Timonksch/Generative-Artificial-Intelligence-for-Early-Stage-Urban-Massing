"""Segmentation metrics and helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch

try:
    from ..models.unet import losses as seg_losses
except ImportError:  # when metrics is imported as top-level package
    from models.unet import losses as seg_losses  # type: ignore


def _squeeze_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 5 and tensor.size(1) == 1:
        return tensor[:, 0]
    return tensor


def _volume_error(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = _squeeze_channel(preds)
    target = _squeeze_channel(targets.to(preds.dtype))
    spatial_sum = preds.sum(dim=(-3, -2, -1))
    target_sum = target.sum(dim=(-3, -2, -1))
    return (spatial_sum - target_sum).abs().mean().item()


def evaluate(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    """Return IoU, precision, recall, F1/Dice, and volume related metrics."""
    iou = seg_losses.iou_score(logits, targets, threshold)
    precision, recall = seg_losses.precision_recall(logits, targets, threshold)

    # F1 score is the harmonic mean of precision and recall
    # For binary segmentation, F1 is equivalent to Dice coefficient
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)

    delta_vol = seg_losses.volumetric_error(logits, targets, threshold)
    volume_error = _volume_error(logits, targets, threshold)
    z_error = seg_losses.z_profile_error(logits, targets, threshold)
    return {
        "iou": iou,
        "dice": f1,  # Dice is equivalent to F1 for binary segmentation
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "delta_vol": delta_vol,
        "volume_error": volume_error,
        "z_profile_error": z_error,
    }


def auto_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresh_grid: Iterable[float],
) -> tuple[float, dict[str, float]]:
    """Select threshold maximizing IoU over a grid."""
    best_thr, best_iou = 0.5, -1.0
    best_metrics: dict[str, float] = {}
    for thr in thresh_grid:
        metrics = evaluate(logits, targets, thr)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thr = thr
            best_metrics = metrics
    return best_thr, best_metrics
