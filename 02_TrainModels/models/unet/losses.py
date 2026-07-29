"""Segmentation losses and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _dilate3d(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    weight = torch.ones(
        (1, 1, kernel_size, kernel_size, kernel_size), device=x.device, dtype=x.dtype
    )
    out = F.conv3d(x, weight, stride=1, padding=pad)
    return (out > 0).to(x.dtype)


def _erode3d(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return 1.0 - _dilate3d(1.0 - x, kernel_size)


def _squeeze_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 5 and tensor.size(1) == 1:
        return tensor[:, 0]
    return tensor


class BCEDiceLoss(nn.Module):
    """Combination of BCE with logits and Dice loss with optional surface weights."""

    def __init__(
        self,
        bce_weight: float,
        dice_weight: float,
        surface_weight: float,
        max_pos_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.surface_weight = surface_weight
        self.max_pos_weight = max_pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        with torch.no_grad():
            pos = targets.sum().clamp_min(1.0)
            neg = (1.0 - targets).sum().clamp_min(1.0)
            pos_w = (neg / pos).clamp(max=self.max_pos_weight)

        if self.surface_weight <= 1.0:
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
        else:
            boundary = _dilate3d(targets) - _erode3d(targets)
            weights = 1.0 + (self.surface_weight - 1.0) * boundary
            bce = (
                F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none", pos_weight=pos_w
                )
                * weights
            ).mean()

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1.0 - (2 * intersection + 1e-6) / (union + 1e-6)

        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum()
    union = (preds + targets).clamp(0, 1).sum()
    return (inter / (union + 1e-6)).item()


@torch.no_grad()
def precision_recall(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[float, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    return precision, recall


@torch.no_grad()
def volumetric_error(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = _squeeze_channel(preds)
    target = _squeeze_channel(targets.to(preds.dtype))
    pred_vol = preds.sum(dim=(-3, -2, -1))
    target_vol = target.sum(dim=(-3, -2, -1))
    rel_err = (pred_vol - target_vol).abs().div(target_vol.clamp_min(1.0))
    return rel_err.mean().item()


@torch.no_grad()
def z_profile_error(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = _squeeze_channel(preds)
    target = _squeeze_channel(targets.to(preds.dtype))
    pred_profile = preds.sum(dim=(-2, -1))
    target_profile = target.sum(dim=(-2, -1))
    return (pred_profile - target_profile).abs().mean().item()
