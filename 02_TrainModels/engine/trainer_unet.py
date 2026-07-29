"""Training loop for UNet variants."""

from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from ..metrics import seg3d as seg_metrics
    from ..models.unet.losses import BCEDiceLoss
    from ..utils.device import DeviceConfig
    from ..utils.io import ensure_dir, save_json
    from ..utils.logging import RunLogger, configure_root_logger
    from ..utils.visuals import save_3d_context_visualization, save_max_projection
except ImportError:
    from metrics import seg3d as seg_metrics  # type: ignore
    from models.unet.losses import BCEDiceLoss  # type: ignore
    from utils.device import DeviceConfig  # type: ignore
    from utils.io import ensure_dir, save_json  # type: ignore
    from utils.logging import RunLogger, configure_root_logger  # type: ignore
    from utils.visuals import save_3d_context_visualization, save_max_projection  # type: ignore


@dataclass
class UNetTrainOptions:
    epochs: int
    log_every: int
    save_every: int
    accum_steps: int
    grad_clip: float
    tta: bool
    tta_mode: str
    auto_thresh: bool
    thresh_grid: tuple[float, ...]
    cond_drop: float
    early_stop_patience: int
    vis_every: int
    volume_reg_weight: float


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"


class UNetTrainer:
    """Full-featured UNet training pipeline."""

    def __init__(  # noqa: PLR0917
        self,
        model: torch.nn.Module,
        loss_fn: BCEDiceLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device_cfg: DeviceConfig,
        run_dir: Path,
        logger: RunLogger,
        options: UNetTrainOptions,
        cond_dim: int,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device_cfg = device_cfg
        self.run_dir = ensure_dir(run_dir)
        self.logger = logger
        self.options = options
        self.cond_dim = cond_dim
        self.volume_reg_weight = max(options.volume_reg_weight, 0.0)
        self.tta_mode = options.tta_mode
        self._progress_last: dict[str, int] = {}
        self._last_epoch_summary: str = ""

        self.vis_dir = ensure_dir(self.run_dir / "vis")

        if not logging.getLogger("urban3d").handlers:
            configure_root_logger()
        self.console = logging.getLogger("urban3d").getChild("trainer.unet")

        self.device = device_cfg.device
        self.model.to(self.device)
        self.use_amp = device_cfg.device_type == "cuda" and device_cfg.amp_dtype is not None
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_iou = -1.0
        self.best_metrics: dict[str, float] = {}
        self.early_stop_counter = 0
        self.console.info(
            "Initialized UNet trainer | device=%s | amp=%s | grad_clip=%.2f | vis_every=%d",
            self.device.type,
            self.use_amp,
            self.options.grad_clip,
            self.options.vis_every,
        )

    def _maybe_forward(self, vox: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if self.cond_dim > 0:
            return self.model(vox, cond)
        return self.model(vox)

    def _apply_tta(self, vox: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if not self.options.tta:
            return self._maybe_forward(vox, cond)

        if self.tta_mode == "rot90":
            preds = []
            for k in range(4):
                aug = vox if k == 0 else torch.rot90(vox, k, (3, 4))
                logits = self._maybe_forward(aug, cond)
                if k != 0:
                    logits = torch.rot90(logits, -k, (3, 4))
                preds.append(logits)
            return torch.stack(preds, dim=0).mean(dim=0)

        flips = [(), (2,), (3,), (2, 3)]
        preds = []
        for dims in flips:
            aug = vox.flip(dims) if dims else vox
            logits = self._maybe_forward(aug, cond)
            if dims:
                logits = logits.flip(dims)
            preds.append(logits)
        return torch.stack(preds, dim=0).mean(dim=0)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
    ) -> dict[str, float]:
        total_epochs = self.options.epochs
        summary = "Starting training for %d epochs | train_batches=%d | val_batches=%d"
        self.console.info(summary, total_epochs, len(train_loader), len(val_loader))

        for epoch in range(start_epoch, total_epochs):
            self.console.info("Epoch %d/%d", epoch + 1, total_epochs)

            train_metrics = self._run_train_epoch(train_loader, epoch)
            val_metrics = self._run_eval_epoch(val_loader, epoch)

            payload = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
            save_json(self.run_dir / "info.json", payload)

            summary_msg = _build_epoch_summary(epoch + 1, train_metrics, val_metrics)
            if summary_msg != getattr(self, "_last_epoch_summary", ""):
                self.console.info(summary_msg)
                self._last_epoch_summary = summary_msg

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # For segmentation runs we optimize checkpointing by validation IoU.
                    self.scheduler.step(val_metrics.get("iou", -1.0))
                else:
                    self.scheduler.step()

            current_iou = val_metrics.get("iou", -1.0)
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                self.best_metrics = val_metrics
                self._save_checkpoint(epoch, is_best=True)
                self.console.info(
                    "New best IoU %.4f achieved at epoch %d | checkpoint updated",
                    current_iou,
                    epoch + 1,
                )
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if (
                    self.options.early_stop_patience
                    and self.early_stop_counter >= self.options.early_stop_patience
                ):
                    self.console.info(
                        "Early stopping triggered after %d epochs without improvement",
                        self.options.early_stop_patience,
                    )
                    return self._finalize()

            if (epoch + 1) % self.options.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)
                self.console.info("Saved periodic checkpoint for epoch %d", epoch + 1)

        return self._finalize()

    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        accum_primary = 0.0
        accum_volume = 0.0
        accum_total = 0.0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        last_grad_norm: float | None = None
        total_batches = len(loader)
        self.optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()

        for step, batch in enumerate(loader):
            vox = batch["voxels"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            cond = batch["cond"].to(self.device, non_blocking=True) if self.cond_dim > 0 else None

            if (
                self.cond_dim > 0
                and cond is not None
                and cond.numel() > 0
                and self.options.cond_drop > 0.0
            ):
                keep = (
                    (torch.rand(cond.size(0), device=cond.device) >= self.options.cond_drop)
                    .float()
                    .unsqueeze(1)
                )
                cond = cond * keep

            amp_ctx = (
                autocast(device_type="cuda", dtype=self.device_cfg.amp_dtype, enabled=True)
                if self.use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = self._maybe_forward(vox, cond)
                primary_loss = self.loss_fn(logits, target)
                volume_loss = (
                    self._volume_regularizer(logits, target)
                    if self.volume_reg_weight > 0.0
                    else None
                )
                combined_loss = primary_loss + (
                    self.volume_reg_weight * volume_loss if volume_loss is not None else 0.0
                )
                loss = combined_loss / self.options.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm_value: float | None = None
            if (step + 1) % self.options.accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                grad_norm_value = self._grad_norm()
                if grad_norm_value is not None:
                    grad_norm_sum += grad_norm_value
                    grad_norm_count += 1
                    last_grad_norm = grad_norm_value
                if self.options.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.grad_clip)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            primary_value = primary_loss.detach().item()
            volume_value = volume_loss.detach().item() if volume_loss is not None else 0.0
            total_value = primary_value + self.volume_reg_weight * volume_value

            accum_primary += primary_value
            accum_total += total_value
            if volume_loss is not None:
                accum_volume += volume_value

            elapsed = max(time.perf_counter() - epoch_start, 1e-6)
            processed_batches = step + 1
            iters_per_sec = processed_batches / elapsed
            remaining = max(total_batches - processed_batches, 0)
            eta_seconds = remaining / iters_per_sec if iters_per_sec > 0 else float("inf")

            global_step = epoch * total_batches + step
            if global_step % self.options.log_every == 0:
                scalars = {
                    "train_loss": total_value,
                    "train_primary_loss": primary_value,
                    "train_lr": self.optimizer.param_groups[0]["lr"]
                    if self.optimizer.param_groups
                    else 0.0,
                    "train_iter_per_sec": iters_per_sec,
                    "train_eta_sec": eta_seconds,
                }
                if volume_loss is not None:
                    scalars["train_volume_loss"] = volume_value
                if last_grad_norm is not None:
                    scalars["train_grad_norm"] = last_grad_norm
                self.logger.log_scalars(global_step, scalars)

            progress_metrics = {
                "loss": total_value,
                "primary": primary_value,
                "it_s": iters_per_sec,
                "eta_s": eta_seconds,
            }
            if volume_loss is not None:
                progress_metrics["volume"] = volume_value
            self._progress(
                "train", step + 1, total_batches, progress_metrics, done=(step + 1) == total_batches
            )

        # Flush gradients for trailing mini-batches when total_batches % accum_steps != 0.
        if total_batches and (total_batches % self.options.accum_steps) != 0:
            grad_norm_value: float | None = None
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm_value = self._grad_norm()
            if grad_norm_value is not None:
                grad_norm_sum += grad_norm_value
                grad_norm_count += 1
                last_grad_norm = grad_norm_value
            if self.options.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.grad_clip)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        denom = max(total_batches, 1)
        epoch_time = max(time.perf_counter() - epoch_start, 1e-6)
        summary: dict[str, float] = {
            "loss": accum_total / denom,
            "primary": accum_primary / denom,
            "iter_per_sec": (total_batches / epoch_time) if total_batches else 0.0,
            "epoch_time_sec": epoch_time,
        }
        if self.volume_reg_weight > 0.0:
            summary["volume"] = accum_volume / denom
        if grad_norm_count:
            summary["grad_norm"] = grad_norm_sum / grad_norm_count
        summary["lr"] = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        return summary

    @torch.no_grad()
    def _run_eval_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        total_batches = len(loader)

        logits_list = []
        targets_list = []
        vis_context: torch.Tensor | None = None

        for idx, batch in enumerate(loader):
            vox = batch["voxels"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            cond = batch["cond"].to(self.device, non_blocking=True) if self.cond_dim > 0 else None

            amp_ctx = (
                autocast(device_type="cuda", dtype=self.device_cfg.amp_dtype, enabled=True)
                if self.use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = self._apply_tta(vox, cond)
                primary_loss = self.loss_fn(logits, target)
                volume_loss = (
                    self._volume_regularizer(logits, target)
                    if self.volume_reg_weight > 0.0
                    else None
                )
                loss = primary_loss + (
                    self.volume_reg_weight * volume_loss if volume_loss is not None else 0.0
                )
            val_loss += loss.item()
            logits_list.append(logits.detach().cpu())
            targets_list.append(target.detach().cpu())
            self._progress(
                "val",
                idx + 1,
                total_batches,
                {"loss": loss.item()},
                done=(idx + 1) == total_batches,
            )

            if self.options.vis_every and vis_context is None:
                context_tensor = self._extract_context(batch)
                if context_tensor is not None:
                    vis_context = context_tensor
                    paths = batch.get("path")
                    if isinstance(paths, (list, tuple)) and paths:
                        str(paths[0])
                    elif isinstance(paths, str):
                        pass

        avg_loss = val_loss / max(total_batches, 1)
        logits_tensor = torch.cat(logits_list, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)

        base_threshold = 0.5
        base_metrics = seg_metrics.evaluate(logits_tensor, targets_tensor, base_threshold)

        if self.options.auto_thresh:
            thr, best_metrics = seg_metrics.auto_threshold(
                logits_tensor, targets_tensor, self.options.thresh_grid
            )
            metrics = {"loss": avg_loss, "threshold": thr, **best_metrics}
        else:
            metrics = {"loss": avg_loss, "threshold": base_threshold, **base_metrics}

        if (
            self.options.vis_every
            and (epoch + 1) % self.options.vis_every == 0
            and logits_tensor.size(0) > 0
        ):
            thresh = metrics.get("threshold", base_threshold)
            self._write_visuals(
                logits_tensor[0],
                targets_tensor[0],
                epoch,
                thresh,
                context_tensor=vis_context,
            )

        log_payload = {f"val_{k}": v for k, v in metrics.items()}
        self.logger.log_scalars(epoch, log_payload)
        return metrics

    def _volume_regularizer(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target_float = target.to(logits.dtype)
        if target_float.dim() == probs.dim() - 1:
            target_float = target_float.unsqueeze(1)
        dims = tuple(range(1, probs.dim()))
        pred_vol = probs.sum(dim=dims)
        target_vol = target_float.sum(dim=dims)
        return ((pred_vol - target_vol).abs() / target_vol.clamp_min(1.0)).mean()

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "best_iou": self.best_iou,
            "best_metrics": self.best_metrics,
            "early_stop_counter": self.early_stop_counter,
        }
        ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(state, self.run_dir / "best.pt")

    def load_checkpoint(self, state: dict[str, object]) -> int:
        epoch = int(state.get("epoch", -1))
        model_state = state.get("model")
        if model_state:
            self.model.load_state_dict(model_state)  # type: ignore[arg-type]

        optimizer_state = state.get("optimizer")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]

        scheduler_state = state.get("scheduler")
        if self.scheduler is not None and scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)  # type: ignore[arg-type]

        scaler_state = state.get("scaler")
        if scaler_state and self.use_amp:
            self.scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]

        self.best_iou = float(state.get("best_iou", self.best_iou))
        best_metrics = state.get("best_metrics")
        if isinstance(best_metrics, dict):
            self.best_metrics = best_metrics

        self.early_stop_counter = int(state.get("early_stop_counter", self.early_stop_counter))

        self.console.info(
            "Restored checkpoint | epoch=%d | best_iou=%.4f | early_stop_counter=%d",
            epoch,
            self.best_iou,
            self.early_stop_counter,
        )
        return max(epoch + 1, 0)

    def _write_visuals(
        self,
        logits_sample: torch.Tensor,
        target_sample: torch.Tensor,
        epoch: int,
        threshold: float,
        context_tensor: torch.Tensor | None = None,
    ) -> None:
        prob = torch.sigmoid(logits_sample)
        name = f"epoch_{epoch + 1:04d}_sample0"
        prob_cpu = prob.detach().cpu()
        target_cpu = target_sample.detach().cpu()
        save_max_projection(self.vis_dir, name, target_cpu, prob_cpu, threshold)
        self.console.info("Saved visualization -> %s", (self.vis_dir / f"{name}_projections.png"))

        if context_tensor is not None:
            context_cpu = context_tensor.detach().cpu()
            stride = self._compute_context_stride(prob_cpu, context_cpu)
            save_3d_context_visualization(
                out_dir=self.vis_dir,
                name=name,
                prediction=prob_cpu,
                context=context_cpu,
                threshold=threshold,
                context_stride=stride,
            )
            self.console.info(
                "Saved 3D context visualization -> %s", (self.vis_dir / f"{name}_3d_context.png")
            )

    def _progress(
        self,
        phase: str,
        current: int,
        total: int,
        metrics: dict[str, float],
        done: bool = False,
    ) -> None:
        total = max(total, 1)
        current = max(min(current, total), 0)
        percent = (current / total) * 100.0

        def _format_value(value: object) -> str:
            if isinstance(value, (int, float)):
                if math.isfinite(value):
                    return f"{value:.4f}"
                return "nan"
            return str(value)

        metrics_str = " | ".join(f"{k}={_format_value(v)}" for k, v in metrics.items())
        line = f"\r[{phase}] {current:>4}/{total:<4} ({percent:5.1f}%) {metrics_str}"
        prev_len = self._progress_last.get(phase, 0)
        padding = max(prev_len - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        if done or current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_last.pop(phase, None)
        else:
            self._progress_last[phase] = len(line)

    def _finalize(self) -> dict[str, float]:
        save_json(self.run_dir / "final_metrics.json", self.best_metrics)
        self.logger.close()
        if self.best_metrics:
            loss = self.best_metrics.get("loss", 0.0)
            iou = self.best_metrics.get("iou", self.best_metrics.get("base_iou", -1.0))
            precision = self.best_metrics.get(
                "precision", self.best_metrics.get("base_precision", 0.0)
            )
            recall = self.best_metrics.get("recall", self.best_metrics.get("base_recall", 0.0))
            msg = (
                f"{BOLD}Best metrics{RESET} | "
                f"{CYAN}loss={loss:.4f}{RESET} | "
                f"{GREEN}IoU={iou:.4f} | precision={precision:.4f} | recall={recall:.4f}{RESET}"
            )
            self.console.info(msg)
        return self.best_metrics

    def _extract_context(
        self, batch: dict[str, torch.Tensor], index: int = 0
    ) -> torch.Tensor | None:
        paths = batch.get("path")
        if paths is None:
            return None
        if isinstance(paths, (list, tuple)):
            if not paths:
                return None
            sample_path = paths[index]
        else:
            sample_path = paths
        if isinstance(sample_path, bytes):
            sample_path = sample_path.decode()
        return self._load_context_volume(Path(sample_path))

    def _load_context_volume(self, sample_path: Path) -> torch.Tensor | None:
        try:
            with np.load(sample_path, allow_pickle=True) as data:
                key = None
                if "Y_neigh" in data:
                    key = "Y_neigh"
                elif "context" in data:
                    key = "context"
                if key is None:
                    return None
                ctx = torch.from_numpy(data[key]).float()
                return ctx
        except Exception as exc:
            self.console.debug("Context visualization skipped for %s: %s", sample_path, exc)
        return None

    @staticmethod
    def _compute_context_stride(prediction: torch.Tensor, context: torch.Tensor) -> int:
        def _shape(t: torch.Tensor) -> torch.Size:
            vol = t
            while vol.dim() > 3 and vol.size(0) == 1:
                vol = vol.squeeze(0)
            if vol.dim() > 3:
                vol = vol[0]
            return vol.shape[-3:]

        pred_shape = _shape(prediction)
        ctx_shape = _shape(context)
        ratios = []
        for ctx_dim, pred_dim in zip(ctx_shape, pred_shape, strict=False):
            if pred_dim > 0:
                ratios.append(max(1, round(ctx_dim / pred_dim)))
        if not ratios:
            return 1
        return max(1, round(sum(ratios) / len(ratios)))

    def _grad_norm(self) -> float | None:
        total = 0.0
        has_grad = False
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                return float("nan")
            total += grad.pow(2).sum().item()
            has_grad = True
        if not has_grad:
            return None
        return math.sqrt(total)


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _build_epoch_summary(
    epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
) -> str:
    train_loss = train_metrics.get("loss", 0.0)
    val_loss = val_metrics.get("loss", 0.0)
    iou = val_metrics.get("iou", val_metrics.get("base_iou", -1.0))
    f1 = val_metrics.get("f1", -1.0)
    precision = val_metrics.get("precision", val_metrics.get("base_precision", 0.0))
    recall = val_metrics.get("recall", val_metrics.get("base_recall", 0.0))
    delta = val_metrics.get("delta_vol", val_metrics.get("base_delta_vol", 0.0))

    summary = (
        f"{BOLD}Epoch {epoch}{RESET} "
        f"{CYAN}| train_loss={train_loss:.4f}{RESET} "
        f"{MAGENTA}| val_loss={val_loss:.4f}{RESET} "
        f"{GREEN}| IoU={iou:.4f} | F1={f1:.4f} | P={precision:.4f} "
        f"| R={recall:.4f} | Δvol={delta:.2f}{RESET}"
    )
    return summary
