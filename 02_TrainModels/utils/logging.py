"""Logging helpers for console, CSV, and TensorBoard output."""

from __future__ import annotations

import csv
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


class RunLogger:
    """Thin wrapper that mirrors metrics to stdout, CSV, and TensorBoard."""

    def __init__(
        self,
        log_path: Path,
        enable_tb: bool = False,
        console: bool = True,
        name: str = "run",
    ) -> None:
        self.log_path = log_path
        self.csv_path = log_path.with_suffix(".csv")
        self._csv_fields = self._load_csv_fields()
        self._csv_header_written = bool(self._csv_fields)
        self._tb_writer = (
            SummaryWriter(log_path.parent / "tb") if enable_tb and SummaryWriter else None
        )
        self._console_logger: logging.Logger | None = None
        if console:
            parent = logging.getLogger("urban3d")
            if not parent.handlers:
                configure_root_logger()
            self._console_logger = parent.getChild(name)

    def close(self) -> None:
        if self._tb_writer:
            self._tb_writer.flush()
            self._tb_writer.close()

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        """Persist metrics to CSV (append) and tensorboard if available."""
        if not scalars:
            return
        self._append_csv(step, scalars)
        if self._tb_writer:
            for key, value in scalars.items():
                self._tb_writer.add_scalar(key, value, global_step=step)
        if self._console_logger:
            formatted = ", ".join(f"{key}={_format_value(value)}" for key, value in scalars.items())
            self._console_logger.info("step %d | %s", step, formatted)

    def _append_csv(self, step: int, scalars: dict[str, float]) -> None:
        row = {**{"step": step}, **scalars}
        self._ensure_csv_schema(row.keys())

        with self.csv_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if not self._csv_header_written:
                writer.writerow(self._csv_fields)
                self._csv_header_written = True
            writer.writerow([row.get(field, "") for field in self._csv_fields])

    def _load_csv_fields(self) -> list[str]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            header = next(reader, [])
        return [field for field in header if field]

    def _ensure_csv_schema(self, keys: Iterable[str]) -> None:
        incoming = list(dict.fromkeys(keys))
        if not incoming:
            return
        if not self._csv_fields:
            self._csv_fields = incoming
            return

        new_fields = [field for field in incoming if field not in self._csv_fields]
        if not new_fields:
            return
        old_fields = list(self._csv_fields)
        self._csv_fields.extend(new_fields)
        if self._csv_header_written:
            self._rewrite_csv_header(old_fields, self._csv_fields)

    def _rewrite_csv_header(self, old_fields: list[str], new_fields: list[str]) -> None:
        if not self.csv_path.exists() or not old_fields:
            return
        rows = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames:
                return
            rows.extend(reader)

        with self.csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=new_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in new_fields})


def configure_root_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger."""
    logger = logging.getLogger("urban3d")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )

    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def tabulate(rows: Iterable[dict[str, float]]) -> str:
    """Return a simple aligned table string for CLI metrics."""
    rows = list(rows)
    if not rows:
        return ""

    headers = rows[0].keys()
    widths = {
        header: max(len(header), *(len(f"{row[header]:.4f}") for row in rows)) for header in headers
    }
    header_line = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    sep_line = "-+-".join("-" * widths[header] for header in headers)
    body = "\n".join(
        " | ".join(f"{row[header]:<{widths[header]}.4f}" for header in headers) for row in rows
    )
    return "\n".join([header_line, sep_line, body])
