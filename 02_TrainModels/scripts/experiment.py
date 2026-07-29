"""Experiment runner that spawns training scripts from JSON configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.args import add_hyphenated_aliases  # noqa: E402
from utils.io import ensure_dir, load_json, save_json  # noqa: E402

SCRIPT_MAP = {
    "unet": "train_unet.py",
    "cond_unet": "train_unet_cond.py",
    "ldm": "train_ldm.py",
}
EVAL_SCRIPT = "infer.py"


def build_command(script: Path, params: dict[str, object]) -> list[str]:
    cmd = [sys.executable, str(script)]
    # Skip keys that are metadata, not training arguments
    skip_keys = {"model", "run_name", "description"}

    for key, value in params.items():
        if key in skip_keys:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def merge_params(global_cfg: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(global_cfg)
    merged.update(override)
    return merged


def build_eval_command(  # noqa: PLR0917
    python_exec: str,
    script: Path,
    model_kind: str,
    checkpoint: Path,
    config_path: Path,
    data_root: str,
    out_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
    threshold: float | None = None,
    force_auto_thresh: bool = False,
) -> list[str]:
    cmd = [
        python_exec,
        str(script),
        "--model",
        model_kind,
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config_path),
        "--data_root",
        data_root,
        "--out_dir",
        str(out_dir),
        "--split",
        split,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    if force_auto_thresh:
        cmd.append("--auto-thresh")
    if threshold is not None:
        cmd.extend(["--threshold", str(float(threshold)), "--no-auto-thresh"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple training runs from a config JSON."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_parent", required=True, type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--post_eval_split", type=str, default="test", choices=("train", "val", "test")
    )
    parser.add_argument(
        "--no_post_eval",
        action="store_true",
        help="Disable automatic post-training inference evaluation.",
    )
    add_hyphenated_aliases(parser)
    args = parser.parse_args()

    config = load_json(args.config)
    name = config["name"]
    global_params = config.get("global", {})
    overrides = config.get("overrides", [])

    exp_dir = ensure_dir(Path(args.out_parent) / name)
    summary = {"name": name, "runs": []}

    for override in overrides:
        params = merge_params(global_params, override)
        model_kind = params.pop("model")
        run_name = params.pop("run_name")
        script_name = SCRIPT_MAP.get(model_kind)
        if not script_name:
            raise ValueError(f"Unsupported model '{model_kind}'")

        run_dir = ensure_dir(exp_dir / run_name)
        params["out_dir"] = str(run_dir)
        script_path = ROOT / "scripts" / script_name
        cmd = build_command(script_path, params)
        print("Launching:", " ".join(cmd), flush=True)
        result = {"run_name": run_name, "model": model_kind, "train_command": cmd}

        if args.dry_run:
            result["status"] = "skipped"
            result["post_eval"] = "skipped"
        else:
            proc = subprocess.run(cmd, check=False)  # noqa: S603 - fixed local script
            result["train_returncode"] = proc.returncode
            result["post_eval"] = "disabled" if args.no_post_eval else "pending"
            if proc.returncode != 0:
                result["status"] = f"failed_train({proc.returncode})"
            elif args.no_post_eval:
                result["status"] = "ok_train_only"
            else:
                checkpoint = run_dir / "best.pt"
                config_path = run_dir / "config.json"
                eval_out = ensure_dir(run_dir / f"eval_{args.post_eval_split}")
                if not checkpoint.exists():
                    result["status"] = "failed_eval(no_best_checkpoint)"
                elif not config_path.exists():
                    result["status"] = "failed_eval(no_run_config)"
                else:
                    eval_batch_size = int(params.get("batch_size", 1) or 1)
                    eval_num_workers = int(params.get("num_workers", 0) or 0)
                    eval_seed = int(params.get("seed", 42) or 42)
                    eval_max_samples = params.get("max_samples")
                    eval_max_samples = (
                        int(eval_max_samples) if eval_max_samples is not None else None
                    )
                    data_root = str(params.get("data_root", ""))
                    if not data_root:
                        result["status"] = "failed_eval(no_data_root)"
                    else:
                        eval_script = ROOT / "scripts" / EVAL_SCRIPT
                        selected_threshold = None
                        wants_auto_thresh = bool(params.get("auto_thresh", False))

                        # Calibrate automatic test thresholds on validation data first.
                        if args.post_eval_split == "test" and wants_auto_thresh:
                            val_eval_out = ensure_dir(run_dir / "eval_val")
                            val_eval_cmd = build_eval_command(
                                python_exec=sys.executable,
                                script=eval_script,
                                model_kind=model_kind,
                                checkpoint=checkpoint,
                                config_path=config_path,
                                data_root=data_root,
                                out_dir=val_eval_out,
                                split="val",
                                batch_size=eval_batch_size,
                                num_workers=eval_num_workers,
                                seed=eval_seed,
                                max_samples=eval_max_samples,
                                threshold=None,
                                force_auto_thresh=True,
                            )
                            print(
                                "Val-threshold calibration:",
                                " ".join(val_eval_cmd),
                                flush=True,
                            )
                            val_eval_proc = subprocess.run(  # noqa: S603 - fixed local script
                                val_eval_cmd, check=False
                            )
                            result["val_eval_command"] = val_eval_cmd
                            result["val_eval_returncode"] = val_eval_proc.returncode
                            if val_eval_proc.returncode != 0:
                                result["status"] = (
                                    f"failed_val_calibration({val_eval_proc.returncode})"
                                )
                                result["post_eval"] = "failed"
                                summary["runs"].append(result)
                                continue
                            val_metrics_path = val_eval_out / "metrics.json"
                            if not val_metrics_path.exists():
                                result["status"] = "failed_val_calibration(no_metrics)"
                                result["post_eval"] = "failed"
                                summary["runs"].append(result)
                                continue
                            val_metrics = load_json(val_metrics_path)
                            result["val_metrics"] = val_metrics
                            if not isinstance(val_metrics, dict) or "threshold" not in val_metrics:
                                result["status"] = "failed_val_calibration(no_threshold)"
                                result["post_eval"] = "failed"
                                summary["runs"].append(result)
                                continue
                            selected_threshold = float(val_metrics["threshold"])

                        eval_cmd = build_eval_command(
                            python_exec=sys.executable,
                            script=eval_script,
                            model_kind=model_kind,
                            checkpoint=checkpoint,
                            config_path=config_path,
                            data_root=data_root,
                            out_dir=eval_out,
                            split=args.post_eval_split,
                            batch_size=eval_batch_size,
                            num_workers=eval_num_workers,
                            seed=eval_seed,
                            max_samples=eval_max_samples,
                            threshold=selected_threshold,
                            force_auto_thresh=False,
                        )
                        print("Post-eval:", " ".join(eval_cmd), flush=True)
                        eval_proc = subprocess.run(  # noqa: S603 - fixed local script
                            eval_cmd, check=False
                        )
                        result["eval_command"] = eval_cmd
                        result["eval_returncode"] = eval_proc.returncode
                        if eval_proc.returncode != 0:
                            result["status"] = f"failed_eval({eval_proc.returncode})"
                            result["post_eval"] = "failed"
                        else:
                            metrics_path = eval_out / "metrics.json"
                            if metrics_path.exists():
                                result["test_metrics"] = load_json(metrics_path)
                            result["status"] = "ok"
                            result["post_eval"] = "ok"
        summary["runs"].append(result)

    best = None
    for run in summary["runs"]:
        metrics_doc = run.get("test_metrics")
        if not isinstance(metrics_doc, dict):
            continue
        metrics = metrics_doc.get("metrics", {})
        if not isinstance(metrics, dict) or "iou" not in metrics:
            continue
        iou = float(metrics["iou"])
        if best is None or iou > best["iou"]:
            best = {"run_name": run["run_name"], "model": run["model"], "iou": iou}
    if best is not None:
        summary["best_by_test_iou"] = best

    save_json(exp_dir / "experiment_summary.json", summary)


if __name__ == "__main__":
    main()
