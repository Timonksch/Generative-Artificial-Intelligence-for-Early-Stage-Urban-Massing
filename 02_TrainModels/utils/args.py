"""Shared CLI argument builders."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def add_hyphenated_aliases(parser: argparse.ArgumentParser) -> None:
    """Add readable hyphen aliases for options whose destination uses underscores."""
    for action in parser._actions:
        for option in tuple(action.option_strings):
            if not option.startswith("--") or "_" not in option:
                continue
            alias = option.replace("_", "-")
            if alias in parser._option_string_actions:
                continue
            action.option_strings.insert(0, alias)
            parser._option_string_actions[alias] = action


def add_common_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("common")
    group.add_argument(
        "--data_root", type=str, required=True, help="Root directory containing NPZ samples."
    )
    group.add_argument(
        "--out_dir", type=str, required=True, help="Directory to store run artifacts."
    )
    group.add_argument("--epochs", type=int, default=50)
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--accum_steps", type=int, default=4)
    group.add_argument("--lr", type=float, default=1e-3)
    group.add_argument("--optimizer", type=str, default="adamw", choices=("adam", "adamw", "sgd"))
    group.add_argument("--weight_decay", type=float, default=0.01)
    group.add_argument(
        "--lr_schedule", type=str, default="none", choices=("none", "cosine", "plateau")
    )
    group.add_argument("--early_stop_patience", type=int, default=0)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_workers", type=int, default=0)
    group.add_argument("--log_every", type=int, default=25)
    group.add_argument("--save_every", type=int, default=5)
    group.add_argument("--grad_clip", type=float, default=1.0)
    group.add_argument("--vis_every", type=int, default=0)
    group.add_argument(
        "--allow_resume_mismatch",
        action="store_true",
        help="Allow auto-resume from existing checkpoints even when stored config differs.",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument("--downsample_stride", type=int, default=0)
    group.add_argument("--crop_size", type=int, default=0)
    group.add_argument("--add_coords", action="store_true")
    group.add_argument("--channels", nargs="+", default=("C0", "C1", "C2", "C3"))
    group.add_argument("--x_indices", nargs="+", type=int, default=None)
    group.add_argument(
        "--split_manifest",
        type=str,
        default=None,
        help="Path to JSON manifest containing explicit train/val/test sample IDs.",
    )
    group.add_argument("--max_samples", type=int, default=None)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("eval")
    group.add_argument("--tta", action="store_true")
    group.add_argument("--tta_mode", type=str, default="rot90", choices=("rot90", "flip"))
    group.add_argument("--auto_thresh", action="store_true")
    group.add_argument("--thresh_grid", nargs="+", type=float, default=(0.3, 0.4, 0.5))


def add_unet_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("unet")
    group.add_argument("--base_ch", type=int, default=16)
    group.add_argument("--depth", type=int, default=4)
    group.add_argument("--surface_weight", type=float, default=1.0)
    group.add_argument("--bce_w", type=float, default=0.5)
    group.add_argument("--dice_w", type=float, default=0.5)
    group.add_argument("--volume_reg_weight", type=float, default=0.1)


def add_cond_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("conditioning")
    group.add_argument("--cond_dim", type=int, default=0)
    group.add_argument("--cond_select", nargs="+", type=int, default=None)
    group.add_argument("--cond_drop", type=float, default=0.0)
    group.add_argument("--cond_stats_path", type=str, default="")


def add_ldm_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ldm")
    group.add_argument(
        "--mode",
        type=str,
        default="diffusion",
        choices=("vae_pretrain", "diffusion", "joint"),
    )
    group.add_argument("--input_res", type=int, default=64)
    group.add_argument("--latent_res", type=int, default=16)
    group.add_argument("--latent_channels", type=int, default=4)
    group.add_argument("--vae_base_ch", type=int, default=32)
    group.add_argument("--kl_weight", type=float, default=1e-4)
    group.add_argument("--kl_anneal_epochs", type=int, default=0)
    group.add_argument("--diff_base_ch", type=int, default=64)
    group.add_argument("--diff_depth", type=int, default=3)
    group.add_argument("--time_dim", type=int, default=256)
    group.add_argument("--train_timesteps", type=int, default=1000)
    group.add_argument("--noise_schedule", type=str, default="linear", choices=("linear", "cosine"))
    group.add_argument("--sampler", type=str, default="ddpm", choices=("ddpm", "ddim"))
    group.add_argument("--sample_timesteps", type=int, default=50)
    group.add_argument("--ddim_eta", type=float, default=0.0)
    group.add_argument("--cfg_scale", type=float, default=1.0)
    group.add_argument("--ema", action="store_true")
    group.add_argument("--ema_decay", type=float, default=0.999)
    group.add_argument("--vae_checkpoint", type=str, default="")
    group.add_argument("--ema_sample_start", type=int, default=30)
    group.add_argument("--sample_every", type=int, default=0)
    group.add_argument("--num_sample_batches", type=int, default=4)
    group.add_argument("--num_sample_conds", type=int, default=3)


def str_list(values: Sequence[int] | None) -> str:
    """Format an optional integer sequence for logging."""
    if values is None:
        return "None"
    return ",".join(str(v) for v in values)
