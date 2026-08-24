"""Create a GIF from a bounded sequence of static image frames."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

_NUMBER_PATTERN = re.compile(r"(\d+)")


def create_gif(
    input_directory: Path,
    output_path: Path,
    *,
    frames_per_second: int = 12,
    pattern: str = "*.png",
    maximum_frames: int = 1000,
) -> Path:
    """Create an animated GIF from naturally sorted image frames.

    Args:
        input_directory: Directory containing source images.
        output_path: Destination ending in ``.gif``.
        frames_per_second: Positive playback rate.
        pattern: Glob selecting source frames.
        maximum_frames: Safety bound for memory use.

    Returns:
        Absolute path of the GIF written.

    Raises:
        ValueError: If arguments are invalid or no frames are found.

    """
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    if output_path.suffix.lower() != ".gif":
        raise ValueError("Animation output must use the .gif extension")
    frame_paths = sorted(input_directory.glob(pattern), key=_natural_key)
    if not frame_paths:
        raise ValueError(f"No frames matching {pattern!r} under {input_directory}")
    if len(frame_paths) > maximum_frames:
        raise ValueError(f"Refusing to load {len(frame_paths)} frames; maximum is {maximum_frames}")
    frames = [_read_rgb_image(path) for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_milliseconds = round(1000 / frames_per_second)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_milliseconds,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()
    return output_path.resolve()


def _natural_key(path: Path) -> tuple[object, ...]:
    """Return a mixed text/number key for deterministic frame ordering."""
    parts = _NUMBER_PATTERN.split(path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _read_rgb_image(path: Path) -> Image.Image:
    """Read one image into an independent RGB buffer."""
    with Image.open(path) as image:
        return image.convert("RGB")
