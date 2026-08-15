"""Video/photos -> a flat directory of images COLMAP can consume."""

import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def prepare_images(input_path: Path, frames_dir: Path, fps: int) -> Path:
    """Return a directory of images ready for COLMAP.

    `input_path` is either a directory of photos (used in place, unchanged)
    or a single video file (extracted to `frames_dir` at `fps`).
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            sys.exit(f"No images ({sorted(IMAGE_EXTS)}) found in {input_path}")
        print(f"Using {len(images)} photos directly from {input_path}")
        return input_path

    if input_path.suffix.lower() not in VIDEO_EXTS:
        sys.exit(f"{input_path} is neither a directory of photos nor a video file {sorted(VIDEO_EXTS)}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"fps={fps}",
        "-q:v", "1",  # highest JPEG quality
        str(frames_dir / "%06d.jpg"),
    ]
    print(f"Extracting {input_path.name} -> {frames_dir} @ {fps} fps")
    subprocess.run(cmd, check=True)

    n = sum(1 for _ in frames_dir.glob("*.jpg"))
    if n == 0:
        sys.exit(f"ffmpeg produced no frames from {input_path}")
    print(f"Extracted {n} frames")
    return frames_dir
