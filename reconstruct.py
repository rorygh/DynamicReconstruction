#!/usr/bin/env python3
"""Phase 1 entry point: video or photo collection -> 3D point cloud.

    python reconstruct.py --input path/to/video.mp4 --output output/myscene
    python reconstruct.py --input path/to/photos/ --output output/myscene --skip-splat
"""

import argparse
from pathlib import Path

from core import DATA_DIR, load_config
from core.frames import prepare_images
from core.sfm import run_sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Video file or directory of photos")
    parser.add_argument("--output", required=True, help="Output directory for this scene")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-splat", action="store_true", help="Stop after COLMAP sparse reconstruction")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    input_path = Path(args.input)
    if input_path.is_file():
        scene_name = input_path.stem
    elif input_path.name == "images":
        # convention used by scripts/download_dataset.sh: data/raw/<scene>/images/
        scene_name = input_path.parent.name
    else:
        scene_name = input_path.name
    workspace_dir = DATA_DIR / "colmap" / scene_name
    frames_dir = DATA_DIR / "frames" / scene_name
    output_dir = Path(args.output)

    image_dir = prepare_images(input_path, frames_dir, fps=cfg["frames"]["fps"])

    matcher = "sequential" if input_path.is_file() else "exhaustive"
    sparse_model, image_dir = run_sparse(
        image_dir, workspace_dir,
        matcher=matcher,
        sequential_overlap=cfg["colmap"]["sequential_overlap"],
        max_image_size=cfg["colmap"]["max_image_size"],
        max_num_features=cfg["colmap"]["max_num_features"],
    )

    if args.skip_splat:
        print(f"Skipping Gaussian Splatting. Sparse model at {sparse_model}")
        return

    from core.splat import train  # deferred: imports torch/gsplat, CUDA-only
    train(
        sparse_model, image_dir, output_dir,
        iterations=cfg["gsplat"]["iterations"],
        sh_degree=cfg["gsplat"]["sh_degree"],
        means_lr=cfg["gsplat"]["means_lr"],
        opacity_lr=cfg["gsplat"]["opacity_lr"],
        scales_lr=cfg["gsplat"]["scales_lr"],
        quats_lr=cfg["gsplat"]["quats_lr"],
        colors_lr=cfg["gsplat"]["colors_lr"],
        densify_until_iter=cfg["gsplat"]["densify_until_iter"],
    )


if __name__ == "__main__":
    main()
