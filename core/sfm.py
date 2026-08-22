"""COLMAP structure-from-motion wrapper: feature extraction -> matching -> sparse mapping."""

import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PLATFORM = platform.system()  # "Linux" on RunPod, "Windows" locally

# Windows dev machines use a prebuilt nocuda binary; RunPod images put `colmap`
# (conda-forge build, see Dockerfile.runpod) on PATH.
_WIN_COLMAP = Path("C:/Program Files/colmap-x64-windows-nocuda/bin/colmap.exe")
COLMAP_BIN = str(_WIN_COLMAP) if PLATFORM == "Windows" and _WIN_COLMAP.exists() else "colmap"


def _run(cmd: list, **kwargs):
    print(f"+ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def run_sparse(
    image_dir: Path,
    workspace_dir: Path,
    matcher: str = "sequential",
    sequential_overlap: int = 10,
    max_image_size: int = 1600,
    max_num_features: int = 8192,
) -> Path:
    """Run COLMAP feature extraction -> matching -> sparse mapping.

    `matcher` is "sequential" (ordered video frames) or "exhaustive"
    (unordered photo collections, where every pair is a match candidate).
    Returns the path to the resulting sparse model directory.
    """
    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    db_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"Platform: {PLATFORM}")

    # COLMAP's own --use_gpu SIFT path needs either a CUDA-compiled colmap
    # binary or a working OpenGL context; neither binary this project installs
    # has the former (Windows nocuda build, RunPod's conda-forge `colmap`
    # package -- see Dockerfile.runpod), and headless pods have no display for
    # the latter. Always run COLMAP's own SIFT on CPU; the CUDA-only step is
    # core/splat.py's gsplat training, not this one.
    _run([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_num_features),
    ])

    if matcher == "sequential":
        _run([
            COLMAP_BIN, "sequential_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0",
            "--SequentialMatching.overlap", str(sequential_overlap),
            "--SequentialMatching.loop_detection", "0",
        ])
    elif matcher == "exhaustive":
        _run([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0",
        ])
    else:
        sys.exit(f"Unknown matcher '{matcher}' (expected 'sequential' or 'exhaustive')")

    _run([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", "8",
    ])

    models = [m for m in sparse_dir.iterdir() if m.is_dir()]
    if not models:
        sys.exit("COLMAP mapper produced no models -- insufficient overlap or matching failed.")
    if len(models) > 1:
        import pycolmap
        model = max(models, key=lambda m: len(pycolmap.Reconstruction(str(m)).images))
        print(f"COLMAP mapper produced {len(models)} disconnected models -- using the largest ({model.name})")
    else:
        model = models[0]
    print(f"Done -- sparse model at {model}")
    return model
