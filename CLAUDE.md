# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What This Is

Phase 1 of an open, automatable pipeline turning a small video or photo collection walking around a *static* object into a 3D point cloud: `core/frames.py` (video → frames or photo passthrough) → `core/sfm.py` (COLMAP sparse SfM) → `core/splat.py` (3D Gaussian Splatting via `gsplat`, CUDA-only). See [docs/architecture.md](docs/architecture.md) for the full design and why COLMAP+gsplat were chosen over OpenMVS / feed-forward (DUSt3R/MASt3R/VGGT) alternatives.

Phase 2 (moving/non-rigid objects) is intentionally **not designed yet** — see the placeholder section in `docs/architecture.md`. Don't build toward it speculatively; when the user asks to start Phase 2, that's a decision session (multi-camera rig vs. monocular casual video + deformable splatting), not an extension of existing code.

## Environment

- No local GPU on the primary dev machine (Windows). `core/sfm.py` runs fine locally (CPU, or the Windows nocuda COLMAP binary). `core/splat.py` hard-exits with a clear message if `torch.cuda.is_available()` is False — that's expected locally; real training happens on RunPod.
- RunPod base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. `uv pip` for installs (see `setup-env.sh` / `Dockerfile.runpod`).
- COLMAP is a system binary, not a pip package. Windows: `C:/Program Files/colmap-x64-windows-nocuda/bin/colmap.exe` (auto-detected). RunPod: baked into `Dockerfile.runpod` via a standalone micromamba/conda-forge env (Ubuntu 22.04's apt colmap is an incompatible v3.x — see `docs/architecture.md`).

## Conventions

- No pluggable backend abstraction for pose/dense reconstruction — COLMAP and gsplat are hardcoded, not swappable via config. Deliberate: avoid premature abstraction until a second backend is actually needed.
- `configs/default.json` uses the JSON-with-sections pattern (`frames` / `colmap` / `gsplat`) shared with sibling projects (see `c:/rory/scripts/EncroachNet/configs/default.json`).
- `data/`, `checkpoints/`, `output/` are gitignored and auto-created by the pipeline.

## Common Commands

```bash
bash setup-env.sh                                    # one-time env bootstrap
bash scripts/download_dataset.sh south-building        # small local smoke-test dataset
python reconstruct.py --input <video_or_photo_dir> --output output/<scene> [--skip-splat]
```

## Git

- Commit email: rory@mcclenagan.net
