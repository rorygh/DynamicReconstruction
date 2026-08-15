# DynamicReconstruction

Open, automatable, GPU-based pipeline: a small video or photo collection walking around an object → a 3D point cloud. Fully scriptable/customizable in Python.

**Repo:** https://github.com/rorygh/DynamicReconstruction

---

## What it does

**Phase 1 (this repo, today):** RGB video or photo collection of a static object → COLMAP structure-from-motion (camera poses + sparse points) → 3D Gaussian Splatting (`gsplat`) → dense point cloud.

```
input:  video.mp4, or a directory of photos
output: output/<scene>/points.ply
```

**Phase 2 (planned, not built):** the same, for a *moving* object — see [docs/architecture.md](docs/architecture.md#phase-2--moving-object-not-yet-designed).

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline diagram and design rationale (why COLMAP + gsplat over OpenMVS / feed-forward alternatives).

---

## Quick Start

### Environment

```bash
bash setup-env.sh
```

Installs into system Python via `uv` (no conda on the RunPod base image). Requires COLMAP on PATH (`Dockerfile.runpod` bakes this in; locally on Windows the pipeline auto-detects `C:/Program Files/colmap-x64-windows-nocuda/bin/colmap.exe`).

### Local smoke test (no GPU needed)

```bash
bash scripts/download_dataset.sh south-building
python reconstruct.py --input data/raw/south-building/images --output output/south-building --skip-splat
```

Runs frame prep + COLMAP sparse reconstruction only — validates the CPU-portable half of the pipeline. See [docs/datasets.md](docs/datasets.md).

### Full pipeline (COLMAP + Gaussian Splatting — requires CUDA, run on RunPod)

```bash
python reconstruct.py --input path/to/video.mp4 --output output/myscene
python reconstruct.py --input path/to/photos/ --output output/myscene
```

### RunPod / Docker

```bash
docker build --platform linux/amd64 -f Dockerfile.runpod -t rorygh/dynamicreconstruction:latest .
docker push rorygh/dynamicreconstruction:latest
```

Or automatically: `.github/workflows/docker-publish.yml` builds and pushes this same image on every push to `main` that touches the Dockerfile/pipeline code. Needs `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repo secrets set once (Settings -> Secrets and variables -> Actions).

Launch a pod with an RTX 3090 / A6000 or similar, CUDA 12.4 base image. Mount a network volume at `/workspace/`.

---

## Project Structure

```
DynamicReconstruction/
├── reconstruct.py           Phase 1 CLI: video/photos -> point cloud
├── setup-env.sh              uv-based environment bootstrap
├── Dockerfile.runpod          RunPod deployment
├── configs/
│   └── default.json           fps / COLMAP / gsplat parameters
├── core/
│   ├── frames.py                video -> frames (or photo passthrough)
│   ├── sfm.py                   COLMAP feature extraction -> matching -> sparse mapping
│   └── splat.py                 3D Gaussian Splatting training + .ply export (CUDA-only)
├── scripts/
│   └── download_dataset.sh    Small sample datasets for local smoke-testing
└── docs/
    ├── architecture.md          Full pipeline design + Phase 2 notes
    └── datasets.md               Dataset survey + licenses
```

## Git

- Remote: https://github.com/rorygh/DynamicReconstruction
- Commit email: rory@mcclenagan.net

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `gsplat` | CUDA-accelerated 3D Gaussian Splatting rasterizer + densification strategy |
| `pycolmap` | Reading COLMAP camera poses / sparse points into Python |
| `plyfile` | Point cloud `.ply` export |
| COLMAP (system binary, not pip) | Structure-from-motion: feature extraction, matching, sparse mapping |
