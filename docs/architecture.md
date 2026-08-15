# Architecture

## Phase 1 — Static Object (video or photo collection → point cloud)

```
video (.mp4/...) or photo directory
              │
              ▼
┌───────────────────────┐
│ core/frames.py          │  Video: ffmpeg -> frames/ at configurable fps
│                          │  Photos: used in place, unchanged
└───────────┬──────────────┘
              │
              ▼
┌───────────────────────┐
│ core/sfm.py              │  COLMAP: feature_extractor -> matcher -> mapper
│                          │  sequential matcher for video, exhaustive for
│                          │  unordered photo collections
└───────────┬──────────────┘
              │  sparse point cloud + camera poses
              ▼
┌───────────────────────┐
│ core/splat.py             │  3D Gaussian Splatting (gsplat): init from
│                          │  COLMAP points, optimize means/scales/quats/
│                          │  opacity/SH color against posed photos
└───────────┬──────────────┘
              ▼
   output/<scene>/points.ply
```

## Why COLMAP + gsplat (not alternatives considered)

- **OpenMVS** (vendored at `c:/rory/scripts/openMVS`): mature classical dense-stereo + textured mesh, CPU/GPU. Left as a documented future backend rather than built now — 3D Gaussian Splatting was chosen as the Phase 1 default because it's GPU-native end-to-end and the modern default for this problem as of 2026 (trains in minutes, not hours; explicit point-cloud representation is a natural byproduct of the splat optimization).
- **Feed-forward transformers (DUSt3R / MASt3R / VGGT)**: no calibration/matching step needed, and notably strong on sparse image sets (<10 images) — a good match for "small collections" in principle. Not used as the Phase 1 default because they're newer/less battle-tested at unbounded image counts than COLMAP, which this project's predecessor ([LogMotion](c:/rory/scripts/LogMotion)) already has working end-to-end. Worth revisiting as an alternate pose backend if COLMAP struggles on genuinely sparse (<10 image) captures.

## COLMAP Binary Notes

- Windows dev machine: prebuilt nocuda binary at `C:/Program Files/colmap-x64-windows-nocuda/bin/colmap.exe` (same binary LogMotion uses).
- RunPod: Ubuntu 22.04's apt `colmap` package is v3.x with incompatible flag names (`--SiftExtraction.*` vs current `--FeatureExtraction.*`, discovered while building LogMotion). `Dockerfile.runpod` instead pulls a modern build via a standalone micromamba/conda-forge env, symlinking out only the `colmap` binary so conda's Python never shadows the base image's torch/CUDA environment.
- `core/sfm.py` always runs COLMAP's own SIFT on CPU (`--*.use_gpu 0`): neither binary this project installs is CUDA-compiled (Windows nocuda build; RunPod's conda-forge `colmap` package resolves to a CPU-only build too), so GPU mode would fall back to an OpenGL context that doesn't exist in a headless pod -- confirmed by a SIGABRT crash (`opengl_utils.cc: Check failed: context_.create()`) when this was tried. `core/splat.py`'s `_require_cuda()` is the actual hard CUDA requirement, since gsplat's rasterizer is a CUDA-only op.

## Phase 2 — Moving Object (not yet designed)

Two candidate strategies, to be decided when this phase starts:

1. **Multi-camera rig**: synchronized cameras capture the moving object from several angles at once; reconstruct per-timestep from simultaneous frames (same Phase 1 pipeline, run once per timestep), then temporally register the resulting point clouds. Simpler and more robust, but requires a multi-camera capture setup.
2. **Monocular casual video**: mask out the moving object, run Phase 1's SfM on the static background only to recover the camera trajectory, then fit a deformable Gaussian splatting model (canonical-space Gaussians + a per-frame deformation field) to the object region alone — since "all we care about is the reconstruction of the moving object," the background reconstruction only exists to anchor camera poses.

Neither is implemented; this section exists so the choice is made deliberately rather than by default.
