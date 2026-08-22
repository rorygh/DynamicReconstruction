# Timing Expectations & Viewer Options (2026-08)

Companion to [pipeline-alternatives.md](pipeline-alternatives.md) (which covers
pipeline/framework alternatives). This one covers: how long each stage
actually takes, and how to look at the result.

## How long each stage takes

**COLMAP sparse (feature extraction -> matching -> mapping)**
- Exhaustive matching is fine up to "several hundred" images; mapping/bundle
  adjustment is 70-90% of total sparse-reconstruction time, not matching itself.
- **GLOMAP** (global SfM, drop-in after COLMAP's feature/match step) is
  reported ~3.5x, and in some benchmarks 1-2 orders of magnitude, faster than
  COLMAP's incremental mapper at comparable accuracy. **FastMap** (2025) claims
  up to 10x faster than both with GPU accel. Neither is adopted here (see
  bottom line), but either is a drop-in swap for the mapping stage alone if
  COLMAP mapping time becomes the bottleneck on a larger capture.

**3D Gaussian Splatting training**
- Reference implementation at 30k iterations: ~20-50 min depending on scene
  (Tanks & Temples ~20 min, Mip-NeRF 360 ~51 min) on datacenter-class GPUs
  (V100/A100/3090-class).
- `gsplat` (what `core/splat.py` uses) is ~15% faster than the reference impl
  and up to 4x lighter on memory at the same iteration count/quality — same
  core math, not a different algorithm.
- Faster training variants exist if 30k iterations of vanilla optimization
  becomes the bottleneck: **Taming 3DGS** (4-5x speedup, few minutes),
  **DashGaussian** (~200s), **FastGS** (~100s) — all restructure densification/
  parallelization, not something to adopt speculatively (see bottom line in
  pipeline-alternatives.md).
- `core/splat.py`'s loop renders **one view per step** (not batched) and has no
  progressive resolution schedule, so it's slower per-iteration than the
  reference impl on the same hardware — expect low-end/laptop GPUs (e.g. a
  16GB RTX 2000 Ada) to take noticeably longer than the benchmarks above at
  the default 30k iterations. For a quick smoke test, cutting iterations to a
  few thousand (and `densify_until_iter` proportionally) is the right lever,
  not touching the training loop itself.

## Viewer options for `output/<scene>/points.ply`

`core/splat.py`'s `_export_ply` only writes `xyz` + `rgb` per point — it does
**not** write scale/rotation/opacity, so the export is a plain colored point
cloud, not a renderable set of splat ellipsoids yet. That caps which viewers
are useful today:

| Viewer | Fit today | Notes |
|---|---|---|
| Any point-cloud viewer (Open3D, MeshLab, CloudCompare, a plain WebGL point renderer) | ✅ works now | All that's needed for the current xyz+rgb export. |
| [SuperSplat](https://superspl.at/) (PlayCanvas, MIT, open source) | Future | Full splat editor/viewer, drag-drop `.ply`, self-hostable single-file HTML export (`@playcanvas/supersplat-viewer`) — the one to reach for once `_export_ply` carries scale/rotation/opacity. |
| [antimatter15/splat](https://github.com/antimatter15/splat) | Future | Minimal no-dependency WebGL viewer, same prerequisite as above. |
| nerfstudio's `viser`-based live viewer | Not adopted | Live training-time view; would mean pulling in nerfstudio (already ruled out in pipeline-alternatives.md). |

## Bottom line

- Nothing here changes the Phase 1 design: COLMAP + gsplat stays hardcoded,
  GLOMAP/FastMap/Taming-3DGS etc. are documented options to revisit only if
  a specific stage becomes the actual bottleneck on real captures.
- Next real lever for richer visualization is exporting full Gaussian
  parameters (scale/rotation/opacity) from `core/splat.py`, not swapping
  viewers — worth a follow-up once Phase 1's `.ply` export needs to look like
  an actual splat instead of a point cloud.
