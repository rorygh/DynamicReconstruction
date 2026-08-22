# Existing Pipelines Surveyed (2026-08)

Prior art check for Phase 1 before extending it further. Short answer: nothing to
adopt wholesale, but a couple of things worth stealing.

## Nerfstudio isn't NeRF-only

Nerfstudio started as a NeRF research framework, but since Jan 2024 it ships
**Splatfacto**, a real 3D Gaussian Splatting method — and Splatfacto is built on
**gsplat**, the same rasterizer `core/splat.py` already calls directly. So the
raw training math/speed is identical; nerfstudio doesn't have a faster core.

What it adds is the scaffolding around that core:

- `ns-process-data`: same video→frames→COLMAP flow as `core/frames.py` +
  `core/sfm.py`, but also ingests Polycam, RealityCapture, Metashape, ODM, and
  **Record3D** (iPhone LiDAR pose capture, skips COLMAP entirely using ARKit
  poses). Relevant later: Record3D-style pose priors and multi-camera/synced
  rig ingestion are exactly the kind of input Phase 2's "multi-camera rig"
  option (`docs/architecture.md`) would need.
- A live web viewer during training (via `viser`) — watch gaussians form in
  real time instead of waiting for `points.ply` at the end.
- Export tooling: mesh via Poisson/TSDF, camera-path renders, `.ply` splat
  export compatible with common viewers (SuperSplat, PlayCanvas).
- A method/config registry (`tyro`-based) for swapping training presets.

That last point is also the reason not to adopt it: it's the plugin
abstraction this repo's CLAUDE.md deliberately avoids ("no pluggable backend
... until a second backend is actually needed"). Nerfstudio is a multi-method
research platform; this repo is a minimal `reconstruct.py --input X` script.
Pulling it in would mean inheriting its full dependency tree (viewer server,
many unused methods) for zero speed gain over the gsplat call already in
place.

**Worth reusing without adopting the framework**: Splatfacto's tuned defaults
(densification schedule, absgrad, opacity-reset cadence) are public in its
config and worth diffing against `core/splat.py`'s hyperparameters if splat
quality/speed ever needs tuning.

## Other tools checked

| Project | What it is | Relevant? |
|---|---|---|
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Original paper's reference impl, own `convert.py` COLMAP wrapper | Canonical reference; gsplat itself reimplements this. No pipeline advantage over what's already here. |
| [OpenSplat](https://github.com/pierotofy/OpenSplat) | From-scratch **C++** (not PyTorch/gsplat) GS trainer; consumes COLMAP/ODM/nerfstudio project output; runs on NVIDIA/AMD/Apple GPU or CPU | Different engine, not just different wrapper — genuinely worth a look *only* if gsplat's PyTorch training loop ever becomes the RunPod bottleneck. Aimed at drone/mapping use cases, not a pipeline fit otherwise. |
| PostShot, Polycam, Luma AI, DJI Terra | Commercial, no-code (local or cloud) | Not automatable via script/CI, out of scope for "open, automatable pipeline." |
| arpm511/guassian_splats_pipeline, NLeSC/3dgs_automation, jacobvanbeets/SplatReady, Nannigalaxy/video-3d-reconstruction-gsplat | Small hobby/research repos doing roughly this same video→COLMAP→GS flow | No maintained advantage over this repo; same scope, less mature. Not worth adopting or depending on. |

## Bottom line

- No existing project beats what's here on raw training speed — everything
  serious sits on gsplat (or a from-scratch equivalent like OpenSplat).
- Nerfstudio is the one project worth returning to, not to replace this
  pipeline now, but as a reference when: (a) tuning `core/splat.py`
  hyperparameters, or (b) Phase 2 design needs multi-camera/pose-prior
  ingestion patterns.
- Staying with the current hardcoded COLMAP+gsplat script remains the right
  call for Phase 1's stated goal (minimal, automatable, CI-friendly).
