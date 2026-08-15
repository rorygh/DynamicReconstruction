# Datasets

## Local smoke-testing (small, real photos, no GPU needed)

`scripts/download_dataset.sh [south-building|gerrard-hall]` — COLMAP's own sample datasets, hosted directly by the COLMAP project ([colmap/colmap releases](https://github.com/colmap/colmap/releases)).

| Scene | Images | Size | Notes |
|-------|--------|------|-------|
| south-building (default) | 128 | 233.9 MB | UNC Chapel Hill building exterior/interior walk |
| gerrard-hall | 100 | 886.1 MB | UNC Chapel Hill hall, higher-resolution photos |

These validate the frame-prep + COLMAP sparse half of the pipeline (`core/frames.py`, `core/sfm.py`) without needing a GPU or a large download. They're building walkthroughs, not discrete objects — fine for pipeline plumbing, not representative of the eventual "walk around an object" use case.

## Object-centric validation (larger, GPU step needed — run on RunPod)

| Dataset | Size | License | Notes |
|---------|------|---------|-------|
| [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) | ~12 GB (all scenes) | CC-BY (Google Research) | Real photos, object-centric orbit trajectories at fixed elevation/radius — closest public match to "walking around an object." Scenes: `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`. Only ships as one multi-scene zip; extract the one scene you want after downloading. |
| [Tanks & Temples](https://www.tanksandtemples.org/download/) | Varies per scene (100s of MB–GBs) | CC BY-NC-SA (research use) | Real captures including small individual objects (e.g. `Ignatius`, `Horse`). Per-scene downloads, not bundled. |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/) | Per-sequence subsets, full set is huge | Meta research license | ~19,000 short object-centric videos across 50 categories — good for later large-scale validation or category-specific fine-tuning, not a quick smoke test. |

## Phase 2 (moving objects) — not yet sourced

No dataset selected; revisit once Phase 2's capture strategy (multi-camera rig vs. monocular casual video, see [architecture.md](architecture.md)) is decided, since the right dataset depends on which.
