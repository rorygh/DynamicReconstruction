# Experimental Prerequisites (2026-08-22)

Tracks what had to be installed for the benchmarking/quality work in
[benchmarks.md](benchmarks.md), separate from `requirements.txt` because
none of this is adopted into the shipped pipeline yet -- see that doc's
bottom line before installing any of it for real use.

## Metrics (installed, pip)

```bash
pip install scikit-image pytorch-msssim
```
- `pytorch-msssim`: differentiable SSIM, used both as a training loss term
  and an eval metric (`ssim()` from the same call, `data_range=1.0`).
- `scikit-image`: available for reference but `pytorch-msssim`'s SSIM ended
  up covering both training-loss and eval use, so scikit-image wasn't
  actually load-bearing in the end -- kept installed, not a hard dependency.

## GLOMAP (conda-forge, standalone env)

```bash
micromamba create -y -n glomap-env -c conda-forge glomap
```
Same pattern `Dockerfile.runpod` already uses for `colmap` (standalone
micromamba env, only the binary gets used, to avoid conda's Python shadowing
the base image's torch/CUDA env). Package: `glomap` on conda-forge, current
release `1.2.0`, with CUDA and CPU-only build variants both published.
Consumes a COLMAP database directly (`glomap mapper --database_path ...
--image_path ... --output_path ...`) -- no separate feature
extraction/matching step, reuses COLMAP's.

## Not available / not attempted

- **FastMap**: no packaged binary found (conda-forge or pip); would require
  building from source. Skipped given the time budget -- GLOMAP alone
  already answers "is a faster SfM mapper worth it here."
- **Headless browser (Playwright/Puppeteer/Chromium)**: not installed in
  this environment, and not installed as part of this work either --
  screenshotting the WebGL viewer artifact to cross-check it programmatically
  wasn't attempted; visual comparison of the viewer relies on the user's own
  browser instead.
- **Open3D**: not installed; not needed once gsplat's own `rasterization()`
  became the render/eval path (see below).

## Eval/render harness (scratch, not shipped)

Lives at `/tmp/.../scratchpad/train_eval.py` in the session that produced it,
not in this repo -- it's an experiment harness (train/test split, SSIM-loss
ablation, PSNR logging, PNG dumping), not pipeline code. Reuses
`core.splat._load_colmap_scene` / `_init_gaussians` directly rather than
duplicating them, and calls `gsplat.rasterization()` (the same real,
CUDA-backed rasterizer `core/splat.py` already uses) to render eval views --
deliberately not a from-scratch renderer, so a quality problem shows up as a
training problem rather than a rendering-code problem.
