# Benchmarks (2026-08-22)

Real numbers from this machine (RTX 2000 Ada, 16GB, 64 CPUs), not internet
research -- that survey is in [pipeline-alternatives.md](pipeline-alternatives.md)
and [timing-and-viewers.md](timing-and-viewers.md). Dataset throughout:
COLMAP's own `south-building` sample (128 real photos), at various subsampled
counts, downsized to 1200px wide unless noted. See
[experimental-prereqs.md](experimental-prereqs.md) for what had to be
installed to produce these.

## COLMAP matcher: exhaustive vs sequential, at small scale

40-image subset, `max_image_size 1200`, `max_num_features 8192`. Both
matchers were killed early once the trend was clear (see finding below), so
these are partial-progress rates, not completed wall-clock totals.

| Matcher | Pairs (of 40 images) | Observed rate |
|---|---|---|
| Exhaustive | C(40,2) = 780 | still in its single matching block after 11 min |
| Sequential (overlap 10) | ~640 (40 images x ~16 neighbors avg, edge-clipped) | ~10-18s/image step, 24/40 done at 11 min |

**Finding**: at this small a scale, sequential's advantage over exhaustive is
much smaller than "O(n) vs O(n^2)" suggests -- an overlap window of 10 on
only 40 images means most images are already within each other's window.

## GLOMAP: a genuinely surprising result

**Install**: `micromamba create -y -n glomap-env -c conda-forge glomap` --
clean, ~1 min, CUDA-enabled build.

**First real blocker -- database schema incompatibility.** Feeding GLOMAP a
database built by this project's own standalone `colmap` binary (v4.0.4,
in `/opt/colmap-env`) failed immediately:
```
E database_sqlite.cc:1599] SQLite error: SQL logic error
terminate: No registered database factory succeeded.
```
Not a corrupt file -- Python's `sqlite3` opened it fine. This is a known
class of issue ([colmap/colmap#2907](https://github.com/colmap/colmap/issues/2907),
similar reports on colmap/glomap) where GLOMAP's bundled COLMAP database
code expects a specific schema version. The fix: let conda-forge resolve
`colmap` and `glomap` **together** in one environment
(`micromamba create -n glomap-full -c conda-forge glomap colmap`) rather
than pairing GLOMAP with whatever `colmap` build this project already had --
the solver picked `colmap 3.8` as the mutually-compatible version (needed a
`faiss` package added by hand afterward; the initial joint-solve pulled it
in incompletely and `colmap` errored with `libfaiss.so: cannot open shared
object file` until `micromamba install -n glomap-full -c conda-forge faiss`
was run). Confirms this repo's own [architecture.md](architecture.md) note
that COLMAP 3.x uses `--SiftExtraction.*`/`--SiftMatching.*` flag names
where 4.x uses `--FeatureExtraction.*`/`--FeatureMatching.*` -- had to
switch flag families to drive the 3.8 binary.

**Second blocker -- crashes in retriangulation.** Even with a compatible
database, `glomap mapper` completed view-graph calibration, rotation
averaging, global positioning, and bundle adjustment, then crashed:
```
Check failed: existing_frame.DataIds() == frame.DataIds()
```
during its optional retriangulation pass. Fixed with `--skip_retriangulation 1`.

**Then the actual result, at both scales, was the opposite of the
"1-2 orders of magnitude faster" claim documented in
[pipeline-alternatives.md](pipeline-alternatives.md):**

| Scale | COLMAP mapper | GLOMAP mapper (`--skip_retriangulation 1`) | Registered / points |
|---|---|---|---|
| 20 images | ~10s | 26.6s | COLMAP: 20/20, 9,270 pts. GLOMAP: 13/20, 5,139 pts |
| 128 images | 8m37s | 32m29s | COLMAP: 128/128, 65,201 pts. GLOMAP: 128/128, 56,663 pts, one bundle-adjustment stage hit `NO_CONVERGENCE` after 201 iterations |

GLOMAP was **~2.6x slower at small scale and ~3.8x slower at full scale**,
registered fewer images at small scale, and produced fewer 3D points at
both scales, on this specific dataset. This doesn't mean GLOMAP is bad --
its own paper's benchmarks are real -- it means **this scene doesn't play
to its strengths** (possibly weaker view-graph connectivity from a
building walkthrough vs. the wide-baseline outdoor scenes GLOMAP is usually
benchmarked on) and the "always faster" framing from secondhand research
doesn't hold universally. Worth re-testing on a more object-centric,
wide-baseline capture before drawing a general conclusion.

**Bottom line**: COLMAP's incremental mapper stays the better default here.
GLOMAP is not a drop-in speed upgrade for this kind of scene without further
tuning (`--BundleAdjustment.*` options weren't explored).

## GPU-accelerated SIFT: possible now, but not actually faster

`docs/architecture.md` documents that GPU SIFT was tried previously and hit
a SIGABRT (`opengl_utils.cc: Check failed: context_.create()`) in a headless
pod with no display. Retested here because the earlier "how fast is COLMAP"
research kept pointing at GPU SIFT as the obvious next speed lever, and this
machine does have real NVIDIA GL/EGL libraries present
(`libGLX_nvidia.so.0`, `libEGL_nvidia.so.0`) even though it's headless.

**It now works** -- installing `xvfb` (`apt-get install xvfb`, not present
by default) and running COLMAP under `xvfb-run -a --server-args="-screen 0
1024x768x24"` gets past both the earlier Qt platform-plugin crash and the
OpenGL-context crash; `sift.cc` logs confirm a genuine `Creating SIFT GPU
feature extractor` (not a silent CPU fallback).

**It's not worth it here.** Same 128-image dataset, same everything else:

| Stage | CPU | GPU (via `xvfb-run`) |
|---|---|---|
| Feature extraction | ~1 min | 2.33 min |
| Sequential matching (overlap 10) | ~6.35 min | 26.1 min |
| **Total** | **~7.35 min** | **~28.5 min** |

GPU was **~3.9x slower**, not faster -- the opposite of the expectation set
by public benchmarks (e.g. a "~128 seconds with CUDA" figure found during
earlier research). The GPU run's `user` CPU-time was 182 minutes despite
"running on the GPU," which points at `xvfb-run`'s virtual-display/driver
round-trip overhead on every SIFT call dominating any real GPU compute
gain -- plausible with only 128 modest-resolution (1200x900) images, where
each GPU call's fixed overhead isn't amortized over enough work per call.
Might flip the other way on a much larger dataset or with a proper (non-Xvfb)
display, but wasn't tested at that scale here.

**Bottom line**: GPU SIFT is achievable now (unlike what the architecture
doc assumed), documented here so nobody re-discovers the Xvfb trick from
scratch -- but don't actually switch `core/sfm.py` to `--*.use_gpu 1` based
on this machine's numbers. Same lesson as the GLOMAP result above: verify
before adopting, secondhand benchmarks don't transfer.

## Training quality ablation: baseline vs SSIM+LR-decay

20-image scene (all 20 registered, 9,270 sparse points), 7,000 iterations,
held out every 5th image (4 test views) not used in the training loss. Both
configs use gsplat's `DefaultStrategy` defaults, which already match the
reference paper's densification schedule (see
[timing-and-viewers.md](timing-and-viewers.md)) -- only the loss and the
position learning-rate schedule differ.

| Config | Loss | Position LR | Final test PSNR | Final test SSIM |
|---|---|---|---|---|
| baseline (current `core/splat.py`) | L1 only | fixed 1.6e-4 | 9.53 | 0.257 |
| improved | 0.8 L1 + 0.2 (1-SSIM) | exponential decay 1.6e-4 -> 1.6e-6 | 10.64 | 0.284 |

`0.2` SSIM weight and the decay schedule match the reference 3DGS paper's
defaults, not arbitrary choices.

**These numbers are bad in an absolute sense** (a healthy NVS result on a
well-covered scene is usually 25-30+ dB; low-to-mid teens is a real problem,
not just "not photorealistic yet"). SSIM+LR-decay helped (+1.1 dB) but
didn't fix the underlying issue. Diagnosis: rendered held-out test views
were checked directly (ASCII-art brightness maps and per-channel
mean/std, since an image-viewer tool was unavailable at the time) -- the
renders clearly reproduce the scene's coarse structure and tone, not noise,
so this isn't a broken renderer. Test PSNR also never improved over the
course of training (bounced between 9-12 dB from step 0 onward) while train
loss dropped steadily -- consistent with **severe overfitting to only 16
training views** of a real building facade rather than a training-loop bug.
This directly supports the "perhaps not enough images" hypothesis: 16 views
is very little multi-view constraint for a gaussian field to resolve true
3D structure instead of memorizing per-training-view appearance.

**Follow-up: full 128-image reconstruction.** Same config (SSIM+LR-decay),
trained on the full 128-image sparse model (62,979 points, all images
registered), held out every 5th image (25 test views vs. the 20-image run's
4):

| Images | Iterations | Final test PSNR | Final test SSIM |
|---|---|---|---|
| 20 (16 train / 4 test) | 7,000 | 10.64 | 0.284 |
| 128 (102 train / 26 test) | 9,000 | 13.02 | 0.338 |

More images measurably helped (+2.4 dB), confirming the hypothesis
directionally -- but the gap to a healthy NVS result (25-30+ dB) is still
enormous, so image count alone doesn't explain it. Test PSNR plateaued by
~step 6,000-7,000 in both the 9,000- and a longer 15,000-iteration run on
the same 128-image data (12.6-13.1 dB either way, no further gain from
training longer), which rules out "just train longer" as the fix too.
`output/south-building-full/points.ply` (56,522 Gaussians) is the artifact
from this run, viewable via `viewers/render_point_viewer.py` and
`viewers/render_reference_viewer.py`.

**Follow-up: lens undistortion.** `core/splat.py` builds its camera matrix
from focal length + principal point only -- it never accounts for lens
distortion, and the pipeline never runs COLMAP's `image_undistorter`. This
scene's camera model is `SIMPLE_RADIAL` with a real distortion coefficient
(k = -0.0196), which works out to roughly 13px of displacement at the image
corners on these 1200px-wide photos -- meaning every training step's
pixel-wise loss was comparing the render against a systematically warped
target. Ran `colmap image_undistorter` on the same 128-image reconstruction
(confirmed it converts the camera to a clean `PINHOLE` model, same 128
images / 62,979 points preserved) and retrained identically (9,000 iters,
SSIM+LR-decay):

| | Final test PSNR | Final test SSIM |
|---|---|---|
| distorted (as shipped) | 13.02 | 0.338 |
| undistorted | 13.16 | 0.405 |

Real but modest: PSNR barely moved (+0.14dB) while SSIM improved
meaningfully (+0.067) -- consistent with undistortion fixing geometric/
structural alignment (what SSIM is sensitive to) without touching the
bigger, separate problem. **Worth keeping as a permanent pipeline fix
regardless** (it's free, and it's simply correct to train on the camera
model that matches the actual pixel content), but it is not the answer to
"why is quality so low" -- the training loop itself remains the prime
suspect: single random view per step with no multi-view consistency term,
vs. the reference implementation's same-in-principle-but-more-carefully-
tuned schedule. Worth a follow-up ablation isolating that specifically, not
attempted here given time budget.

## Bottom line

- Sequential vs exhaustive COLMAP matching: real savings only show up at
  larger image counts than tested here; re-test at 150-200+ images before
  changing `reconstruct.py`'s hardcoded exhaustive-for-photos default.
- GLOMAP: not adopted. Slower and slightly less complete than COLMAP's own
  mapper on this dataset, despite the opposite expectation from secondhand
  research -- a good example of why this doc exists instead of trusting the
  survey alone.
- GPU SIFT: now achievable headlessly (`xvfb-run`, undocumented before
  this session) but ~3.9x *slower* than CPU on this dataset/machine -- do
  not switch `core/sfm.py`'s hardcoded `--*.use_gpu 0` based on this.
- Quality: the "doesn't look photorealistic" complaint has a concrete,
  measured cause (PSNR ~10-13dB, not a vibe), not a rendering bug. Image
  count helped some (+2.4dB, 20->128 images) and lens undistortion helped
  SSIM specifically (+0.067) but barely PSNR -- neither closes the real
  gap. Test PSNR plateaus by step ~6,000-7,000 regardless of iteration
  budget, pointing at the simplified single-view-per-step training loop
  itself as the next thing to fix, not more data or more training time.
