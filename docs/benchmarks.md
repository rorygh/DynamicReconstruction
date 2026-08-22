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

**Follow-up (in progress at the time of writing)**: training the same
config on the full 128-image reconstruction (65,201 sparse points, all
images registered) to directly test whether more views closes this gap --
see the pipeline output / follow-up note once that run completes.

## Bottom line

- Sequential vs exhaustive COLMAP matching: real savings only show up at
  larger image counts than tested here; re-test at 150-200+ images before
  changing `reconstruct.py`'s hardcoded exhaustive-for-photos default.
- GLOMAP: not adopted. Slower and slightly less complete than COLMAP's own
  mapper on this dataset, despite the opposite expectation from secondhand
  research -- a good example of why this doc exists instead of trusting the
  survey alone.
- Quality: the "doesn't look photorealistic" complaint has a concrete,
  measured cause (PSNR ~10dB, not a vibe) that traces to sparse training
  views, not a rendering bug -- pending the full-128-image re-run for
  confirmation.
