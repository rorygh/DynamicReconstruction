# Segmenting Gaussian-Splat / 3D-Generated Scenes (2026 survey)

Research note, not implemented code. Directly relevant to
[Phase 2's](architecture.md#phase-2--moving-object-not-yet-designed) monocular
option, which needs to mask a moving object out of the background before
anything else happens — this is the landscape of tools for that step (and
its 3D-native alternatives).

## Overview paper

[**A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and
Generation**](https://arxiv.org/abs/2508.09977) (TPAMI 2026, He et al.;
curated list at
[heshuting555/Awesome-3DGS-Applications](https://github.com/heshuting555/Awesome-3DGS-Applications)).
Splits 3DGS segmentation work into: open-vocabulary, interactive/promptable,
semantic, instance, panoptic, referring/spatial, and language-aware feature
learning. The rest of this doc follows that split, merged down to the
distinctions that matter for picking an approach.

## Two fundamentally different strategies

**A. 2D-first, lift-to-3D.** Run a mature 2D/video segmenter on the posed
images, then fuse the per-frame masks into the 3D representation.
- [**Gaussian Grouping**](https://arxiv.org/abs/2312.00860): runs
  [SAM](https://github.com/facebookresearch/segment-anything) per frame, uses
  the zero-shot video tracker DEVA to keep object IDs consistent across
  views, then trains 3DGS with an extra per-Gaussian "identity" channel
  supervised by those tracked masks — segmentation falls out of the trained
  scene for free afterward.
- Plain multi-view mask fusion (no learned identity channel) is the naive
  version of the same idea, and its known failure mode is exactly what
  Gaussian Grouping/DEVA exist to fix: 2D masks from independent frames
  aren't mutually consistent (an object's mask can flip ID or boundary
  frame-to-frame), so naive fusion needs a tracker or 3D-consistency term
  bolted on. SAM2 (native video segmentation with memory-based tracking)
  is the current default front-end for this step.
- Upside: reuses best-in-class, heavily-trained 2D models; no custom 3D
  training objective. Downside: quality caps out at 2D mask quality + how
  good the cross-frame ID association is.

**B. Native 3D / feature-embedded.** Bake a feature (not just RGB) into every
Gaussian during training, then segment by querying that feature field
directly in 3D — no per-frame mask-fusion step at all.
- [**Feature 3DGS**](https://arxiv.org/abs/2312.03203): parallel rasterizer +
  lightweight decoder distills arbitrary 2D feature fields (SAM, CLIP-LSeg)
  straight into the Gaussians.
- [**LangSplat**](https://langsplat.github.io/): distills CLIP language
  embeddings per-Gaussian through a scene-specific autoencoder (keeps memory
  down), enabling open-vocabulary text queries directly against the splat.
- [**OmniSeg3D**](https://arxiv.org/abs/2311.11666): hierarchical contrastive
  learning over a feature field for click-free, multi-granularity 3D
  segmentation.
- Upside: segmentation/query is a direct lookup once trained, and it's
  inherently 3D/view-consistent by construction. Downside: adds a whole
  extra training objective (and compute) on top of the base 3DGS fit, and
  early open-vocab variants trailed 2D SOTA in raw mask quality.

## Interactive / promptable (fast, post-hoc)

[**SAGA (Segment Any 3D Gaussians)**](https://jumpat.github.io/SAGA/):
distills SAM's masks into a per-Gaussian "affinity" feature with a
scale-gate to resolve multi-granularity ambiguity (whole object vs. part),
then a point/scribble/mask prompt segments the matching 3D Gaussians in ~4ms
— roughly 1000x faster than prior optimization-based 3D segmenters, because
the expensive distillation is done once at train time and querying at
inference is just a feature lookup. [Click-Gaussian] and [GaussianCut] are
the same category (click-to-select on an already-trained splat).

## Instance & panoptic

Object-level (not just class-level) separation: [**InstanceGaussian**](https://arxiv.org/abs/2411.19235),
[**ObjectGS**](https://arxiv.org/abs/2506.15991), [**PanoGS**](https://arxiv.org/abs/2503.18107)
(semantic + instance jointly). Useful when the goal is "isolate this one
object" rather than "label every pixel by class" — closer to what Phase 2's
foreground/background split actually needs than dense semantic segmentation is.

## Where RaDe-GS fits

[**RaDe-GS**](https://baowenz.github.io/radegs/) is **not a segmentation
method** — it's a better depth/normal rasterizer for 3DGS (closed-form
ray–Gaussian intersection instead of the usual alpha-compositing depth
approximation), giving surface-reconstruction accuracy close to
implicit-surface methods (Neuralangelo, GOF) at 3DGS training speed
(~5 min on DTU). It matters here as an *input quality* lever, not a
competing pipeline stage: several of the methods above lean on multi-view
consistency or clean object boundaries (mask fusion, panoptic boundaries),
and both degrade when the underlying depth/normals are noisy — which is
exactly what plain 3DGS alpha-compositing tends to produce (`core/splat.py`'s
own export is `xyz`+`rgb` only, no depth/normal signal at all right now).
The RaDe-GS paper is explicit that the technique "can be easily integrated
into any Gaussian-Splatting-based method," so it's a drop-in geometry
upgrade to sit underneath whichever segmentation strategy gets picked, not
an alternative to be chosen between.

## Bottom line for this repo

- Nothing here is implemented or should be adopted speculatively — this is
  reconnaissance for the Phase 2 design session, per
  [CLAUDE.md](../CLAUDE.md)'s instruction not to build toward Phase 2 ahead
  of that decision.
- If/when Phase 2 picks the monocular casual-video path (mask the moving
  object out, reconstruct background for pose, deform-fit the object), the
  concrete choice at that point is 2D-first (SAM2 + a tracker, simplest,
  reuses mature models) vs. native 3D feature field (LangSplat/Feature-3DGS
  style, more upfront training cost, view-consistent by construction) —
  this doc is what that tradeoff call should be made against.
- RaDe-GS-style depth rasterization is an independent, compatible upgrade to
  `core/splat.py` regardless of which segmentation route gets picked later.
