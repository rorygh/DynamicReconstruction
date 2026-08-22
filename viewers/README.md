# Viewers

Two ways to turn a trained `output/<scene>/points.ply` into a self-contained
HTML file you can open in a browser (or publish as an artifact) -- no server,
no build step.

```bash
python viewers/render_point_viewer.py output/myscene/points.ply --out myscene.html --title "My Scene"
python viewers/render_reference_viewer.py output/myscene/points.ply --out myscene_ref.html
```

- **`render_point_viewer.py`** -- this project's own WebGL renderer
  (`template.html`): projects each Gaussian's real 3D covariance
  (scale+rotation) through the camera every frame into a 2D screen-space
  ellipse, shades it with the exact analytic gaussian falloff (the conic /
  inverse covariance), and depth-sorts back-to-front (O(n) counting sort,
  re-triggered on camera movement, not every frame) for correct alpha
  blending.
- **`render_reference_viewer.py`** -- feeds the same data into
  [antimatter15/splat](https://github.com/antimatter15/splat)'s real,
  unmodified renderer (vendored under `vendor/antimatter15-splat/`, MIT
  licensed), via a fetch-intercept shim rather than any change to their code.
  Useful as an independent cross-check: if a scene looks equally good (or
  equally bad) in both viewers, the issue is upstream in training/data, not
  either renderer's shader math.

Both read the standard 3DGS ply fields `core/splat.py` exports
(`scale_0-2`, `rot_0-3`, `opacity`, `f_dc_0-2`) -- not just xyz+rgb -- so
either viewer works on any scene the pipeline produces.

See [docs/timing-and-viewers.md](../docs/timing-and-viewers.md) and
[docs/benchmarks.md](../docs/benchmarks.md) for how these were evaluated.
