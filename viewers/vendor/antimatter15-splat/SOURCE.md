# Vendored from antimatter15/splat

- Repo: https://github.com/antimatter15/splat
- Author: Kevin Kwok
- License: MIT (see `LICENSE` in this directory)
- Retrieved: 2026-08-22, from the `main` branch's `main.js` and `index.html`

Vendored (not fetched at generation time) so the viewer generator works
offline and reproducibly, and isn't silently affected by upstream changes.
Files are unmodified from upstream -- `render_reference_viewer.py` inlines
them as-is and only injects a small fetch-intercept shim so the code loads
our own splat data instead of making a network request.
