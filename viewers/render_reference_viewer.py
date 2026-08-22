#!/usr/bin/env python3
"""Render a trained splat .ply into a self-contained HTML viewer using
antimatter15/splat's real, unmodified renderer (vendored in
viewers/vendor/antimatter15-splat/, MIT licensed).

Feeds our own data in via a fetch-intercept shim rather than modifying their
code -- see vendor/antimatter15-splat/SOURCE.md. Useful as an independent
cross-check against render_point_viewer.py's custom renderer: if a scene
looks equally good/bad in both, the issue is the training, not the viewer.

    python viewers/render_reference_viewer.py output/myscene/points.ply --out myscene_reference.html
"""
import re
import sys
import base64
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData

VENDOR_DIR = Path(__file__).parent / "vendor" / "antimatter15-splat"


def ply_to_dot_splat_bytes(ply_path: Path) -> bytes:
    """Convert our standard 3DGS ply export into antimatter15/splat's exact
    32-byte-per-row .splat binary layout (see their processPlyBuffer)."""
    p = PlyData.read(str(ply_path))
    v = p["vertex"]
    n = len(v)

    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float64)).astype(np.float32)

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
    sh_c0 = 0.28209479177387814
    rgb = np.clip((0.5 + sh_c0 * f_dc) * 255, 0, 255).astype(np.uint8)
    opacity_raw = v["opacity"].astype(np.float64)
    alpha = np.clip((1.0 / (1.0 + np.exp(-opacity_raw))) * 255, 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    qlen = np.linalg.norm(quat, axis=1, keepdims=True)
    rot = np.clip((quat / qlen) * 128 + 128, 0, 255).astype(np.uint8)

    # importance presort, matches antimatter15/splat's own processPlyBuffer
    size = scale[:, 0].astype(np.float64) * scale[:, 1] * scale[:, 2] * (alpha.astype(np.float64) / 255.0)
    order = np.argsort(-size)
    pos, scale, rgba, rot = pos[order], scale[order], rgba[order], rot[order]

    row_dtype = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    rows = np.zeros(n, dtype=row_dtype)
    rows["pos"], rows["scale"], rows["rgba"], rows["rot"] = pos, scale, rgba, rot
    return rows.tobytes()


def build_html(ply_path: Path, title: str) -> str:
    splat_bytes = ply_to_dot_splat_bytes(ply_path)
    splat_b64 = base64.b64encode(splat_bytes).decode("ascii")

    index_html = (VENDOR_DIR / "index.html").read_text()
    main_js = (VENDOR_DIR / "main.js").read_text()

    css = re.search(r"<style>(.*?)</style>", index_html, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", index_html, re.S).group(1)
    body = body.replace('<script src="main.js"></script>', "")
    body = re.sub(r"<script>\s*if\(location\.host\.includes.*?</script>", "", body, flags=re.S)

    shim = f"""
const EMBEDDED_SPLAT_B64 = "{splat_b64}";
window.fetch = async function(url, opts){{
  const bin = atob(EMBEDDED_SPLAT_B64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Response(bytes, {{status: 200, headers: {{'content-length': String(bytes.length)}}}});
}};
"""

    return (
        f"<title>{title}</title>\n"
        f"<style>\n{css}\n</style>\n"
        f"{body}"
        f"<script>\n{shim}\n</script>\n"
        f"<script>\n{main_js}\n</script>\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="Path to a points.ply exported by core/splat.py")
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--title", default="Reference Splat Viewer")
    args = ap.parse_args()

    html = build_html(Path(args.ply), args.title)
    Path(args.out).write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
