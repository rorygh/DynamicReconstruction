#!/usr/bin/env python3
"""Render a trained splat .ply into a self-contained HTML viewer using this
project's own WebGL renderer (viewers/template.html): real per-Gaussian
anisotropic covariance projected through the camera each frame, exact conic
falloff shading, and an O(n) back-to-front depth sort re-triggered on camera
movement. See docs/timing-and-viewers.md for how this compares to
render_reference_viewer.py's antimatter15/splat cross-check.

    python viewers/render_point_viewer.py output/myscene/points.ply --out myscene.html --title "My Scene"
"""
import argparse
from pathlib import Path

import numpy as np
import base64
from plyfile import PlyData

TEMPLATE = Path(__file__).parent / "template.html"


def compute_covariance(scale_raw, quat_raw):
    """World-space 3x3 covariance Sigma = R*S*S^T*R^T per Gaussian, packed as
    two vec3 attributes (Sigma00,01,02) and (Sigma11,12,22)."""
    scale = np.exp(scale_raw.astype(np.float64))
    quat = quat_raw.astype(np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    n = scale.shape[0]
    R = np.zeros((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    M = R * scale[:, np.newaxis, :]
    sigma = np.matmul(M, np.transpose(M, (0, 2, 1)))
    cov0 = np.stack([sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2]], axis=1).astype(np.float32)
    cov1 = np.stack([sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2]], axis=1).astype(np.float32)
    return cov0, cov1


def build_html(ply_path: Path, title: str, stats: dict) -> str:
    p = PlyData.read(str(ply_path))
    v = p["vertex"]
    n = len(v)

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(f_dc * 0.28209479177387814 + 0.5, 0.0, 1.0).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))).astype(np.float32)
    cov0, cov1 = compute_covariance(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
    )

    def b64(arr):
        return base64.b64encode(arr.tobytes()).decode("ascii")

    html = TEMPLATE.read_text()
    html = html.replace("__POSITIONS_B64__", b64(positions))
    html = html.replace("__COLORS_B64__", b64(colors))
    html = html.replace("__COV0_B64__", b64(cov0))
    html = html.replace("__COV1_B64__", b64(cov1))
    html = html.replace("__OPACITY_B64__", b64(opacity))
    html = html.replace("__SCENE_DATE__", stats.get("date", ""))
    html = html.replace("__N_IMAGES__", stats.get("n_images", ""))
    html = html.replace("__N_REGISTERED__", stats.get("n_registered", ""))
    html = html.replace("__N_SPARSE__", stats.get("n_sparse", ""))
    html = html.replace("__N_GAUSSIANS__", f"{n:,}")
    html = html.replace("__N_ITERS__", stats.get("n_iters", ""))
    html = html.replace("__WALL_TIME__", stats.get("wall_time", ""))
    html = html.replace("<title>South Building Splat</title>", f"<title>{title}</title>")
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="Path to a points.ply exported by core/splat.py")
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--title", default="Splat Viewer")
    ap.add_argument("--date", default="")
    ap.add_argument("--n-images", default="")
    ap.add_argument("--n-registered", default="")
    ap.add_argument("--n-sparse", default="")
    ap.add_argument("--n-iters", default="")
    ap.add_argument("--wall-time", default="")
    args = ap.parse_args()

    stats = {
        "date": args.date, "n_images": args.n_images, "n_registered": args.n_registered,
        "n_sparse": args.n_sparse, "n_iters": args.n_iters, "wall_time": args.wall_time,
    }
    html = build_html(Path(args.ply), args.title, stats)
    Path(args.out).write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
