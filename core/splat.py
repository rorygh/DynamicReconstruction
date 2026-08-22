"""3D Gaussian Splatting training on a COLMAP sparse model -> exported point cloud.

CUDA-only (gsplat's rasterizer is a custom CUDA op). Mirrors the GPU guard
pattern used for COLMAP dense reconstruction in the LogMotion prototype:
hard-exit locally with a clear message, actual training happens on RunPod.
"""

import math
import sys
from pathlib import Path

import numpy as np


def _require_cuda():
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch not installed -- run setup-env.sh first.")
    if not torch.cuda.is_available():
        sys.exit(
            "Gaussian Splatting training requires CUDA -- run this step on RunPod.\n"
            "(COLMAP sparse reconstruction has already completed and is CPU-portable.)"
        )


def _load_colmap_scene(sparse_model_dir: Path, image_dir: Path):
    """Read COLMAP cameras/images/points3D via pycolmap into plain tensors."""
    import pycolmap
    import torch

    recon = pycolmap.Reconstruction(str(sparse_model_dir))

    viewmats, Ks, image_paths, widths, heights = [], [], [], [], []
    for image in recon.images.values():
        cam = recon.cameras[image.camera_id]
        fx, fy, cx, cy = cam.focal_length_x, cam.focal_length_y, cam.principal_point_x, cam.principal_point_y

        cam_from_world = image.cam_from_world()
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = cam_from_world.rotation.matrix()
        w2c[:3, 3] = cam_from_world.translation
        viewmats.append(w2c)

        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
        Ks.append(K)

        image_paths.append(Path(image_dir) / image.name)
        widths.append(cam.width)
        heights.append(cam.height)

    points3d = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float32)
    colors3d = np.array([p.color for p in recon.points3D.values()], dtype=np.float32) / 255.0

    return {
        "viewmats": torch.from_numpy(np.stack(viewmats)),
        "Ks": torch.from_numpy(np.stack(Ks)),
        "image_paths": image_paths,
        "width": widths[0],
        "height": heights[0],
        "points3d": torch.from_numpy(points3d),
        "colors3d": torch.from_numpy(colors3d),
    }


def _init_gaussians(points3d, colors3d, sh_degree: int, device):
    import torch

    n = points3d.shape[0]
    extent = (points3d.max(dim=0).values - points3d.min(dim=0).values).norm().item()
    init_scale = max(extent / (2.0 * n ** (1 / 3)), 1e-4)

    means = points3d.clone().to(device).requires_grad_(True)
    scales = torch.full((n, 3), math.log(init_scale), device=device).requires_grad_(True)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)
    opacities = torch.full((n,), _inverse_sigmoid(0.1), device=device).requires_grad_(True)

    sh0 = ((colors3d.to(device) - 0.5) / 0.28209479177387814).unsqueeze(1)  # (N, 1, 3)
    sh0 = sh0.requires_grad_(True)
    n_sh_rest = (sh_degree + 1) ** 2 - 1
    shN = torch.zeros((n, n_sh_rest, 3), device=device).requires_grad_(True)

    return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}


def _inverse_sigmoid(x: float) -> float:
    return math.log(x / (1 - x))


def train(
    sparse_model_dir: Path,
    image_dir: Path,
    out_dir: Path,
    iterations: int = 30000,
    sh_degree: int = 3,
    means_lr: float = 1.6e-4,
    opacity_lr: float = 0.05,
    scales_lr: float = 5e-3,
    quats_lr: float = 1e-3,
    colors_lr: float = 2.5e-3,
    densify_until_iter: int = 15000,
) -> Path:
    """Train a 3D Gaussian Splatting scene and export the point cloud to .ply."""
    _require_cuda()

    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from PIL import Image

    device = torch.device("cuda")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_colmap_scene(sparse_model_dir, image_dir)
    params = _init_gaussians(scene["points3d"], scene["colors3d"], sh_degree, device)

    lrs = {
        "means": means_lr, "scales": scales_lr, "quats": quats_lr,
        "opacities": opacity_lr, "sh0": colors_lr, "shN": colors_lr / 20,
    }
    optimizers = {name: torch.optim.Adam([p], lr=lrs[name], eps=1e-15) for name, p in params.items()}

    strategy = DefaultStrategy(refine_stop_iter=densify_until_iter)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()

    viewmats = scene["viewmats"].to(device)
    Ks = scene["Ks"].to(device)
    n_cams = viewmats.shape[0]

    print(f"Training {params['means'].shape[0]} gaussians on {n_cams} views for {iterations} iterations")
    for step in range(iterations):
        cam_idx = step % n_cams
        target = torch.from_numpy(np.asarray(Image.open(scene["image_paths"][cam_idx]).convert("RGB"))).to(device).float() / 255.0

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        cur_degree = min(sh_degree, step // 1000)
        render, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmats[cam_idx: cam_idx + 1],
            Ks=Ks[cam_idx: cam_idx + 1],
            width=scene["width"],
            height=scene["height"],
            sh_degree=cur_degree,
            packed=False,  # strategy.step_post_backward(..., packed=False) needs dense [C, N, ...] outputs
        )
        render = render[0, ..., :3]

        loss = torch.nn.functional.l1_loss(render, target)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        for opt in optimizers.values():
            opt.zero_grad()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)

        if step % 1000 == 0:
            print(f"  step {step:6d}  loss {loss.item():.4f}  gaussians {params['means'].shape[0]}")

    ply_path = out_dir / "points.ply"
    _export_ply(params, ply_path)
    print(f"Done -- {params['means'].shape[0]} gaussians exported to {ply_path}")
    return ply_path


def _export_ply(params, ply_path: Path):
    """Write the standard 3DGS splat PLY: raw (pre-activation) per-Gaussian
    position/normal/color(SH dc)/opacity/scale/rotation -- the format read by
    SuperSplat, antimatter15/splat, and the reference INRIA viewer, so the
    export doubles as a real point cloud AND a real splat file."""
    from plyfile import PlyData, PlyElement

    means = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(-1, 3)
    opacities = params["opacities"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()

    n = means.shape[0]
    vertex = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["nx"] = vertex["ny"] = vertex["nz"] = 0.0
    vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    vertex["opacity"] = opacities
    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    )

    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(ply_path))
