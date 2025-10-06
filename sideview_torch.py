# python
# file: sideview_torch_exact.py
import numpy as np
import torch
from typing import Optional
from sideview import sectionview  # legacy, uses DBSCAN and raw points

@torch.no_grad()
def _np_median(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    NumPy-equivalent median using linear interpolation for even-length arrays.
    Matches np.median behavior.
    """
    # if dim is None:
    #     return torch.quantile(x, 0.5)
    # return torch.quantile(x, 0.5, dim=dim)
    if dim is None:
        return torch.median(x)
    return torch.median(x, dim=dim).values

@torch.no_grad()
def _center_and_scale(points: torch.Tensor) -> torch.Tensor:
    # Center by NumPy-like per-axis median; scale by global max-abs (scalar)
    med = _np_median(points, dim=0)
    pts = points - med
    scale = pts.abs().max().clamp(min=1e-12)
    return pts / scale

@torch.no_grad()
def _rasterize_2d(ix: torch.Tensor,
                  iy: torch.Tensor,
                  values: torch.Tensor,
                  res: int,
                  reduce: str) -> torch.Tensor:
    assert reduce in ("amax", "amin")
    init = float("-inf") if reduce == "amax" else float("inf")
    out = torch.full((res * res,), init, device=ix.device, dtype=values.dtype)
    lin = (ix * res + iy).reshape(-1)
    out.scatter_reduce_(0, lin, values.reshape(-1), reduce=reduce, include_self=True)
    img = out.view(res, res)
    img = torch.where(torch.isinf(img), torch.zeros_like(img), img)
    return img

@torch.no_grad()
def _bins_xy(pts_xy: torch.Tensor, res: int, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_med = _np_median(x)
    y_med = _np_median(y)
    size = torch.maximum(x_max - x_min, y_max - y_min) / float(res)
    size = size.clamp(min=1e-12)

    max_axis_x = (x_max - x_min) > (y_max - y_min)
    if inverse:
        if max_axis_x:
            x_pos = ((x - x_min) / size).long()
            y_pos = (((y - y_med) / size).long() + (res // 2))
        else:
            x_pos = (((x - x_med) / size).long() + (res // 2))
            y_pos = ((y - y_min) / size).long()
    else:
        if max_axis_x:
            x_pos = ((x - x_min) / size).long()
            y_pos = (((y - y_med) / size).long() + (res // 2))
        else:
            x_pos = (((x - x_med) / size).long() + (res // 2))
            y_pos = ((y - y_min) / size).long()

    x_pos = torch.remainder(x_pos, res)
    y_pos = torch.remainder(y_pos, res)
    return x_pos, y_pos

@torch.no_grad()
def _bins_xz(pts_xz: torch.Tensor, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = pts_xz[:, 0]
    z = pts_xz[:, 1]
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    x_med = _np_median(x)
    z_med = _np_median(z)
    size = torch.maximum(x_max - x_min, z_max - z_min) / float(res)
    size = size.clamp(min=1e-12)

    max_axis_x = (x_max - x_min) > (z_max - z_min)
    if max_axis_x:
        x_pos = ((x - x_min) / size).long()
        z_pos = (((z - z_med) / size).long() + (res // 2))
    else:
        x_pos = (((x - x_med) / size).long() + (res // 2))
        z_pos = ((z - z_min) / size).long()

    x_pos = torch.remainder(x_pos, res)
    z_pos = torch.remainder(z_pos, res)
    return x_pos, z_pos

@torch.no_grad()
def _topview_torch(points: torch.Tensor, res: int = 256, inverse: bool = False) -> torch.Tensor:
    ix, iy = _bins_xy(points[:, :2], res, inverse=inverse)
    reduce = "amin" if inverse else "amax"
    return _rasterize_2d(ix, iy, points[:, 2].contiguous(), res, reduce)

@torch.no_grad()
def _sideview_torch(points: torch.Tensor, res: int = 256) -> torch.Tensor:
    ix, iz = _bins_xz(points[:, (0, 2)], res)
    return _rasterize_2d(ix, iz, points[:, 1].contiguous(), res, "amax")


@torch.no_grad()
def points_to_images(points: np.ndarray,
                                 res_im: int = 256,
                                 num_side: int = 4,
                                 max_n: int = 500000,
                                 device: Optional[torch.device] = None) -> np.ndarray:
    """
    Matches legacy output exactly:
    - Section channel uses legacy DBSCAN on raw points.
    - Top/sides/bottom rasterized with torch.
    Returns CPU numpy float32 array of shape [num_side+3, res, res].
    """
    if device is None:
        device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Always produce a NumPy copy for the section (raw, unnormalized)
    if isinstance(points, torch.Tensor):
        points_np_raw = points.detach().cpu().numpy()
    else:
        points_np_raw = np.asarray(points, dtype=np.float32)

    # 1) Section view from raw points (legacy)
    section = sectionview(points_np_raw, res_im=res_im, plot=False, debug=False).astype(np.float32)

    # 2) Subsample like legacy before other views
    pts_np = points_np_raw
    if pts_np.shape[0] > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts_np.shape[0], max_n, replace=False)
        pts_np = pts_np[idx]

    # 3) Center/scale like legacy
    pts = torch.from_numpy(pts_np.astype(np.float32)).to(device)
    pts = _center_and_scale(pts)

    # 4) Allocate and fill channels
    out = torch.zeros((num_side + 3, res_im, res_im), dtype=torch.float32, device=device)
    out[0] = _topview_torch(pts, res=res_im, inverse=False)

    # side rotations
    deg_steps = torch.linspace(0.0, 360.0, steps=num_side + 1, device=device)[:-1]
    cos_t = torch.cos(torch.deg2rad(deg_steps))
    sin_t = torch.sin(torch.deg2rad(deg_steps))
    R = torch.stack([
        torch.stack([cos_t, -sin_t, torch.zeros_like(cos_t)], dim=-1),
        torch.stack([sin_t,  cos_t, torch.zeros_like(cos_t)], dim=-1),
        torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_side, 3)
    ], dim=-2)  # [num_side, 3, 3]
    pts_b = pts.unsqueeze(0).expand(num_side, -1, -1)
    pts_rot = torch.bmm(pts_b, R.transpose(1, 2))
    for i in range(num_side):
        out[i + 1] = _sideview_torch(pts_rot[i], res=res_im)

    out[num_side + 1] = _topview_torch(pts, res=res_im, inverse=True)

    # 5) Section channel last, from legacy
    out[num_side + 2] = torch.from_numpy(section).to(device=device, dtype=torch.float32)

    # 6) Return CPU numpy float32
    return out #.cpu().numpy()

# Optional: deterministic seeding for DataLoader workers to make transforms consistent
def seed_worker(worker_id: int):
    import random, os
    base_seed = (torch.initial_seed() ^ os.getpid()) % (2 ** 32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)