# python
import math
from typing import Optional
import torch

def _rasterize_2d(idx_x: torch.Tensor,
                  idx_y: torch.Tensor,
                  values: torch.Tensor,
                  res: int,
                  reduce: str = "amax") -> torch.Tensor:
    """
    idx_x, idx_y: int64 pixel indices in [0, res)
    values: float32 per-point values to aggregate
    reduce: 'amax' (max) or 'amin' (min)
    """
    assert reduce in ("amax", "amin")
    device = idx_x.device
    init_val = float("-inf") if reduce == "amax" else float("inf")
    out = torch.full((res * res,), init_val, device=device, dtype=values.dtype)
    lin_idx = (idx_x * res + idx_y).view(-1)
    vals = values.view(-1)
    out.scatter_reduce_(0, lin_idx, vals, reduce=reduce, include_self=True)
    out = out.view(res, res)
    # Replace untouched pixels (±inf) with 0
    if reduce == "amax":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    else:
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out

def _center_and_scale(points: torch.Tensor) -> torch.Tensor:
    # center by median and scale by global max abs (avoid div by zero)
    med = points.median(dim=0).values
    pts = points - med
    scale = pts.abs().max()
    scale = torch.clamp(scale, min=1e-12)
    return pts / scale

def _compute_xy_bins(pts_xy: torch.Tensor, res: int, inverse: bool) -> torch.Tensor:
    # Mimics original logic choosing pixel size by longer axis and half-shift on shorter axis
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_med = x.median()
    y_med = y.median()
    size = torch.maximum(x_max - x_min, y_max - y_min) / float(res)
    size = torch.clamp(size, min=1e-12)

    if inverse:
        # bottom view
        max_axis_x = (x_max - x_min) > (y_max - y_min)
        if max_axis_x:
            x_pos = ((x - x_min) / size).long()
            y_pos = (((y - y_med) / size).long() + (res // 2))
        else:
            x_pos = (((x - x_med) / size).long() + (res // 2))
            y_pos = ((y - y_min) / size).long()
    else:
        # top view
        max_axis_x = (x_max - x_min) > (y_max - y_min)
        if max_axis_x:
            x_pos = ((x - x_min) / size).long()
            y_pos = (((y - y_med) / size).long() + (res // 2))
        else:
            x_pos = (((x - x_med) / size).long() + (res // 2))
            y_pos = ((y - y_min) / size).long()

    # wrap indices like original
    x_pos = torch.remainder(x_pos, res)
    y_pos = torch.remainder(y_pos, res)
    return torch.stack([x_pos, y_pos], dim=1)

def _compute_xz_bins(pts_xz: torch.Tensor, res: int) -> torch.Tensor:
    x = pts_xz[:, 0]
    z = pts_xz[:, 1]
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    x_med = x.median()
    z_med = z.median()
    size = torch.maximum(x_max - x_min, z_max - z_min) / float(res)
    size = torch.clamp(size, min=1e-12)

    max_axis_x = (x_max - x_min) > (z_max - z_min)
    if max_axis_x:
        x_pos = ((x - x_min) / size).long()
        z_pos = (((z - z_med) / size).long() + (res // 2))
    else:
        x_pos = (((x - x_med) / size).long() + (res // 2))
        z_pos = ((z - z_min) / size).long()

    x_pos = torch.remainder(x_pos, res)
    z_pos = torch.remainder(z_pos, res)
    return torch.stack([x_pos, z_pos], dim=1)

def topview_torch(points: torch.Tensor, res: int = 256, inverse: bool = False) -> torch.Tensor:
    """
    points: [N,3] float32 torch tensor (x,y,z), assumed already centered/scaled
    Returns: [res,res] float32 image; depth is max(z) for top, min(z) for bottom.
    """
    bins = _compute_xy_bins(points[:, :2], res, inverse=inverse)
    x_pos, y_pos = bins[:, 0], bins[:, 1]
    depth = points[:, 2].contiguous()
    reduce = "amin" if inverse else "amax"
    return _rasterize_2d(x_pos, y_pos, depth, res, reduce=reduce)

def sideview_torch(points: torch.Tensor, res: int = 256) -> torch.Tensor:
    """
    points: [N,3] float32 torch tensor (x,y,z), assumed already centered/scaled
    Returns: [res,res] float32 image; depth is max(y) per (x,z) bin.
    """
    bins = _compute_xz_bins(points[:, (0, 2)], res)
    x_pos, z_pos = bins[:, 0], bins[:, 1]
    depth = points[:, 1].contiguous()
    return _rasterize_2d(x_pos, z_pos, depth, res, reduce="amax")

def points_to_images_torch(points_np,
                           res_im: int = 256,
                           num_side: int = 4,
                           max_n: int = 500000,
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Vectorized torch version of points→multi‑view raster.
    Returns: [num_side + 3, res_im, res_im] float32 tensor.
    """
    if device is None:
        device = torch.device("cpu")

    # Accept numpy or torch input
    if isinstance(points_np, torch.Tensor):
        pts = points_np.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        pts = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)

    # Subsample if needed
    if pts.shape[0] > max_n:
        idx = torch.randperm(pts.shape[0], device=device)[:max_n]
        pts = pts.index_select(0, idx)

    # Center & scale in torch
    pts = _center_and_scale(pts)

    # Allocate views
    views = torch.zeros((num_side + 3, res_im, res_im), device=device, dtype=torch.float32)

    # Top and bottom
    views[0] = topview_torch(pts, res=res_im, inverse=False)
    views[num_side + 1] = topview_torch(pts, res=res_im, inverse=True)

    # Side views with Z‑axis rotations
    deg_steps = torch.linspace(0.0, 360.0, steps=num_side + 1, device=device)[:-1]
    cos_t = torch.cos(torch.deg2rad(deg_steps))
    sin_t = torch.sin(torch.deg2rad(deg_steps))
    R = torch.stack([
        torch.stack([cos_t, -sin_t, torch.zeros_like(cos_t)], dim=-1),
        torch.stack([sin_t,  cos_t, torch.zeros_like(cos_t)], dim=-1),
        torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_side, 3)
    ], dim=-2)  # [num_side, 3, 3]

    # Apply each rotation and rasterize
    # Batch‑apply via matmul broadcasting: [num_side, N, 3] = [num_side, N, 3] @ [num_side, 3, 3]^T
    pts_b = pts.unsqueeze(0).expand(num_side, -1, -1)          # [num_side, N, 3]
    pts_rot = torch.bmm(pts_b, R.transpose(1, 2))              # [num_side, N, 3]
    for i in range(num_side):
        views[i + 1] = sideview_torch(pts_rot[i], res=res_im)

    # DBH section view (fast torch version without DBSCAN denoising)
    section_mask = (pts[:, 2] < 1.5) & (pts[:, 2] > 1.0)
    if section_mask.any():
        section_pts = pts[section_mask]
        # Center/scale section for stability
        section_pts = _center_and_scale(section_pts)
        views[num_side + 2] = sideview_torch(section_pts, res=res_im)
    else:
        views[num_side + 2].zero_()

    return views