# python
import time
import math
import numpy as np
import torch
import laspy
import matplotlib.pyplot as plt
import augmentation as au

from sideview import (
    topview as topview_np,
    sideview as sideview_np,
    points_to_images as points_to_images_np,
)
from sideview_torch import (
    _topview_torch as topview_torch,
    _sideview_torch as sideview_torch_fn,
    points_to_images as points_to_images_torch,
)

LAS_PATH = r"T:\Puliti_Reference_Dataset\down\train\03567.las"
# Set to None to use all points (may be large)
SUBSAMPLE_POINTS = 500_000

def load_las_points(path=LAS_PATH, subsample=SUBSAMPLE_POINTS):
    # las = laspy.read(path)
    # pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    # if subsample and pts.shape[0] > subsample:
    #     idx = np.random.default_rng(0).choice(pts.shape[0], subsample, replace=False)
    #     pts = pts[idx]
    pts = au.augment(path)
    return pts

_LAS_POINTS = None  # cached

def get_points():
    global _LAS_POINTS
    if _LAS_POINTS is None:
        print(f"Loading LAS once from: {LAS_PATH}")
        _LAS_POINTS = load_las_points()
        print(f"Loaded { _LAS_POINTS.shape[0] } points.")
    return _LAS_POINTS

def center_scale(points: np.ndarray) -> np.ndarray:
    pts = points - np.median(points, axis=0)
    denom = np.max(np.abs(pts))
    if denom < 1e-12:
        denom = 1.0
    return pts / denom

def rotation_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r), -math.sin(r), 0],
                     [math.sin(r),  math.cos(r), 0],
                     [0,            0,           1]], dtype=np.float32)

def compare_arrays(a: np.ndarray, b: np.ndarray, name: str, atol=1e-6):
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    frac = (diff > atol).sum() / diff.size
    print(f"{name:>12}: max={max_diff:.3e}  mean={mean_diff:.3e}  frac>|{atol}|={frac:.3e}")
    return max_diff, mean_diff, frac

def plot_views(views_np, views_torch, num_side, res):
    # Select channels for inspection
    top = views_np[0]
    top_t = views_torch[0]
    side0 = views_np[3]
    side0_t = views_torch[3]
    bottom = views_np[num_side + 1]
    bottom_t = views_torch[num_side + 1]
    section = views_np[num_side + 2]
    section_t = views_torch[num_side + 2]

    diff_top = top - top_t
    diff_side0 = side0 - side0_t
    diff_bottom = bottom - bottom_t
    diff_section = section - section_t

    fig, axes = plt.subplots(4, 3, figsize=(14, 9))
    cmap = "viridis"
    axes[0,0].set_title("Top (np)")
    axes[0,0].imshow(top, cmap=cmap)
    axes[0,1].set_title("Top (torch)")
    axes[0,1].imshow(top_t, cmap=cmap)
    axes[0,2].set_title("Top diff")
    im0 = axes[0,2].imshow(diff_top, cmap="coolwarm")
    fig.colorbar(im0, ax=axes[0,2])
    #axes[0,3].axis("off")

    axes[1,0].set_title("Side0 (np)")
    axes[1,0].imshow(side0, cmap=cmap)
    axes[1,1].set_title("Side0 (torch)")
    axes[1,1].imshow(side0_t, cmap=cmap)
    axes[1,2].set_title("Side0 diff")
    im1 = axes[1,2].imshow(diff_side0, cmap="coolwarm")
    fig.colorbar(im1, ax=axes[1,2])
    #axes[1,3].axis("off")

    axes[2,0].set_title("Bottom (np)")
    axes[2,0].imshow(bottom, cmap=cmap)
    axes[2,1].set_title("Bottom (torch)")
    axes[2,1].imshow(bottom_t, cmap=cmap)
    axes[2,2].set_title("Bottom diff")
    im2 = axes[2,2].imshow(diff_bottom, cmap="coolwarm")
    fig.colorbar(im2, ax=axes[2,2])

    axes[3,0].set_title("Section (np)")
    axes[3,0].imshow(section, cmap=cmap)
    axes[3,1].set_title("Section (torch)")
    axes[3,1].imshow(section_t, cmap=cmap)
    axes[3,2].set_title("Bottom diff")
    im2 = axes[3,2].imshow(diff_section, cmap="coolwarm")
    fig.colorbar(im2, ax=axes[3,2])

    plt.tight_layout()
    plt.show()

def test_individual_views(res=256, num_side=4, device="cpu"):
    print("=== Individual view equivalence (top / side rotations / bottom) ===")
    raw_points = get_points()
    proc_points = center_scale(raw_points)

    # Top
    top_np_img = topview_np(proc_points.copy(), res_im=res, inverse=False)
    top_torch_img = topview_torch(torch.from_numpy(proc_points), res=res, inverse=False).numpy()
    compare_arrays(top_np_img, top_torch_img, "top")

    # Bottom
    bot_np_img = topview_np(proc_points.copy(), res_im=res, inverse=True)
    bot_torch_img = topview_torch(torch.from_numpy(proc_points), res=res, inverse=True).numpy()
    compare_arrays(bot_np_img, bot_torch_img, "bottom")

    # Side rotations
    for deg in [0, 45, 90, 135]:
        R = rotation_z(deg)
        rot_pts = proc_points @ R.T
        side_np_img = sideview_np(rot_pts.copy(), res_im=res)
        side_torch_img = sideview_torch_fn(torch.from_numpy(rot_pts), res=res).numpy()
        compare_arrays(side_np_img, side_torch_img, f"side_{deg:03d}")

def test_full_pipeline(res=256, num_side=4, device="cpu", check_section=True, show_plot=True):
    print("\n=== Full pipeline equivalence (channels excluding DBH section) ===")
    raw_points = get_points()

    # Legacy
    views_np = points_to_images_np(raw_points.copy(), res_im=res, num_side=num_side,
                                   plot=False, max_n=500000)

    # Torch
    dev = torch.device(device)
    views_torch = points_to_images_torch(raw_points.copy(), res_im=res,
                                         num_side=num_side, device=dev)

    for ch in list(range(0, num_side + 2)):
        compare_arrays(views_np[ch], views_torch[ch].cpu().numpy(), f"chan_{ch:02d}")

    if check_section:
        print("\n(DBH section comparison - expected differences due to DBSCAN omission)")
        compare_arrays(views_np[num_side + 2], views_torch[num_side + 2].cpu().numpy(),
                       f"chan_{num_side+2:02d}", atol=1e-3)

    if show_plot:
        plot_views(views_np, views_torch.cpu().numpy(), num_side, res)

def benchmark(res=256, num_side=4, repeats=5, device="cpu"):
    print("\n=== Benchmark (averaged over repeats) ===")
    base_points = get_points()

    # Warmup torch
    _ = points_to_images_torch(base_points.copy(), res_im=res, num_side=num_side,
                               device=torch.device(device))

    # Legacy
    t_np = []
    for r in range(repeats):
        # permute for fairness
        perm = np.random.default_rng(r).permutation(base_points.shape[0])
        pts = base_points[perm]
        t0 = time.perf_counter()
        _ = points_to_images_np(pts.copy(), res_im=res, num_side=num_side,
                                plot=False, max_n=500000)
        t_np.append(time.perf_counter() - t0)
    print(f"Legacy numpy/pandas: {np.mean(t_np):.4f}s ± {np.std(t_np):.4f}s")

    # Torch
    t_torch = []
    for r in range(repeats):
        perm = np.random.default_rng(r).permutation(base_points.shape[0])
        pts = base_points[perm]
        t0 = time.perf_counter()
        _ = points_to_images_torch(pts.copy(), res_im=res, num_side=num_side,
                                   device=torch.device(device))
        if device != "cpu":
            torch.cuda.synchronize()
        t_torch.append(time.perf_counter() - t0)
    print(f"Torch rasterization ({device}): {np.mean(t_torch):.4f}s ± {np.std(t_torch):.4f}s")

def main():
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_individual_views(res=256, num_side=4, device=device)
    test_full_pipeline(res=256, num_side=4, device=device, check_section=True, show_plot=True)
    benchmark(res=256, num_side=4, repeats=10, device=device)

if __name__ == "__main__":
    main()