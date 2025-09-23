import os
import math
import numpy as np
import torch
import laspy
from typing import Tuple, Dict, Any

from parallel_densenet import TrainDataset_AllChannels  # adjust import if file name differs

def _to_numpy_views(x: Any) -> np.ndarray:
    """
    Accepts:
      - torch.Tensor with shape (V,H,W) or (V,1,H,W)
      - numpy array with same shapes
    Returns: (V,H,W) float32 numpy array.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu()
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:,0]
        return arr.to(torch.float32).numpy()
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:,0]
    return arr.astype(np.float32, copy=False)

def validate_projection_backends(
    las_path: str,
    res: int = 256,
    n_sides: int = 4,
    max_samples: int = None,
    img_tol: float = 1e-5,
    section_tol: float = 1e-3,
    verbose: bool = True,
    train_height_mean = 15.2046,
    train_height_sd = 9.5494
) -> bool:
    """
    Compare outputs of TrainDataset_AllChannels for projection_backend 'numpy' vs 'torch'.
    Returns True if all non-section channels differ <= img_tol and section channel <= section_tol.
    """
    rng_seed = 1234
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    if verbose:
        print(f"Loading LAS: {las_path}")
    las = laspy.read(las_path)

    ds_np = TrainDataset_AllChannels(
        las, img_trans=None, pc_rotate=False, height_noise=0.0,
        res=res, n_sides=n_sides, projection_backend="numpy", test=True, height_mean=train_height_mean, height_sd=train_height_sd
    )
    ds_torch = TrainDataset_AllChannels(
        las, img_trans=None, pc_rotate=False, height_noise=0.0,
        res=res, n_sides=n_sides, projection_backend="torch", test=True, height_mean=train_height_mean, height_sd=train_height_sd
    )

    if len(ds_np) != len(ds_torch):
        raise ValueError("Dataset length mismatch")

    n = len(ds_np) if max_samples is None else min(len(ds_np), max_samples)
    if n == 0:
        raise ValueError("Empty dataset after filtering")

    # Accumulators
    channel_count = n_sides + 3
    abs_diff_sum = np.zeros(channel_count, dtype=np.float64)
    abs_diff_max = np.zeros(channel_count, dtype=np.float64)
    pixel_counts = np.zeros(channel_count, dtype=np.int64)

    height_abs_diff_sum = 0.0
    height_abs_diff_max = 0.0

    all_within = True

    for idx in range(n):
        sample_np = ds_np[idx]
        sample_torch = ds_torch[idx]

        # Expected tuple: (views, label, height) or (views, height, label)
        # Heuristic: locate tensor/array with 3 or 4 dims as views, scalar as height.
        views_candidates = [s for s in sample_np if (isinstance(s, (np.ndarray, torch.Tensor)) and (s.ndim in (3,4)))]
        if not views_candidates:
            raise RuntimeError("Could not locate image tensor in dataset sample.")
        views_np = _to_numpy_views(views_candidates[0])
        views_t = _to_numpy_views([s for s in sample_torch if (isinstance(s, (np.ndarray, torch.Tensor)) and (s.ndim in (3,4)))][0])

        if views_np.shape != views_t.shape:
            raise ValueError(f"Shape mismatch at idx {idx}: {views_np.shape} vs {views_t.shape}")

        if views_np.shape[0] != channel_count:
            raise ValueError(f"Unexpected channel count {views_np.shape[0]} (expected {channel_count})")

        # Extract height (float)
        heights_np = [s for s in sample_np if np.isscalar(s) or (isinstance(s, (np.ndarray, torch.Tensor)) and np.array(s).ndim == 0)]
        heights_t  = [s for s in sample_torch if np.isscalar(s) or (isinstance(s, (np.ndarray, torch.Tensor)) and np.array(s).ndim == 0)]
        if not heights_np or not heights_t:
            # Height may be embedded differently; skip height check if absent
            height_np = height_t = None
        else:
            height_np = float(np.array(heights_np[0]))
            height_t  = float(np.array(heights_t[0]))
            h_diff = abs(height_np - height_t)
            height_abs_diff_sum += h_diff
            height_abs_diff_max = max(height_abs_diff_max, h_diff)

        # Per-channel diffs
        for ch in range(channel_count):
            a = views_np[ch]
            b = views_t[ch]
            diff = np.abs(a - b)
            d_sum = diff.sum()
            d_max = diff.max()
            abs_diff_sum[ch] += d_sum
            abs_diff_max[ch] = max(abs_diff_max[ch], d_max)
            pixel_counts[ch] += diff.size

        # Per-sample pass/fail (excluding section for strict tolerance)
        for ch in range(channel_count):
            tol = section_tol if ch == channel_count - 1 else img_tol
            if abs_diff_max[ch] > tol:
                all_within = False

    mean_abs_diff = abs_diff_sum / np.maximum(pixel_counts, 1)

    if verbose:
        print("\n=== Projection Backend Comparison Summary ===")
        for ch in range(channel_count):
            tol = section_tol if ch == channel_count - 1 else img_tol
            tag = "section" if ch == channel_count - 1 else f"chan_{ch:02d}"
            print(f"{tag:>8} | mean_abs={mean_abs_diff[ch]:.3e}  max_abs={abs_diff_max[ch]:.3e}  tol={tol:.1e}")

        if heights_np and heights_t:
            print(f"height   | mean_abs={height_abs_diff_sum / n:.3e}  max_abs={height_abs_diff_max:.3e}")

        print(f"\nResult: {'PASS' if all_within else 'FAIL'}")
        # add a plot if failed
        if not all_within:
            try:
                import matplotlib.pyplot as plt
                # Plot last sample's views for visual inspection
                def plot_views(views_np, views_torch, n_sides, res):
                    top_np = views_np[0]
                    top_t = views_torch[0]
                    side0_np = views_np[3]
                    side0_t = views_torch[3]
                    bottom_np = views_np[n_sides + 1]
                    bottom_t = views_torch[n_sides + 1]
                    section_np = views_np[n_sides + 2]
                    section_t = views_torch[n_sides + 2]

                    diff_top = top_np - top_t
                    diff_side0 = side0_np - side0_t
                    diff_bottom = bottom_np - bottom_t
                    diff_section = section_np - section_t

                    fig, axes = plt.subplots(4, 3, figsize=(14, 9))
                    cmap = "viridis"
                    axes[0,0].set_title("Top (np)")
                    axes[0,0].imshow(top_np, cmap=cmap)
                    axes[0,1].set_title("Top (torch)")
                    axes[0,1].imshow(top_t, cmap=cmap)
                    axes[0,2].set_title("Top diff")
                    axes[0,2].imshow(diff_top, cmap='bwr', vmin=-img_tol*10, vmax=img_tol*10)

                    axes[1,0].set_title("Side0 (np)")
                    axes[1,0].imshow(side0_np, cmap=cmap)
                    axes[1,1].set_title("Side0 (torch)")
                    axes[1,1].imshow(side0_t, cmap=cmap)
                    axes[1,2].set_title("Side0 diff")
                    axes[1,2].imshow(diff_side0, cmap='bwr', vmin=-img_tol*10, vmax=img_tol*10)

                    axes[2,0].set_title("Bottom (np)")
                    axes[2,0].imshow(bottom_np, cmap=cmap)
                    axes[2,1].set_title("Bottom (torch)")
                    axes[2,1].imshow(bottom_t, cmap=cmap)
                    axes[2,2].set_title("Bottom diff")
                    axes[2,2].imshow(diff_bottom, cmap='bwr', vmin=-img_tol*10, vmax=img_tol*10)
                    axes[3,0].set_title("Section (np)")
                    axes[3,0].imshow(section_np, cmap=cmap)
                    axes[3,1].set_title("Section (torch)")
                    axes[3,1].imshow(section_t, cmap=cmap)
                    axes[3,2].set_title("Section diff")
                    axes[3,2].imshow(diff_section, cmap='bwr', vmin=-section_tol*10, vmax=section_tol*10)
                    plt.tight_layout()
                    plt.show()
                plot_views(views_np, views_t, n_sides, res)
            except ImportError:
                print("matplotlib not available, skipping plot")
    return all_within


if __name__ == "__main__":
    ok = validate_projection_backends(
        las_path=r"C:\TLS\docker\input\mini.las",
        res=256,
        n_sides=4,
        max_samples=None,
        img_tol=1e-5,
        section_tol=1e-3,
        verbose=True
    )
    exit(0 if ok else 1)