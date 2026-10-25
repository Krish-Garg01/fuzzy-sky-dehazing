"""Guided Image Filter implementation in NumPy + OpenCV box filters (He et al., TPAMI 2013).

Used for O(N) edge-preserving refinement of transmission and membership maps.
"""

from __future__ import annotations

import cv2
import numpy as np


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an RGB float32 [0, 1] image to a 2D grayscale float32 [0, 1] guide.

    Uses standard ITU-R BT.601 luminance weights: 0.299 R + 0.587 G + 0.114 B.

    Args:
        img: RGB image of shape (H, W, 3), dtype float32, in [0, 1].

    Returns:
        Grayscale image of shape (H, W), dtype float32, in [0, 1].
    """
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB or (H, W) grayscale image, got {img.shape}")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.tensordot(img.astype(np.float32, copy=False), weights, axes=([2], [0]))
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def _box_filter(arr: np.ndarray, r: int) -> np.ndarray:
    """Apply a normalised (2r + 1) x (2r + 1) mean box filter."""
    ksize = (2 * r + 1, 2 * r + 1)
    return cv2.boxFilter(
        arr,
        ddepth=-1,
        ksize=ksize,
        normalize=True,
        borderType=cv2.BORDER_DEFAULT,
    )


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    r: int,
    eps: float,
) -> np.ndarray:
    """Perform edge-preserving guided filtering (He et al., TPAMI 2013).

    Internal statistics are accumulated in float64 to avoid precision loss in
    `var_I = mean_II - mean_I^2` when `eps` is small (e.g., 1e-4).

    Args:
        guide: Grayscale guide image of shape (H, W) (or RGB (H, W, 3), converted to gray),
               dtype float32, range [0, 1].
        src: Source 2D map to filter of shape (H, W), dtype float32.
        r: Window radius (window side length is 2*r + 1; must be >= 1).
        eps: Regularisation parameter controlling smoothness vs edge preservation (> 0).

    Returns:
        Filtered 2D map of shape (H, W), dtype float32.
    """
    if r < 1:
        raise ValueError(f"Guided filter radius r must be >= 1, got {r}")
    if eps <= 0.0:
        raise ValueError(f"Guided filter eps must be > 0, got {eps}")

    I_guide = rgb_to_gray(guide) if guide.ndim == 3 else guide
    if I_guide.ndim != 2 or src.ndim != 2:
        raise ValueError(f"Expected 2D guide and 2D src, got {I_guide.shape} and {src.shape}")
    if I_guide.shape != src.shape:
        raise ValueError(f"guide shape {I_guide.shape} != src shape {src.shape}")

    I = I_guide.astype(np.float64)
    p = src.astype(np.float64)

    mean_I = _box_filter(I, r)
    mean_p = _box_filter(p, r)
    mean_Ip = _box_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(I * I, r)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    q = mean_a * I + mean_b
    return q.astype(np.float32)
