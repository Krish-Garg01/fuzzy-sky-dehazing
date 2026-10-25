"""Dark Channel Prior (DCP) core primitives (He et al., TPAMI 2011).

All image arrays are float32, shape (H, W, 3), RGB order, range [0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np


def dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """Compute the dark channel of an RGB image.

    Takes the per-pixel minimum across the RGB channels, followed by
    morphological erosion with a square (patch x patch) structuring element.

    Args:
        img: Input RGB image of shape (H, W, 3), dtype float32.
        patch: Side length of the square local patch Omega(x) (must be >= 1).

    Returns:
        Dark channel map of shape (H, W), dtype float32.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {img.shape}")
    if patch < 1:
        raise ValueError(f"patch size must be >= 1, got {patch}")

    min_c = np.min(img.astype(np.float32, copy=False), axis=2)
    if patch == 1:
        return min_c

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(min_c, kernel, borderType=cv2.BORDER_REPLICATE)
    return dark.astype(np.float32, copy=False)


def estimate_atmospheric_light(
    img: np.ndarray,
    dark: np.ndarray,
    top_frac: float = 0.001,
) -> np.ndarray:
    """Estimate global atmospheric light A using the standard He et al. method.

    1. Select the top `top_frac` (default 0.1%) brightest pixels in `dark`.
    2. Among those candidates, pick the pixel with the largest channel sum in `img`.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        dark: Dark channel of `img`, shape (H, W), dtype float32, in [0, 1].
        top_frac: Fraction of brightest dark-channel pixels to consider (0 < top_frac <= 1).

    Returns:
        Atmospheric light vector `A` of shape (3,), dtype float32, in [0, 1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {img.shape}")
    if dark.shape != img.shape[:2]:
        raise ValueError(f"dark shape {dark.shape} does not match img spatial shape {img.shape[:2]}")
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")

    num_pixels = dark.size
    n_top = max(1, int(np.ceil(num_pixels * top_frac)))

    dark_flat = dark.reshape(-1)
    img_flat = img.reshape(-1, 3).astype(np.float32, copy=False)

    # Stable sort ensures deterministic tie-breaking across platforms
    top_indices = np.argsort(dark_flat, kind="stable")[-n_top:]
    candidate_pixels = img_flat[top_indices]
    intensities = np.sum(candidate_pixels, axis=1)
    best_idx = int(np.argmax(intensities))

    A = candidate_pixels[best_idx].copy()
    return A.astype(np.float32)


def transmission_dcp(
    img: np.ndarray,
    A: np.ndarray,
    omega: float = 0.95,
    patch: int = 15,
) -> np.ndarray:
    """Estimate coarse patch-based DCP transmission t_DCP(x).

    Formula:
        t_DCP(x) = 1 - omega * min_{y in Omega(x)} min_c (I_c(y) / A_c)

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        A: Global atmospheric light of shape (3,), dtype float32.
        omega: Haze retention factor in (0, 1], default 0.95.
        patch: Patch size for dark channel computation, default 15.

    Returns:
        Coarse transmission map `t_DCP` of shape (H, W), dtype float32, clipped to [0, 1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {img.shape}")
    A_arr = np.asarray(A, dtype=np.float32).reshape(1, 1, 3)
    A_safe = np.maximum(A_arr, 1e-6)

    norm_img = img.astype(np.float32, copy=False) / A_safe
    dark_norm = dark_channel(norm_img, patch=patch)
    t_dcp = 1.0 - float(omega) * dark_norm
    return np.clip(t_dcp, 0.0, 1.0).astype(np.float32)


def recover(
    img: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    t0: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover scene radiance J(x) from hazy image I(x), atmospheric light A, and transmission t(x).

    Formula:
        J_raw(x) = (I(x) - A) / max(t(x), t0) + A
        J_clipped(x) = clip(J_raw(x), 0, 1)

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        A: Atmospheric light vector of shape (3,), dtype float32.
        t: Transmission map of shape (H, W), dtype float32, in [0, 1].
        t0: Lower bound for transmission to limit noise amplification, default 0.1.

    Returns:
        Tuple `(J_clipped, J_raw)` where both have shape (H, W, 3) and dtype float32.
        `J_clipped` is in [0, 1]; `J_raw` is unclipped (used for the saturation metric).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {img.shape}")
    if t.shape != img.shape[:2]:
        raise ValueError(f"Transmission shape {t.shape} does not match image {img.shape[:2]}")

    A_3d = np.asarray(A, dtype=np.float32).reshape(1, 1, 3)
    t_clamped = np.maximum(t.astype(np.float32, copy=False), float(t0))[:, :, np.newaxis]

    J_raw = ((img.astype(np.float32, copy=False) - A_3d) / t_clamped + A_3d).astype(np.float32)
    J_clipped = np.clip(J_raw, 0.0, 1.0).astype(np.float32)
    return J_clipped, J_raw
