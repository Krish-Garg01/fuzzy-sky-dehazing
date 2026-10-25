"""Full-reference, region-masked, saturation, and horizon-seam evaluation metrics.

All full-reference metrics compare the clipped output J against the clear ground truth G
(both float32 RGB in [0, 1]), except `saturated_fraction`, which inspects the unclipped `J_raw`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fsdehaze.guided_filter import rgb_to_gray
from fsdehaze.pipelines import Result

MAX_PSNR_DB: float = 100.0


def psnr(G: np.ndarray, J: np.ndarray) -> float:
    """Compute full-image Peak Signal-to-Noise Ratio (dB), capped at 100.0 dB.

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        J: Dehazed RGB image (clipped), shape (H, W, 3), float32 in [0, 1].

    Returns:
        PSNR in dB (float in (-inf, 100.0]).
    """
    if G.shape != J.shape:
        raise ValueError(f"Shape mismatch in psnr: {G.shape} vs {J.shape}")
    mse = float(np.mean((G.astype(np.float64) - J.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return MAX_PSNR_DB
    val = float(peak_signal_noise_ratio(G, J, data_range=1.0))
    return min(val, MAX_PSNR_DB)


def ssim(G: np.ndarray, J: np.ndarray) -> float:
    """Compute full-image Structural Similarity Index (SSIM).

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        J: Dehazed RGB image (clipped), shape (H, W, 3), float32 in [0, 1].

    Returns:
        Mean SSIM over channels in [-1, 1] (typically [0, 1]).
    """
    if G.shape != J.shape:
        raise ValueError(f"Shape mismatch in ssim: {G.shape} vs {J.shape}")
    return float(
        structural_similarity(
            G.astype(np.float32, copy=False),
            J.astype(np.float32, copy=False),
            channel_axis=-1,
            data_range=1.0,
        )
    )


def ciede2000_map(G: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute per-pixel CIEDE2000 colour difference map in CIE Lab space.

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        J: Dehazed RGB image (clipped), shape (H, W, 3), float32 in [0, 1].

    Returns:
        Per-pixel deltaE_00 array of shape (H, W), dtype float32 (>= 0).
    """
    if G.shape != J.shape:
        raise ValueError(f"Shape mismatch in ciede2000_map: {G.shape} vs {J.shape}")
    lab_G = rgb2lab(np.clip(G, 0.0, 1.0).astype(np.float32, copy=False))
    lab_J = rgb2lab(np.clip(J, 0.0, 1.0).astype(np.float32, copy=False))
    de = deltaE_ciede2000(lab_G, lab_J)
    return de.astype(np.float32)


def masked_psnr(G: np.ndarray, J: np.ndarray, mask: np.ndarray) -> float:
    """Compute PSNR (dB) restricted to pixels where `mask` is True.

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        J: Dehazed RGB image (clipped), shape (H, W, 3), float32 in [0, 1].
        mask: Boolean mask of shape (H, W).

    Returns:
        Masked PSNR in dB (capped at 100.0 dB), or `np.nan` if `mask` has no True pixels.
    """
    if G.shape != J.shape:
        raise ValueError(f"Shape mismatch in masked_psnr: {G.shape} vs {J.shape}")
    m = np.asarray(mask, dtype=bool)
    if m.shape != G.shape[:2]:
        raise ValueError(f"Mask shape {m.shape} does not match image {G.shape[:2]}")
    if not np.any(m):
        return float("nan")

    diff = G[m].astype(np.float64) - J[m].astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-10:
        return MAX_PSNR_DB
    val = 10.0 * float(np.log10(1.0 / mse))
    return min(val, MAX_PSNR_DB)


def masked_mean(val_map: np.ndarray, mask: np.ndarray) -> float:
    """Compute the mean of a 2D map over pixels where `mask` is True.

    Args:
        val_map: 2D array of shape (H, W).
        mask: Boolean mask of shape (H, W).

    Returns:
        Mean value over `mask`, or `np.nan` if `mask` has no True pixels.
    """
    m = np.asarray(mask, dtype=bool)
    if val_map.shape[:2] != m.shape:
        raise ValueError(f"Map shape {val_map.shape} does not match mask {m.shape}")
    if not np.any(m):
        return float("nan")
    return float(np.mean(val_map[m], dtype=np.float64))


def saturated_fraction(J_raw: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute the fraction of pixels where unclipped `J_raw` has any channel < 0 or > 1.

    Args:
        J_raw: Unclipped recovered RGB image of shape (H, W, 3), dtype float32.
        mask: Optional boolean region mask of shape (H, W). If None, evaluates full image.

    Returns:
        Fraction in [0.0, 1.0] of pixels in the region that are out of [0, 1],
        or `np.nan` if `mask` is provided and empty.
    """
    if J_raw.ndim != 3 or J_raw.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array for J_raw, got {J_raw.shape}")

    sat_pixel = np.any((J_raw < 0.0) | (J_raw > 1.0), axis=2)
    if mask is None:
        return float(np.mean(sat_pixel, dtype=np.float64))

    m = np.asarray(mask, dtype=bool)
    if m.shape != J_raw.shape[:2]:
        raise ValueError(f"Mask shape {m.shape} does not match J_raw {J_raw.shape[:2]}")
    if not np.any(m):
        return float("nan")
    return float(np.mean(sat_pixel[m], dtype=np.float64))


def boundary_band(mask: np.ndarray, width: int = 10) -> np.ndarray:
    """Construct a boolean band of pixels within `width` px of the sky/non-sky boundary.

    Uses Euclidean distance transforms (`scipy.ndimage.distance_transform_edt`) on both
    `mask` and `~mask`. If the mask is all-True or all-False (no boundary exists),
    returns an all-False array.

    Args:
        mask: Binary sky mask of shape (H, W), dtype bool.
        width: Half-width of the horizon boundary band in pixels (must be >= 1).

    Returns:
        Boolean array of shape (H, W) marking pixels within `width` px of the boundary.
    """
    if width < 1:
        raise ValueError(f"Boundary band width must be >= 1, got {width}")
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {m.shape}")

    # A boundary exists only if both sky (True) and non-sky (False) pixels are present
    if not np.any(m) or np.all(m):
        return np.zeros_like(m, dtype=bool)

    dist_inside = distance_transform_edt(m)
    dist_outside = distance_transform_edt(~m)
    band = (dist_inside <= float(width)) & (dist_outside <= float(width))
    return band.astype(bool)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on the grayscale luminance of an RGB image."""
    gray = rgb_to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy).astype(np.float32)


def seam_errors(
    G: np.ndarray,
    J: np.ndarray,
    mask: np.ndarray,
    width: int = 10,
) -> tuple[float, float]:
    """Compute horizon-seam L1 appearance error and gradient-magnitude discrepancy in band B.

    Definitions:
        - `seam_l1`: mean |J - G| over all pixels and channels inside `boundary_band(mask, width)`
        - `seam_grad`: mean ||grad J| - |grad G|| over all pixels inside `boundary_band(mask, width)`

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        J: Dehazed RGB image (clipped), shape (H, W, 3), float32 in [0, 1].
        mask: Ground-truth binary sky mask of shape (H, W), dtype bool.
        width: Boundary band width in pixels, default 10.

    Returns:
        Tuple `(seam_l1, seam_grad)`, or `(np.nan, np.nan)` if no boundary exists.
    """
    if G.shape != J.shape:
        raise ValueError(f"Shape mismatch in seam_errors: {G.shape} vs {J.shape}")

    band = boundary_band(mask, width=width)
    if not np.any(band):
        return float("nan"), float("nan")

    abs_diff = np.mean(np.abs(J.astype(np.float32) - G.astype(np.float32)), axis=2)
    seam_l1 = float(np.mean(abs_diff[band], dtype=np.float64))

    grad_G = _gradient_magnitude(G)
    grad_J = _gradient_magnitude(J)
    grad_diff = np.abs(grad_J - grad_G)
    seam_grad = float(np.mean(grad_diff[band], dtype=np.float64))

    return seam_l1, seam_grad


def evaluate_all(
    G: np.ndarray,
    result: Result,
    sky_mask: np.ndarray | None = None,
    seam_width: int = 10,
) -> dict[str, Any]:
    """Compute the complete metric suite for a single dehazed image.

    Args:
        G: Ground-truth clear RGB image, shape (H, W, 3), float32 in [0, 1].
        result: Pipeline `Result` containing `J`, `J_raw`, `A`, `info`.
        sky_mask: Optional boolean ground-truth sky mask of shape (H, W).
        seam_width: Horizon boundary band width in pixels (default 10).

    Returns:
        Dictionary of metrics:
            - `psnr`: Full-image PSNR (dB)
            - `ssim`: Full-image SSIM
            - `ciede2000`: Full-image mean CIEDE2000
            - `sky_psnr`: Sky-only PSNR (dB)
            - `sky_ciede2000`: Sky-only mean CIEDE2000
            - `nonsky_psnr`: Non-sky PSNR (dB)
            - `nonsky_ciede2000`: Non-sky mean CIEDE2000
            - `sat_frac_full`: Unclipped saturated pixel fraction [0, 1] (full image)
            - `sat_pct_full`: Unclipped saturated pixel percentage [0, 100] (full image)
            - `sat_frac_sky`: Unclipped saturated pixel fraction [0, 1] (sky region)
            - `sat_pct_sky`: Unclipped saturated pixel percentage [0, 100] (sky region)
            - `seam_l1`: Horizon-band mean |J - G|
            - `seam_grad`: Horizon-band mean ||grad J| - |grad G||
            - `sky_fraction`: Fraction of image pixels belonging to GT sky mask
            - `sky_present`: Whether the method's no-sky guard detected sky (bool)
    """
    J = result.J
    J_raw = result.J_raw

    full_psnr = psnr(G, J)
    full_ssim = ssim(G, J)
    de_map = ciede2000_map(G, J)
    full_de = float(np.mean(de_map, dtype=np.float64))
    sat_full = saturated_fraction(J_raw, mask=None)

    if sky_mask is not None:
        m_sky = np.asarray(sky_mask, dtype=bool)
        m_nonsky = ~m_sky
        sky_frac = float(np.mean(m_sky, dtype=np.float64))
        sky_p = masked_psnr(G, J, m_sky)
        sky_de = masked_mean(de_map, m_sky)
        nonsky_p = masked_psnr(G, J, m_nonsky)
        nonsky_de = masked_mean(de_map, m_nonsky)
        sat_sky = saturated_fraction(J_raw, mask=m_sky)
        s_l1, s_grad = seam_errors(G, J, m_sky, width=seam_width)
    else:
        sky_frac = float("nan")
        sky_p = float("nan")
        sky_de = float("nan")
        nonsky_p = float("nan")
        nonsky_de = float("nan")
        sat_sky = float("nan")
        s_l1 = float("nan")
        s_grad = float("nan")

    sky_present = bool(result.info.get("sky_present", False))

    return {
        "psnr": full_psnr,
        "ssim": full_ssim,
        "ciede2000": full_de,
        "sky_psnr": sky_p,
        "sky_ciede2000": sky_de,
        "nonsky_psnr": nonsky_p,
        "nonsky_ciede2000": nonsky_de,
        "sat_frac_full": sat_full,
        "sat_pct_full": sat_full * 100.0,
        "sat_frac_sky": sat_sky,
        "sat_pct_sky": sat_sky * 100.0 if not np.isnan(sat_sky) else float("nan"),
        "seam_l1": s_l1,
        "seam_grad": s_grad,
        "sky_fraction": sky_frac,
        "sky_present": sky_present,
    }
