"""Per-pixel sky feature extraction and robust percentile normalisation.

Features supported:
- f1 (`brightness`): V = max_c I_c
- f2 (`low_saturation`): 1 - S (where S is HSV saturation)
- f3 (`texture`): Gaussian blur (sigma=1.5) -> Sobel magnitude -> 15x15 local mean
- f4 (`vertical_pos`): 1 - y / H (1 at top of image, 0 at bottom)
- f5 (`dark_channel`, optional): normalised 15x15 dark channel of I
- f6 (`blueness`, optional): B - R channel difference
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from fsdehaze.dcp import dark_channel
from fsdehaze.guided_filter import rgb_to_gray

DEFAULT_FEATURES: tuple[str, ...] = (
    "brightness",
    "low_saturation",
    "texture",
    "vertical_pos",
)


def percentile_normalise(
    arr: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Normalise a 2D array to [0, 1] using 1st/99th percentile min-max clipping.

    If the dynamic range `p_high - p_low` is near zero (< 1e-6, e.g., constant image),
    falls back to clipping `arr` directly to [0, 1].

    Args:
        arr: 2D float array of shape (H, W).
        p_low: Lower percentile (default 1.0).
        p_high: Upper percentile (default 99.0).

    Returns:
        Normalised 2D array of shape (H, W), dtype float32, values in [0, 1].
    """
    f = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(f, [p_low, p_high])
    span = float(hi - lo)
    if span < 1e-6:
        return np.clip(f, 0.0, 1.0).astype(np.float32)
    normed = (f - float(lo)) / span
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def _compute_single_feature(img: np.ndarray, name: str, dcp_patch: int = 15) -> np.ndarray:
    """Compute a single normalised 2D feature map of shape (H, W) in [0, 1]."""
    H, W = img.shape[:2]

    if name == "brightness":
        v = np.max(img, axis=2)
        return percentile_normalise(v)

    if name == "low_saturation":
        v = np.max(img, axis=2)
        min_c = np.min(img, axis=2)
        sat = (v - min_c) / np.maximum(v, 1e-6)
        low_sat = 1.0 - sat
        return percentile_normalise(low_sat)

    if name == "texture":
        gray = rgb_to_gray(img)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)
        mean_grad = cv2.boxFilter(
            grad_mag,
            ddepth=-1,
            ksize=(15, 15),
            normalize=True,
            borderType=cv2.BORDER_DEFAULT,
        )
        return percentile_normalise(mean_grad)

    if name == "vertical_pos":
        # 1.0 at the top row (y=0), 0.0 at the bottom row (y=H-1)
        y_norm = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, np.newaxis]
        return np.broadcast_to(y_norm, (H, W)).copy()

    if name == "dark_channel":
        dc = dark_channel(img, patch=dcp_patch)
        return percentile_normalise(dc)

    if name == "blueness":
        # img is RGB: channel 2 is Blue, channel 0 is Red
        blue_minus_red = img[:, :, 2] - img[:, :, 0]
        return percentile_normalise(blue_minus_red)

    raise ValueError(
        f"Unknown sky feature {name!r}. Supported: "
        "'brightness', 'low_saturation', 'texture', 'vertical_pos', 'dark_channel', 'blueness'."
    )


def compute_features(
    img: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-pixel sky feature tensor and feature names.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Configuration dictionary (reads `cfg['sky']['features']` and `cfg['dcp']['patch']`).

    Returns:
        Tuple `(features, names)` where:
            - `features` has shape `(H, W, F)`, dtype float32, with each channel in [0, 1].
            - `names` is the list of `F` feature names in channel order.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {img.shape}")

    sky_cfg = (cfg or {}).get("sky", {})
    dcp_cfg = (cfg or {}).get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))

    feature_names = [str(n) for n in sky_cfg.get("features", DEFAULT_FEATURES)]
    if not feature_names:
        raise ValueError("Feature list in cfg['sky']['features'] cannot be empty.")

    maps = [_compute_single_feature(img, name, dcp_patch=patch) for name in feature_names]
    features = np.stack(maps, axis=-1).astype(np.float32)
    return features, feature_names
