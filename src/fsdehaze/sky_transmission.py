"""Sky transmission estimation (`t_sky`) via floor or tolerance mechanisms."""

from __future__ import annotations

from typing import Any

import numpy as np


def t_sky_floor(t_dcp: np.ndarray, t_floor: float = 0.8) -> np.ndarray:
    """Compute floor-clamped sky transmission map.

    Formula:
        t_sky(x) = max(t_DCP(x), t_floor)

    Stops the atmospheric scattering division `(I - A) / t` from blowing up in sky regions
    where `I` is close to `A` and `t_DCP` approaches zero.

    Args:
        t_dcp: Raw DCP transmission map of shape (H, W), dtype float32, in [0, 1].
        t_floor: Minimum transmission floor in the sky (default 0.8, typically in [0.5, 1.0]).

    Returns:
        Sky transmission map `t_sky` of shape (H, W), dtype float32, in [0, 1].
    """
    floor_val = float(np.clip(t_floor, 0.0, 1.0))
    return np.clip(np.maximum(t_dcp.astype(np.float32, copy=False), floor_val), 0.0, 1.0).astype(
        np.float32
    )


def t_sky_tolerance(
    img: np.ndarray,
    A: np.ndarray,
    t_dcp: np.ndarray,
    K: float = 0.08,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Compute tolerance-based sky transmission map.

    Formula:
        t_sky(x) = clip( max(t_DCP(x), K / (mean_c |I_c(x) - A_c| + eps)), 0, 1 )

    Where pixel intensity `I(x)` is close to atmospheric light `A` (as in bright hazy sky),
    `mean_c |I_c(x) - A_c|` is small, pushing `t_sky(x)` toward 1.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        A: Estimated atmospheric light vector of shape (3,), dtype float32.
        t_dcp: Raw DCP transmission map of shape (H, W), dtype float32, in [0, 1].
        K: Tolerance scaling parameter (default 0.08, typically in [0.02, 0.30]).
        eps: Small constant to prevent division by zero (default 1e-6).

    Returns:
        Sky transmission map `t_sky` of shape (H, W), dtype float32, in [0, 1].
    """
    a_vec = np.asarray(A, dtype=np.float32).reshape(1, 1, 3)
    mean_diff = np.mean(np.abs(img.astype(np.float32, copy=False) - a_vec), axis=2)
    tol_term = float(K) / (mean_diff + float(eps))
    combined = np.maximum(t_dcp.astype(np.float32, copy=False), tol_term)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def compute_t_sky(
    img: np.ndarray,
    A: np.ndarray,
    t_dcp: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Dispatch `t_sky` computation (`floor` or `tolerance`) from configuration and overrides.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        A: Atmospheric light vector of shape (3,), dtype float32.
        t_dcp: Raw DCP transmission map of shape (H, W), dtype float32.
        cfg: Base configuration dictionary (`cfg['sky']`).
        params: Optional dictionary overriding `t_sky_mode`, `t_floor`, or `K`.

    Returns:
        Sky transmission map `t_sky` of shape (H, W), dtype float32, in [0, 1].
    """
    sky_cfg = cfg.get("sky", {})
    p = params or {}

    mode = str(p.get("t_sky_mode", sky_cfg.get("t_sky_mode", "floor"))).lower()
    if mode == "floor":
        t_floor = float(p.get("t_floor", sky_cfg.get("t_floor", 0.8)))
        return t_sky_floor(t_dcp, t_floor=t_floor)
    if mode == "tolerance":
        k_val = float(p.get("K", p.get("tolerance_K", sky_cfg.get("tolerance_K", 0.08))))
        return t_sky_tolerance(img, A, t_dcp, K=k_val)

    raise ValueError(f"Unknown t_sky_mode {mode!r}; expected 'floor' or 'tolerance'.")
