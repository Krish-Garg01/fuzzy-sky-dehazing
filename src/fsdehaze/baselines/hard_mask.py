"""Hard binary sky mask baselines (B1 hard threshold on sky score, B2 hardened FCM mask)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fsdehaze.sky_features import compute_features
from fsdehaze.sky_membership import estimate_sky_membership


def compute_sky_score_map(
    img: np.ndarray,
    cfg: dict[str, Any],
    precomputed_features: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute a normalised per-pixel sky score in [0, 1] from the 4 sky features.

    Formula:
        sky_score(x) = (1 / F) * sum_{j=1}^F s_j(x)
    where `s_j(x) = 1 - f_j(x)` for `texture` and `f_j(x)` for all other features
    (`brightness`, `low_saturation`, `vertical_pos`).

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        precomputed_features: Optional `(features, names)` tuple to avoid recomputing features.

    Returns:
        Tuple `(score_map, names)` where `score_map` has shape (H, W), dtype float32, in [0, 1].
    """
    if precomputed_features is not None:
        feat_map, names = precomputed_features
    else:
        feat_map, names = compute_features(img, cfg)

    H, W, F = feat_map.shape
    score = np.zeros((H, W), dtype=np.float32)
    for j, name in enumerate(names):
        ch = feat_map[:, :, j]
        if name == "texture":
            score += 1.0 - ch
        else:
            score += ch
    score /= float(max(F, 1))
    return np.clip(score, 0.0, 1.0).astype(np.float32), names


def estimate_hard_threshold_mask(
    img: np.ndarray,
    cfg: dict[str, Any],
    tau: float = 0.65,
    precomputed_features: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate binary sky mask M = 1[sky_score > tau] with the same no-sky guard as FCM (Method B1).

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        tau: Threshold in [0, 1] applied to `sky_score(x)` (tuned via grid search on `ots_subset`).
        precomputed_features: Optional `(features, names)` tuple.

    Returns:
        Tuple `(mask, info)` where `mask` is a boolean array of shape (H, W) and `info` stores metadata.
    """
    H, W = img.shape[:2]
    score_map, _ = compute_sky_score_map(img, cfg, precomputed_features=precomputed_features)
    raw_mask = score_map > float(tau)

    guard_cfg = cfg.get("sky", {}).get("guard", {})
    min_brightness = float(guard_cfg.get("min_brightness", 0.5))
    min_area_frac = float(guard_cfg.get("min_area_frac", 0.01))
    max_mean_y = float(guard_cfg.get("max_mean_y", 0.6))

    area_frac = float(np.mean(raw_mask))
    raw_v = np.max(img, axis=2)
    mean_brightness = float(np.mean(raw_v[raw_mask])) if area_frac > 0.0 else 0.0
    y_grid = np.broadcast_to(
        np.linspace(0.0, 1.0, H, dtype=np.float32)[:, np.newaxis],
        (H, W),
    )
    mean_y = float(np.mean(y_grid[raw_mask])) if area_frac > 0.0 else 1.0

    guard_triggered = (
        mean_brightness < min_brightness
        or area_frac < min_area_frac
        or mean_y > max_mean_y
    )

    if guard_triggered:
        mask = np.zeros((H, W), dtype=bool)
        sky_present = False
    else:
        mask = raw_mask
        sky_present = True

    info: dict[str, Any] = {
        "sky_present": sky_present,
        "tau": float(tau),
        "sky_score": score_map,
        "area_frac": area_frac,
        "cluster_brightness": mean_brightness,
        "mean_y": mean_y,
    }
    return mask, info


def estimate_hard_fcm_mask(
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    threshold: float = 0.5,
    precomputed_features: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate hardened FCM binary sky mask M = 1[mu_raw > 0.5] (Method B2).

    Uses the exact same Fuzzy C-Means clustering and no-sky guard as `estimate_sky_membership`,
    replacing the smoothstep + guided-filter soft ramp with a crisp binary step `mu_raw > 0.5`.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        params: Optional parameter overrides.
        threshold: Hard cutoff on `mu_raw` (default 0.5).
        precomputed_features: Optional `(features, names)` tuple.

    Returns:
        Tuple `(mask, info)` where `mask` is a boolean array of shape (H, W).
    """
    H, W = img.shape[:2]
    _, info = estimate_sky_membership(
        img,
        cfg,
        params=params,
        precomputed_features=precomputed_features,
    )
    if not info["sky_present"]:
        mask = np.zeros((H, W), dtype=bool)
    else:
        mask = np.asarray(info["mu_raw"] > float(threshold), dtype=bool)
    return mask, info
