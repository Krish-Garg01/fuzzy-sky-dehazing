"""End-to-end dehazing pipelines (B0, B1, B2, F0/F1).

All methods share the exact same atmospheric-light estimator, DCP parameters,
and final guided-filter settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fsdehaze.baselines.hard_mask import (
    estimate_hard_fcm_mask,
    estimate_hard_threshold_mask,
)
from fsdehaze.config import load_config
from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.guided_filter import guided_filter, rgb_to_gray
from fsdehaze.sky_membership import estimate_sky_membership
from fsdehaze.sky_transmission import compute_t_sky


@dataclass
class Result:
    """Standardised output container for all dehazing pipelines.

    Attributes:
        J: Clipped dehazed RGB image, shape (H, W, 3), float32 in [0, 1].
        J_raw: Unclipped dehazed RGB image, shape (H, W, 3), float32 (for saturation metric).
        t: Final refined transmission map, shape (H, W), float32 in [0, 1].
        A: Estimated atmospheric light vector, shape (3,), float32 in [0, 1].
        mu: Soft sky membership map, shape (H, W), float32 in [0, 1], or None.
        mask: Hard binary sky mask, shape (H, W), bool, or None.
        info: Dictionary of intermediate maps and metadata (e.g., t_dcp, t_blend, dark, sky_present).
    """

    J: np.ndarray
    J_raw: np.ndarray
    t: np.ndarray
    A: np.ndarray
    mu: np.ndarray | None = None
    mask: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


def _blend_and_refine_transmission(
    img: np.ndarray,
    A: np.ndarray,
    t_dcp: np.ndarray,
    weight_map: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend `t_sky` and `t_dcp` using `weight_map` in [0, 1] and apply guided filter refinement.

    Supports `gf_order='after_blend'` (default) and `gf_order='before_blend'` (Ablation 10).

    Returns:
        Tuple `(t_final, t_blend, t_sky)` each of shape (H, W), dtype float32.
    """
    gf_cfg = cfg.get("guided_filter", {})
    gf_r = int(gf_cfg.get("r", 60))
    gf_eps = float(gf_cfg.get("eps", 1.0e-4))

    sky_cfg = cfg.get("sky", {})
    p = params or {}
    gf_order = str(p.get("gf_order", sky_cfg.get("gf_order", "after_blend"))).lower()

    gray = rgb_to_gray(img)
    w = weight_map.astype(np.float32, copy=False)

    if gf_order == "before_blend":
        t_dcp_ref = np.clip(guided_filter(gray, t_dcp, r=gf_r, eps=gf_eps), 0.0, 1.0).astype(
            np.float32
        )
        t_sky = compute_t_sky(img, A, t_dcp_ref, cfg, params=params)
        t_blend = (w * t_sky + (1.0 - w) * t_dcp_ref).astype(np.float32)
        t_final = np.clip(t_blend, 0.0, 1.0).astype(np.float32)
        return t_final, t_blend, t_sky

    # Default: blend first, then refine with shared guided filter (r=60, eps=1e-4)
    t_sky = compute_t_sky(img, A, t_dcp, cfg, params=params)
    t_blend = (w * t_sky + (1.0 - w) * t_dcp).astype(np.float32)
    t_refined = guided_filter(gray, t_blend, r=gf_r, eps=gf_eps)
    t_final = np.clip(t_refined, 0.0, 1.0).astype(np.float32)
    return t_final, t_blend, t_sky


def dehaze_dcp(img: np.ndarray, cfg: dict[str, Any]) -> Result:
    """Run the plain Dark Channel Prior baseline (Method B0).

    Pipeline:
        1. dark = dark_channel(img, patch)
        2. A = estimate_atmospheric_light(img, dark, top_frac)
        3. t_dcp = transmission_dcp(img, A, omega, patch)
        4. t = guided_filter(rgb_to_gray(img), t_dcp, r, eps)
        5. J, J_raw = recover(img, A, t, t0)

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Configuration dictionary containing `dcp` and `guided_filter` sections.

    Returns:
        Result containing dehazed image `J`, unclipped `J_raw`, refined `t`, `A`, and debug info.
    """
    dcp_cfg = cfg.get("dcp", {})
    gf_cfg = cfg.get("guided_filter", {})

    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    t0 = float(dcp_cfg.get("t0", 0.1))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    gf_r = int(gf_cfg.get("r", 60))
    gf_eps = float(gf_cfg.get("eps", 1.0e-4))

    dark = dark_channel(img, patch=patch)
    A = estimate_atmospheric_light(img, dark, top_frac=top_frac)
    t_dcp = transmission_dcp(img, A, omega=omega, patch=patch)

    gray = rgb_to_gray(img)
    t_refined = guided_filter(gray, t_dcp, r=gf_r, eps=gf_eps)
    t_final = np.clip(t_refined, 0.0, 1.0).astype(np.float32)

    J, J_raw = recover(img, A, t_final, t0=t0)

    return Result(
        J=J,
        J_raw=J_raw,
        t=t_final,
        A=A,
        mu=None,
        mask=None,
        info={
            "method": "B0",
            "dark": dark,
            "t_dcp": t_dcp,
            "t_blend": t_dcp,
            "sky_present": False,
        },
    )


def _resolve_default_tau(cfg: dict[str, Any], tau: float | None) -> float:
    """Resolve hard threshold `tau` from explicit argument, `results/best_tau.yaml`, or config."""
    if tau is not None:
        return float(tau)
    sky_cfg = cfg.get("sky", {})
    if "hard_tau" in sky_cfg:
        return float(sky_cfg["hard_tau"])
    best_tau_path = Path("results/best_tau.yaml")
    if best_tau_path.is_file():
        try:
            loaded = load_config(best_tau_path)
            if "tau" in loaded:
                return float(loaded["tau"])
        except Exception:
            pass
    return 0.65


def dehaze_hard_threshold(
    img: np.ndarray,
    cfg: dict[str, Any],
    tau: float | None = None,
    params: dict[str, Any] | None = None,
) -> Result:
    """Run DCP + hard-thresholded sky mask baseline (Method B1).

    Computes `M = 1[sky_score > tau]` from the 4 normalised sky features, blends
    `t_blend = M * t_sky + (1 - M) * t_DCP`, refines with the shared guided filter, and recovers `J`.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        tau: Threshold in [0, 1] on `sky_score` (if None, loads `results/best_tau.yaml` or 0.65).
        params: Optional parameter overrides for `t_sky`.

    Returns:
        Result with `mask=M`, `mu=None`, dehazed `J`, `J_raw`, `t`, `A`, and `info`.
    """
    dcp_cfg = cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    t0 = float(dcp_cfg.get("t0", 0.1))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    resolved_tau = _resolve_default_tau(cfg, tau)

    dark = dark_channel(img, patch=patch)
    A = estimate_atmospheric_light(img, dark, top_frac=top_frac)
    t_dcp = transmission_dcp(img, A, omega=omega, patch=patch)

    mask, mask_info = estimate_hard_threshold_mask(img, cfg, tau=resolved_tau)
    mask_f = mask.astype(np.float32)

    t_final, t_blend, t_sky = _blend_and_refine_transmission(
        img, A, t_dcp, mask_f, cfg, params=params
    )
    J, J_raw = recover(img, A, t_final, t0=t0)

    info = {
        **mask_info,
        "method": "B1",
        "dark": dark,
        "t_dcp": t_dcp,
        "t_sky": t_sky,
        "t_blend": t_blend,
    }
    return Result(
        J=J,
        J_raw=J_raw,
        t=t_final,
        A=A,
        mu=None,
        mask=mask,
        info=info,
    )


def dehaze_hard_fcm(
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Result:
    """Run DCP + hardened FCM sky mask baseline (Method B2).

    Uses the exact same Fuzzy C-Means clustering and no-sky guard as `dehaze_fuzzy`,
    but replaces the soft membership `mu` with the binary step mask `M = 1[mu_raw > 0.5]`.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        params: Optional parameter overrides.

    Returns:
        Result with `mask=M`, `mu=None`, dehazed `J`, `J_raw`, `t`, `A`, and `info`.
    """
    dcp_cfg = cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    t0 = float(dcp_cfg.get("t0", 0.1))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    dark = dark_channel(img, patch=patch)
    A = estimate_atmospheric_light(img, dark, top_frac=top_frac)
    t_dcp = transmission_dcp(img, A, omega=omega, patch=patch)

    mask, fcm_info = estimate_hard_fcm_mask(img, cfg, params=params, threshold=0.5)
    mask_f = mask.astype(np.float32)

    t_final, t_blend, t_sky = _blend_and_refine_transmission(
        img, A, t_dcp, mask_f, cfg, params=params
    )
    J, J_raw = recover(img, A, t_final, t0=t0)

    info = {
        **fcm_info,
        "method": "B2",
        "dark": dark,
        "t_dcp": t_dcp,
        "t_sky": t_sky,
        "t_blend": t_blend,
    }
    return Result(
        J=J,
        J_raw=J_raw,
        t=t_final,
        A=A,
        mu=None,
        mask=mask,
        info=info,
    )


def dehaze_fuzzy(
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    precomputed: dict[str, Any] | None = None,
) -> Result:
    """Run the proposed Fuzzy Sky-Aware Dehazing pipeline (Method F0 with defaults, F1 with GA params).

    Pipeline:
        1. dark = dark_channel(img, patch)
        2. A = estimate_atmospheric_light(img, dark, top_frac)
        3. t_dcp = transmission_dcp(img, A, omega, patch)
        4. mu, info = estimate_sky_membership(img, cfg, params)
        5. t_sky = compute_t_sky(img, A, t_dcp, cfg, params)
        6. t_blend = mu * t_sky + (1 - mu) * t_dcp
        7. t = guided_filter(rgb_to_gray(img), t_blend, gf_r, gf_eps)
        8. J, J_raw = recover(img, A, t, t0)

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        params: Optional dictionary of GA-tuned or ablation parameters.
            If `params` contains `"force_mu"`, that array/scalar is used as `mu(x)`.
        precomputed: Optional cache dictionary containing precomputed `"dark"`, `"A"`,
            `"t_dcp"`, and/or `"features"` `(feat_map, feat_names)` for fast GA tuning.

    Returns:
        Result containing dehazed `J`, unclipped `J_raw`, refined `t`, `A`, soft `mu`, and `info`.
    """
    dcp_cfg = cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    t0 = float(dcp_cfg.get("t0", 0.1))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    cache = precomputed or {}
    dark = cache.get("dark")
    if dark is None:
        dark = dark_channel(img, patch=patch)

    A = cache.get("A")
    if A is None:
        A = estimate_atmospheric_light(img, dark, top_frac=top_frac)

    t_dcp = cache.get("t_dcp")
    if t_dcp is None:
        t_dcp = transmission_dcp(img, A, omega=omega, patch=patch)

    p = params or {}
    if "force_mu" in p and p["force_mu"] is not None:
        forced = p["force_mu"]
        if np.ndim(forced) == 0:
            mu = np.full(img.shape[:2], float(forced), dtype=np.float32)
        else:
            mu = np.asarray(forced, dtype=np.float32)
        mem_info: dict[str, Any] = {
            "sky_present": bool(np.any(mu > 0.0)),
            "mu_raw": mu,
        }
    else:
        mu, mem_info = estimate_sky_membership(
            img,
            cfg,
            params=params,
            precomputed_features=cache.get("features"),
        )

    t_final, t_blend, t_sky = _blend_and_refine_transmission(
        img, A, t_dcp, mu, cfg, params=params
    )
    J, J_raw = recover(img, A, t_final, t0=t0)

    info = {
        **mem_info,
        "method": "F1" if params else "F0",
        "dark": dark,
        "t_dcp": t_dcp,
        "t_sky": t_sky,
        "t_blend": t_blend,
    }
    return Result(
        J=J,
        J_raw=J_raw,
        t=t_final,
        A=A,
        mu=mu,
        mask=None,
        info=info,
    )
