"""End-to-end dehazing pipelines (B0, B1, B2, F0/F1).

All methods share the exact same atmospheric-light estimator, DCP parameters,
and final guided-filter settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.guided_filter import guided_filter, rgb_to_gray


@dataclass
class Result:
    """Standardised output container for all dehazing pipelines.

    Attributes:
        J: Clipped dehazed RGB image, shape (H, W, 3), float32 in [0, 1].
        J_raw: Unclipped dehazed RGB image, shape (H, W, 3), float32 (for saturation metric).
        t: Final refined transmission map, shape (H, W), float32 in [0, 1].
        A: Estimated atmospheric light vector, shape (3,), float32 in [0, 1].
        mu: Soft sky membership map, shape (H, W), float32 in [0, 1], or None.
        mask: Hard binary sky mask, shape (H, W), bool/float32, or None.
        info: Dictionary of intermediate maps and metadata (e.g., t_dcp, dark, sky_present).
    """

    J: np.ndarray
    J_raw: np.ndarray
    t: np.ndarray
    A: np.ndarray
    mu: np.ndarray | None = None
    mask: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


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
            "sky_present": False,
        },
    )


def dehaze_hard_threshold(
    img: np.ndarray,
    cfg: dict[str, Any],
    tau: float = 0.5,
) -> Result:
    """Run DCP + hard-thresholded sky mask baseline (Method B1, Phase 4)."""
    raise NotImplementedError("dehaze_hard_threshold will be implemented in Phase 4.")


def dehaze_hard_fcm(img: np.ndarray, cfg: dict[str, Any]) -> Result:
    """Run DCP + hardened FCM sky mask baseline (Method B2, Phase 4)."""
    raise NotImplementedError("dehaze_hard_fcm will be implemented in Phase 4.")


def dehaze_fuzzy(
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Result:
    """Run proposed Fuzzy Sky-Aware Dehazing pipeline (Method F0/F1, Phase 4)."""
    raise NotImplementedError("dehaze_fuzzy will be implemented in Phase 4.")
