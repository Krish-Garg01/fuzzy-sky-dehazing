"""Unit tests for the Guided Filter implementation."""

from __future__ import annotations

import cv2
import numpy as np

from fsdehaze.guided_filter import guided_filter, rgb_to_gray


def test_guided_filter_constant_input_returns_constant() -> None:
    """Filtering a constant source map must return the same constant everywhere."""
    rng = np.random.default_rng(42)
    guide = rng.uniform(0.0, 1.0, size=(64, 80)).astype(np.float32)
    const_val = 0.37
    src = np.full((64, 80), const_val, dtype=np.float32)

    out = guided_filter(guide, src, r=8, eps=1e-4)
    assert out.shape == (64, 80)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, const_val, atol=1e-5)


def test_guided_filter_step_edge_sharper_than_box_blur() -> None:
    """Guided filter aligned with a step-edge guide must preserve the edge sharper than a box blur."""
    H, W = 80, 100
    step = np.zeros((H, W), dtype=np.float32)
    step[:, W // 2 :] = 1.0

    rng = np.random.default_rng(42)
    noisy_src = np.clip(step + rng.normal(0.0, 0.05, size=(H, W)), 0.0, 1.0).astype(np.float32)

    r = 10
    gf_out = guided_filter(step, noisy_src, r=r, eps=1e-4)
    box_out = cv2.boxFilter(noisy_src, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)

    # Measure edge gradient magnitude right across the vertical boundary (x = W//2 - 1 to W//2)
    gf_jump = float(np.mean(np.abs(gf_out[:, W // 2] - gf_out[:, W // 2 - 1])))
    box_jump = float(np.mean(np.abs(box_out[:, W // 2] - box_out[:, W // 2 - 1])))

    assert gf_jump > 0.85, f"Guided filter blurred the step edge too much: jump={gf_jump:.4f}"
    assert gf_jump > 4.0 * box_jump, (
        f"Guided filter jump ({gf_jump:.4f}) should be much sharper than box blur ({box_jump:.4f})"
    )


def test_guided_filter_accepts_rgb_guide() -> None:
    """Passing a 3-channel RGB guide should match passing its grayscale conversion."""
    rng = np.random.default_rng(123)
    rgb = rng.uniform(0.0, 1.0, size=(48, 64, 3)).astype(np.float32)
    src = rng.uniform(0.0, 1.0, size=(48, 64)).astype(np.float32)

    out_rgb = guided_filter(rgb, src, r=5, eps=1e-3)
    out_gray = guided_filter(rgb_to_gray(rgb), src, r=5, eps=1e-3)
    np.testing.assert_allclose(out_rgb, out_gray, atol=1e-6)


def test_guided_filter_crosscheck_with_opencv_ximgproc_if_available() -> None:
    """Cross-check against cv2.ximgproc.guidedFilter when opencv-contrib is installed."""
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        return

    rng = np.random.default_rng(99)
    guide = rng.uniform(0.0, 1.0, size=(60, 70)).astype(np.float32)
    src = rng.uniform(0.0, 1.0, size=(60, 70)).astype(np.float32)

    ours = guided_filter(guide, src, r=7, eps=1e-3)
    ref = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=7, eps=1e-3)
    np.testing.assert_allclose(ours, ref, atol=1e-4)
