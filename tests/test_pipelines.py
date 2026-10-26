"""Unit tests for sky transmission (`fsdehaze.sky_transmission`) and dehazing pipelines (`fsdehaze.pipelines`)."""

from __future__ import annotations

import numpy as np

from fsdehaze.config import load_config
from fsdehaze.pipelines import (
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)
from fsdehaze.sky_transmission import compute_t_sky, t_sky_floor, t_sky_tolerance


def _make_synthetic_hazy_sky_scene(height: int = 100, width: int = 120, seed: int = 42) -> np.ndarray:
    """Create a synthetic outdoor hazy scene with bright sky on top and textured dark ground below."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)
    half = height // 2

    # Sky region (top half): smooth bright blue-white
    y_top = np.linspace(0.0, 1.0, half, dtype=np.float32)[:, np.newaxis]
    img[:half, :, 0] = 0.84 + 0.06 * y_top
    img[:half, :, 1] = 0.89 + 0.05 * y_top
    img[:half, :, 2] = 0.95 + 0.03 * y_top

    # Ground region (bottom half): textured colourful foreground attenuated by haze
    checker = ((np.indices((height - half, width)).sum(axis=0) // 4) % 2).astype(np.float32)
    noise = rng.uniform(0.0, 0.10, size=(height - half, width, 3)).astype(np.float32)
    img[half:, :, 0] = 0.20 + 0.18 * checker + noise[:, :, 0]
    img[half:, :, 1] = 0.30 + 0.20 * (1.0 - checker) + noise[:, :, 1]
    img[half:, :, 2] = 0.15 + 0.12 * checker + noise[:, :, 2]

    return np.clip(img, 0.0, 1.0)


def test_t_sky_floor_and_tolerance() -> None:
    """t_sky_floor and t_sky_tolerance return float32 maps in [0, 1] >= t_dcp."""
    img = _make_synthetic_hazy_sky_scene(60, 80)
    t_dcp = np.full((60, 80), 0.12, dtype=np.float32)
    A = np.array([0.90, 0.92, 0.96], dtype=np.float32)

    tf = t_sky_floor(t_dcp, t_floor=0.80)
    assert tf.shape == (60, 80)
    assert tf.dtype == np.float32
    np.testing.assert_allclose(tf, 0.80, atol=1e-6)

    tt = t_sky_tolerance(img, A, t_dcp, K=0.08)
    assert tt.shape == (60, 80)
    assert tt.dtype == np.float32
    assert np.all(tt >= t_dcp)
    assert np.all(tt <= 1.0)
    # Sky region (close to A) must get higher tolerance transmission than dark ground region
    assert float(np.mean(tt[:25, :])) > float(np.mean(tt[35:, :]))

    cfg = load_config("configs/default.yaml")
    t_disp_floor = compute_t_sky(img, A, t_dcp, cfg, params={"t_sky_mode": "floor", "t_floor": 0.75})
    np.testing.assert_allclose(t_disp_floor, 0.75, atol=1e-6)


def test_dehaze_fuzzy_with_forced_zero_mu_equals_dehaze_dcp_exactly() -> None:
    """With mu = 0 forced, dehaze_fuzzy equals dehaze_dcp bit-for-bit."""
    img = _make_synthetic_hazy_sky_scene(96, 128)
    cfg = load_config("configs/default.yaml")

    res_b0 = dehaze_dcp(img, cfg)
    res_f0_zero = dehaze_fuzzy(img, cfg, params={"force_mu": 0.0})

    np.testing.assert_array_equal(res_f0_zero.A, res_b0.A)
    np.testing.assert_array_equal(res_f0_zero.t, res_b0.t)
    np.testing.assert_array_equal(res_f0_zero.J_raw, res_b0.J_raw)
    np.testing.assert_array_equal(res_f0_zero.J, res_b0.J)


def test_no_sky_image_reduces_all_methods_to_dcp_exactly() -> None:
    """When the no-sky guard triggers on a dark ground image, B1, B2, and F0 reduce to B0."""
    rng = np.random.default_rng(77)
    dark_img = rng.uniform(0.05, 0.35, size=(80, 100, 3)).astype(np.float32)
    cfg = load_config("configs/default.yaml")

    res_b0 = dehaze_dcp(dark_img, cfg)
    res_b1 = dehaze_hard_threshold(dark_img, cfg, tau=0.65)
    res_b2 = dehaze_hard_fcm(dark_img, cfg)
    res_f0 = dehaze_fuzzy(dark_img, cfg)

    assert res_b1.info["sky_present"] is False
    assert res_b2.info["sky_present"] is False
    assert res_f0.info["sky_present"] is False

    np.testing.assert_array_equal(res_b1.J, res_b0.J)
    np.testing.assert_array_equal(res_b2.J, res_b0.J)
    np.testing.assert_array_equal(res_f0.J, res_b0.J)


def test_b1_b2_f0_shapes_bounds_and_sky_transmission_lift() -> None:
    """B1, B2, and F0 preserve shapes/bounds and lift sky transmission above plain DCP on sky scenes."""
    img = _make_synthetic_hazy_sky_scene(100, 120)
    cfg = load_config("configs/default.yaml")

    res_b0 = dehaze_dcp(img, cfg)
    res_b1 = dehaze_hard_threshold(img, cfg, tau=0.60)
    res_b2 = dehaze_hard_fcm(img, cfg)
    res_f0 = dehaze_fuzzy(img, cfg)

    for res in (res_b0, res_b1, res_b2, res_f0):
        assert res.J.shape == img.shape
        assert res.J.dtype == np.float32
        assert res.t.shape == img.shape[:2]
        assert res.t.dtype == np.float32
        assert float(np.min(res.J)) >= 0.0 and float(np.max(res.J)) <= 1.0
        assert float(np.min(res.t)) >= 0.0 and float(np.max(res.t)) <= 1.0

    # Plain DCP has low transmission in the bright sky (< 0.25), whereas B1/B2/F0 lift it (> 0.65)
    sky_t_b0 = float(np.mean(res_b0.t[:35, :]))
    sky_t_b1 = float(np.mean(res_b1.t[:35, :]))
    sky_t_b2 = float(np.mean(res_b2.t[:35, :]))
    sky_t_f0 = float(np.mean(res_f0.t[:35, :]))

    assert sky_t_b0 < 0.25
    assert sky_t_b1 > 0.65
    assert sky_t_b2 > 0.65
    assert sky_t_f0 > 0.65


def test_ga_repair_and_individual_to_params() -> None:
    """repair_individual clips bounds, enforces b - a >= 0.1, and maps to valid dehaze_fuzzy params."""
    from fsdehaze.tuning.ga import GENE_NAMES, individual_to_params, repair_individual

    ga_cfg = load_config("configs/ga.yaml")
    bounds = ga_cfg["bounds"]

    # Out-of-bounds genes with inverted smoothstep (a=0.49, b=0.50) and all-zero feature weights
    bad_ind = [0.5, 0.0, 0.0, 0.0, 0.0, 0.49, 0.50, 25.4, 1.5, -0.1]
    repaired = repair_individual(bad_ind, bounds)

    for idx, name in enumerate(GENE_NAMES):
        lo, hi = bounds[name]
        assert lo <= repaired[idx] <= hi

    params = individual_to_params(repaired)
    assert params["b"] - params["a"] >= 0.10 - 1e-6
    assert isinstance(params["r_mu"], int)
    assert sum(params["feature_weights"]) >= 0.20

