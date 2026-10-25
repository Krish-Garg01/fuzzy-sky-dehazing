"""Unit tests for sky feature extraction and soft sky membership (`fsdehaze.sky_membership`)."""

from __future__ import annotations

import numpy as np

from fsdehaze.config import load_config
from fsdehaze.sky_features import compute_features, percentile_normalise
from fsdehaze.sky_membership import estimate_sky_membership, smoothstep


def _make_synthetic_sky_ground_image(height: int = 120, width: int = 160, seed: int = 42) -> np.ndarray:
    """Create a synthetic image with a smooth bright blue-white sky on top and textured dark ground below."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)
    half = height // 2

    # Top half: smooth bright blue-white sky (high brightness, low saturation, low texture, high vertical_pos)
    y_top = np.linspace(0.0, 1.0, half, dtype=np.float32)[:, np.newaxis]
    img[:half, :, 0] = 0.82 + 0.08 * y_top  # R
    img[:half, :, 1] = 0.88 + 0.06 * y_top  # G
    img[:half, :, 2] = 0.95 + 0.03 * y_top  # B

    # Bottom half: dark textured colorful ground (low brightness, higher saturation, high texture, low vertical_pos)
    checker = ((np.indices((height - half, width)).sum(axis=0) // 3) % 2).astype(np.float32)
    noise = rng.uniform(0.0, 0.15, size=(height - half, width, 3)).astype(np.float32)
    img[half:, :, 0] = 0.12 + 0.18 * checker + noise[:, :, 0]
    img[half:, :, 1] = 0.25 + 0.22 * (1.0 - checker) + noise[:, :, 1]
    img[half:, :, 2] = 0.08 + 0.10 * checker + noise[:, :, 2]

    return np.clip(img, 0.0, 1.0)


def test_compute_features_shapes_and_bounds() -> None:
    """compute_features returns (H, W, F) float32 tensor in [0, 1] for all 6 supported features."""
    img = _make_synthetic_sky_ground_image(80, 100)
    cfg = {
        "dcp": {"patch": 15},
        "sky": {
            "features": [
                "brightness",
                "low_saturation",
                "texture",
                "vertical_pos",
                "dark_channel",
                "blueness",
            ]
        },
    }
    feats, names = compute_features(img, cfg)
    assert feats.shape == (80, 100, 6)
    assert feats.dtype == np.float32
    assert len(names) == 6
    assert float(np.min(feats)) >= 0.0
    assert float(np.max(feats)) <= 1.0


def test_percentile_normalise_constant_array() -> None:
    """Constant arrays do not produce NaN or Inf during percentile normalisation."""
    const = np.full((32, 32), 0.4, dtype=np.float32)
    normed = percentile_normalise(const)
    assert normed.shape == (32, 32)
    assert np.all(np.isfinite(normed))
    np.testing.assert_allclose(normed, 0.4, atol=1e-6)


def test_smoothstep_endpoints_and_monotonicity() -> None:
    """smoothstep maps <= a to 0, >= b to 1, midpoint to 0.5, and is monotonically non-decreasing."""
    x = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    y = smoothstep(x, a=0.2, b=0.8)
    assert y.dtype == np.float32
    assert np.all(y[x <= 0.2] == 0.0)
    assert np.all(y[x >= 0.8] == 1.0)
    np.testing.assert_allclose(smoothstep(np.array([0.5], dtype=np.float32), a=0.2, b=0.8), [0.5], atol=1e-6)
    assert np.all(np.diff(y) >= -1e-7)


def test_membership_high_on_sky_low_on_ground() -> None:
    """Synthetic sky-over-ground image gives high mu on top and low mu on bottom."""
    img = _make_synthetic_sky_ground_image(120, 160)
    cfg = load_config("configs/default.yaml")
    mu, info = estimate_sky_membership(img, cfg)

    assert mu.shape == (120, 160)
    assert mu.dtype == np.float32
    assert info["sky_present"] is True
    assert float(np.min(mu)) >= 0.0
    assert float(np.max(mu)) <= 1.0

    # Avoid the immediate horizon band (rows 50..70) when checking pure sky vs pure ground means
    mean_top = float(np.mean(mu[:45, :]))
    mean_bottom = float(np.mean(mu[75:, :]))
    assert mean_top > 0.80, f"Expected high sky membership in top region, got {mean_top:.3f}"
    assert mean_bottom < 0.15, f"Expected low sky membership in bottom region, got {mean_bottom:.3f}"


def test_no_sky_guard_dark_and_bottom_bright_images() -> None:
    """Images without sky (dark textured image or bright object at bottom) trigger the no-sky guard."""
    rng = np.random.default_rng(42)
    cfg = load_config("configs/default.yaml")

    # Case 1: Dark textured foliage/ground scene (brightness < 0.5 everywhere)
    dark_img = rng.uniform(0.05, 0.35, size=(100, 120, 3)).astype(np.float32)
    mu_dark, info_dark = estimate_sky_membership(dark_img, cfg)
    assert info_dark["sky_present"] is False
    assert np.all(mu_dark == 0.0)

    # Case 2: Bright smooth patch only at the very bottom of the image (mean_y > 0.6)
    bottom_bright = rng.uniform(0.08, 0.25, size=(100, 120, 3)).astype(np.float32)
    bottom_bright[80:, :, :] = 0.92
    mu_bot, info_bot = estimate_sky_membership(bottom_bright, cfg)
    assert info_bot["sky_present"] is False
    assert np.all(mu_bot == 0.0)
