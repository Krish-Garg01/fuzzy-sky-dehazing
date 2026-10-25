"""Unit tests for io_utils, DCP primitives, and the B0 dehaze_dcp pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fsdehaze.config import load_config
from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.io_utils import load_image, load_mask, pair_dataset, save_image
from fsdehaze.pipelines import dehaze_dcp


def test_dark_channel_all_white_is_one() -> None:
    """Dark channel of an all-white image must be 1 everywhere."""
    white = np.ones((48, 64, 3), dtype=np.float32)
    dark = dark_channel(white, patch=15)
    assert dark.shape == (48, 64)
    assert dark.dtype == np.float32
    np.testing.assert_allclose(dark, 1.0, atol=1e-6)


def test_dark_channel_zero_channel_gives_zero() -> None:
    """If every pixel has at least one zero channel, dark channel is 0 everywhere."""
    img = np.ones((40, 50, 3), dtype=np.float32)
    img[:, :, 2] = 0.0  # Blue channel is 0 everywhere
    dark = dark_channel(img, patch=15)
    np.testing.assert_allclose(dark, 0.0, atol=1e-6)


def test_synthetic_haze_atmospheric_light_and_recovery() -> None:
    """Synthesise I = J*t + A*(1-t) with known J, constant t=0.6, A=0.9 and verify A & recovery."""
    rng = np.random.default_rng(42)
    H, W = 128, 128

    # Construct a smooth haze-free image J where every 15x15 patch has at least one dark channel
    yy, xx = np.meshgrid(
        np.linspace(0.1, 0.7, H, dtype=np.float32),
        np.linspace(0.1, 0.7, W, dtype=np.float32),
        indexing="ij",
    )
    J_true = np.stack([yy, xx, 0.5 * (yy + xx)], axis=-1).astype(np.float32)
    # Zero out one channel on a regular grid (spacing 5 px < patch 15 px) so DCP holds strictly
    J_true[::5, ::5, 0] = 0.0
    J_true[2::5, 2::5, 1] = 0.0
    J_true[4::5, 4::5, 2] = 0.0

    # Place a small bright sky/haze patch where J ~= A so top 0.1% dark channel finds A
    A_true = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    t_true_scalar = 0.6
    t_map = np.full((H, W), t_true_scalar, dtype=np.float32)
    # Top-left 20x20 corner is pure atmospheric veil (t -> 0)
    t_map[:20, :20] = 0.0

    I_hazy = (
        J_true * t_map[:, :, np.newaxis]
        + A_true.reshape(1, 1, 3) * (1.0 - t_map[:, :, np.newaxis])
    ).astype(np.float32)

    dark = dark_channel(I_hazy, patch=15)
    A_est = estimate_atmospheric_light(I_hazy, dark, top_frac=0.001)

    assert A_est.shape == (3,)
    assert A_est.dtype == np.float32
    np.testing.assert_allclose(A_est, A_true, atol=0.02)

    # Estimate transmission with omega=1.0 (exact ASM without 0.95 aerial perspective retention)
    t_est = transmission_dcp(I_hazy, A_est, omega=1.0, patch=15)
    # In interior non-sky region (away from the 20x20 sky corner), t_est should be close to 0.6
    non_sky_t = t_est[35:, 35:]
    np.testing.assert_allclose(non_sky_t, t_true_scalar, atol=0.02)

    # Recover J and check recovery error in non-sky region
    J_rec, J_raw = recover(I_hazy, A_est, t_est, t0=0.1)
    assert J_rec.shape == (H, W, 3)
    assert J_raw.shape == (H, W, 3)
    mae_non_sky = float(np.mean(np.abs(J_rec[35:, 35:] - J_true[35:, 35:])))
    assert mae_non_sky < 0.02, f"Expected non-sky recovery MAE < 0.02, got {mae_non_sky:.4f}"


def test_dehaze_dcp_pipeline_end_to_end() -> None:
    """Verify dehaze_dcp (B0) runs end-to-end and returns valid shapes/ranges."""
    cfg = load_config("configs/default.yaml")
    H, W = 96, 120
    J_true = np.zeros((H, W, 3), dtype=np.float32)
    J_true[:, :, 0] = 0.2
    J_true[:, :, 1] = 0.5
    J_true[::4, ::4, 2] = 0.0
    A_true = np.array([0.85, 0.88, 0.92], dtype=np.float32)
    t_true = np.full((H, W, 1), 0.55, dtype=np.float32)
    t_true[:18, :] = 0.02  # Sky band at top

    I_hazy = (J_true * t_true + A_true.reshape(1, 1, 3) * (1.0 - t_true)).astype(np.float32)
    res = dehaze_dcp(I_hazy, cfg)

    assert res.J.shape == (H, W, 3)
    assert res.J.dtype == np.float32
    assert float(res.J.min()) >= 0.0
    assert float(res.J.max()) <= 1.0
    assert res.t.shape == (H, W)
    assert res.A.shape == (3,)
    assert res.mu is None
    assert res.mask is None
    assert "t_dcp" in res.info
    assert "dark" in res.info


def test_io_utils_load_save_and_pairing(tmp_path: Path) -> None:
    """Verify RGB conversion, grayscale/RGBA handling, mask loading, and strict pairing."""
    # 1. RGB round-trip
    rgb = np.zeros((16, 20, 3), dtype=np.float32)
    rgb[:, :, 0] = 1.0  # Pure red in RGB
    rgb[:, :, 1] = 0.5
    rgb[:, :, 2] = 0.0
    out_file = tmp_path / "red.png"
    save_image(out_file, rgb)
    loaded = load_image(out_file)
    assert loaded.shape == (16, 20, 3)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded[:, :, 0], 1.0, atol=1.0 / 255.0)
    np.testing.assert_allclose(loaded[:, :, 2], 0.0, atol=1.0 / 255.0)

    # 2. Grayscale and BGRA handling
    gray_u8 = np.full((12, 14), 128, dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    cv2.imwrite(str(gray_path), gray_u8)
    loaded_gray = load_image(gray_path)
    assert loaded_gray.shape == (12, 14, 3)

    bgra_u8 = np.zeros((12, 14, 4), dtype=np.uint8)
    bgra_u8[:, :, 2] = 255  # Red in BGRA
    bgra_u8[:, :, 3] = 128  # Alpha
    bgra_path = tmp_path / "bgra.png"
    cv2.imwrite(str(bgra_path), bgra_u8)
    loaded_bgra = load_image(bgra_path)
    assert loaded_bgra.shape == (12, 14, 3)
    np.testing.assert_allclose(loaded_bgra[:, :, 0], 1.0, atol=1e-5)

    # 3. Mask loading
    mask_u8 = np.zeros((12, 14), dtype=np.uint8)
    mask_u8[:6, :] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask_u8)
    mask = load_mask(mask_path)
    assert mask.dtype == bool
    assert mask.shape == (12, 14)
    assert mask[:6, :].all() and not mask[6:, :].any()

    # 4. Dataset pairing and assertions
    hazy_dir = tmp_path / "hazy"
    clear_dir = tmp_path / "clear"
    save_image(hazy_dir / "0001_0.8_0.2.png", rgb)
    save_image(clear_dir / "0001.png", rgb)

    pairs = pair_dataset(hazy_dir, clear_dir, id_rule="prefix_before_underscore")
    assert len(pairs) == 1
    assert pairs[0].image_id == "0001"

    # Size mismatch must trigger AssertionError
    wrong_size = np.zeros((10, 10, 3), dtype=np.float32)
    save_image(hazy_dir / "0002_0.8_0.2.png", wrong_size)
    save_image(clear_dir / "0002.png", rgb)
    with pytest.raises(AssertionError, match="Size mismatch"):
        pair_dataset(hazy_dir, clear_dir, id_rule="prefix_before_underscore")


def test_run_benchmark_script_synthetic_pairs(tmp_path: Path) -> None:
    """Run scripts/run_benchmark.py on a synthetic hazy/clear dataset and verify CSV output."""
    import subprocess
    import sys

    hazy_dir = tmp_path / "synth" / "hazy"
    clear_dir = tmp_path / "synth" / "clear"
    out_dir = tmp_path / "results"

    rng = np.random.default_rng(42)
    for idx in range(1, 4):
        clear = rng.uniform(0.0, 0.7, size=(64, 80, 3)).astype(np.float32)
        clear[::4, ::4, 0] = 0.0
        A = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        t = np.full((64, 80, 1), 0.6, dtype=np.float32)
        t[:12, :] = 0.05
        hazy = (clear * t + A.reshape(1, 1, 3) * (1.0 - t)).astype(np.float32)

        save_image(clear_dir / f"{idx:04d}.png", clear)
        save_image(hazy_dir / f"{idx:04d}_0.8_0.2.png", hazy)

    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--hazy-dir",
        str(hazy_dir),
        "--clear-dir",
        str(clear_dir),
        "--methods",
        "B0",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "PSNR:" in proc.stdout
    assert "SSIM:" in proc.stdout

    csv_file = out_dir / "per_image" / "synth_B0.csv"
    assert csv_file.is_file()

