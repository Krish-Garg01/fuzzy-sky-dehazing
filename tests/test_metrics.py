"""Unit tests for evaluation metrics, runner, and statistical utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fsdehaze.config import load_config
from fsdehaze.eval.runner import run_methods
from fsdehaze.eval.stats import (
    bootstrap_ci,
    paired_tests,
    summarise,
    to_latex_table,
    to_markdown_table,
)
from fsdehaze.io_utils import pair_dataset, save_image
from fsdehaze.metrics import (
    MAX_PSNR_DB,
    boundary_band,
    ciede2000_map,
    evaluate_all,
    masked_mean,
    masked_psnr,
    psnr,
    saturated_fraction,
    seam_errors,
    ssim,
)
from fsdehaze.pipelines import Result


def test_identical_images_give_ssim_one_and_capped_psnr() -> None:
    """Identical images must yield PSNR == 100.0 dB, SSIM == 1.0, and CIEDE2000 == 0.0."""
    rng = np.random.default_rng(42)
    img = rng.uniform(0.1, 0.9, size=(64, 80, 3)).astype(np.float32)

    assert psnr(img, img) == MAX_PSNR_DB
    np.testing.assert_allclose(ssim(img, img), 1.0, atol=1e-6)
    de = ciede2000_map(img, img)
    assert de.shape == (64, 80)
    np.testing.assert_allclose(de, 0.0, atol=1e-5)


def test_masked_metrics_ignore_pixels_outside_mask() -> None:
    """masked_psnr and masked_mean must be invariant to large corruptions outside the mask."""
    rng = np.random.default_rng(7)
    G = rng.uniform(0.2, 0.8, size=(48, 60, 3)).astype(np.float32)
    J1 = np.clip(G + 0.05, 0.0, 1.0).astype(np.float32)
    J2 = J1.copy()

    mask = np.zeros((48, 60), dtype=bool)
    mask[:20, :] = True  # Top 20 rows are sky

    # Corrupt J2 heavily outside the mask (rows 20..48)
    J2[20:, :, :] = 1.0 - J2[20:, :, :]

    p1 = masked_psnr(G, J1, mask)
    p2 = masked_psnr(G, J2, mask)
    np.testing.assert_allclose(p1, p2, atol=1e-6)

    de1 = masked_mean(ciede2000_map(G, J1), mask)
    de2 = masked_mean(ciede2000_map(G, J2), mask)
    np.testing.assert_allclose(de1, de2, atol=1e-5)

    # Empty mask returns NaN
    empty = np.zeros((48, 60), dtype=bool)
    assert np.isnan(masked_psnr(G, J1, empty))
    assert np.isnan(masked_mean(ciede2000_map(G, J1), empty))


def test_saturated_fraction_counts_out_of_range_pixels_in_j_raw() -> None:
    """saturated_fraction must count pixels where any channel of J_raw is < 0 or > 1."""
    J_raw = np.full((10, 10, 3), 0.5, dtype=np.float32)
    # 15 pixels < 0 in channel 0
    J_raw[:3, :5, 0] = -0.1
    # 10 pixels > 1 in channel 2 (non-overlapping)
    J_raw[5:7, :5, 2] = 1.25

    frac_all = saturated_fraction(J_raw, mask=None)
    np.testing.assert_allclose(frac_all, 25.0 / 100.0, atol=1e-6)

    sky_mask = np.zeros((10, 10), dtype=bool)
    sky_mask[:5, :] = True  # Top 50 pixels contain the 15 negative pixels
    frac_sky = saturated_fraction(J_raw, mask=sky_mask)
    np.testing.assert_allclose(frac_sky, 15.0 / 50.0, atol=1e-6)


def test_boundary_band_and_seam_errors() -> None:
    """boundary_band must isolate pixels within w px of the horizon; seam_errors must detect jumps."""
    H, W = 60, 40
    mask = np.zeros((H, W), dtype=bool)
    mask[:30, :] = True  # Horizontal boundary between row 29 and row 30

    band = boundary_band(mask, width=5)
    assert band.shape == (H, W)
    # Rows 25..29 (5 rows inside mask) and rows 30..34 (5 rows outside mask) have edt <= 5
    expected_band = np.zeros((H, W), dtype=bool)
    expected_band[25:35, :] = True
    assert np.array_equal(band, expected_band)

    # Identical images have 0 seam error
    G = np.full((H, W, 3), 0.5, dtype=np.float32)
    l1_zero, grad_zero = seam_errors(G, G, mask, width=5)
    np.testing.assert_allclose(l1_zero, 0.0, atol=1e-7)
    np.testing.assert_allclose(grad_zero, 0.0, atol=1e-7)

    # Artificial hard-mask step jump at the horizon increases both L1 and gradient seam error
    J_step = G.copy()
    J_step[:30, :, :] = 0.2
    l1_step, grad_step = seam_errors(G, J_step, mask, width=5)
    assert l1_step > 0.1
    assert grad_step > 0.05

    # All-True or all-False mask has no boundary
    assert not np.any(boundary_band(np.ones((H, W), dtype=bool), width=5))
    l1_nan, grad_nan = seam_errors(G, J_step, np.zeros((H, W), dtype=bool), width=5)
    assert np.isnan(l1_nan) and np.isnan(grad_nan)


def test_evaluate_all_and_stats_and_resumable_runner(tmp_path: Path) -> None:
    """Verify evaluate_all, summarise, paired_tests, table formatters, and resumable runner."""
    H, W = 48, 64
    G = np.full((H, W, 3), 0.6, dtype=np.float32)
    J_raw = G.copy()
    J_raw[:10, :10, 0] = -0.2  # 100 saturated sky pixels
    J = np.clip(J_raw, 0.0, 1.0)
    sky_mask = np.zeros((H, W), dtype=bool)
    sky_mask[:24, :] = True

    res = Result(
        J=J,
        J_raw=J_raw,
        t=np.full((H, W), 0.5, dtype=np.float32),
        A=np.array([0.9, 0.9, 0.9], dtype=np.float32),
        info={"sky_present": True},
    )
    metrics_dict = evaluate_all(G, res, sky_mask=sky_mask, seam_width=5)
    assert "sky_psnr" in metrics_dict
    assert "nonsky_psnr" in metrics_dict
    assert metrics_dict["nonsky_psnr"] == MAX_PSNR_DB
    np.testing.assert_allclose(metrics_dict["sat_frac_sky"], 100.0 / (24 * 64), atol=1e-6)

    # Paired tests & bootstrap CI check
    df_a = pd.DataFrame(
        {
            "dataset": ["test"] * 20,
            "method": ["F0"] * 20,
            "image_id": [f"{i:04d}" for i in range(20)],
            "psnr": np.linspace(20.0, 25.0, 20),
            "ciede2000": np.linspace(5.0, 8.0, 20),
            "sky_fraction": [0.3] * 15 + [0.05] * 5,
            "sky_present": [True] * 18 + [False] * 2,
        }
    )
    df_b = df_a.copy()
    df_b["method"] = "B0"
    df_b["psnr"] = df_a["psnr"] - 1.5  # F0 beats B0 on PSNR by +1.5 dB everywhere
    df_b["ciede2000"] = df_a["ciede2000"] + 2.0  # F0 has lower (better) dE2000 by -2.0 everywhere

    summary = summarise(pd.concat([df_a, df_b], ignore_index=True))
    assert set(summary["subset"]) == {"all", "sky_heavy"}

    ptest = paired_tests(df_a, df_b, metrics=["psnr", "ciede2000"], n_resamples=500, seed=42)
    psnr_row = ptest[ptest["metric"] == "psnr"].iloc[0]
    de_row = ptest[ptest["metric"] == "ciede2000"].iloc[0]
    np.testing.assert_allclose(psnr_row["mean_diff"], 1.5, atol=1e-6)
    np.testing.assert_allclose(psnr_row["win_rate_a"], 1.0, atol=1e-6)
    np.testing.assert_allclose(de_row["mean_diff"], -2.0, atol=1e-6)
    np.testing.assert_allclose(de_row["win_rate_a"], 1.0, atol=1e-6)

    ci_l, ci_h = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=42)
    assert 1.0 <= ci_l <= 2.5 <= ci_h <= 4.0

    md_str = to_markdown_table(ptest)
    tex_str = to_latex_table(ptest, caption="Comparison", label="tab:comp")
    assert "| metric |" in md_str
    assert r"\begin{tabular}" in tex_str

    # Resumable runner test
    hazy_dir = tmp_path / "hazy"
    clear_dir = tmp_path / "clear"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx in (1, 2):
        save_image(clear_dir / f"{idx:04d}.png", G)
        save_image(hazy_dir / f"{idx:04d}_0.8_0.2.png", J)
        cv2.imwrite(str(mask_dir / f"{idx:04d}.png"), (sky_mask.astype(np.uint8) * 255))

    pairs = pair_dataset(hazy_dir, clear_dir, sky_mask_dir=mask_dir)
    cfg = load_config("configs/default.yaml")
    out_dir = tmp_path / "results"

    # Run on first pair only, then resume with both pairs
    run_methods(pairs[:1], ["B0"], cfg, out_dir, dataset_name="tiny", resume=True)
    res_dfs = run_methods(pairs, ["B0"], cfg, out_dir, dataset_name="tiny", resume=True)
    assert len(res_dfs["B0"]) == 2
