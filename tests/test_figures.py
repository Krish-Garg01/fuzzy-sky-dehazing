"""Tests for the Phase 7 figure helpers (synthetic data only; no datasets required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsdehaze.pipelines import Result
from fsdehaze.viz import figures as F


def _sky_mask(h: int = 80, w: int = 100, y_h: int = 30) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[:y_h] = True
    return m


def test_find_horizon_column_and_crop_box() -> None:
    mask = _sky_mask(y_h=30)
    x, y = F.find_horizon_column(mask)
    assert 0 <= x < mask.shape[1]
    assert y == 30
    y0, y1, x0, x1 = F.horizon_crop_box(mask, crop_h=40, crop_w=50)
    assert (y1 - y0, x1 - x0) == (40, 50)
    assert y0 <= 30 < y1


def test_find_step_column_prefers_largest_hard_step() -> None:
    h, w, y_h = 80, 100, 40
    gt = _sky_mask(h, w, y_h)
    hard_mask = gt.copy()
    t_blend = np.full((h, w), 0.3, dtype=np.float32)
    t_blend[:y_h] = 0.5
    t_blend[:y_h, 70] = 0.9  # biggest jump at column 70
    res = Result(J=np.zeros((h, w, 3), np.float32), J_raw=np.zeros((h, w, 3), np.float32),
                 t=t_blend, A=np.ones(3, np.float32), mask=hard_mask, info={"t_blend": t_blend})
    x, y = F.find_step_column(gt, res)
    assert (x, y) == (70, y_h)


def test_chart_figures_write_pngs(tmp_path) -> None:
    rows = []
    for subset, n in [("all", 10), ("sky_heavy", 6)]:
        for i, m in enumerate(F.MAIN_METHODS):
            rows.append({"subset": subset, "method": m, "n_images": n,
                         "seam_grad_mean": 0.1 - 0.01 * i, "seam_l1_mean": 0.1 - 0.01 * i,
                         "sky_ciede2000_mean": 10 - i, "sky_psnr_mean": 15 + i})
    summary = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    assert F.plot_seam_sky_bars(summary, tmp_path / "f6.png").stat().st_size > 0

    ab = [{"subset": s, "method": m, "description": m, "delta_psnr_vs_F0": d, "seam_grad_mean": 0.05}
          for s in ["all", "sky_heavy"] for m, d in [("F0", 0.0), ("F1", 1.0), ("A3_no_f4", -0.2)]]
    ab_csv = tmp_path / "ab.csv"
    pd.DataFrame(ab).to_csv(ab_csv, index=False)
    assert F.plot_ablation_chart(ab_csv, tmp_path / "f7.png").stat().st_size > 0

    hist = pd.DataFrame({"generation": [0, 1, 2], "best_fitness": [25.0, 25.5, 26.0],
                         "mean_fitness": [22.0, 24.0, 25.5], "std_fitness": [2.0, 1.0, 0.3],
                         "val_fitness": [25.2, 25.6, 26.1], "f0_train_fitness": [25.4] * 3,
                         "f0_val_fitness": [25.6] * 3})
    hist_csv = tmp_path / "ga.csv"
    hist.to_csv(hist_csv, index=False)
    assert F.plot_ga_convergence(hist_csv, tmp_path / "f8.png").stat().st_size > 0
    assert F.plot_pipeline_diagram(tmp_path / "f1.png").stat().st_size > 0
