"""Resumable evaluation runner across datasets and dehazing methods."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from fsdehaze.io_utils import Pair, load_image, load_mask, save_image
from fsdehaze.metrics import evaluate_all
from fsdehaze.pipelines import (
    Result,
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)

REQUIRED_METRIC_COLS: set[str] = {
    "psnr",
    "ssim",
    "ciede2000",
    "sky_psnr",
    "sky_ciede2000",
    "nonsky_psnr",
    "sat_pct_sky",
    "seam_l1",
    "seam_grad",
}


def dispatch_method(
    method: str,
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    tau: float | None = None,
) -> Result:
    """Run the specified dehazing method on a single RGB float32 [0, 1] image."""
    m = method.upper()
    if m in ("B0", "DCP"):
        return dehaze_dcp(img, cfg)
    if m in ("B1", "HARD_THRESHOLD"):
        return dehaze_hard_threshold(img, cfg, tau=tau)
    if m in ("B2", "HARD_FCM"):
        return dehaze_hard_fcm(img, cfg, params=params)
    if m in ("F0", "FUZZY"):
        return dehaze_fuzzy(img, cfg, params=None)
    if m == "F1":
        return dehaze_fuzzy(img, cfg, params=params)
    raise ValueError(f"Unknown method identifier: {method!r}")


def run_methods(
    pairs: Sequence[Pair],
    methods: Sequence[str],
    cfg: dict[str, Any],
    out_dir: str | Path,
    dataset_name: str = "dataset",
    params: dict[str, Any] | None = None,
    tau: float | None = None,
    save_outputs: bool = False,
    save_preview_count: int = 10,
    resume: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run a list of methods on paired images, computing all metrics and saving resumable CSVs.

    Writes one row per image to `<out_dir>/per_image/<dataset_name>_<method>.csv`.
    If `resume=True` and the CSV already exists with all required metric columns,
    previously evaluated `image_id` rows are preserved and skipped.

    Args:
        pairs: Sequence of validated `Pair` entries.
        methods: Sequence of method IDs (e.g. `['B0', 'B1', 'B2', 'F0', 'F1']`).
        cfg: Base configuration dictionary.
        out_dir: Output root directory (usually `results/`).
        dataset_name: Name of the dataset for file naming and logging.
        params: Optional GA-tuned parameter dictionary for `F1`.
        tau: Optional tuned threshold for `B1`.
        save_outputs: Whether to save dehazed images for every pair in the dataset.
        save_preview_count: Number of initial pairs for which side-by-side comparison panels are saved.
        resume: Whether to resume from an existing per-image CSV if present.

    Returns:
        Dictionary mapping each `method` to its complete per-image `pd.DataFrame`.
    """
    out_root = Path(out_dir)
    per_image_dir = out_root / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    seam_width = int(cfg.get("eval", {}).get("seam_band_px", 10))
    results_by_method: dict[str, pd.DataFrame] = {}

    for method in methods:
        csv_path = per_image_dir / f"{dataset_name}_{method}.csv"
        preview_dir = out_root / "images" / dataset_name / method
        preview_dir.mkdir(parents=True, exist_ok=True)

        existing_rows: list[dict[str, Any]] = []
        done_ids: set[str] = set()

        if resume and csv_path.is_file():
            try:
                prev_df = pd.read_csv(csv_path, dtype={"image_id": str})
                if REQUIRED_METRIC_COLS.issubset(set(prev_df.columns)):
                    existing_rows = prev_df.to_dict(orient="records")
                    done_ids = {str(r["image_id"]) for r in existing_rows}
            except Exception:
                existing_rows = []
                done_ids = set()

        rows = list(existing_rows)

        for idx, pair in enumerate(tqdm(pairs, desc=f"{dataset_name} [{method}]", leave=False)):
            if str(pair.image_id) in done_ids:
                continue

            hazy = load_image(pair.hazy_path)
            clear = load_image(pair.clear_path)
            sky_mask = (
                load_mask(pair.sky_mask_path)
                if pair.sky_mask_path is not None and pair.sky_mask_path.is_file()
                else None
            )

            t_start = time.perf_counter()
            res = dispatch_method(method, hazy, cfg, params=params, tau=tau)
            elapsed = time.perf_counter() - t_start

            metric_dict = evaluate_all(clear, res, sky_mask=sky_mask, seam_width=seam_width)

            if save_outputs or idx < save_preview_count:
                save_image(preview_dir / f"{pair.image_id}_dehazed.png", res.J)
                panel = np.concatenate([hazy, res.J, clear], axis=1)
                save_image(preview_dir / f"{pair.image_id}_compare_hazy_dehazed_clear.png", panel)
                save_image(preview_dir / f"{pair.image_id}_t_final.png", res.t)

            row: dict[str, Any] = {
                "dataset": dataset_name,
                "method": method,
                "image_id": str(pair.image_id),
                "hazy_file": pair.hazy_path.name,
                "clear_file": pair.clear_path.name,
                **metric_dict,
                "A_r": float(res.A[0]),
                "A_g": float(res.A[1]),
                "A_b": float(res.A[2]),
                "runtime_s": elapsed,
                "seed": seed,
            }
            rows.append(row)

            # Incremental flush every 25 images so interrupted runs remain resumable
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(csv_path, index=False)

        df = pd.DataFrame(rows)
        if not df.empty and "image_id" in df.columns:
            df["image_id"] = df["image_id"].astype(str)
            df = df.sort_values("image_id").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        results_by_method[method] = df

    return results_by_method
