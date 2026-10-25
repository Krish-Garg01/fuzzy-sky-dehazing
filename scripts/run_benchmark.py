"""Run dehazing benchmark on paired hazy/clear folders or configured datasets.

Supports both:
1. Direct folder mode:
   python scripts/run_benchmark.py --hazy-dir path/to/hazy --clear-dir path/to/clear
2. Dataset config mode (matches README.md):
   python scripts/run_benchmark.py --config configs/default.yaml --datasets sots_outdoor --methods B0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# Ensure `src` is importable when running `python scripts/run_benchmark.py` without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fsdehaze.config import load_config
from fsdehaze.io_utils import Pair, load_image, pair_dataset, save_image
from fsdehaze.pipelines import (
    Result,
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)


def compute_psnr(clear: np.ndarray, dehazed: np.ndarray) -> float:
    """Compute PSNR (dB) on float32 [0, 1] images, capping infinite PSNR at 100.0 dB."""
    mse = float(np.mean((clear.astype(np.float64) - dehazed.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return 100.0
    val = float(peak_signal_noise_ratio(clear, dehazed, data_range=1.0))
    return min(val, 100.0)


def compute_ssim(clear: np.ndarray, dehazed: np.ndarray) -> float:
    """Compute SSIM on float32 [0, 1] RGB images."""
    return float(
        structural_similarity(
            clear,
            dehazed,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def run_single_method(
    method: str,
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> Result:
    """Dispatch a single image to the requested dehazing pipeline."""
    m = method.upper()
    if m in ("B0", "DCP"):
        return dehaze_dcp(img, cfg)
    if m in ("B1", "HARD_THRESHOLD"):
        return dehaze_hard_threshold(img, cfg)
    if m in ("B2", "HARD_FCM"):
        return dehaze_hard_fcm(img, cfg)
    if m in ("F0", "FUZZY"):
        return dehaze_fuzzy(img, cfg, params=None)
    if m == "F1":
        return dehaze_fuzzy(img, cfg, params=params)
    raise ValueError(f"Unsupported method ID: {method!r}")


def evaluate_pairs(
    dataset_name: str,
    pairs: list[Pair],
    methods: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
    params: dict[str, Any] | None = None,
    save_outputs: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run methods over a list of image pairs, saving per-image CSVs and printing summaries."""
    per_image_dir = out_dir / "per_image"
    summary_dir = out_dir / "summary"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))

    results_by_method: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for method in methods:
        csv_path = per_image_dir / f"{dataset_name}_{method}.csv"
        rows: list[dict[str, Any]] = []
        preview_dir = out_dir / "images" / dataset_name / method
        preview_dir.mkdir(parents=True, exist_ok=True)

        for idx, pair in enumerate(tqdm(pairs, desc=f"{dataset_name} [{method}]", leave=False)):
            hazy = load_image(pair.hazy_path)
            clear = load_image(pair.clear_path)

            t0 = time.perf_counter()
            res = run_single_method(method, hazy, cfg, params=params)
            elapsed = time.perf_counter() - t0

            psnr_val = compute_psnr(clear, res.J)
            ssim_val = compute_ssim(clear, res.J)

            # Always save the first 10 visual check panels (hazy | dehazed | clear), or all if --save-images
            if save_outputs or idx < 10:
                save_image(preview_dir / f"{pair.image_id}_dehazed.png", res.J)
                panel = np.concatenate([hazy, res.J, clear], axis=1)
                save_image(preview_dir / f"{pair.image_id}_compare_hazy_dehazed_clear.png", panel)
                save_image(preview_dir / f"{pair.image_id}_t_final.png", res.t)

            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "image_id": pair.image_id,
                    "hazy_file": pair.hazy_path.name,
                    "clear_file": pair.clear_path.name,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "A_r": float(res.A[0]),
                    "A_g": float(res.A[1]),
                    "A_b": float(res.A[2]),
                    "runtime_s": elapsed,
                    "seed": seed,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        results_by_method[method] = df

        mean_psnr = float(df["psnr"].mean())
        std_psnr = float(df["psnr"].std(ddof=0))
        med_psnr = float(df["psnr"].median())
        mean_ssim = float(df["ssim"].mean())
        std_ssim = float(df["ssim"].std(ddof=0))
        med_ssim = float(df["ssim"].median())
        med_time = float(df["runtime_s"].median())

        summary_rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "n_images": len(df),
                "psnr_mean": round(mean_psnr, 4),
                "psnr_std": round(std_psnr, 4),
                "psnr_median": round(med_psnr, 4),
                "ssim_mean": round(mean_ssim, 4),
                "ssim_std": round(std_ssim, 4),
                "ssim_median": round(med_ssim, 4),
                "runtime_median_s": round(med_time, 4),
                "seed": seed,
            }
        )

        print(
            f"[{dataset_name} | {method}] N={len(df)} | "
            f"PSNR: {mean_psnr:.2f} +/- {std_psnr:.2f} dB (median {med_psnr:.2f}) | "
            f"SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f} (median {med_ssim:.4f}) | "
            f"Median time: {med_time:.3f}s\n"
            f"  -> Per-image CSV : {csv_path}\n"
            f"  -> Visual outputs: {preview_dir}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = summary_dir / f"{dataset_name}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  -> Summary CSV   : {summary_csv_path}")

    return results_by_method


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running benchmark evaluations."""
    parser = argparse.ArgumentParser(
        description="Run dehazing methods on paired hazy/clear datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to main hyperparameter YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to dataset paths YAML config (default: configs/datasets.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names from datasets.yaml (e.g. sots_outdoor ohaze)",
    )
    parser.add_argument(
        "--hazy-dir",
        type=Path,
        default=None,
        help="Direct path to a folder of hazy images (overrides --datasets)",
    )
    parser.add_argument(
        "--clear-dir",
        type=Path,
        default=None,
        help="Direct path to a folder of clear ground-truth images (overrides --datasets)",
    )
    parser.add_argument(
        "--id-rule",
        type=str,
        default="prefix_before_underscore",
        choices=["prefix_before_underscore", "exact_stem", "ohaze"],
        help="Filename pairing rule when using --hazy-dir / --clear-dir",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["B0"],
        help="Methods to evaluate (default: B0)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional tuned parameters YAML for F1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for per-image CSVs (default: results)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save dehazed output images under results/images/",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    params = load_config(args.params) if args.params is not None else None

    if args.hazy_dir is not None or args.clear_dir is not None:
        if args.hazy_dir is None or args.clear_dir is None:
            parser.error("Both --hazy-dir and --clear-dir must be provided together.")
        pairs = pair_dataset(
            hazy_dir=args.hazy_dir,
            clear_dir=args.clear_dir,
            id_rule=args.id_rule,
        )
        dataset_label = args.hazy_dir.parent.name or "custom"
        evaluate_pairs(
            dataset_name=dataset_label,
            pairs=pairs,
            methods=args.methods,
            cfg=cfg,
            out_dir=args.out_dir,
            params=params,
            save_outputs=args.save_images,
        )
        return 0

    datasets_cfg = load_config(args.datasets_config)
    dataset_names = args.datasets if args.datasets is not None else ["sots_outdoor"]

    for ds_name in dataset_names:
        if ds_name not in datasets_cfg:
            raise KeyError(f"Dataset {ds_name!r} not found in {args.datasets_config}")
        ds_info = datasets_cfg[ds_name]
        pairs = pair_dataset(
            hazy_dir=ds_info["hazy_dir"],
            clear_dir=ds_info["clear_dir"],
            id_rule=ds_info.get("id_rule", "prefix_before_underscore"),
            sky_mask_dir=ds_info.get("sky_mask_dir"),
        )
        evaluate_pairs(
            dataset_name=ds_name,
            pairs=pairs,
            methods=args.methods,
            cfg=cfg,
            out_dir=args.out_dir,
            params=params,
            save_outputs=args.save_images,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
