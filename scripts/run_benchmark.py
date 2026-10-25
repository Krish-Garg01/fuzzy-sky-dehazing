"""Run dehazing benchmark on paired hazy/clear folders or configured datasets.

Supports both:
1. Direct folder mode:
   python scripts/run_benchmark.py --hazy-dir path/to/hazy --clear-dir path/to/clear
2. Dataset config mode:
   python scripts/run_benchmark.py --config configs/default.yaml --datasets sots_outdoor ohaze --methods B0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fsdehaze.config import load_config
from fsdehaze.eval.runner import run_methods
from fsdehaze.eval.stats import summarise
from fsdehaze.io_utils import Pair, pair_dataset


def evaluate_dataset(
    dataset_name: str,
    pairs: list[Pair],
    methods: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
    params: dict[str, Any] | None = None,
    save_outputs: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """Run requested methods on `pairs`, save per-image and summary CSVs, and print results."""
    results_by_method = run_methods(
        pairs=pairs,
        methods=methods,
        cfg=cfg,
        out_dir=out_dir,
        dataset_name=dataset_name,
        params=params,
        save_outputs=save_outputs,
        resume=resume,
    )

    combined_df = pd.concat(list(results_by_method.values()), ignore_index=True)
    sky_heavy_thresh = float(cfg.get("eval", {}).get("sky_heavy_threshold", 0.20))
    summary_df = summarise(combined_df, sky_heavy_threshold=sky_heavy_thresh)

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_dir / f"{dataset_name}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    for _, row in summary_df.iterrows():
        subset = row["subset"]
        method = row["method"]
        n_imgs = int(row["n_images"])
        psnr_m = row.get("psnr_mean", float("nan"))
        psnr_s = row.get("psnr_std", float("nan"))
        ssim_m = row.get("ssim_mean", float("nan"))
        de_m = row.get("ciede2000_mean", float("nan"))
        sky_p = row.get("sky_psnr_mean", float("nan"))
        sky_de = row.get("sky_ciede2000_mean", float("nan"))
        nsky_p = row.get("nonsky_psnr_mean", float("nan"))
        sat_sky = row.get("sat_pct_sky_mean", float("nan"))
        seam_l1 = row.get("seam_l1_mean", float("nan"))
        med_t = row.get("runtime_s_median", float("nan"))

        print(
            f"[{dataset_name} ({subset}) | {method}] N={n_imgs} | "
            f"PSNR: {psnr_m:.2f}+/-{psnr_s:.2f} | SSIM: {ssim_m:.4f} | dE00: {de_m:.2f} | "
            f"SkyPSNR: {sky_p:.2f} | SkydE00: {sky_de:.2f} | NonSkyPSNR: {nsky_p:.2f} | "
            f"SkySat%: {sat_sky:.2f}% | SeamL1: {seam_l1:.4f} | Time: {med_t:.3f}s"
        )

    print(f"  -> Summary CSV saved to: {summary_csv_path}")
    return summary_df


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
        "--sky-mask-dir",
        type=Path,
        default=None,
        help="Optional path to ground-truth sky masks when using --hazy-dir / --clear-dir",
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
        help="Also save dehazed output images for all pairs under results/images/",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute all images from scratch instead of resuming existing CSVs",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    params = load_config(args.params) if args.params is not None else None
    resume = not args.no_resume

    if args.hazy_dir is not None or args.clear_dir is not None:
        if args.hazy_dir is None or args.clear_dir is None:
            parser.error("Both --hazy-dir and --clear-dir must be provided together.")
        pairs = pair_dataset(
            hazy_dir=args.hazy_dir,
            clear_dir=args.clear_dir,
            id_rule=args.id_rule,
            sky_mask_dir=args.sky_mask_dir,
        )
        dataset_label = args.hazy_dir.parent.name or "custom"
        evaluate_dataset(
            dataset_name=dataset_label,
            pairs=pairs,
            methods=args.methods,
            cfg=cfg,
            out_dir=args.out_dir,
            params=params,
            save_outputs=args.save_images,
            resume=resume,
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
        evaluate_dataset(
            dataset_name=ds_name,
            pairs=pairs,
            methods=args.methods,
            cfg=cfg,
            out_dir=args.out_dir,
            params=params,
            save_outputs=args.save_images,
            resume=resume,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
