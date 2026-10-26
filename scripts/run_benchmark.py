"""Run dehazing benchmark on paired hazy/clear folders or configured datasets.

Supports:
1. Direct folder mode:
   python scripts/run_benchmark.py --hazy-dir path/to/hazy --clear-dir path/to/clear
2. Dataset config mode (with optional --include-ablations):
   python scripts/run_benchmark.py --config configs/default.yaml --datasets sots_outdoor ohaze --methods B0 B1 B2 F0 F1 --include-ablations
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
from fsdehaze.eval.runner import ABLATION_METHODS, run_methods
from fsdehaze.eval.stats import (
    paired_tests,
    summarise,
    to_latex_table,
    to_markdown_table,
)
from fsdehaze.io_utils import Pair, pair_dataset


def _compute_significance_tables(
    results_by_method: dict[str, pd.DataFrame],
    sky_heavy_threshold: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired Wilcoxon + bootstrap CI + win-rate comparisons (F1 vs B0/B1/B2/F0, and F0 vs B2)."""
    comparisons = [
        ("F1", "B0"),
        ("F1", "B1"),
        ("F1", "B2"),
        ("F1", "F0"),
        ("F0", "B2"),
        ("F0", "B1"),
        ("F0", "B0"),
    ]
    sig_frames: list[pd.DataFrame] = []
    for m_a, m_b in comparisons:
        if m_a not in results_by_method or m_b not in results_by_method:
            continue
        df_a = results_by_method[m_a]
        df_b = results_by_method[m_b]

        # 1. All images
        res_all = paired_tests(df_a, df_b, n_resamples=n_resamples, seed=seed)
        if not res_all.empty:
            res_all.insert(0, "method_b", m_b)
            res_all.insert(0, "method_a", m_a)
            res_all.insert(0, "subset", "all")
            sig_frames.append(res_all)

        # 2. Sky-heavy subset (> 20% sky)
        if "sky_fraction" in df_a.columns:
            heavy_ids = set(
                df_a.loc[df_a["sky_fraction"] > sky_heavy_threshold, "image_id"].astype(str)
            )
            if len(heavy_ids) >= 2:
                df_a_h = df_a[df_a["image_id"].astype(str).isin(heavy_ids)]
                df_b_h = df_b[df_b["image_id"].astype(str).isin(heavy_ids)]
                res_h = paired_tests(df_a_h, df_b_h, n_resamples=n_resamples, seed=seed)
                if not res_h.empty:
                    res_h.insert(0, "method_b", m_b)
                    res_h.insert(0, "method_a", m_a)
                    res_h.insert(0, "subset", "sky_heavy")
                    sig_frames.append(res_h)

    if not sig_frames:
        return pd.DataFrame()
    return pd.concat(sig_frames, ignore_index=True)


def write_ablation_summary_table(
    summary_df: pd.DataFrame, summary_dir: Path, dataset_name: str
) -> None:
    """Write a dedicated ablation comparison CSV and Markdown table if F0 and ablations exist."""
    methods_present = set(summary_df["method"].unique())
    if "F0" not in methods_present or not any(m in methods_present for m in ABLATION_METHODS):
        return

    ablation_map = [
        ("A1_B0_plain_dcp", "B0", "Plain DCP (no sky correction)"),
        ("A2a_B1_hard_thresh", "B1", "Hard threshold on (f1, f2), tuned tau=0.65"),
        ("A2b_B2_hard_fcm", "B2", "Hard FCM mask M = 1[mu_raw > 0.5]"),
        ("Ref_F0_soft_fuzzy", "F0", "Soft fuzzy transmission blend (hand-set defaults)"),
        ("Ref_F1_fuzzy_ga", "F1", "Soft fuzzy transmission blend + GA-tuned params"),
        ("A3_no_f4", "A3_no_f4", "Drop spatial vertical prior f4 (use f1..f3 only)"),
        ("A4_no_f3", "A4_no_f3", "Drop gradient feature f3 (use f1, f2, f4 only)"),
        ("A5a_with_f5", "A5a_with_f5", "Add local intensity variance f5 (5 features)"),
        ("A5b_with_f6", "A5b_with_f6", "Add normalized dark channel f6 (5 features)"),
        ("A6a_c2", "A6a_c2", "FCM cluster count c = 2"),
        ("A6b_c4", "A6b_c4", "FCM cluster count c = 4"),
        ("A7_tolerance", "A7_tolerance", "Additive tolerance lift vs multiplicative blend"),
        ("A8_no_mu_gf", "A8_no_mu_gf", "Skip guided filter on membership mu_raw"),
        ("A9a_m1.5", "A9a_m1.5", "FCM fuzziness exponent m = 1.5"),
        ("A9b_m3.0", "A9b_m3.0", "FCM fuzziness exponent m = 3.0"),
        ("A10_gf_before_blend", "A10_gf_before_blend", "Guided filter before blend instead of after"),
    ]
    rows: list[dict[str, Any]] = []
    for subset in ["all", "sky_heavy"]:
        sub = summary_df[summary_df["subset"] == subset].set_index("method")
        if "F0" not in sub.index:
            continue
        f0 = sub.loc["F0"]
        f1 = sub.loc["F1"] if "F1" in sub.index else f0
        for ab_id, m_code, desc in ablation_map:
            if m_code not in sub.index:
                continue
            r = sub.loc[m_code]
            rows.append(
                {
                    "subset": subset,
                    "ablation_id": ab_id,
                    "method": m_code,
                    "description": desc,
                    "n_images": int(r["n_images"]),
                    "psnr_mean": float(r["psnr_mean"]),
                    "delta_psnr_vs_F0": float(r["psnr_mean"] - f0["psnr_mean"]),
                    "delta_psnr_vs_F1": float(r["psnr_mean"] - f1["psnr_mean"]),
                    "ssim_mean": float(r["ssim_mean"]),
                    "ciede2000_mean": float(r["ciede2000_mean"]),
                    "sky_psnr_mean": float(r["sky_psnr_mean"]),
                    "sky_ciede2000_mean": float(r["sky_ciede2000_mean"]),
                    "nonsky_psnr_mean": float(r["nonsky_psnr_mean"]),
                    "sat_pct_sky_mean": float(r["sat_pct_sky_mean"]),
                    "seam_l1_mean": float(r["seam_l1_mean"]),
                    "seam_grad_mean": float(r["seam_grad_mean"]),
                    "runtime_s_mean": float(r["runtime_s_mean"]),
                }
            )
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(summary_dir / f"{dataset_name}_ablations.csv", index=False)

    md_lines = [f"# {dataset_name} Ablation Study Summary\n"]
    for subset, title in [("all", "All Images"), ("sky_heavy", "Sky-Heavy Images")]:
        sub_df = out_df[out_df["subset"] == subset]
        if sub_df.empty:
            continue
        n_val = int(sub_df.iloc[0]["n_images"])
        md_lines.append(f"## {title} (N = {n_val})\n")
        md_lines.append(
            "| Ablation | Method | Description | PSNR (dB) | dPSNR vs F0 | dPSNR vs F1 | "
            "SSIM | Sky PSNR | Sky dE00 | NonSky PSNR | SkySat% | Seam L1 | Seam Grad |"
        )
        md_lines.append(
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for _, r in sub_df.iterrows():
            md_lines.append(
                f"| {r['ablation_id']} | `{r['method']}` | {r['description']} | "
                f"{r['psnr_mean']:.2f} | {r['delta_psnr_vs_F0']:+.2f} | {r['delta_psnr_vs_F1']:+.2f} | "
                f"{r['ssim_mean']:.4f} | {r['sky_psnr_mean']:.2f} | {r['sky_ciede2000_mean']:.2f} | "
                f"{r['nonsky_psnr_mean']:.2f} | {r['sat_pct_sky_mean']:.2f}% | "
                f"{r['seam_l1_mean']:.4f} | {r['seam_grad_mean']:.4f} |"
            )
        md_lines.append("")
    (summary_dir / f"{dataset_name}_ablations.md").write_text("\n".join(md_lines), encoding="utf-8")


def evaluate_dataset(
    dataset_name: str,
    pairs: list[Pair],
    methods: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
    params: dict[str, Any] | None = None,
    save_outputs: bool = False,
    save_preview_count: int = 5,
    resume: bool = True,
) -> pd.DataFrame:
    """Run requested methods on `pairs`, save per-image, summary, and significance CSVs, and print results."""
    results_by_method = run_methods(
        pairs=pairs,
        methods=methods,
        cfg=cfg,
        out_dir=out_dir,
        dataset_name=dataset_name,
        params=params,
        save_outputs=save_outputs,
        save_preview_count=save_preview_count,
        resume=resume,
    )

    combined_df = pd.concat(list(results_by_method.values()), ignore_index=True)
    eval_cfg = cfg.get("eval", {})
    sky_heavy_thresh = float(eval_cfg.get("sky_heavy_threshold", 0.20))
    n_resamples = int(eval_cfg.get("bootstrap_resamples", 10000))
    seed = int(cfg.get("seed", 42))

    summary_df = summarise(combined_df, sky_heavy_threshold=sky_heavy_thresh)

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_dir / f"{dataset_name}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Also write Markdown and LaTeX tables
    md_path = summary_dir / f"{dataset_name}_summary.md"
    tex_path = summary_dir / f"{dataset_name}_summary.tex"
    md_content = (
        f"### {dataset_name} — Summary Table\n\n"
        + to_markdown_table(summary_df)
        + "\n"
    )
    md_path.write_text(md_content, encoding="utf-8")
    tex_path.write_text(
        to_latex_table(summary_df, caption=f"Results on {dataset_name}."),
        encoding="utf-8",
    )

    # Compute paired significance tests (Wilcoxon + bootstrap CIs + win rates)
    sig_df = _compute_significance_tables(
        results_by_method,
        sky_heavy_threshold=sky_heavy_thresh,
        n_resamples=n_resamples,
        seed=seed,
    )
    if not sig_df.empty:
        sig_csv_path = summary_dir / f"{dataset_name}_significance.csv"
        sig_df.to_csv(sig_csv_path, index=False)

    write_ablation_summary_table(summary_df, summary_dir, dataset_name)

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
        seam_gr = row.get("seam_grad_mean", float("nan"))
        med_t = row.get("runtime_s_median", float("nan"))

        print(
            f"[{dataset_name} ({subset}) | {method}] N={n_imgs} | "
            f"PSNR: {psnr_m:.2f}+/-{psnr_s:.2f} | SSIM: {ssim_m:.4f} | dE00: {de_m:.2f} | "
            f"SkyPSNR: {sky_p:.2f} | SkydE00: {sky_de:.2f} | NonSkyPSNR: {nsky_p:.2f} | "
            f"SkySat%: {sat_sky:.2f}% | SeamL1: {seam_l1:.4f} | SeamGrad: {seam_gr:.4f} | Time: {med_t:.3f}s"
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
        "--include-ablations",
        action="store_true",
        help="Also run all 11 ablation configurations (A3..A10) on sots_outdoor",
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
        "--preview-count",
        type=int,
        default=5,
        help="Number of preview images per method to save (default: 5)",
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
            check_sizes=False,
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
            save_preview_count=args.preview_count,
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
            check_sizes=False,
        )
        run_list = list(args.methods)
        if args.include_ablations and ds_name == "sots_outdoor":
            for ab in ABLATION_METHODS:
                if ab not in run_list:
                    run_list.append(ab)

        evaluate_dataset(
            dataset_name=ds_name,
            pairs=pairs,
            methods=run_list,
            cfg=cfg,
            out_dir=args.out_dir,
            params=params,
            save_outputs=args.save_images,
            save_preview_count=args.preview_count,
            resume=resume,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
