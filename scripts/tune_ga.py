"""Run DEAP Genetic Algorithm tuning on `ots_subset` and plot GA convergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsdehaze.config import load_config
from fsdehaze.io_utils import pair_dataset
from fsdehaze.tuning.ga import GENE_NAMES, DEFAULT_GENE_VALUES, run_ga


def plot_ga_convergence(history_df: pd.DataFrame, out_path: Path) -> None:
    """Plot best/mean training fitness and validation fitness across GA generations."""
    gens = history_df["generation"].to_numpy()
    best_tr = history_df["best_fitness"].to_numpy()
    mean_tr = history_df["mean_fitness"].to_numpy()
    std_tr = history_df["std_fitness"].to_numpy()
    val_fit = history_df["val_fitness"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)

    # Panel 1: Population fitness convergence on tuning set
    ax1.plot(gens, best_tr, "o-", color="#1f77b4", linewidth=2.2, markersize=5, label="Hall-of-Fame Best (Train)")
    ax1.plot(gens, mean_tr, "s--", color="#2ca02c", linewidth=1.8, markersize=4, label="Population Mean (Train)")
    ax1.fill_between(
        gens,
        mean_tr - 0.5 * std_tr,
        np.minimum(best_tr, mean_tr + 0.5 * std_tr),
        color="#2ca02c",
        alpha=0.18,
        label="Population +/- 0.5 SD",
    )
    if "f0_train_fitness" in history_df.columns:
        f0_tr = float(history_df["f0_train_fitness"].iloc[0])
        ax1.axhline(f0_tr, color="gray", linestyle=":", linewidth=1.6, label=f"F0 Hand-Set Default ({f0_tr:.2f})")

    ax1.set_xlabel("Generation", fontsize=10.5)
    ax1.set_ylabel("Fitness F (0.5 PSNR + 0.5 SkyPSNR - Pen)", fontsize=10.5)
    ax1.set_title("GA Convergence on Tuning Set (OTS-Subset Train, N=96)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel 2: Train vs Validation PSNR & SkyPSNR of Best Individual
    ax2.plot(gens, history_df["best_train_psnr"], "o-", color="#1f77b4", linewidth=2.0, label="Train Full PSNR (dB)")
    ax2.plot(gens, history_df["val_psnr"], "s--", color="#ff7f0e", linewidth=2.0, label="Val Full PSNR (dB)")
    ax2.plot(gens, history_df["best_train_sky_psnr"], "^-", color="#9467bd", linewidth=2.0, label="Train Sky PSNR (dB)")
    ax2.plot(gens, history_df["val_sky_psnr"], "d--", color="#d62728", linewidth=2.0, label="Val Sky PSNR (dB)")

    ax2.set_xlabel("Generation", fontsize=10.5)
    ax2.set_ylabel("PSNR (dB)", fontsize=10.5)
    ax2.set_title("Generalisation Check (Train N=96 vs. Held-Out Val N=24)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune Fuzzy Sky-Aware Dehazing parameters via DEAP GA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ga.yaml"),
        help="Path to GA configuration YAML",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to base pipeline configuration YAML",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets configuration YAML",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ots_subset",
        help="Tuning dataset entry in datasets.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/best_params.yaml"),
        help="Output YAML path for best GA parameters",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/summary/ga_history.csv"),
        help="Output CSV path for per-generation history",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/figures/ga_convergence.png"),
        help="Output PNG path for GA convergence figure",
    )
    args = parser.parse_args(argv)

    cfg_ga = load_config(args.config)
    base_cfg = load_config(args.base_config)
    ds_cfg = load_config(args.datasets_config)[args.dataset]

    pairs = pair_dataset(
        hazy_dir=ds_cfg["hazy_dir"],
        clear_dir=ds_cfg["clear_dir"],
        id_rule=ds_cfg["id_rule"],
        sky_mask_dir=ds_cfg.get("sky_mask_dir"),
        check_sizes=False,
    )

    best_params, history_df = run_ga(
        tune_pairs=pairs,
        cfg_ga=cfg_ga,
        base_cfg=base_cfg,
        out_yaml=args.out,
        history_csv=args.history,
    )

    plot_ga_convergence(history_df, args.plot)

    print(f"\nSaved best GA parameters to: {args.out}")
    print(f"Saved GA generation history to: {args.history}")
    print(f"Saved GA convergence figure to: {args.plot}")
    print("\nTuned Genes (F1) vs. Hand-Set Defaults (F0):")
    for g in GENE_NAMES:
        print(f"  {g:8s}: {best_params[g]!s:>8s}  (default F0: {DEFAULT_GENE_VALUES[g]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
