"""Generate all report figures (1-10) and extra tables (4-6) from saved results (Phase 7).

Usage:
    python scripts/make_figures.py                 # all figures + tables
    python scripts/make_figures.py --only 4 5      # just the hero crops and the profile
    python scripts/make_figures.py --charts-only   # CSV-based figures only (no datasets needed)

Inputs:
    configs/default.yaml, configs/datasets.yaml, configs/ga.yaml
    results/best_params.yaml, results/best_tau.yaml
    results/summary/*.csv (written by scripts/run_benchmark.py and scripts/tune_ga.py)

Outputs:
    results/figures/fig01..fig10_*.png
    results/summary/{sots_outdoor,ohaze}_significance.md, runtime_summary.*, ga_params_table.*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fsdehaze.config import load_config
from fsdehaze.io_utils import pair_dataset
from fsdehaze.viz import figures as F

# Images chosen from the per-image CSVs (see report): sky-heavy SOTS scenes with a clear
# horizon where the soft/hard difference is visible, plus honest failure cases.
SOTS_DCP_FAILURE = ["1037", "0004", "1020"]
SOTS_MEMBERSHIP = ["1790", "1924", "0292"]
SOTS_HERO = ["1037", "1790", "1924"]
SOTS_PROFILE = "1924"
OHAZE_GRID = ["30_outdoor", "02_outdoor", "32_outdoor"]
CHART_FIGS = {1, 6, 7, 8}


def _load_pairs(datasets_cfg: dict, name: str):
    info = datasets_cfg[name]
    return pair_dataset(
        hazy_dir=info["hazy_dir"],
        clear_dir=info["clear_dir"],
        id_rule=info.get("id_rule", "prefix_before_underscore"),
        sky_mask_dir=info.get("sky_mask_dir"),
        check_sizes=False,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--datasets-config", type=Path, default=Path("configs/datasets.yaml"))
    ap.add_argument("--ga-config", type=Path, default=Path("configs/ga.yaml"))
    ap.add_argument("--params", type=Path, default=Path("results/best_params.yaml"))
    ap.add_argument("--tau", type=Path, default=Path("results/best_tau.yaml"))
    ap.add_argument("--summary-dir", type=Path, default=Path("results/summary"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    ap.add_argument("--only", type=int, nargs="+", default=None, help="Figure numbers to generate (1-10)")
    ap.add_argument("--charts-only", action="store_true", help="Only CSV-based figures 1, 6, 7, 8 + tables")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    best_params = load_config(args.params)
    tau = float(load_config(args.tau)["tau"])
    ga_cfg = load_config(args.ga_config)
    out, sd = args.out_dir, args.summary_dir

    wanted = set(args.only) if args.only else set(range(1, 11))
    if args.charts_only:
        wanted &= CHART_FIGS

    sots = ohaze = None
    if wanted - CHART_FIGS:
        ds_cfg = load_config(args.datasets_config)
        sots = _load_pairs(ds_cfg, "sots_outdoor")
        if wanted & {9, 10}:
            ohaze = _load_pairs(ds_cfg, "ohaze")

    jobs = {
        1: lambda: F.plot_pipeline_diagram(out / "fig01_pipeline_diagram.png"),
        2: lambda: F.plot_dcp_failure(sots, cfg, best_params, out / "fig02_dcp_failure.png", SOTS_DCP_FAILURE),
        3: lambda: F.plot_membership_maps(sots, cfg, best_params, tau, out / "fig03_membership_maps.png", SOTS_MEMBERSHIP),
        4: lambda: F.plot_hero_crops(sots, cfg, best_params, tau, out / "fig04_hero_horizon_crops.png", SOTS_HERO),
        5: lambda: F.plot_transmission_profile(sots, cfg, best_params, tau, out / "fig05_transmission_profile.png", SOTS_PROFILE),
        6: lambda: F.plot_seam_sky_bars(sd / "sots_outdoor_summary.csv", out / "fig06_seam_sky_bars.png"),
        7: lambda: F.plot_ablation_chart(sd / "sots_outdoor_ablations.csv", out / "fig07_ablation_chart.png"),
        8: lambda: F.plot_ga_convergence(sd / "ga_history.csv", out / "fig08_ga_convergence.png"),
        9: lambda: F.plot_failure_cases(
            [
                (sots, "1869", "SOTS 1869: smooth soil/trunk taken as sky"),
                (sots, "0113", "SOTS 0113: dusk sky; F1 lifts lit buildings"),
                (sots, "0297", "SOTS 0297: no-sky guard fires (F1 = B0)"),
                (ohaze, "20_outdoor", "O-HAZE 20: dense smoke taken as sky"),
            ],
            cfg, best_params, out / "fig09_failure_cases.png",
        ),
        10: lambda: F.plot_ohaze_grid(ohaze, cfg, best_params, tau, out / "fig10_ohaze_grid.png", OHAZE_GRID),
    }

    failures = 0
    for n in sorted(wanted):
        try:
            print(f"[fig {n:02d}] -> {jobs[n]()}", flush=True)
        except Exception as exc:  # keep going so one bad figure does not block the rest
            failures += 1
            print(f"[fig {n:02d}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    sky = cfg.get("sky", {})
    w = sky.get("feature_weights", [1.0, 1.0, 1.0, 1.0])
    f0_defaults = {
        "m": sky.get("fuzzifier", 2.0), "w_f1": w[0], "w_f2": w[1], "w_f3": w[2], "w_f4": w[3],
        "a": sky.get("smoothstep", {}).get("a", 0.2), "b": sky.get("smoothstep", {}).get("b", 0.8),
        "r_mu": sky.get("mu_gf", {}).get("r", 20), "t_floor": sky.get("t_floor", 0.8),
        "K": sky.get("tolerance_K", 0.08),
    }
    for p in F.write_extra_tables(sd, best_params, ga_cfg.get("bounds", {}), f0_defaults):
        print(f"[table] -> {p}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
