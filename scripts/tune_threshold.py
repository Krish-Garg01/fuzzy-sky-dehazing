"""Grid-search tuning of the hard sky-score threshold `tau` for Method B1 on `ots_subset`.

Uses the exact same tuning split and fitness objective as the GA (`configs/ga.yaml`):
    F = 0.5 * PSNR_full + 0.5 * PSNR_sky - lambda_sat * saturated_fraction_sky
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fsdehaze.baselines.hard_mask import compute_sky_score_map
from fsdehaze.config import load_config
from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.guided_filter import guided_filter, rgb_to_gray
from fsdehaze.io_utils import Pair, load_image, load_mask, pair_dataset
from fsdehaze.metrics import masked_psnr, psnr, saturated_fraction
from fsdehaze.sky_transmission import compute_t_sky


@dataclass
class CachedB1Sample:
    """Precomputed `tau`-independent intermediates for one tuning image."""

    image_id: str
    img: np.ndarray
    gray: np.ndarray
    clear: np.ndarray
    sky_mask: np.ndarray
    A: np.ndarray
    t_dcp: np.ndarray
    t_sky: np.ndarray
    sky_score: np.ndarray
    raw_v: np.ndarray
    y_grid: np.ndarray


def _prepare_sample(pair: Pair, cfg: dict[str, Any]) -> CachedB1Sample:
    img = load_image(pair.hazy_path)
    clear = load_image(pair.clear_path)
    H, W = img.shape[:2]
    if pair.sky_mask_path is not None and pair.sky_mask_path.is_file():
        sky_mask = load_mask(pair.sky_mask_path)
    else:
        sky_mask = np.zeros((H, W), dtype=bool)

    dcp_cfg = cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    dark = dark_channel(img, patch=patch)
    A = estimate_atmospheric_light(img, dark, top_frac=top_frac)
    t_dcp = transmission_dcp(img, A, omega=omega, patch=patch)
    t_sky = compute_t_sky(img, A, t_dcp, cfg)
    sky_score, _ = compute_sky_score_map(img, cfg)

    gray = rgb_to_gray(img)
    raw_v = np.max(img, axis=2)
    y_grid = np.broadcast_to(
        np.linspace(0.0, 1.0, H, dtype=np.float32)[:, np.newaxis],
        (H, W),
    )
    return CachedB1Sample(
        image_id=pair.image_id,
        img=img,
        gray=gray,
        clear=clear,
        sky_mask=sky_mask,
        A=A,
        t_dcp=t_dcp,
        t_sky=t_sky,
        sky_score=sky_score,
        raw_v=raw_v,
        y_grid=y_grid,
    )


def _evaluate_tau_on_samples(
    tau: float,
    samples: list[CachedB1Sample],
    cfg: dict[str, Any],
    lambda_sat: float,
) -> dict[str, float]:
    dcp_cfg = cfg.get("dcp", {})
    t0 = float(dcp_cfg.get("t0", 0.1))
    gf_cfg = cfg.get("guided_filter", {})
    gf_r = int(gf_cfg.get("r", 60))
    gf_eps = float(gf_cfg.get("eps", 1.0e-4))

    guard_cfg = cfg.get("sky", {}).get("guard", {})
    min_brightness = float(guard_cfg.get("min_brightness", 0.5))
    min_area_frac = float(guard_cfg.get("min_area_frac", 0.01))
    max_mean_y = float(guard_cfg.get("max_mean_y", 0.6))

    fitness_vals: list[float] = []
    psnr_vals: list[float] = []
    sky_psnr_vals: list[float] = []
    sat_sky_vals: list[float] = []
    sky_iou_vals: list[float] = []

    for s in samples:
        raw_mask = s.sky_score > float(tau)
        area_frac = float(np.mean(raw_mask))
        mean_b = float(np.mean(s.raw_v[raw_mask])) if area_frac > 0.0 else 0.0
        mean_y = float(np.mean(s.y_grid[raw_mask])) if area_frac > 0.0 else 1.0

        if mean_b < min_brightness or area_frac < min_area_frac or mean_y > max_mean_y:
            mask_b = np.zeros_like( raw_mask, dtype=bool)
            mask_f = np.zeros_like(s.t_dcp, dtype=np.float32)
        else:
            mask_b = raw_mask
            mask_f = raw_mask.astype(np.float32)

        t_blend = (mask_f * s.t_sky + (1.0 - mask_f) * s.t_dcp).astype(np.float32)
        t_final = np.clip(guided_filter(s.gray, t_blend, r=gf_r, eps=gf_eps), 0.0, 1.0).astype(
            np.float32
        )
        J, J_raw = recover(s.img, s.A, t_final, t0=t0)

        p_full = psnr(s.clear, J)
        if np.any(s.sky_mask):
            p_sky = masked_psnr(s.clear, J, s.sky_mask)
            sat_sky = saturated_fraction(J_raw, s.sky_mask)
            inter = int(np.sum(mask_b & s.sky_mask))
            union = int(np.sum(mask_b | s.sky_mask))
            sky_iou_vals.append(float(inter) / float(max(union, 1)))
        else:
            p_sky = p_full
            sat_sky = saturated_fraction(J_raw, None)

        fit = 0.5 * p_full + 0.5 * p_sky - float(lambda_sat) * sat_sky
        fitness_vals.append(fit)
        psnr_vals.append(p_full)
        sky_psnr_vals.append(p_sky)
        sat_sky_vals.append(sat_sky)

    return {
        "fitness": float(np.mean(fitness_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "sky_psnr": float(np.mean(sky_psnr_vals)),
        "sat_frac_sky": float(np.mean(sat_sky_vals)),
        "sky_iou": float(np.mean(sky_iou_vals)) if sky_iou_vals else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grid search hard-mask threshold tau for Method B1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to default config YAML",
    )
    parser.add_argument(
        "--ga-config",
        type=Path,
        default=Path("configs/ga.yaml"),
        help="Path to GA config YAML (for lambda_sat, val_fraction, seed)",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets YAML",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ots_subset",
        help="Tuning dataset name in datasets.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/best_tau.yaml"),
        help="Output YAML path for best tau",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ga_cfg = load_config(args.ga_config)
    ds_cfg = load_config(args.datasets_config)[args.dataset]

    seed = int(ga_cfg.get("seed", cfg.get("seed", 42)))
    val_fraction = float(ga_cfg.get("val_fraction", 0.2))
    lambda_sat = float(ga_cfg.get("lambda_sat", 20.0))

    pairs = pair_dataset(
        hazy_dir=ds_cfg["hazy_dir"],
        clear_dir=ds_cfg["clear_dir"],
        id_rule=ds_cfg["id_rule"],
        sky_mask_dir=ds_cfg.get("sky_mask_dir"),
        check_sizes=False,
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    val_idx = set(indices[:n_val].tolist())

    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]

    print(
        f"Precomputing B1 features on {args.dataset}: {len(train_pairs)} train, {len(val_pairs)} val (seed={seed})..."
    )
    train_samples = [_prepare_sample(p, cfg) for p in tqdm(train_pairs, desc="Cache train", leave=False)]
    val_samples = [_prepare_sample(p, cfg) for p in tqdm(val_pairs, desc="Cache val", leave=False)]

    # Grid search tau in [0.65, 0.85] where sky_score > tau isolates sky without flooding foreground
    grid = np.round(np.linspace(0.65, 0.85, 17), 4)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for tau in grid:
        tr = _evaluate_tau_on_samples(float(tau), train_samples, cfg, lambda_sat=lambda_sat)
        va = _evaluate_tau_on_samples(float(tau), val_samples, cfg, lambda_sat=lambda_sat)
        row = {
            "tau": float(tau),
            "train_fitness": tr["fitness"],
            "train_psnr": tr["psnr"],
            "train_sky_psnr": tr["sky_psnr"],
            "train_sat_pct_sky": tr["sat_frac_sky"] * 100.0,
            "train_sky_iou": tr["sky_iou"],
            "val_fitness": va["fitness"],
            "val_psnr": va["psnr"],
            "val_sky_psnr": va["sky_psnr"],
            "val_sat_pct_sky": va["sat_frac_sky"] * 100.0,
            "val_sky_iou": va["sky_iou"],
        }
        rows.append(row)
        if best_row is None or row["train_fitness"] > best_row["train_fitness"]:
            best_row = row

    assert best_row is not None
    df = pd.DataFrame(rows)
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = summary_dir / "tune_threshold_grid.csv"
    df.to_csv(grid_csv, index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau": float(best_row["tau"]),
        "dataset": args.dataset,
        "seed": seed,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "lambda_sat": lambda_sat,
        "train_fitness": float(best_row["train_fitness"]),
        "train_psnr": float(best_row["train_psnr"]),
        "train_sky_psnr": float(best_row["train_sky_psnr"]),
        "val_fitness": float(best_row["val_fitness"]),
        "val_psnr": float(best_row["val_psnr"]),
        "val_sky_psnr": float(best_row["val_sky_psnr"]),
    }
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"\nBest B1 threshold tau = {best_row['tau']:.4f} saved to {args.out}")
    print(
        f"  Train (N={len(train_samples)}): Fitness={best_row['train_fitness']:.3f}, "
        f"PSNR={best_row['train_psnr']:.2f} dB, SkyPSNR={best_row['train_sky_psnr']:.2f} dB, "
        f"SkySat%={best_row['train_sat_pct_sky']:.2f}%"
    )
    print(
        f"  Val   (N={len(val_samples)}): Fitness={best_row['val_fitness']:.3f}, "
        f"PSNR={best_row['val_psnr']:.2f} dB, SkyPSNR={best_row['val_sky_psnr']:.2f} dB, "
        f"SkySat%={best_row['val_sat_pct_sky']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
