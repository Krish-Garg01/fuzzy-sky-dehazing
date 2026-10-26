"""Resumable evaluation runner across datasets, baselines, and ablation variants."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

from fsdehaze.config import load_config
from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.io_utils import Pair, load_image, load_mask, save_image
from fsdehaze.metrics import evaluate_all, prepare_gt_cache
from fsdehaze.pipelines import (
    Result,
    _blend_and_refine_transmission,
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)
from fsdehaze.sky_features import compute_features
from fsdehaze.sky_membership import estimate_sky_membership, smoothstep

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

ABLATION_METHODS: tuple[str, ...] = (
    "A3_no_f4",
    "A4_no_f3",
    "A5a_with_f5",
    "A5b_with_f6",
    "A6a_c2",
    "A6b_c4",
    "A7_tolerance",
    "A8_no_mu_gf",
    "A9a_m1.5",
    "A9b_m3.0",
    "A10_gf_before_blend",
)


def _resolve_f1_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is not None:
        return params
    default_path = Path("results/best_params.yaml")
    if default_path.is_file():
        return load_config(default_path)
    return None


def _slice_features(
    all_feats: tuple[np.ndarray, list[str]],
    wanted_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    feat_map, names = all_feats
    idxs = [names.index(n) for n in wanted_names]
    return feat_map[:, :, idxs], list(wanted_names)


def dispatch_method(
    method: str,
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    tau: float | None = None,
    precomputed: dict[str, Any] | None = None,
) -> Result:
    """Run a main method (`B0..F1`) or ablation (`A3..A10`) on a single RGB float32 [0, 1] image."""
    m = method.strip()
    m_up = m.upper()
    cache = dict(precomputed or {})

    if m_up in ("B0", "DCP"):
        return dehaze_dcp(img, cfg, precomputed=cache)

    if m_up in ("B1", "HARD_THRESHOLD"):
        return dehaze_hard_threshold(img, cfg, tau=tau, precomputed=cache)

    if m_up in ("B2", "HARD_FCM"):
        if "f0_membership" in cache:
            _, f0_info = cache["f0_membership"]
            t0 = float(cfg.get("dcp", {}).get("t0", 0.1))
            if not f0_info["sky_present"]:
                mask = np.zeros(img.shape[:2], dtype=bool)
            else:
                mask = np.asarray(f0_info["mu_raw"] > 0.5, dtype=bool)
            mask_f = mask.astype(np.float32)
            t_final, t_blend, t_sky = _blend_and_refine_transmission(
                img, cache["A"], cache["t_dcp"], mask_f, cfg, params=None
            )
            J, J_raw = recover(img, cache["A"], t_final, t0=t0)
            return Result(
                J=J,
                J_raw=J_raw,
                t=t_final,
                A=cache["A"],
                mu=None,
                mask=mask,
                info={
                    **f0_info,
                    "method": "B2",
                    "dark": cache["dark"],
                    "t_dcp": cache["t_dcp"],
                    "t_sky": t_sky,
                    "t_blend": t_blend,
                },
            )
        return dehaze_hard_fcm(img, cfg, params=None, precomputed=cache)

    if m_up in ("F0", "FUZZY"):
        if "f0_membership" in cache:
            f0_mu, f0_info = cache["f0_membership"]
            res = dehaze_fuzzy(img, cfg, params={"force_mu": f0_mu}, precomputed=cache)
            res.info["sky_present"] = f0_info["sky_present"]
            res.info["method"] = "F0"
            return res
        return dehaze_fuzzy(img, cfg, params=None, precomputed=cache)

    if m_up == "F1":
        return dehaze_fuzzy(
            img, cfg, params=_resolve_f1_params(params), precomputed=cache
        )

    # Ablation variants (Section 8.2)
    all_feats = cache.get("all_features")
    if m == "A3_no_f4":
        sub_cache = dict(cache)
        if all_feats is not None:
            sub_cache["features"] = _slice_features(
                all_feats, ["brightness", "low_saturation", "texture"]
            )
            cfg_sub = cfg
        else:
            cfg_sub = {
                **cfg,
                "sky": {
                    **cfg.get("sky", {}),
                    "features": ["brightness", "low_saturation", "texture"],
                },
            }
        return dehaze_fuzzy(
            img,
            cfg_sub,
            params={"feature_weights": [1.0, 1.0, 1.0], "fcm_samples": 8000},
            precomputed=sub_cache,
        )

    if m == "A4_no_f3":
        sub_cache = dict(cache)
        if all_feats is not None:
            sub_cache["features"] = _slice_features(
                all_feats, ["brightness", "low_saturation", "vertical_pos"]
            )
            cfg_sub = cfg
        else:
            cfg_sub = {
                **cfg,
                "sky": {
                    **cfg.get("sky", {}),
                    "features": ["brightness", "low_saturation", "vertical_pos"],
                },
            }
        return dehaze_fuzzy(
            img,
            cfg_sub,
            params={"feature_weights": [1.0, 1.0, 1.0], "fcm_samples": 8000},
            precomputed=sub_cache,
        )

    if m == "A5a_with_f5":
        sub_cache = dict(cache)
        wanted = ["brightness", "low_saturation", "texture", "vertical_pos", "dark_channel"]
        if all_feats is not None:
            sub_cache["features"] = _slice_features(all_feats, wanted)
            cfg_sub = cfg
        else:
            cfg_sub = {**cfg, "sky": {**cfg.get("sky", {}), "features": wanted}}
        return dehaze_fuzzy(
            img,
            cfg_sub,
            params={"feature_weights": [1.0, 1.0, 1.0, 1.0, 1.0], "fcm_samples": 8000},
            precomputed=sub_cache,
        )

    if m == "A5b_with_f6":
        sub_cache = dict(cache)
        wanted = ["brightness", "low_saturation", "texture", "vertical_pos", "blueness"]
        if all_feats is not None:
            sub_cache["features"] = _slice_features(all_feats, wanted)
            cfg_sub = cfg
        else:
            cfg_sub = {**cfg, "sky": {**cfg.get("sky", {}), "features": wanted}}
        return dehaze_fuzzy(
            img,
            cfg_sub,
            params={"feature_weights": [1.0, 1.0, 1.0, 1.0, 1.0], "fcm_samples": 8000},
            precomputed=sub_cache,
        )

    if m == "A6a_c2":
        return dehaze_fuzzy(img, cfg, params={"clusters": 2, "fcm_samples": 8000}, precomputed=cache)

    if m == "A6b_c4":
        return dehaze_fuzzy(img, cfg, params={"clusters": 4, "fcm_samples": 8000}, precomputed=cache)

    if m == "A7_tolerance":
        if "f0_membership" in cache:
            f0_mu, f0_info = cache["f0_membership"]
            res = dehaze_fuzzy(
                img,
                cfg,
                params={"force_mu": f0_mu, "t_sky_mode": "tolerance", "K": 0.08},
                precomputed=cache,
            )
            res.info["sky_present"] = f0_info["sky_present"]
            return res
        return dehaze_fuzzy(
            img, cfg, params={"t_sky_mode": "tolerance", "K": 0.08}, precomputed=cache
        )

    if m == "A8_no_mu_gf":
        if "f0_membership" in cache:
            _, f0_info = cache["f0_membership"]
            if not f0_info["sky_present"]:
                mu_no_gf = np.zeros(img.shape[:2], dtype=np.float32)
            else:
                mu_no_gf = smoothstep(f0_info["mu_raw"], a=0.2, b=0.8)
            res = dehaze_fuzzy(
                img, cfg, params={"force_mu": mu_no_gf}, precomputed=cache
            )
            res.info["sky_present"] = f0_info["sky_present"]
            return res
        return dehaze_fuzzy(img, cfg, params={"skip_mu_gf": True}, precomputed=cache)

    if m == "A9a_m1.5":
        return dehaze_fuzzy(img, cfg, params={"m": 1.5, "fcm_samples": 8000}, precomputed=cache)

    if m == "A9b_m3.0":
        return dehaze_fuzzy(img, cfg, params={"m": 3.0, "fcm_samples": 8000}, precomputed=cache)

    if m == "A10_gf_before_blend":
        if "f0_membership" in cache:
            f0_mu, f0_info = cache["f0_membership"]
            res = dehaze_fuzzy(
                img,
                cfg,
                params={"force_mu": f0_mu, "gf_order": "before_blend"},
                precomputed=cache,
            )
            res.info["sky_present"] = f0_info["sky_present"]
            return res
        return dehaze_fuzzy(
            img, cfg, params={"gf_order": "before_blend"}, precomputed=cache
        )

    raise ValueError(f"Unknown method identifier: {method!r}")


def _process_single_pair(
    idx: int,
    pair: Pair,
    pending_methods: list[str],
    cfg: dict[str, Any],
    dataset_name: str,
    params: dict[str, Any] | None,
    tau: float | None,
    save_outputs: bool,
    save_preview_count: int,
    preview_dirs: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Evaluate all `pending_methods` on a single `pair` and return `{method: row_dict}`."""
    img_id = str(pair.image_id)
    seed = int(cfg.get("seed", 42))
    seam_width = int(cfg.get("eval", {}).get("seam_band_px", 10))
    dcp_cfg = cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    f0_sharing_set = {
        "B2",
        "HARD_FCM",
        "F0",
        "FUZZY",
        "A7_tolerance",
        "A8_no_mu_gf",
        "A10_gf_before_blend",
    }
    extra_feat_set = {"A3_no_f4", "A4_no_f3", "A5a_with_f5", "A5b_with_f6"}

    hazy = load_image(pair.hazy_path)
    clear = load_image(pair.clear_path)
    sky_mask = (
        load_mask(pair.sky_mask_path)
        if pair.sky_mask_path is not None and pair.sky_mask_path.is_file()
        else None
    )

    gt_cache = prepare_gt_cache(clear, sky_mask=sky_mask, seam_width=seam_width)

    # 1. Shared DCP backbone (measured via thread CPU clock for multi-thread accuracy)
    t0_dcp = time.thread_time()
    dark = dark_channel(hazy, patch=patch)
    A = estimate_atmospheric_light(hazy, dark, top_frac=top_frac)
    t_dcp = transmission_dcp(hazy, A, omega=omega, patch=patch)
    dcp_elapsed = max(time.thread_time() - t0_dcp, 1e-4)

    # 2. Shared feature maps
    needs_extra = any(m in extra_feat_set for m in pending_methods)
    needs_sky_feats = any(m.upper() not in ("B0", "DCP") for m in pending_methods)
    if needs_extra:
        t0_feat = time.thread_time()
        cfg_all6 = {
            **cfg,
            "sky": {
                **cfg.get("sky", {}),
                "features": [
                    "brightness",
                    "low_saturation",
                    "texture",
                    "vertical_pos",
                    "dark_channel",
                    "blueness",
                ],
            },
        }
        all_feats = compute_features(hazy, cfg_all6)
        feats_4 = _slice_features(
            all_feats, ["brightness", "low_saturation", "texture", "vertical_pos"]
        )
        feat_elapsed = max((time.thread_time() - t0_feat) * (4.0 / 6.0), 1e-4)
    elif needs_sky_feats:
        t0_feat = time.thread_time()
        feats_4 = compute_features(hazy, cfg)
        all_feats = None
        feat_elapsed = max(time.thread_time() - t0_feat, 1e-4)
    else:
        feats_4 = None
        all_feats = None
        feat_elapsed = 0.0

    # 3. Shared default FCM membership
    f0_user_count = sum(
        1 for m in pending_methods if (m in f0_sharing_set or m.upper() in f0_sharing_set)
    )
    if f0_user_count >= 1 and feats_4 is not None:
        t0_f0_mem = time.thread_time()
        f0_mem = estimate_sky_membership(
            hazy, cfg, params=None, precomputed_features=feats_4
        )
        f0_mem_elapsed = max(time.thread_time() - t0_f0_mem, 1e-4)
    else:
        f0_mem = None
        f0_mem_elapsed = 0.0

    precomputed: dict[str, Any] = {
        "dark": dark,
        "A": A,
        "t_dcp": t_dcp,
        "features": feats_4,
        "all_features": all_feats,
    }
    if f0_mem is not None:
        precomputed["f0_membership"] = f0_mem

    out_rows: dict[str, dict[str, Any]] = {}
    for method in pending_methods:
        t_start = time.thread_time()
        res = dispatch_method(
            method, hazy, cfg, params=params, tau=tau, precomputed=precomputed
        )
        method_elapsed = max(time.thread_time() - t_start, 1e-4)

        if method.upper() in ("B0", "DCP"):
            total_runtime = dcp_elapsed + method_elapsed
        elif (
            method in f0_sharing_set or method.upper() in f0_sharing_set
        ) and f0_mem is not None:
            total_runtime = dcp_elapsed + feat_elapsed + f0_mem_elapsed + method_elapsed
        else:
            total_runtime = dcp_elapsed + feat_elapsed + method_elapsed

        metric_dict = evaluate_all(
            clear, res, sky_mask=sky_mask, seam_width=seam_width, gt_cache=gt_cache
        )

        if save_outputs or idx < save_preview_count:
            p_dir = preview_dirs[method]
            p_dir.mkdir(parents=True, exist_ok=True)
            save_image(p_dir / f"{pair.image_id}_dehazed.png", res.J)
            panel = np.concatenate([hazy, res.J, clear], axis=1)
            save_image(p_dir / f"{pair.image_id}_compare_hazy_dehazed_clear.png", panel)
            save_image(p_dir / f"{pair.image_id}_t_final.png", res.t)
            if res.mu is not None:
                save_image(p_dir / f"{pair.image_id}_mu.png", res.mu)
            elif res.mask is not None:
                save_image(p_dir / f"{pair.image_id}_mask.png", res.mask.astype(np.float32))

        out_rows[method] = {
            "dataset": dataset_name,
            "method": method,
            "image_id": img_id,
            "hazy_file": pair.hazy_path.name,
            "clear_file": pair.clear_path.name,
            **metric_dict,
            "A_r": float(res.A[0]),
            "A_g": float(res.A[1]),
            "A_b": float(res.A[2]),
            "runtime_s": total_runtime,
            "seed": seed,
        }

    return out_rows


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
    n_jobs: int = -1,
    chunk_size: int = 25,
) -> dict[str, pd.DataFrame]:
    """Run a list of methods on paired images in parallel chunks, saving resumable CSVs."""
    out_root = Path(out_dir)
    per_image_dir = out_root / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: dict[str, Path] = {}
    preview_dirs: dict[str, Path] = {}
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    done_by_method: dict[str, set[str]] = {}

    for method in methods:
        csv_path = per_image_dir / f"{dataset_name}_{method}.csv"
        preview_dir = out_root / "images" / dataset_name / method
        if save_outputs or save_preview_count > 0:
            preview_dir.mkdir(parents=True, exist_ok=True)

        csv_paths[method] = csv_path
        preview_dirs[method] = preview_dir
        rows_by_method[method] = []
        done_by_method[method] = set()

        if resume and csv_path.is_file():
            try:
                prev_df = pd.read_csv(csv_path, dtype={"image_id": str})
                if REQUIRED_METRIC_COLS.issubset(set(prev_df.columns)):
                    rows_by_method[method] = prev_df.to_dict(orient="records")
                    done_by_method[method] = {
                        str(r["image_id"]) for r in rows_by_method[method]
                    }
            except Exception:
                rows_by_method[method] = []
                done_by_method[method] = set()

    work_items: list[tuple[int, Pair, list[str]]] = []
    for idx, pair in enumerate(pairs):
        img_id = str(pair.image_id)
        pending = [m for m in methods if img_id not in done_by_method[m]]
        if pending:
            work_items.append((idx, pair, pending))

    if work_items:
        resolved_params = _resolve_f1_params(params)
        for start in tqdm(
            range(0, len(work_items), chunk_size),
            desc=f"{dataset_name} ({len(methods)} methods)",
            leave=False,
        ):
            batch = work_items[start : start + chunk_size]
            batch_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_single_pair)(
                    idx,
                    pair,
                    pending,
                    cfg,
                    dataset_name,
                    resolved_params,
                    tau,
                    save_outputs,
                    save_preview_count,
                    preview_dirs,
                )
                for idx, pair, pending in batch
            )
            for pair_out in batch_results:
                for m, row in pair_out.items():
                    rows_by_method[m].append(row)

            # Flush all modified method CSVs after every batch
            for m in methods:
                if rows_by_method[m]:
                    pd.DataFrame(rows_by_method[m]).to_csv(csv_paths[m], index=False)

    results_by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        df = pd.DataFrame(rows_by_method[method])
        if not df.empty and "image_id" in df.columns:
            df["image_id"] = df["image_id"].astype(str)
            df = df.sort_values("image_id").reset_index(drop=True)
        df.to_csv(csv_paths[method], index=False)
        results_by_method[method] = df

    return results_by_method
