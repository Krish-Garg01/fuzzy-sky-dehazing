"""Soft sky membership estimation via Fuzzy C-Means, no-sky guard, smoothstep, and guided filtering."""

from __future__ import annotations

from typing import Any

import numpy as np

from fsdehaze.fcm import FuzzyCMeans
from fsdehaze.guided_filter import guided_filter, rgb_to_gray
from fsdehaze.sky_features import compute_features


def smoothstep(x: np.ndarray, a: float = 0.2, b: float = 0.8) -> np.ndarray:
    """Apply cubic Hermite smoothstep mapping from [a, b] to [0, 1].

    Formula:
        s = clip((x - a) / (b - a), 0, 1)
        smoothstep(x) = s^2 * (3 - 2*s)

    Args:
        x: Input array of arbitrary shape, dtype float32/float64.
        a: Lower edge (values <= a map to 0).
        b: Upper edge (values >= b map to 1; requires b > a).

    Returns:
        Smoothly remapped array of same shape as `x`, dtype float32, in [0, 1].
    """
    span = max(float(b) - float(a), 1e-6)
    s = np.clip((x.astype(np.float32, copy=False) - float(a)) / span, 0.0, 1.0)
    return (s * s * (3.0 - 2.0 * s)).astype(np.float32)


def _resolve_sky_params(
    cfg: dict[str, Any],
    params: dict[str, Any] | None,
    n_features: int,
) -> dict[str, Any]:
    """Merge default `cfg['sky']` settings with optional GA/ablation `params` overrides."""
    sky_cfg = dict(cfg.get("sky", {}))
    p = params or {}

    clusters = int(p.get("clusters", sky_cfg.get("clusters", 3)))
    fuzzifier = float(p.get("m", p.get("fuzzifier", sky_cfg.get("fuzzifier", 2.0))))
    fcm_samples = int(p.get("fcm_samples", sky_cfg.get("fcm_samples", 20000)))
    fcm_max_iter = int(p.get("fcm_max_iter", sky_cfg.get("fcm_max_iter", 100)))
    fcm_tol = float(p.get("fcm_tol", sky_cfg.get("fcm_tol", 1.0e-4)))

    # Feature weights: support either list `feature_weights` or individual genes `w_f1..w_f4`
    default_weights = list(sky_cfg.get("feature_weights", [1.0] * n_features))
    if len(default_weights) < n_features:
        default_weights.extend([1.0] * (n_features - len(default_weights)))
    weights = [float(w) for w in p.get("feature_weights", default_weights[:n_features])]
    for idx in range(min(4, n_features)):
        gene_key = f"w_f{idx + 1}"
        if gene_key in p:
            weights[idx] = float(p[gene_key])

    # Avoid all-zero weights degenerating FCM
    if sum(abs(w) for w in weights) < 1e-6:
        weights = [1.0] * n_features

    ss_cfg = sky_cfg.get("smoothstep", {})
    a = float(p.get("a", ss_cfg.get("a", 0.2)))
    b = float(p.get("b", ss_cfg.get("b", 0.8)))
    if b - a < 0.05:
        b = min(1.0, a + 0.1)

    mu_gf_cfg = sky_cfg.get("mu_gf", {})
    r_mu = int(round(float(p.get("r_mu", mu_gf_cfg.get("r", 20)))))
    eps_mu = float(p.get("eps_mu", mu_gf_cfg.get("eps", 1.0e-3)))
    skip_mu_gf = bool(p.get("skip_mu_gf", sky_cfg.get("skip_mu_gf", False)))

    guard_cfg = sky_cfg.get("guard", {})
    min_brightness = float(p.get("min_brightness", guard_cfg.get("min_brightness", 0.5)))
    min_area_frac = float(p.get("min_area_frac", guard_cfg.get("min_area_frac", 0.01)))
    max_mean_y = float(p.get("max_mean_y", guard_cfg.get("max_mean_y", 0.6)))

    return {
        "clusters": clusters,
        "fuzzifier": fuzzifier,
        "fcm_samples": fcm_samples,
        "fcm_max_iter": fcm_max_iter,
        "fcm_tol": fcm_tol,
        "weights": np.asarray(weights, dtype=np.float64),
        "a": a,
        "b": b,
        "r_mu": r_mu,
        "eps_mu": eps_mu,
        "skip_mu_gf": skip_mu_gf,
        "min_brightness": min_brightness,
        "min_area_frac": min_area_frac,
        "max_mean_y": max_mean_y,
    }


def _score_clusters(
    centers_unscaled: np.ndarray,
    feature_names: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute sky score for each cluster centre in [0, 1] feature coordinates.

    Matches Section 4.4:
        sky_score_k = f1_k + f2_k + (1 - f3_k) + f4_k
    weighted by active feature weights so zero-weighted features do not dominate.
    """
    c = centers_unscaled.shape[0]
    scores = np.zeros(c, dtype=np.float64)
    for j, name in enumerate(feature_names):
        w = float(weights[j]) if j < len(weights) else 1.0
        col = centers_unscaled[:, j]
        if name == "texture":
            scores += w * (1.0 - col)
        else:
            scores += w * col
    return scores


def estimate_sky_membership(
    img: np.ndarray,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    precomputed_features: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate soft per-pixel sky membership mu(x) in [0, 1].

    Steps:
        1. Extract normalised per-pixel sky features (or reuse `precomputed_features`).
        2. Scale features by `feature_weights` and subsample up to `fcm_samples` pixels (seeded).
        3. Fit `FuzzyCMeans` on the subsample and predict memberships across all pixels.
        4. Select the sky cluster via `sky_score_k` and evaluate the no-sky guard.
        5. If sky is present, apply `smoothstep(a, b)` and edge-aware guided filtering.

    Args:
        img: Hazy RGB image of shape (H, W, 3), dtype float32, in [0, 1].
        cfg: Base configuration dictionary.
        params: Optional dictionary overriding fuzzy/membership parameters (e.g. from GA).
        precomputed_features: Optional `(features, names)` tuple to avoid recomputing features.

    Returns:
        Tuple `(mu, info)` where:
            - `mu` has shape `(H, W)`, dtype float32, values in `[0, 1]`.
            - `info` contains `sky_present`, `sky_cluster`, `centers`, `mu_raw`, `n_iter`, etc.
    """
    H, W = img.shape[:2]
    seed = int(cfg.get("seed", 42))

    if precomputed_features is not None:
        feat_map, feat_names = precomputed_features
    else:
        feat_map, feat_names = compute_features(img, cfg)

    n_features = len(feat_names)
    resolved = _resolve_sky_params(cfg, params, n_features=n_features)
    weights = resolved["weights"]

    X_unscaled = feat_map.reshape(-1, n_features).astype(np.float64)
    X_weighted = X_unscaled * weights[np.newaxis, :]
    N_total = X_weighted.shape[0]

    n_sub = min(N_total, resolved["fcm_samples"])
    rng = np.random.default_rng(seed)
    if n_sub < N_total:
        sub_idx = rng.choice(N_total, size=n_sub, replace=False)
        X_sub = X_weighted[sub_idx]
        X_sub_unscaled = X_unscaled[sub_idx]
    else:
        X_sub = X_weighted
        X_sub_unscaled = X_unscaled

    fcm = FuzzyCMeans(
        c=resolved["clusters"],
        m=resolved["fuzzifier"],
        tol=resolved["fcm_tol"],
        max_iter=resolved["fcm_max_iter"],
        seed=seed,
    )
    fcm.fit(X_sub)

    # Compute unscaled [0, 1] centre coordinates for interpretable scoring & guard checks
    U_sub = fcm.predict_membership(X_sub)
    um_sub = U_sub ** resolved["fuzzifier"]
    denom_sub = np.maximum(np.sum(um_sub, axis=0)[:, np.newaxis], 1e-12)
    centers_unscaled = (um_sub.T @ X_sub_unscaled) / denom_sub

    scores = _score_clusters(centers_unscaled, feat_names, weights)
    sky_cluster = int(np.argmax(scores))

    U_all = fcm.predict_membership(X_weighted)
    mu_raw = U_all[:, sky_cluster].reshape(H, W).astype(np.float32)

    # No-sky guard (Section 4.5)
    hard_labels = np.argmax(U_all, axis=1).reshape(H, W)
    sky_hard_mask = hard_labels == sky_cluster
    area_frac = float(np.mean(sky_hard_mask))

    raw_v = np.max(img, axis=2)
    raw_cluster_brightness = float(np.mean(raw_v[sky_hard_mask])) if area_frac > 0.0 else 0.0
    if "brightness" in feat_names:
        b_idx = feat_names.index("brightness")
        norm_center_brightness = float(centers_unscaled[sky_cluster, b_idx])
    else:
        norm_center_brightness = raw_cluster_brightness

    y_grid = np.broadcast_to(
        np.linspace(0.0, 1.0, H, dtype=np.float32)[:, np.newaxis],
        (H, W),
    )
    mean_y = float(np.mean(y_grid[sky_hard_mask])) if area_frac > 0.0 else 1.0

    guard_triggered = (
        norm_center_brightness < resolved["min_brightness"]
        or raw_cluster_brightness < resolved["min_brightness"]
        or area_frac < resolved["min_area_frac"]
        or mean_y > resolved["max_mean_y"]
    )

    if guard_triggered:
        sky_present = False
        mu = np.zeros((H, W), dtype=np.float32)
    else:
        sky_present = True
        mu_s = smoothstep(mu_raw, a=resolved["a"], b=resolved["b"])
        if not resolved["skip_mu_gf"] and resolved["r_mu"] >= 1:
            gray = rgb_to_gray(img)
            mu_filt = guided_filter(gray, mu_s, r=resolved["r_mu"], eps=resolved["eps_mu"])
            mu = np.clip(mu_filt, 0.0, 1.0).astype(np.float32)
        else:
            mu = mu_s

    info: dict[str, Any] = {
        "sky_present": sky_present,
        "sky_cluster": sky_cluster,
        "centers": fcm.centers_.astype(np.float32) if fcm.centers_ is not None else None,
        "centers_unscaled": centers_unscaled.astype(np.float32),
        "cluster_scores": scores.astype(np.float32),
        "mu_raw": mu_raw,
        "n_iter": fcm.n_iter_,
        "objective": fcm.objective_,
        "area_frac": area_frac,
        "cluster_brightness": raw_cluster_brightness,
        "norm_center_brightness": norm_center_brightness,
        "mean_y": mean_y,
    }
    return mu, info
