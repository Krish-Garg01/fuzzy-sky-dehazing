"""Publication and report figure generators (Phase 7).

Reproducible generators for the report figures and tables in SPEC.md Section 15:

  fig01_pipeline_diagram.png           Block diagram of the two-branch pipeline.
  fig02_dcp_failure.png                Hazy | dark channel | t_DCP | B0 | F1 | GT.
  fig03_membership_maps.png            GT sky mask vs B1 mask vs B2 mask vs F0 mu vs F1 mu.
  fig04_hero_horizon_crops.png         Full image with crop box + zoomed horizon crops per method.
  fig05_transmission_profile.png       1-D vertical transmission profile across the horizon.
  fig06_seam_sky_bars.png              Seam-error and sky-quality bars for B0, B1, B2, F0, F1.
  fig07_ablation_chart.png             Ablation chart (PSNR and seam gradient error).
  fig08_ga_convergence.png             GA best / mean fitness per generation + F0 vs F1.
  fig09_failure_cases.png              Failure cases and no-sky guard behaviour.
  fig10_ohaze_grid.png                 O-HAZE qualitative grid.

Image-based figures re-run the pipelines on a handful of selected images; chart figures
read only the CSVs under `results/`. Ground-truth sky masks are used for display and
metrics only, never as an input to any method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsdehaze.io_utils import Pair, load_image, load_mask  # noqa: E402
from fsdehaze.metrics import evaluate_all, prepare_gt_cache  # noqa: E402
from fsdehaze.pipelines import (  # noqa: E402
    Result,
    dehaze_dcp,
    dehaze_fuzzy,
    dehaze_hard_fcm,
    dehaze_hard_threshold,
)

MAIN_METHODS = ["B0", "B1", "B2", "F0", "F1"]
METHOD_LABELS = {
    "B0": "B0 (plain DCP)",
    "B1": "B1 (hard threshold)",
    "B2": "B2 (hard FCM)",
    "F0": "F0 (soft fuzzy)",
    "F1": "F1 (fuzzy + GA)",
}
METHOD_COLORS = {
    "B0": "#78909c",
    "B1": "#fb8c00",
    "B2": "#e53935",
    "F0": "#1e88e5",
    "F1": "#2e7d32",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_pair(pairs: Sequence[Pair], image_id: str) -> Pair:
    """Return the pair whose `image_id` equals `image_id` (raises KeyError if absent)."""
    for p in pairs:
        if str(p.image_id) == str(image_id):
            return p
    raise KeyError(f"Image id {image_id!r} not found among {len(pairs)} pairs")


def load_triplet(pair: Pair) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (hazy, clear, gt_sky_mask) for a pair; missing mask -> all-False."""
    hazy = load_image(pair.hazy_path)
    clear = load_image(pair.clear_path)
    if pair.sky_mask_path is not None and Path(pair.sky_mask_path).is_file():
        mask = load_mask(pair.sky_mask_path)
    else:
        mask = np.zeros(hazy.shape[:2], dtype=bool)
    return hazy, clear, mask


def run_methods_on_image(
    hazy: np.ndarray,
    cfg: dict[str, Any],
    best_params: dict[str, Any] | None,
    tau: float,
    methods: Sequence[str] = MAIN_METHODS,
) -> dict[str, Result]:
    """Run the requested methods on one image, all sharing the same DCP backbone."""
    runners: dict[str, Callable[[], Result]] = {
        "B0": lambda: dehaze_dcp(hazy, cfg),
        "B1": lambda: dehaze_hard_threshold(hazy, cfg, tau=tau),
        "B2": lambda: dehaze_hard_fcm(hazy, cfg),
        "F0": lambda: dehaze_fuzzy(hazy, cfg, params=None),
        "F1": lambda: dehaze_fuzzy(hazy, cfg, params=best_params),
    }
    return {m: runners[m]() for m in methods}


def score_results(
    clear: np.ndarray, mask: np.ndarray, results: dict[str, Result]
) -> dict[str, dict[str, Any]]:
    """Compute the metric suite for every result against the clear image."""
    cache = prepare_gt_cache(clear, mask, seam_width=10)
    return {m: evaluate_all(clear, r, mask, gt_cache=cache) for m, r in results.items()}


def weight_map(result: Result) -> np.ndarray:
    """Return the sky weight map used for blending (soft `mu` or binary `mask`)."""
    if result.mu is not None:
        return np.asarray(result.mu, dtype=np.float32)
    if result.mask is not None:
        return np.asarray(result.mask, dtype=np.float32)
    return np.zeros(result.J.shape[:2], dtype=np.float32)


def find_horizon_column(
    mask: np.ndarray, x_range: tuple[float, float] = (0.15, 0.85)
) -> tuple[int, int]:
    """Find a column with a single clean sky-above / ground-below transition.

    Returns:
        (x, y_horizon) where `y_horizon` is the first non-sky row below the sky in column x.
        Falls back to the image centre if the mask has no usable transition.
    """
    h, w = mask.shape
    best: tuple[float, int, int] | None = None
    for x in range(int(x_range[0] * w), int(x_range[1] * w), max(1, w // 60)):
        col = mask[:, x].astype(np.int8)
        changes = np.flatnonzero(np.diff(col) != 0)
        if changes.size != 1 or col[0] != 1:
            continue  # want exactly one sky->ground transition starting with sky at the top
        y = int(changes[0]) + 1
        # Prefer horizons near the middle of the image (room for a symmetric window)
        score = -abs(y - h / 2.0)
        if best is None or score > best[0]:
            best = (score, x, y)
    if best is None:
        return w // 2, h // 2
    return best[1], best[2]


def find_step_column(gt_mask: np.ndarray, hard: Result, tol_px: int = 15) -> tuple[int, int]:
    """Pick a scanline where the hard (B2) mask has one clean sky->ground step at the GT horizon.

    Among columns where both the GT mask and the hard mask switch exactly once from sky (top)
    to ground, with the two transitions within `tol_px` rows, choose the column with the largest
    jump in the hard method's blended transmission across the step. This is the column where a
    hard mask produces the most visible seam, i.e. where step vs ramp is easiest to see.
    Falls back to `find_horizon_column` if no such column exists.
    """
    hm = weight_map(hard) > 0.5
    t_blend = hard.info["t_blend"]
    h, w = gt_mask.shape
    best: tuple[float, int, int] | None = None
    for x in range(int(0.1 * w), int(0.9 * w)):
        tr = []
        for col in (gt_mask[:, x], hm[:, x]):
            c = col.astype(np.int8)
            ch = np.flatnonzero(np.diff(c) != 0)
            tr.append(int(ch[0]) + 1 if ch.size == 1 and c[0] == 1 else None)
        y_gt, y_hd = tr
        if y_gt is None or y_hd is None or abs(y_gt - y_hd) > tol_px:
            continue
        if y_hd < 30 or y_hd > h - 30:
            continue
        jump = float(np.mean(t_blend[y_hd - 8:y_hd - 2, x]) - np.mean(t_blend[y_hd + 2:y_hd + 8, x]))
        if best is None or jump > best[0]:
            best = (jump, x, y_hd)
    if best is None:
        return find_horizon_column(gt_mask)
    return best[1], best[2]


def horizon_crop_box(
    mask: np.ndarray, crop_h: int = 120, crop_w: int = 180
) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) of a crop window centred on the detected horizon."""
    h, w = mask.shape
    crop_h, crop_w = min(crop_h, h), min(crop_w, w)
    x, y = find_horizon_column(mask)
    y0 = int(np.clip(y - crop_h // 2, 0, h - crop_h))
    x0 = int(np.clip(x - crop_w // 2, 0, w - crop_w))
    return y0, y0 + crop_h, x0, x0 + crop_w


def _show(ax: plt.Axes, arr: np.ndarray, cmap: str | None = None, xlabel: str = "") -> None:
    """imshow helper: RGB clipped to [0, 1]; scalar maps rendered with a fixed [0, 1] range."""
    if cmap is None:
        ax.imshow(np.clip(arr, 0.0, 1.0))
    else:
        ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)


def _save(fig: plt.Figure, out_path: str | Path, dpi: int = 180) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = pred > 0.5
    union = float(np.logical_or(pb, gt).sum())
    return float(np.logical_and(pb, gt).sum()) / union if union > 0 else 1.0


# --------------------------------------------------------------------------- #
# Figure 1: pipeline diagram
# --------------------------------------------------------------------------- #
def plot_pipeline_diagram(out_path: str | Path) -> Path:
    """Figure 1: block diagram of the shared-DCP + fuzzy-membership pipeline."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, sub: str, fc: str, ec: str) -> None:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=fc, edgecolor=ec, linewidth=1.6,
            )
        )
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7.8, color="#37474f")

    def arrow(x0: float, y0: float, x1: float, y1: float, dashed: bool = False) -> None:
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#37474f", ls="--" if dashed else "-"),
        )

    blue, green, purple, orange = ("#e3f2fd", "#1565c0"), ("#e8f5e9", "#2e7d32"), ("#f3e5f5", "#6a1b9a"), ("#fff3e0", "#e65100")

    box(0.2, 2.25, 1.6, 1.1, "Hazy image I", "RGB float32 [0,1]", *blue)
    # DCP branch
    ax.text(2.2, 5.25, "Shared DCP backbone (identical for every method)", fontsize=9, fontweight="bold", color=blue[1])
    box(2.2, 3.9, 2.1, 1.1, "Dark channel + A", "15x15 min, top 0.1%", *blue)
    box(4.7, 3.9, 2.1, 1.1, "t_DCP", "1 - 0.95 min(I/A)", *blue)
    box(7.2, 3.9, 2.1, 1.1, "t_sky", "max(t_DCP, t_floor)", *blue)
    # Sky branch
    ax.text(2.2, 1.55, "Fuzzy sky membership (no training, no GT)", fontsize=9, fontweight="bold", color=green[1])
    box(2.2, 0.25, 2.1, 1.1, "Sky features", "brightness, 1-sat,\nsmoothness, 1-y/H", *green)
    box(4.7, 0.25, 2.1, 1.1, "Fuzzy C-Means", "c=3 + no-sky guard", *green)
    box(7.2, 0.25, 2.1, 1.1, "Soft mu(x)", "smoothstep + guided filter", *green)
    # Fusion
    box(9.8, 2.95, 3.0, 1.15, "Blend + guided filter", "t = GF((1-mu) t_DCP + mu t_sky)", *purple)
    box(9.8, 1.35, 3.0, 1.15, "Recover J", "(I - A) / max(t, 0.1) + A", *purple)
    # Offline GA
    box(4.7, 2.25, 4.6, 1.05, "Offline GA tuning (F1 only)", "10 genes, tuned on OTS subset; test sets never seen", *orange)

    arrow(1.8, 3.0, 2.2, 4.4)
    arrow(1.8, 2.6, 2.2, 0.8)
    arrow(4.3, 4.45, 4.7, 4.45)
    arrow(6.8, 4.45, 7.2, 4.45)
    arrow(4.3, 0.8, 4.7, 0.8)
    arrow(6.8, 0.8, 7.2, 0.8)
    arrow(9.3, 4.45, 9.8, 3.75)
    arrow(5.75, 3.9, 9.8, 3.45)
    arrow(9.3, 0.8, 9.8, 3.1)
    arrow(11.3, 2.95, 11.3, 2.5)
    arrow(7.0, 2.25, 7.0, 1.35, dashed=True)
    arrow(8.25, 3.3, 8.25, 3.9, dashed=True)
    fig.tight_layout()
    return _save(fig, out_path, dpi=200)


# --------------------------------------------------------------------------- #
# Figure 2: DCP failure
# --------------------------------------------------------------------------- #
def plot_dcp_failure(
    pairs: Sequence[Pair], cfg: dict[str, Any], best_params: dict[str, Any],
    out_path: str | Path, image_ids: Sequence[str],
) -> Path:
    """Figure 2: hazy | dark channel | t_DCP | B0 | F1 | GT for sky-heavy images."""
    titles = ["Hazy input", "Dark channel", "t_DCP (raw)", "B0 (plain DCP)", "F1 (proposed)", "Ground truth"]
    fig, axes = plt.subplots(len(image_ids), 6, figsize=(15, 2.7 * len(image_ids)), squeeze=False)
    for r, img_id in enumerate(image_ids):
        hazy, clear, mask = load_triplet(find_pair(pairs, img_id))
        res = run_methods_on_image(hazy, cfg, best_params, tau=0.65, methods=["B0", "F1"])
        sc = score_results(clear, mask, res)
        t_dcp = res["B0"].info["t_dcp"]
        sky_t = float(np.median(t_dcp[mask])) if mask.any() else float("nan")
        panels = [
            (hazy, None, f"{img_id}  (GT sky {100 * mask.mean():.0f}%)"),
            (res["B0"].info["dark"], "magma", "bright = dense haze"),
            (t_dcp, "inferno", f"median t in sky = {sky_t:.2f}"),
            (res["B0"].J, None, f"PSNR {sc['B0']['psnr']:.1f} / sky {sc['B0']['sky_psnr']:.1f} dB"),
            (res["F1"].J, None, f"PSNR {sc['F1']['psnr']:.1f} / sky {sc['F1']['sky_psnr']:.1f} dB"),
            (clear, None, ""),
        ]
        for c, (arr, cmap, lbl) in enumerate(panels):
            _show(axes[r, c], arr, cmap, lbl)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 3: membership maps
# --------------------------------------------------------------------------- #
def plot_membership_maps(
    pairs: Sequence[Pair], cfg: dict[str, Any], best_params: dict[str, Any], tau: float,
    out_path: str | Path, image_ids: Sequence[str],
) -> Path:
    """Figure 3: GT sky mask vs B1 / B2 binary masks vs F0 / F1 soft memberships."""
    titles = ["Hazy input", "GT sky mask (eval only)", "B1 hard mask", "B2 hard FCM mask", "F0 soft mu", "F1 soft mu"]
    fig, axes = plt.subplots(len(image_ids), 6, figsize=(15, 2.7 * len(image_ids)), squeeze=False)
    for r, img_id in enumerate(image_ids):
        hazy, _, mask = load_triplet(find_pair(pairs, img_id))
        res = run_methods_on_image(hazy, cfg, best_params, tau, methods=["B1", "B2", "F0", "F1"])
        panels = [(hazy, None, img_id), (mask.astype(np.float32), "viridis", f"sky {100 * mask.mean():.0f}%")]
        for m in ["B1", "B2", "F0", "F1"]:
            wm = weight_map(res[m])
            panels.append((wm, "viridis", f"IoU(>0.5) = {_iou(wm, mask):.2f}"))
        for c, (arr, cmap, lbl) in enumerate(panels):
            _show(axes[r, c], arr, cmap, lbl)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 4: hero horizon crops
# --------------------------------------------------------------------------- #
def plot_hero_crops(
    pairs: Sequence[Pair], cfg: dict[str, Any], best_params: dict[str, Any], tau: float,
    out_path: str | Path, image_ids: Sequence[str], crop_h: int = 110, crop_w: int = 170,
) -> Path:
    """Figure 4: full hazy image with crop box, then zoomed horizon crops for every method + GT."""
    cols = ["Hazy (crop boxed)", "Hazy", *[METHOD_LABELS[m] for m in ["B0", "B1", "B2", "F1"]], "Ground truth"]
    fig, axes = plt.subplots(len(image_ids), len(cols), figsize=(17, 2.6 * len(image_ids)), squeeze=False)
    for r, img_id in enumerate(image_ids):
        hazy, clear, mask = load_triplet(find_pair(pairs, img_id))
        res = run_methods_on_image(hazy, cfg, best_params, tau, methods=["B0", "B1", "B2", "F1"])
        sc = score_results(clear, mask, res)
        y0, y1, x0, x1 = horizon_crop_box(mask, crop_h, crop_w)

        _show(axes[r, 0], hazy, xlabel=img_id)
        axes[r, 0].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2, ec="#ff1744", fc="none"))
        _show(axes[r, 1], hazy[y0:y1, x0:x1])
        for c, m in enumerate(["B0", "B1", "B2", "F1"], start=2):
            lbl = f"PSNR {sc[m]['psnr']:.1f} | seam-grad {sc[m]['seam_grad']:.3f}"
            _show(axes[r, c], res[m].J[y0:y1, x0:x1], xlabel=lbl)
        _show(axes[r, len(cols) - 1], clear[y0:y1, x0:x1])
        if r == 0:
            for c, t in enumerate(cols):
                axes[r, c].set_title(t, fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path, dpi=200)


# --------------------------------------------------------------------------- #
# Figure 5: transmission profile across the horizon
# --------------------------------------------------------------------------- #
def plot_transmission_profile(
    pairs: Sequence[Pair], cfg: dict[str, Any], best_params: dict[str, Any], tau: float,
    out_path: str | Path, image_id: str, half_window: int = 60,
) -> Path:
    """Figure 5: vertical 1-D profile of t_blend / t_final across the horizon (step vs ramp)."""
    hazy, _, mask = load_triplet(find_pair(pairs, image_id))
    res = run_methods_on_image(hazy, cfg, best_params, tau)
    x, y_h = find_step_column(mask, res["B2"])
    h = hazy.shape[0]
    y0, y1 = max(0, y_h - half_window), min(h, y_h + half_window)
    ys = np.arange(y0, y1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    _show(axes[0], hazy, xlabel=f"{image_id}: scanline x = {x}, horizon y = {y_h}")
    axes[0].plot([x, x], [y0, y1], color="#ffeb3b", lw=3)
    axes[0].axhline(y_h, color="#00e5ff", ls=":", lw=1.2)
    axes[0].set_title("Scanline location", fontsize=10, fontweight="bold")

    for ax, key, title in [
        (axes[1], "t_blend", "Blended transmission before guided filter"),
        (axes[2], "t_final", "Final transmission after guided filter"),
    ]:
        for m in MAIN_METHODS:
            prof = res[m].info["t_blend"] if key == "t_blend" else res[m].t
            ax.plot(ys, prof[y0:y1, x], label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                    lw=2.2 if m.startswith("F") else 1.6, ls=":" if m == "B0" else "-")
        if key == "t_final":
            ax.plot(ys, weight_map(res["F1"])[y0:y1, x], color="#8e24aa", ls="-.", lw=1.4, label="F1 mu(x)")
        ax.axvline(y_h, color="#263238", ls="--", lw=1)
        ax.axvspan(y_h - 10, y_h + 10, color="#fff59d", alpha=0.4, label="seam band (+/-10 px)")
        ax.set_xlabel("row y  (sky  ->  ground)")
        ax.set_ylabel("transmission")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return _save(fig, out_path, dpi=200)


# --------------------------------------------------------------------------- #
# Figure 6: seam & sky bars
# --------------------------------------------------------------------------- #
def plot_seam_sky_bars(summary_csv: str | Path, out_path: str | Path) -> Path:
    """Figure 6: seam-grad, seam-L1, sky dE2000 and sky PSNR bars for B0-F1 (all vs sky-heavy)."""
    df = pd.read_csv(summary_csv)
    metrics = [
        ("seam_grad_mean", "Horizon seam gradient error (lower = better)"),
        ("seam_l1_mean", "Horizon seam L1 error (lower = better)"),
        ("sky_ciede2000_mean", "Sky CIEDE2000 (lower = better)"),
        ("sky_psnr_mean", "Sky PSNR, dB (higher = better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    x = np.arange(len(MAIN_METHODS))
    wd = 0.38
    for ax, (col, title) in zip(axes.flat, metrics):
        for k, (subset, color) in enumerate([("all", "#5c6bc0"), ("sky_heavy", "#26a69a")]):
            sub = df[df["subset"] == subset].set_index("method").reindex(MAIN_METHODS)
            vals = sub[col].to_numpy(dtype=float)
            n = int(sub["n_images"].iloc[0])
            bars = ax.bar(x + (k - 0.5) * wd, vals, wd, color=color, label=f"{subset} (N={n})")
            fmt = "{:.3f}" if col.startswith("seam") else "{:.1f}"
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(MAIN_METHODS)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SOTS-outdoor: seam and sky-region quality", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 7: ablation chart
# --------------------------------------------------------------------------- #
def plot_ablation_chart(ablations_csv: str | Path, out_path: str | Path) -> Path:
    """Figure 7: PSNR delta vs F0 and seam-grad error for each variant (B0 omitted for scale).

    Every ablation A3-A10 changes exactly one setting of the F0 hand-set defaults, so F0 (not
    the GA-tuned F1) is the correct reference; F1 is shown as an extra bar for context.
    """
    df = pd.read_csv(ablations_csv)
    df = df[df["method"] != "B0"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    order = (
        df[df["subset"] == "sky_heavy"].sort_values("delta_psnr_vs_F0", ascending=False)["method"].tolist()
    )
    desc = df.drop_duplicates("method").set_index("method")["description"]
    labels = [f"{m}: {desc[m]}" for m in order]
    y = np.arange(len(order))
    h = 0.4
    for k, (subset, color) in enumerate([("all", "#42a5f5"), ("sky_heavy", "#26a69a")]):
        sub = df[df["subset"] == subset].set_index("method").reindex(order)
        axes[0].barh(y + (k - 0.5) * h, sub["delta_psnr_vs_F0"], h, color=color, label=subset)
        axes[1].barh(y + (k - 0.5) * h, sub["seam_grad_mean"], h, color=color, label=subset)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("PSNR difference vs F0 (dB)")
    axes[0].set_title("Whole-image PSNR relative to F0 (the ablation base)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("horizon seam gradient error (lower = better)")
    axes[1].set_title("Horizon seam gradient error", fontsize=10, fontweight="bold")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SOTS-outdoor ablations: one change from F0 each (B0 omitted: -5.5 dB vs F0)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 8: GA convergence
# --------------------------------------------------------------------------- #
def plot_ga_convergence(ga_history_csv: str | Path, out_path: str | Path) -> Path:
    """Figure 8: best / mean (+/- std) fitness per generation with train/val reference lines."""
    hist = pd.read_csv(ga_history_csv)
    g = hist["generation"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(g, hist["best_fitness"], "o-", color="#1565c0", lw=2, label="best so far (train)")
    ax.plot(g, hist["mean_fitness"], "s--", color="#2e7d32", lw=1.5, label="population mean (train)")
    ax.fill_between(g, hist["mean_fitness"] - hist["std_fitness"], hist["mean_fitness"] + hist["std_fitness"],
                    color="#66bb6a", alpha=0.2, label="mean +/- 1 std")
    ax.plot(g, hist["val_fitness"], "^-", color="#8e24aa", lw=1.5, label="best individual on held-out val")
    ax.axhline(hist["f0_train_fitness"].iloc[0], color="#d32f2f", ls=":", label="F0 defaults (train)")
    ax.axhline(hist["f0_val_fitness"].iloc[0], color="#d32f2f", ls="-.", alpha=0.6, label="F0 defaults (val)")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness = 0.5 PSNR + 0.5 sky PSNR - 20 sat_sky")
    ax.set_xticks(g)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("GA tuning on the OTS subset", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 9: failure cases
# --------------------------------------------------------------------------- #
def plot_failure_cases(
    cases: Sequence[tuple[Sequence[Pair], str, str]], cfg: dict[str, Any], best_params: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Figure 9: rows of (pairs, image_id, caption) showing hazy | GT mask | F1 mu | B0 | F1 | GT."""
    titles = ["Hazy input", "GT sky mask", "F1 soft mu", "B0 (plain DCP)", "F1 (proposed)", "Ground truth"]
    fig, axes = plt.subplots(len(cases), 6, figsize=(15, 2.8 * len(cases)), squeeze=False)
    for r, (pairs, img_id, caption) in enumerate(cases):
        hazy, clear, mask = load_triplet(find_pair(pairs, img_id))
        res = run_methods_on_image(hazy, cfg, best_params, tau=0.65, methods=["B0", "F1"])
        sc = score_results(clear, mask, res)
        guard = "guard fired: mu = 0" if not res["F1"].info.get("sky_present", True) else f"mean mu = {weight_map(res['F1']).mean():.2f}"
        panels = [
            (hazy, None, caption),
            (mask.astype(np.float32), "viridis", f"GT sky {100 * mask.mean():.1f}%"),
            (weight_map(res["F1"]), "viridis", guard),
            (res["B0"].J, None, f"PSNR {sc['B0']['psnr']:.2f} dB"),
            (res["F1"].J, None, f"PSNR {sc['F1']['psnr']:.2f} dB"),
            (clear, None, ""),
        ]
        for c, (arr, cmap, lbl) in enumerate(panels):
            _show(axes[r, c], arr, cmap, lbl)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 10: O-HAZE grid
# --------------------------------------------------------------------------- #
def plot_ohaze_grid(
    pairs: Sequence[Pair], cfg: dict[str, Any], best_params: dict[str, Any], tau: float,
    out_path: str | Path, image_ids: Sequence[str],
) -> Path:
    """Figure 10: real-haze qualitative grid (hazy | B0 | B1 | B2 | F1 | GT)."""
    methods = ["B0", "B1", "B2", "F1"]
    titles = ["Real hazy input", *[METHOD_LABELS[m] for m in methods], "Ground truth"]
    fig, axes = plt.subplots(len(image_ids), 6, figsize=(15, 2.6 * len(image_ids)), squeeze=False)
    for r, img_id in enumerate(image_ids):
        hazy, clear, mask = load_triplet(find_pair(pairs, img_id))
        res = run_methods_on_image(hazy, cfg, best_params, tau, methods=methods)
        sc = score_results(clear, mask, res)
        _show(axes[r, 0], hazy, xlabel=f"{img_id} (GT sky {100 * mask.mean():.0f}%)")
        for c, m in enumerate(methods, start=1):
            _show(axes[r, c], res[m].J, xlabel=f"PSNR {sc[m]['psnr']:.2f} | dE {sc[m]['ciede2000']:.1f}")
        _show(axes[r, 5], clear)
        if r == 0:
            for c, t in enumerate(titles):
                axes[r, c].set_title(t, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Tables 4-6 (Tables 1-3 are written by scripts/run_benchmark.py)
# --------------------------------------------------------------------------- #
def write_extra_tables(
    summary_dir: str | Path, best_params: dict[str, Any], ga_bounds: dict[str, list[float]],
    f0_defaults: dict[str, float],
) -> list[Path]:
    """Write significance (Table 4, Markdown), runtime (Table 5) and GA parameter (Table 6) tables."""
    summary_dir = Path(summary_dir)
    written: list[Path] = []

    for ds in ["sots_outdoor", "ohaze"]:
        sig_csv = summary_dir / f"{ds}_significance.csv"
        if not sig_csv.is_file():
            continue
        sig = pd.read_csv(sig_csv)
        lines = [
            f"# Paired significance tests: {ds}\n",
            "| Subset | A vs B | Metric | N | Mean A | Mean B | Diff (A-B) | 95% CI | Wilcoxon p | Win rate A |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |",
        ]
        for _, r in sig.iterrows():
            lines.append(
                f"| {r['subset']} | {r['method_a']} vs {r['method_b']} | {r['metric']} | {int(r['n_pairs'])} | "
                f"{r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['mean_diff']:+.4f} | "
                f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['wilcoxon_p']:.2e} | {100 * r['win_rate_a']:.1f}% |"
            )
        out = summary_dir / f"{ds}_significance.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(out)

    rows = []
    for ds in ["sots_outdoor", "ohaze"]:
        s = summary_dir / f"{ds}_summary.csv"
        if s.is_file():
            df = pd.read_csv(s)
            df = df[(df["subset"] == "all") & (df["method"].isin(MAIN_METHODS))]
            for _, r in df.iterrows():
                rows.append({"dataset": ds, "method": r["method"], "n_images": int(r["n_images"]),
                             "runtime_mean_s": r["runtime_s_mean"], "runtime_median_s": r["runtime_s_median"]})
    if rows:
        rt = pd.DataFrame(rows)
        rt.to_csv(summary_dir / "runtime_summary.csv", index=False)
        lines = ["# Per-image runtime (CPU, parallel workers)\n",
                 "| Dataset | Method | N | Mean (s) | Median (s) |", "| :--- | :--- | ---: | ---: | ---: |"]
        for _, r in rt.iterrows():
            lines.append(f"| {r['dataset']} | {r['method']} | {r['n_images']} | {r['runtime_mean_s']:.3f} | {r['runtime_median_s']:.3f} |")
        out = summary_dir / "runtime_summary.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += [summary_dir / "runtime_summary.csv", out]

    prow = []
    for gene, (lo, hi) in ga_bounds.items():
        prow.append({"gene": gene, "search_min": lo, "search_max": hi,
                     "F0_default": f0_defaults.get(gene, float("nan")), "F1_tuned": best_params.get(gene, float("nan"))})
    if prow:
        pt = pd.DataFrame(prow)
        pt.to_csv(summary_dir / "ga_params_table.csv", index=False)
        lines = ["# GA-tuned parameters (F1) vs hand-set defaults (F0)\n",
                 "| Gene | Search range | F0 default | F1 tuned |", "| :--- | :--- | ---: | ---: |"]
        for _, r in pt.iterrows():
            lines.append(f"| {r['gene']} | [{r['search_min']}, {r['search_max']}] | {r['F0_default']:.4g} | {r['F1_tuned']:.4g} |")
        out = summary_dir / "ga_params_table.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += [summary_dir / "ga_params_table.csv", out]
    return written
