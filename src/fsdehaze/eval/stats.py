"""Statistical summaries, paired Wilcoxon tests, bootstrap CIs, and table formatters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HIGHER_IS_BETTER: set[str] = {
    "psnr",
    "ssim",
    "sky_psnr",
    "nonsky_psnr",
}

LOWER_IS_BETTER: set[str] = {
    "ciede2000",
    "sky_ciede2000",
    "nonsky_ciede2000",
    "sat_frac_full",
    "sat_pct_full",
    "sat_frac_sky",
    "sat_pct_sky",
    "seam_l1",
    "seam_grad",
    "runtime_s",
}

DEFAULT_METRICS: tuple[str, ...] = (
    "psnr",
    "ssim",
    "ciede2000",
    "sky_psnr",
    "sky_ciede2000",
    "nonsky_psnr",
    "sat_pct_sky",
    "seam_l1",
    "seam_grad",
    "runtime_s",
)


def bootstrap_ci(
    diffs: np.ndarray | Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute a seeded percentile bootstrap confidence interval for the mean.

    Args:
        diffs: 1D array or sequence of per-image differences (NaNs are dropped).
        n_resamples: Number of bootstrap resamples (default 10,000).
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for a 95% confidence interval).

    Returns:
        Tuple `(ci_low, ci_high)` of the (1 - alpha) confidence interval for the mean.
    """
    arr = np.asarray(diffs, dtype=np.float64).reshape(-1)
    valid = arr[~np.isnan(arr)]
    n = valid.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or np.allclose(valid, valid[0]):
        v = float(valid[0])
        return v, v

    rng = np.random.default_rng(seed)
    # Sample in batches to bound peak memory while remaining fast
    batch_size = min(n_resamples, 2000)
    means: list[np.ndarray] = []
    remaining = n_resamples
    while remaining > 0:
        b = min(batch_size, remaining)
        idx = rng.integers(0, n, size=(b, n))
        means.append(np.mean(valid[idx], axis=1))
        remaining -= b

    boot_means = np.concatenate(means)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(boot_means, [low_pct, high_pct])
    return float(ci_low), float(ci_high)


def _summarise_subset(
    df: pd.DataFrame,
    subset_name: str,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    """Summarise each method in a filtered DataFrame."""
    rows: list[dict[str, object]] = []
    if df.empty:
        return rows

    methods = df["method"].unique() if "method" in df.columns else ["unknown"]
    datasets = df["dataset"].unique() if "dataset" in df.columns else ["dataset"]
    dataset_label = str(datasets[0])

    for method in methods:
        mdf = df[df["method"] == method] if "method" in df.columns else df
        n_imgs = len(mdf)
        if "sky_present" in mdf.columns and n_imgs > 0:
            no_sky_rate = float(np.mean(~mdf["sky_present"].astype(bool)))
        else:
            no_sky_rate = float("nan")

        row: dict[str, object] = {
            "dataset": dataset_label,
            "subset": subset_name,
            "method": str(method),
            "n_images": int(n_imgs),
            "no_sky_guard_rate": round(no_sky_rate, 4) if not np.isnan(no_sky_rate) else float("nan"),
        }

        for col in metrics:
            if col not in mdf.columns:
                continue
            vals = mdf[col].dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                row[f"{col}_median"] = float("nan")
            else:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals, ddof=0))
                row[f"{col}_median"] = float(np.median(vals))

        rows.append(row)

    return rows


def summarise(
    df: pd.DataFrame,
    sky_heavy_threshold: float = 0.20,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Compute mean, std, median, and no-sky guard trigger rate for all and sky-heavy images.

    Args:
        df: Per-image DataFrame containing one or more methods.
        sky_heavy_threshold: Minimum `sky_fraction` to qualify as sky-heavy (default 0.20).
        metrics: Sequence of metric column names to aggregate.

    Returns:
        Summary DataFrame with rows for `subset='all'` and `subset='sky_heavy'`.
    """
    rows = _summarise_subset(df, "all", metrics)
    if "sky_fraction" in df.columns:
        heavy_df = df[df["sky_fraction"] > float(sky_heavy_threshold)]
        if not heavy_df.empty:
            rows.extend(_summarise_subset(heavy_df, "sky_heavy", metrics))
    return pd.DataFrame(rows)


def paired_tests(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    n_resamples: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests, bootstrap 95% CIs, and win rates between two methods.

    Computes `diff = A - B` aligned strictly by `image_id`.

    Args:
        df_a: Per-image DataFrame for Method A (e.g., proposed fuzzy method).
        df_b: Per-image DataFrame for Method B (e.g., baseline method).
        metrics: Metric columns to compare (defaults to `DEFAULT_METRICS` present in both).
        n_resamples: Bootstrap resamples for the 95% CI of mean difference.
        seed: Random seed for bootstrap resampling.

    Returns:
        DataFrame with one row per metric:
        `metric, n_pairs, mean_a, mean_b, mean_diff, std_diff, median_diff, ci95_low, ci95_high, wilcoxon_p, win_rate_a`.
    """
    if "image_id" not in df_a.columns or "image_id" not in df_b.columns:
        raise ValueError("Both df_a and df_b must contain an 'image_id' column for paired comparison.")

    merged = pd.merge(
        df_a,
        df_b,
        on="image_id",
        suffixes=("_a", "_b"),
        how="inner",
    ).sort_values("image_id")

    if merged.empty:
        raise ValueError("No overlapping image_id values between df_a and df_b.")

    metric_list = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    records: list[dict[str, object]] = []

    for m in metric_list:
        col_a = f"{m}_a"
        col_b = f"{m}_b"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue

        pair_vals = merged[[col_a, col_b]].dropna()
        n_valid = len(pair_vals)
        if n_valid == 0:
            continue

        a_arr = pair_vals[col_a].to_numpy(dtype=np.float64)
        b_arr = pair_vals[col_b].to_numpy(dtype=np.float64)
        diffs = a_arr - b_arr

        mean_a = float(np.mean(a_arr))
        mean_b = float(np.mean(b_arr))
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=0))
        med_diff = float(np.median(diffs))
        ci_low, ci_high = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)

        if np.allclose(diffs, 0.0):
            p_val = 1.0
        else:
            try:
                stat_res = wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
                p_val = float(stat_res.pvalue)
            except ValueError:
                p_val = 1.0

        if m in LOWER_IS_BETTER:
            wins = float(np.mean(diffs < 0.0))
        else:
            wins = float(np.mean(diffs > 0.0))

        records.append(
            {
                "metric": m,
                "n_pairs": int(n_valid),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "median_diff": med_diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_p": p_val,
                "win_rate_a": wins,
            }
        )

    return pd.DataFrame(records)


def _format_cell(val: object, float_fmt: str) -> str:
    """Format a single table cell."""
    if val is None:
        return ""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "—"
        return format(float(val), float_fmt)
    return str(val)


def to_markdown_table(df: pd.DataFrame, float_fmt: str = ".2f") -> str:
    """Render a DataFrame as a GitHub-Flavored Markdown table.

    Args:
        df: Input DataFrame.
        float_fmt: Format specifier for floating-point values (default '.2f').

    Returns:
        Markdown table string.
    """
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        cells = [_format_cell(row[c], float_fmt) for c in df.columns]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines])


def to_latex_table(
    df: pd.DataFrame,
    float_fmt: str = ".2f",
    caption: str = "",
    label: str = "",
) -> str:
    """Render a DataFrame as a LaTeX table environment.

    Args:
        df: Input DataFrame.
        float_fmt: Format specifier for floating-point values (default '.2f').
        caption: Optional LaTeX caption.
        label: Optional LaTeX label.

    Returns:
        LaTeX table string.
    """
    if df.empty:
        return ""
    cols = [str(c).replace("_", r"\_") for c in df.columns]
    col_spec = "l" + "r" * (len(cols) - 1)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        cells = [_format_cell(row[c], float_fmt).replace("%", r"\%").replace("_", r"\_") for c in df.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
