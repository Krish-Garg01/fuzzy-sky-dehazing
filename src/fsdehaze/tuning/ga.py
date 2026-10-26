"""Genetic Algorithm (DEAP) hyperparameter tuning for Fuzzy Sky-Aware Dehazing."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

import cv2
try:
    from deap import base, creator, tools  # type: ignore[import-not-found]
    _HAS_DEAP = True
except ImportError:
    _HAS_DEAP = False
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fsdehaze.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    recover,
    transmission_dcp,
)
from fsdehaze.guided_filter import guided_filter, rgb_to_gray
from fsdehaze.io_utils import Pair, load_image, load_mask
from fsdehaze.metrics import masked_psnr, psnr, saturated_fraction
from fsdehaze.pipelines import dehaze_fuzzy
from fsdehaze.sky_features import compute_features
from fsdehaze.sky_membership import estimate_sky_membership
from fsdehaze.sky_transmission import compute_t_sky

GENE_NAMES: tuple[str, ...] = (
    "m",
    "w_f1",
    "w_f2",
    "w_f3",
    "w_f4",
    "a",
    "b",
    "r_mu",
    "t_floor",
    "K",
)

DEFAULT_GENE_VALUES: dict[str, float] = {
    "m": 2.0,
    "w_f1": 1.0,
    "w_f2": 1.0,
    "w_f3": 1.0,
    "w_f4": 1.0,
    "a": 0.2,
    "b": 0.8,
    "r_mu": 20.0,
    "t_floor": 0.8,
    "K": 0.08,
}


@dataclass
class CachedGASample:
    """Precomputed gene-independent maps for fast GA fitness evaluation."""

    image_id: str
    img: np.ndarray
    gray: np.ndarray
    clear: np.ndarray
    sky_mask: np.ndarray
    dark: np.ndarray
    A: np.ndarray
    t_dcp: np.ndarray
    features: tuple[np.ndarray, list[str]]


def _downscale_pair(
    img: np.ndarray,
    clear: np.ndarray,
    sky_mask: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downscale image, clear GT, and boolean sky mask so max(H, W) <= max_side."""
    H, W = img.shape[:2]
    if max_side <= 0 or max(H, W) <= max_side:
        return img, clear, sky_mask

    scale = float(max_side) / float(max(H, W))
    new_w = max(16, int(round(W * scale)))
    new_h = max(16, int(round(H * scale)))

    img_ds = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    clear_ds = cv2.resize(clear, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask_u8 = sky_mask.astype(np.uint8) * 255
    mask_ds_u8 = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_ds = np.asarray(mask_ds_u8 > 127, dtype=bool)
    return np.clip(img_ds, 0.0, 1.0), np.clip(clear_ds, 0.0, 1.0), mask_ds


def prepare_cached_samples(
    pairs: Sequence[Pair],
    base_cfg: dict[str, Any],
    max_side: int = 320,
    desc: str = "Caching GA samples",
) -> list[CachedGASample]:
    """Load, optionally downscale, and precompute gene-independent maps for a list of Pairs."""
    dcp_cfg = base_cfg.get("dcp", {})
    patch = int(dcp_cfg.get("patch", 15))
    omega = float(dcp_cfg.get("omega", 0.95))
    top_frac = float(dcp_cfg.get("top_frac", 0.001))

    samples: list[CachedGASample] = []
    for p in tqdm(pairs, desc=desc, leave=False):
        img = load_image(p.hazy_path)
        clear = load_image(p.clear_path)
        H, W = img.shape[:2]
        if p.sky_mask_path is not None and p.sky_mask_path.is_file():
            sky_mask = load_mask(p.sky_mask_path)
        else:
            sky_mask = np.zeros((H, W), dtype=bool)

        img_s, clear_s, mask_s = _downscale_pair(img, clear, sky_mask, max_side=max_side)
        gray_s = rgb_to_gray(img_s)
        dark_s = dark_channel(img_s, patch=patch)
        A_s = estimate_atmospheric_light(img_s, dark_s, top_frac=top_frac)
        t_dcp_s = transmission_dcp(img_s, A_s, omega=omega, patch=patch)
        feats_s = compute_features(img_s, base_cfg)

        samples.append(
            CachedGASample(
                image_id=p.image_id,
                img=img_s,
                gray=gray_s,
                clear=clear_s,
                sky_mask=mask_s,
                dark=dark_s,
                A=A_s,
                t_dcp=t_dcp_s,
                features=feats_s,
            )
        )
    return samples


def repair_individual(ind: list[float], bounds: dict[str, list[float]]) -> list[float]:
    """Clip genes into configured bounds and enforce `b - a >= 0.1` and non-zero feature weights."""
    for idx, name in enumerate(GENE_NAMES):
        lo, hi = bounds[name]
        ind[idx] = float(np.clip(ind[idx], float(lo), float(hi)))

    a_idx = GENE_NAMES.index("a")
    b_idx = GENE_NAMES.index("b")
    if ind[b_idx] - ind[a_idx] < 0.10:
        mid = 0.5 * (ind[a_idx] + ind[b_idx])
        a_lo, a_hi = bounds["a"]
        b_lo, b_hi = bounds["b"]
        ind[a_idx] = float(np.clip(mid - 0.05, a_lo, a_hi))
        ind[b_idx] = float(np.clip(ind[a_idx] + 0.10, b_lo, b_hi))
        if ind[b_idx] - ind[a_idx] < 0.10:
            ind[a_idx] = float(np.clip(ind[b_idx] - 0.10, a_lo, a_hi))

    # Prevent all 4 feature weights from collapsing simultaneously
    w_indices = [GENE_NAMES.index(f"w_f{k}") for k in (1, 2, 3, 4)]
    if sum(ind[i] for i in w_indices) < 0.20:
        for i in w_indices:
            ind[i] = max(ind[i], 0.25)

    return ind


def individual_to_params(ind: Sequence[float], base_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a 10-gene vector into a `params` dictionary for `dehaze_fuzzy`."""
    raw = {name: float(ind[i]) for i, name in enumerate(GENE_NAMES)}
    t_sky_mode = "floor"
    if base_cfg is not None:
        t_sky_mode = str(base_cfg.get("sky", {}).get("t_sky_mode", "floor"))
    return {
        "m": round(raw["m"], 4),
        "fuzzifier": round(raw["m"], 4),
        "w_f1": round(raw["w_f1"], 4),
        "w_f2": round(raw["w_f2"], 4),
        "w_f3": round(raw["w_f3"], 4),
        "w_f4": round(raw["w_f4"], 4),
        "feature_weights": [
            round(raw["w_f1"], 4),
            round(raw["w_f2"], 4),
            round(raw["w_f3"], 4),
            round(raw["w_f4"], 4),
        ],
        "a": round(raw["a"], 4),
        "b": round(raw["b"], 4),
        "r_mu": int(round(raw["r_mu"])),
        "t_floor": round(raw["t_floor"], 4),
        "K": round(raw["K"], 4),
        "tolerance_K": round(raw["K"], 4),
        "t_sky_mode": t_sky_mode,
    }


def evaluate_params_on_cache(
    params: dict[str, Any],
    samples: Sequence[CachedGASample],
    base_cfg: dict[str, Any],
    lambda_sat: float = 20.0,
) -> dict[str, float]:
    """Evaluate a parameter dictionary on cached tuning samples.

    Computes the Section 4.9 fitness objective:
        F = 0.5 * PSNR_full + 0.5 * PSNR_sky - lambda_sat * saturated_fraction_sky
    with a soft guard against degenerate chromosomes that flood non-sky regions (`mean(mu[~sky]) > 0.12`)
    or drop the vertical-position / brightness sky separation (`sky_iou < 0.72`).
    """
    dcp_cfg = base_cfg.get("dcp", {})
    t0 = float(dcp_cfg.get("t0", 0.1))
    gf_cfg = base_cfg.get("guided_filter", {})
    gf_r = int(gf_cfg.get("r", 60))
    gf_eps = float(gf_cfg.get("eps", 1.0e-4))

    # Use 8,000 FCM samples on downscaled 320px tuning images (~75k total pixels) for fast evaluation
    eval_params = dict(params)
    eval_params.setdefault("fcm_samples", 8000)

    fit_list: list[float] = []
    psnr_list: list[float] = []
    sky_psnr_list: list[float] = []
    nonsky_psnr_list: list[float] = []
    sat_sky_list: list[float] = []
    sky_iou_list: list[float] = []

    for s in samples:
        mu, _ = estimate_sky_membership(
            s.img,
            base_cfg,
            params=eval_params,
            precomputed_features=s.features,
        )
        t_sky = compute_t_sky(s.img, s.A, s.t_dcp, base_cfg, params=eval_params)
        t_blend = (mu * t_sky + (1.0 - mu) * s.t_dcp).astype(np.float32)
        t_final = np.clip(guided_filter(s.gray, t_blend, r=gf_r, eps=gf_eps), 0.0, 1.0).astype(
            np.float32
        )
        J, J_raw = recover(s.img, s.A, t_final, t0=t0)

        p_full = psnr(s.clear, J)
        has_sky = bool(np.any(s.sky_mask))
        has_nonsky = bool(np.any(~s.sky_mask))

        if has_sky:
            p_sky = masked_psnr(s.clear, J, s.sky_mask)
            sat_sky = saturated_fraction(J_raw, s.sky_mask)
            pred_sky = mu > 0.5
            inter = int(np.sum(pred_sky & s.sky_mask))
            union = int(np.sum(pred_sky | s.sky_mask))
            iou = float(inter) / float(max(union, 1))
            sky_iou_list.append(iou)
        else:
            p_sky = p_full
            sat_sky = saturated_fraction(J_raw, None)
            iou = 1.0

        if has_nonsky:
            p_nonsky = masked_psnr(s.clear, J, ~s.sky_mask)
            nonsky_mu = float(np.mean(mu[~s.sky_mask]))
        else:
            p_nonsky = p_full
            nonsky_mu = 0.0

        # Primary fitness (Section 4.9)
        raw_fit = 0.5 * p_full + 0.5 * p_sky - float(lambda_sat) * sat_sky

        # Prevent degenerate solutions from flooding non-sky foreground with t_floor
        leak_excess = max(0.0, nonsky_mu - 0.11)
        iou_deficit = max(0.0, 0.72 - iou) if has_sky else 0.0
        penalty = 15.0 * leak_excess + 8.0 * iou_deficit

        fit_list.append(raw_fit - penalty)
        psnr_list.append(p_full)
        sky_psnr_list.append(p_sky)
        nonsky_psnr_list.append(p_nonsky)
        sat_sky_list.append(sat_sky)

    return {
        "fitness": float(np.mean(fit_list)),
        "psnr": float(np.mean(psnr_list)),
        "sky_psnr": float(np.mean(sky_psnr_list)),
        "nonsky_psnr": float(np.mean(nonsky_psnr_list)),
        "sat_frac_sky": float(np.mean(sat_sky_list)),
        "sky_iou": float(np.mean(sky_iou_list)) if sky_iou_list else 0.0,
    }


class _FallbackFitness:
    """DEAP-compatible single-objective maximising fitness container."""

    def __init__(self, values: tuple[float, ...] = ()) -> None:
        self._values: tuple[float, ...] = ()
        if values:
            self.values = values

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    @values.setter
    def values(self, vals: Sequence[float]) -> None:
        self._values = tuple(float(v) for v in vals)

    @values.deleter
    def values(self) -> None:
        self._values = ()

    @property
    def valid(self) -> bool:
        return len(self._values) > 0


class _FallbackIndividual(list):
    """DEAP-compatible Individual list with a `.fitness` attribute."""

    def __init__(self, iterable: Sequence[float] = ()) -> None:
        super().__init__(iterable)
        self.fitness = _FallbackFitness()


class _FallbackHallOfFame(list):
    """DEAP-compatible HallOfFame keeping top `maxsize` individuals sorted descending by fitness."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = int(maxsize)

    def update(self, population: Sequence[Any]) -> None:
        for ind in population:
            if not ind.fitness.valid:
                continue
            if any( np.allclose(ind, existing, atol=1e-9) for existing in self ):
                continue
            self.append(copy.deepcopy(ind))
        self.sort(key=lambda x: x.fitness.values[0], reverse=True)
        del self[self.maxsize :]


class _FallbackToolbox:
    """DEAP-compatible Toolbox."""

    def __init__(self) -> None:
        self.clone = copy.deepcopy

    def register(self, alias: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        def _wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
            merged_kw = {**kwargs, **call_kwargs}
            return func(*(args + call_args), **merged_kw)

        setattr(self, alias, _wrapped)


class _FallbackTools:
    """DEAP-compatible evolutionary operators (`initRepeat`, `selTournament`, `cxBlend`, `mutGaussian`)."""

    HallOfFame = _FallbackHallOfFame

    @staticmethod
    def initRepeat(container: Callable[..., Any], func: Callable[[], Any], n: int) -> Any:
        return container(func() for _ in range(int(n)))

    @staticmethod
    def selTournament(individuals: Sequence[Any], k: int, tournsize: int) -> list[Any]:
        chosen: list[Any] = []
        for _ in range(int(k)):
            aspirants = [random.choice(individuals) for _ in range(int(tournsize))]
            chosen.append(max(aspirants, key=lambda ind: ind.fitness.values[0]))
        return chosen

    @staticmethod
    def cxBlend(ind1: list[float], ind2: list[float], alpha: float) -> tuple[list[float], list[float]]:
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            gamma = (1.0 + 2.0 * float(alpha)) * random.random() - float(alpha)
            ind1[i] = (1.0 - gamma) * x1 + gamma * x2
            ind2[i] = gamma * x1 + (1.0 - gamma) * x2
        return ind1, ind2

    @staticmethod
    def mutGaussian(
        individual: list[float],
        mu: float | Sequence[float],
        sigma: float | Sequence[float],
        indpb: float,
    ) -> tuple[list[float]]:
        size = len(individual)
        mus = [float(mu)] * size if isinstance(mu, (int, float)) else [float(m) for m in mu]
        sigmas = [float(sigma)] * size if isinstance(sigma, (int, float)) else [float(s) for s in sigma]
        for i in range(size):
            if random.random() < float(indpb):
                individual[i] += random.gauss(mus[i], sigmas[i])
        return (individual,)


def _setup_deap_classes() -> tuple[Any, Any, Any]:
    if _HAS_DEAP:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        return base.Toolbox, creator.Individual, tools
    return _FallbackToolbox, _FallbackIndividual, _FallbackTools


def run_ga(
    tune_pairs: Sequence[Pair],
    cfg_ga: dict[str, Any],
    base_cfg: dict[str, Any],
    out_yaml: str | Path = "results/best_params.yaml",
    history_csv: str | Path = "results/summary/ga_history.csv",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run DEAP Genetic Algorithm to tune fuzzy sky-dehazing parameters on `tune_pairs`.

    Splits `tune_pairs` into a seeded training subset and validation subset (`val_fraction=0.2`),
    precomputes gene-independent maps at `downscale_max_side`, evolves `population_size`
    individuals over `generations` using tournament selection, blend crossover (`cxBlend`),
    and Gaussian mutation (`mutGaussian`), logs generation history, and saves `best_params.yaml`.

    Args:
        tune_pairs: Sequence of paired tuning images (e.g. 120 pairs from `ots_subset`).
        cfg_ga: GA configuration dictionary loaded from `configs/ga.yaml`.
        base_cfg: Base pipeline configuration dictionary loaded from `configs/default.yaml`.
        out_yaml: Output path for best parameters YAML.
        history_csv: Output path for per-generation convergence history CSV.

    Returns:
        Tuple `(best_params, history_df)`.
    """
    toolbox_cls, individual_cls, ga_tools = _setup_deap_classes()

    seed = int(cfg_ga.get("seed", base_cfg.get("seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    pop_size = int(cfg_ga.get("population_size", 24))
    n_gens = int(cfg_ga.get("generations", 15))
    tournsize = int(cfg_ga.get("tournament_size", 3))
    cx_prob = float(cfg_ga.get("cx_prob", 0.7))
    cx_alpha = float(cfg_ga.get("cx_alpha", 0.5))
    mut_prob = float(cfg_ga.get("mut_prob", 0.3))
    mut_sigma_frac = float(cfg_ga.get("mut_sigma_frac", 0.10))
    hof_size = int(cfg_ga.get("hall_of_fame", 2))
    max_side = int(cfg_ga.get("downscale_max_side", 320))
    lambda_sat = float(cfg_ga.get("lambda_sat", 20.0))
    val_fraction = float(cfg_ga.get("val_fraction", 0.2))
    n_jobs = int(cfg_ga.get("n_jobs", -1))
    bounds: dict[str, list[float]] = cfg_ga["bounds"]

    # Deterministic train / validation split
    indices = np.arange(len(tune_pairs))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(tune_pairs) * val_fraction)))
    val_idx_set = set(indices[:n_val].tolist())

    train_pairs = [p for i, p in enumerate(tune_pairs) if i not in val_idx_set]
    val_pairs = [p for i, p in enumerate(tune_pairs) if i in val_idx_set]

    print(
        f"Preparing GA cache on {len(train_pairs)} train and {len(val_pairs)} val images "
        f"(downscale_max_side={max_side}, seed={seed})..."
    )
    train_cache = prepare_cached_samples(
        train_pairs, base_cfg, max_side=max_side, desc="Cache GA train"
    )
    val_cache = prepare_cached_samples(
        val_pairs, base_cfg, max_side=max_side, desc="Cache GA val"
    )

    # Evaluate hand-set F0 baseline on the exact same cached train/val splits for reference
    f0_ind = [DEFAULT_GENE_VALUES[k] for k in GENE_NAMES]
    f0_params = individual_to_params(f0_ind, base_cfg)
    f0_train = evaluate_params_on_cache(f0_params, train_cache, base_cfg, lambda_sat=lambda_sat)
    f0_val = evaluate_params_on_cache(f0_params, val_cache, base_cfg, lambda_sat=lambda_sat)
    print(
        f"Baseline F0 (default params): Train Fitness={f0_train['fitness']:.3f} "
        f"(PSNR={f0_train['psnr']:.2f} dB, SkyPSNR={f0_train['sky_psnr']:.2f} dB, SkyIoU={f0_train['sky_iou']:.3f}) | "
        f"Val Fitness={f0_val['fitness']:.3f} (PSNR={f0_val['psnr']:.2f} dB, SkyPSNR={f0_val['sky_psnr']:.2f} dB)"
    )

    toolbox = toolbox_cls()

    def _random_individual() -> Any:
        genes = [
            float(rng.uniform(bounds[name][0], bounds[name][1]))
            for name in GENE_NAMES
        ]
        repair_individual(genes, bounds)
        return individual_cls(genes)

    toolbox.register("individual", _random_individual)
    toolbox.register("population", ga_tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", ga_tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", ga_tools.cxBlend, alpha=cx_alpha)

    sigmas = [
        float(mut_sigma_frac * (bounds[name][1] - bounds[name][0]))
        for name in GENE_NAMES
    ]
    toolbox.register("mutate", ga_tools.mutGaussian, mu=0.0, sigma=sigmas, indpb=0.35)

    def _eval_batch(inds: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        param_dicts = [individual_to_params(ind, base_cfg) for ind in inds]
        return list(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(evaluate_params_on_cache)(p, train_cache, base_cfg, lambda_sat)
                for p in param_dicts
            )
        )

    pop = toolbox.population(n=pop_size)
    hof = ga_tools.HallOfFame(hof_size)
    history_rows: list[dict[str, Any]] = []

    for gen in range(0, n_gens + 1):
        t_gen = time.perf_counter()

        # At mid-run (gen == 2), ensure the default F0 anchors one slot if random init hasn't surpassed it yet
        if gen == 2 and len(hof) > 0 and hof[0].fitness.values[0] < f0_train["fitness"]:
            worst_idx = int(np.argmin([ind.fitness.values[0] for ind in pop]))
            seeded_f0 = individual_cls(list(f0_ind))
            pop[worst_idx] = seeded_f0

        invalid_inds = [ind for ind in pop if not ind.fitness.valid]
        if invalid_inds:
            metrics_list = _eval_batch(invalid_inds)
            for ind, m_dict in zip(invalid_inds, metrics_list):
                ind.fitness.values = (m_dict["fitness"],)
                ind.meta = m_dict  # type: ignore[attr-defined]

        hof.update(pop)
        best_ind = hof[0]
        best_meta: dict[str, float] = getattr(
            best_ind,
            "meta",
            evaluate_params_on_cache(
                individual_to_params(best_ind, base_cfg), train_cache, base_cfg, lambda_sat
            ),
        )
        val_meta = evaluate_params_on_cache(
            individual_to_params(best_ind, base_cfg), val_cache, base_cfg, lambda_sat
        )

        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=np.float64)
        elapsed = time.perf_counter() - t_gen

        row = {
            "generation": gen,
            "n_evals": len(invalid_inds),
            "best_fitness": float(best_ind.fitness.values[0]),
            "gen_max_fitness": float(np.max(fits)),
            "mean_fitness": float(np.mean(fits)),
            "std_fitness": float(np.std(fits)),
            "min_fitness": float(np.min(fits)),
            "best_train_psnr": best_meta["psnr"],
            "best_train_sky_psnr": best_meta["sky_psnr"],
            "best_train_sat_pct_sky": best_meta["sat_frac_sky"] * 100.0,
            "best_train_sky_iou": best_meta["sky_iou"],
            "val_fitness": val_meta["fitness"],
            "val_psnr": val_meta["psnr"],
            "val_sky_psnr": val_meta["sky_psnr"],
            "val_sat_pct_sky": val_meta["sat_frac_sky"] * 100.0,
            "val_sky_iou": val_meta["sky_iou"],
            "f0_train_fitness": f0_train["fitness"],
            "f0_val_fitness": f0_val["fitness"],
            "elapsed_s": elapsed,
        }
        history_rows.append(row)
        print(
            f"[Gen {gen:02d}/{n_gens:02d}] best={row['best_fitness']:.3f} | "
            f"mean={row['mean_fitness']:.3f} +/- {row['std_fitness']:.3f} | "
            f"train PSNR={row['best_train_psnr']:.2f} / SkyPSNR={row['best_train_sky_psnr']:.2f} | "
            f"val={row['val_fitness']:.3f} (PSNR={row['val_psnr']:.2f} / SkyPSNR={row['val_sky_psnr']:.2f}) "
            f"[{elapsed:.1f}s]"
        )

        # Save incremental history CSV after every generation
        hist_path = Path(history_csv)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(hist_path, index=False)

        if gen == n_gens:
            break

        # Elitism: preserve Hall-of-Fame individuals + tournament selection for remaining slots
        offspring = toolbox.select(pop, len(pop) - len(hof))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < cx_prob:
                toolbox.mate(child1, child2)
                repair_individual(child1, bounds)
                repair_individual(child2, bounds)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if rng.random() < mut_prob:
                toolbox.mutate(mutant)
                repair_individual(mutant, bounds)
                del mutant.fitness.values

        elites = [toolbox.clone(e) for e in hof]
        pop[:] = elites + offspring

    best_params = individual_to_params(hof[0], base_cfg)
    final_train = evaluate_params_on_cache(best_params, train_cache, base_cfg, lambda_sat=lambda_sat)
    final_val = evaluate_params_on_cache(best_params, val_cache, base_cfg, lambda_sat=lambda_sat)

    payload: dict[str, Any] = {
        **best_params,
        "meta": {
            "seed": seed,
            "population_size": pop_size,
            "generations": n_gens,
            "n_train": len(train_cache),
            "n_val": len(val_cache),
            "downscale_max_side": max_side,
            "lambda_sat": lambda_sat,
            "train_fitness": final_train["fitness"],
            "train_psnr": final_train["psnr"],
            "train_sky_psnr": final_train["sky_psnr"],
            "train_sky_iou": final_train["sky_iou"],
            "val_fitness": final_val["fitness"],
            "val_psnr": final_val["psnr"],
            "val_sky_psnr": final_val["sky_psnr"],
            "val_sky_iou": final_val["sky_iou"],
            "f0_train_fitness": f0_train["fitness"],
            "f0_val_fitness": f0_val["fitness"],
        },
    }

    out_path = Path(out_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    history_df = pd.DataFrame(history_rows)
    return best_params, history_df
