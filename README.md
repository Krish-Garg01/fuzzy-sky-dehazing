# Fuzzy Sky-Aware Dehazing

**Training-free single-image dehazing with a soft (fuzzy) sky membership.**

The Dark Channel Prior (DCP) removes haze well on most outdoor scenes but breaks on sky. This project replaces the usual hard sky mask with a **Fuzzy C-Means soft membership** `mu(x) in [0,1]` and blends sky and non-sky transmission estimates continuously, so the horizon has no visible seam. Fuzzy and membership parameters are tuned with a **Genetic Algorithm**.

> **Status:** in development. Results tables below are filled in after running the benchmark.

---

## Table of contents

- [Motivation](#motivation)
- [Method](#method)
- [Features](#features)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Results](#results)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Limitations](#limitations)
- [Related work](#related-work)
- [Citation and references](#citation-and-references)
- [License](#license)

---

## Motivation

Hazy images follow the atmospheric scattering model:

```
I(x) = J(x) * t(x) + A * (1 - t(x))
```

DCP estimates the transmission `t` assuming every small patch of a haze-free outdoor image has a near-zero colour channel. Sky, water and white walls violate this assumption: they are bright in all channels, so DCP reads them as "extremely hazy", estimates `t` close to 0.05, and the recovery step `J = (I - A) / max(t, t0) + A` amplifies noise and colour casts roughly tenfold.

The standard fix is a **hard sky mask**, but two almost identical pixels on either side of the threshold then receive completely different treatment, leaving a visible seam at the horizon.

**Hypothesis tested here:** a fuzzy blend reduces colour distortion and pixel saturation in sky regions relative to both plain DCP and hard-masked DCP, without degrading non-sky restoration quality.

## Method

```
hazy image -> atmospheric light A, t_DCP
           -> per-pixel sky features (brightness, low saturation, texture, vertical position)
           -> Fuzzy C-Means -> sky cluster -> soft membership mu(x)
           -> t_sky(x) = max(t_DCP, t_floor)
           -> t_blend = mu * t_sky + (1 - mu) * t_DCP
           -> guided-filter refinement -> t_final
           -> J = (I - A) / max(t_final, t0) + A
```

Key design points:

- **Soft, not hard:** `mu` is continuous, so the transition at the horizon is a ramp.
- **No-sky guard:** if no real sky is found, `mu = 0` and the method reduces exactly to plain DCP.
- **Fair comparison:** every method shares the same atmospheric-light estimator, DCP parameters and guided filter. Only the sky treatment differs.
- **GA tuning:** only fuzzy-specific parameters are tuned (fuzzifier, feature weights, membership edges, `t_floor`), on a set disjoint from the test data.

All hyperparameter ranges and defaults are configured in [`configs/default.yaml`](configs/default.yaml) and [`configs/ga.yaml`](configs/ga.yaml).

## Features

- DCP baseline (He et al.) with guided-filter refinement
- Own NumPy implementation of Fuzzy C-Means (cross-checked against `scikit-fuzzy` in tests)
- Hard-mask baselines: threshold mask and hardened-FCM mask (same clustering as the fuzzy method)
- Genetic-algorithm tuning with DEAP
- Evaluation: PSNR, SSIM, CIEDE2000, sky-only and non-sky metrics, saturated-pixel percentage, horizon-seam error, runtime
- Paired statistics (Wilcoxon, bootstrap CIs, win rates) on all images and on a sky-heavy subset
- One-command scripts for benchmarks, tuning and figures

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Krish-Garg01/fuzzy-sky-dehazing.git
cd fuzzy-sky-dehazing
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Optional tooling (only for sky ground-truth masks and no-reference metrics):

```bash
pip install -r requirements-tools.txt    # torch, transformers, pyiqa
```

A laptop CPU is enough for the core method; no GPU is required.

## Data setup

Datasets are not included. Download them yourself and respect their licenses (research and coursework use).

| Dataset | Use | Source |
|---|---|---|
| RESIDE SOTS-outdoor | Main test set (500 synthetic pairs) | github.com/BookerDeWitt/RESIDE |
| RESIDE OTS (small subset, about 100-200 images) | GA and threshold tuning | same |
| O-HAZE | Real haze with ground truth (45 pairs) | data.vision.ee.ethz.ch/cvl/ntire18/o-haze/ |
| RTTS / HSTS subset | Qualitative and no-reference evaluation | RESIDE |

Expected layout:

```
data/
  reside/sots_outdoor/{hazy,clear}/
  reside/ots_subset/{hazy,clear}/
  ohaze/{hazy,GT}/
  rtts_subset/
  sky_masks/            # generated, see below
```

Edit paths in `configs/datasets.yaml`. Then generate ground-truth sky masks (evaluation only; the dehazing method never uses them):

```bash
python scripts/make_sky_gt.py --dataset sots_outdoor --preview
```

Always inspect the preview overlays: the masks are pseudo ground truth from a pretrained segmenter.

## Quick start

Dehaze a single image and save debug maps (`t_dcp`, `mu`, `t_final`):

```bash
python -m fsdehaze.cli dehaze \
    --input examples/hazy.jpg \
    --method fuzzy \
    --params results/best_params.yaml \
    --out out.png --save-debug
```

Methods: `dcp` (B0), `hard_threshold` (B1), `hard_fcm` (B2), `fuzzy` (F0 defaults, F1 with tuned `--params`).

Python API:

```python
from fsdehaze.io_utils import load_image, save_image
from fsdehaze.pipelines import dehaze_fuzzy
from fsdehaze.config import load_config

cfg = load_config("configs/default.yaml")
result = dehaze_fuzzy(load_image("hazy.jpg"), cfg)
save_image("dehazed.png", result.J)
```

## Reproducing the experiments

```bash
# 1. tune the hard-mask threshold (fair baseline)
python scripts/tune_threshold.py --config configs/default.yaml

# 2. tune fuzzy parameters with the GA
python scripts/tune_ga.py --config configs/ga.yaml

# 3. run all methods on all datasets
python scripts/run_benchmark.py --config configs/default.yaml \
    --datasets sots_outdoor ohaze --methods B0 B1 B2 F0 F1

# 4. figures and tables
python scripts/make_figures.py --results results/per_image --out results/figures
```

All runs are seeded (default `42`), resumable, and write one CSV row per image per method to `results/per_image/`.

| ID | Method |
|---|---|
| B0 | Plain DCP |
| B1 | DCP + hard threshold sky mask |
| B2 | DCP + hardened FCM mask (same clustering as fuzzy) |
| B3 | Learning-based reference (SFD-Net or substitute) |
| F0 | Fuzzy blend, default parameters |
| F1 | Fuzzy blend, GA-tuned (proposed) |

## Results

*To be filled in after running the benchmark. Do not report numbers that were not produced by the scripts above.*

**SOTS-outdoor, all images**

| Method | PSNR | SSIM | dE2000 | Sky PSNR | Sky dE | Non-sky PSNR | Saturated % (sky) | Seam error |
|---|---|---|---|---|---|---|---|---|
| B0 DCP | | | | | | | | |
| B1 Hard threshold | | | | | | | | |
| B2 Hard FCM | | | | | | | | |
| F0 Fuzzy (default) | | | | | | | | |
| F1 Fuzzy + GA | | | | | | | | |

Also report the **sky-heavy subset** (sky fraction above 20%), O-HAZE results, the ablation table, and the zoomed horizon-crop figure (hazy | B0 | B1 | B2 | F1 | ground truth).

Published plain-DCP numbers on SOTS-outdoor vary by implementation (roughly 18.5-19.1 dB PSNR); use that as a sanity range for your B0.

## Project structure

```
fuzzy-sky-dehazing/
├── README.md
├── configs/                 # default.yaml, ga.yaml, datasets.yaml
├── src/fsdehaze/            # dcp, guided_filter, fcm, sky_*, pipelines, metrics, tuning, eval, viz
├── scripts/                 # make_sky_gt, run_benchmark, tune_ga, tune_threshold, make_figures
├── tests/                   # pytest unit tests
├── notebooks/
├── data/                    # git-ignored (except sky_fraction.csv metadata)
├── results/                 # per-image CSVs, summary tables, visual outputs
└── report/                  # project documentation and figures
```

## Configuration

All hyperparameters live in `configs/default.yaml` (DCP patch, `omega`, `t0`, guided-filter settings, FCM clusters/fuzzifier, feature weights, smoothstep edges, `t_floor`, no-sky guard thresholds, evaluation settings). GA settings are in `configs/ga.yaml`.

## Testing

```bash
pytest -q
```

Tests cover DCP on synthetic haze with known `t` and `A`, guided-filter properties, FCM correctness (including a cross-check with `scikit-fuzzy`), sky membership on synthetic scenes, the invariant that `mu = 0` reproduces plain DCP exactly, and metric correctness.

## Limitations

- DCP-family methods are far below deep networks on synthetic benchmarks; this project targets the sky failure mode, not state-of-the-art PSNR.
- Sky ground-truth masks are pseudo labels from a segmenter unless annotated by hand.
- FCM can mistake bright facades, snow or white walls for sky; the no-sky guard and position feature reduce but do not eliminate this.
- SOTS is synthetic; real haze (O-HAZE, RTTS) is smaller or lacks ground truth.
- Night-time, non-homogeneous haze and video are out of scope.

## Related work

Sky handling in DCP and fuzzy-membership dehazing have prior work, including Park et al. (fuzzy membership + variational refinement, 2017), Lv et al. (sky segmentation + haze density, 2022), Shu et al. (weighted fusion of foreground and sky transmission, 2019), and several segmentation-based DCP variants. This project's contribution is the combination of multi-feature FCM soft membership, continuous blending and GA tuning, evaluated against hard masks with identical clustering.

## Citation and references

- K. He, J. Sun, X. Tang. *Single Image Haze Removal Using Dark Channel Prior.* TPAMI 2011.
- K. He, J. Sun, X. Tang. *Guided Image Filtering.* TPAMI 2013.
- J. C. Bezdek et al. *FCM: The Fuzzy C-Means Clustering Algorithm.* Computers & Geosciences, 1984.
- B. Li et al. *Benchmarking Single Image Dehazing and Beyond (RESIDE).* TIP 2019.
- C. Ancuti et al. *O-HAZE.* CVPR Workshops (NTIRE) 2018.
- F.-A. Fortin et al. *DEAP: Evolutionary Algorithms Made Easy.* JMLR 2012.

If you use this code, cite this repository:

```bibtex
@misc{fuzzy_sky_dehazing,
  title  = {Fuzzy Sky-Aware Transmission Refinement for Training-Free Single-Image Dehazing},
  author = {Krish Garg},
  year   = {2026}
}
```

## License

Code: MIT. Datasets keep their own licenses and are not redistributed here.

## Acknowledgements

RESIDE and O-HAZE authors for the benchmarks; the original DCP and guided-filter authors.
