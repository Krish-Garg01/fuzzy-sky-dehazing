# Fuzzy Sky-Aware Dehazing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Training-free single-image dehazing using a Fuzzy C-Means soft sky membership, continuous transmission blending, and Genetic Algorithm tuning.**

---

## Project Overview

Outdoor photographs taken in haze or fog follow the atmospheric scattering model $I(x) = J(x)\,t(x) + A\,(1 - t(x))$. The classical **Dark Channel Prior (DCP)** removes haze effectively on textured foreground scenes by assuming every small patch has at least one dark colour channel. However, this assumption fails on **sky regions**, which are bright in all three channels ($I(x) \approx A$). As a result, plain DCP underestimates sky transmission near zero ($t \approx 0.05 \to t_0 = 0.10$) and amplifies noise, JPEG blocks, and colour casts roughly tenfold during recovery.

While binary (hard) sky masks are commonly used to separate sky from non-sky regions, a hard threshold introduces a step discontinuity at the boundary—leaving a visible halo or "cut-out" seam along the horizon.

### Core Approach
This project replaces hard sky segmentation with a **continuous Fuzzy C-Means (FCM) soft sky membership** $\mu(x) \in [0, 1]$ and blends sky and non-sky transmission estimates smoothly before edge-preserving refinement:

1. **Multi-Feature Sky Cues (`sky_features.py`):** Extracts per-pixel brightness, low HSV saturation, local Sobel gradient smoothness, and normalized vertical position.
2. **Fuzzy C-Means Soft Membership (`fcm.py`, `sky_membership.py`):** Clusters pixel features via a custom NumPy FCM implementation, identifies the sky cluster automatically, applies a **no-sky guard** (falling back to $\mu = 0$ when no sky is present), and refines the membership with a `smoothstep(a, b)` ramp and guided filter.
3. **Continuous Transmission Blending (`sky_transmission.py`, `pipelines.py`):** Blends a protected sky transmission map $t_{\text{sky}}(x) = \max(t_{\text{DCP}}(x), t_{\text{floor}})$ with $t_{\text{DCP}}(x)$ using $t_{\text{blend}} = \mu\,t_{\text{sky}} + (1 - \mu)\,t_{\text{DCP}}$, followed by guided-filter refinement ($r = 60, \epsilon = 10^{-4}$).
4. **Genetic Algorithm Tuning (`tuning/ga.py`):** Tunes the fuzzy-specific parameters (fuzzifier $m$, feature weights, smoothstep bounds, smoothing radius, and $t_{\text{floor}}$) with DEAP on a disjoint outdoor tuning set.
5. **Controlled Evaluation (`metrics.py`, `eval/`):** Compares the soft fuzzy blend (`F0`, `F1`) against plain DCP (`B0`), a tuned hard-threshold mask (`B1`), and a hardened FCM mask (`B2`, using the exact same clustering with $\mu_{\text{raw}} > 0.5$) across full-image, sky-only, non-sky, pixel saturation, and horizon-seam metrics.

```text
hazy image -> atmospheric light A, coarse t_DCP
           -> per-pixel sky features (brightness, low saturation, texture, vertical position)
           -> Fuzzy C-Means -> sky cluster -> soft membership mu(x) in [0, 1] (with no-sky guard)
           -> t_sky(x) = max(t_DCP, t_floor)
           -> t_blend = mu * t_sky + (1 - mu) * t_DCP
           -> guided-filter refinement -> t_final
           -> J = clip((I - A) / max(t_final, t0) + A, 0, 1)
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Krish-Garg01/fuzzy-sky-dehazing.git
cd fuzzy-sky-dehazing
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Optional tooling (used only offline for SegFormer sky masks or no-reference metrics):

```bash
pip install -r requirements-tools.txt
```

---

## Dataset Layout

Place benchmark datasets under `data/` (paths are configured in `configs/datasets.yaml`):

```text
data/
  reside/sots_outdoor/{hazy,clear}/     # 500 synthetic outdoor test pairs
  reside/ots_subset/{hazy,clear}/       # 120 outdoor tuning pairs
  ohaze/{hazy,GT}/                      # 45 real outdoor haze pairs
  sky_masks/                            # evaluation-only binary sky masks
```

Generate evaluation sky masks from the clear images:

```bash
python scripts/make_sky_gt.py --datasets sots_outdoor ots_subset ohaze --preview
```

---

## Quick Start

**CLI:**

```bash
python -m fsdehaze.cli dehaze \
    --input path/to/hazy.jpg \
    --method dcp \
    --out results/out.png \
    --save-debug
```

Methods: `dcp` (`B0`), `hard_threshold` (`B1`), `hard_fcm` (`B2`), `fuzzy` (`F0` default, `F1` with `--params results/best_params.yaml`).

**Python API:**

```python
from fsdehaze.config import load_config
from fsdehaze.io_utils import load_image, save_image
from fsdehaze.pipelines import dehaze_dcp

cfg = load_config("configs/default.yaml")
result = dehaze_dcp(load_image("path/to/hazy.jpg"), cfg)
save_image("results/out.png", result.J)
```

---

## Running the Experiments

```bash
# 1. Tune hard-mask threshold (B1)
python scripts/tune_threshold.py --config configs/default.yaml

# 2. Tune fuzzy parameters with Genetic Algorithm (F1)
python scripts/tune_ga.py --config configs/ga.yaml

# 3. Run benchmark across datasets (saves CSVs and visual outputs under results/)
python scripts/run_benchmark.py --config configs/default.yaml \
    --datasets sots_outdoor ohaze --methods B0 B1 B2 F0 F1

# 4. Generate figures and summary tables
python scripts/make_figures.py --results results/per_image --out results/figures
```

---

## Project Structure

```text
fuzzy-sky-dehazing/
├── README.md
├── configs/                 # default.yaml, ga.yaml, datasets.yaml
├── src/fsdehaze/            # dcp, guided_filter, fcm, sky_*, pipelines, metrics, tuning, eval, viz
├── scripts/                 # make_sky_gt, run_benchmark, tune_ga, tune_threshold, make_figures
├── tests/                   # pytest unit tests
├── data/                    # datasets (git-ignored) and sky_fraction.csv
└── results/                 # per_image CSVs, summary tables, visual outputs
```

## Testing

```bash
pytest -q
```

## License

MIT License. Benchmark datasets retain their original licenses and are not redistributed here.
