# Fuzzy Sky-Aware Dehazing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Training-free single-image dehazing using a Fuzzy C-Means soft sky membership and continuous transmission blending.**

The Dark Channel Prior (DCP) removes haze well on most outdoor scenes but breaks on bright sky regions, while standard binary sky masks leave a visible step-edge seam at the horizon. This project estimates a continuous **Fuzzy C-Means (FCM) soft sky membership** `mu(x) in [0, 1]` over pixel brightness, saturation, texture, and vertical position, and blends sky and non-sky transmission maps smoothly before guided-filter refinement. Fuzzy parameters are tuned with a **Genetic Algorithm (DEAP)**.

![SOTS-Outdoor 0001 B0 Comparison](results/images/sots_outdoor/B0/0001_compare_hazy_dehazed_clear.png)
*Example (`SOTS-outdoor 0001`): Hazy input (left), Plain DCP B0 output showing sky over-correction (middle), Clear ground truth (right).*

---

## Pipeline Overview

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

## Running the Benchmark

```bash
# 1. Tune hard-mask threshold (B1)
python scripts/tune_threshold.py --config configs/default.yaml

# 2. Tune fuzzy parameters with Genetic Algorithm (F1)
python scripts/tune_ga.py --config configs/ga.yaml

# 3. Run benchmark across datasets
python scripts/run_benchmark.py --config configs/default.yaml \
    --datasets sots_outdoor ohaze --methods B0 B1 B2 F0 F1

# 4. Generate figures and summary tables
python scripts/make_figures.py --results results/per_image --out results/figures
```

---

## Results

**SOTS-outdoor (500 test pairs)**

| Method | PSNR (dB) | SSIM | dE2000 | Sky PSNR | Sky dE2000 | Non-Sky PSNR | Sky Sat. % | Seam Error |
|---|---|---|---|---|---|---|---|---|
| **B0 Plain DCP (all)** | 17.05 | 0.8574 | 10.75 | 15.30 | 14.83 | 20.68 | 4.12% | 0.1459 |
| **B0 Plain DCP (sky-heavy)** | 15.97 | 0.8410 | 11.85 | 13.10 | 18.22 | 21.27 | 4.55% | 0.1584 |
| **B1 Hard Threshold** | | | | | | | | |
| **B2 Hard FCM** | | | | | | | | |
| **F0 Fuzzy (Default)** | | | | | | | | |
| **F1 Fuzzy + GA** | | | | | | | | |

**O-HAZE (45 real outdoor pairs)**

| Method | PSNR (dB) | SSIM | dE2000 | Sky PSNR | Sky dE2000 | Non-Sky PSNR | Sky Sat. % | Seam Error |
|---|---|---|---|---|---|---|---|---|
| **B0 Plain DCP (all)** | 15.86 | 0.6448 | 18.11 | 14.67 | 17.00 | 16.03 | 2.40% | 0.1608 |
| **B1 Hard Threshold** | | | | | | | | |
| **B2 Hard FCM** | | | | | | | | |
| **F0 Fuzzy (Default)** | | | | | | | | |
| **F1 Fuzzy + GA** | | | | | | | | |

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
