# MultiPruneEnsemble

**Pruning-ensemble with gradient-weighted integration for decision trees.**  
This repository is the experiment code for building multiple pruned subtrees from a parent CART tree, ensembling them, and learning softmax weights on a validation split using cross-entropy. Metrics used throughout: **Accuracy, F1, AUC**.

> Tested locally on **Python 3.8.10**. First run will download datasets from **PMLB**; after that, datasets are cached in `data/` and runs are offline.

---

## 1) Introduction

MultiPruneEnsemble implements a practical pipeline for decision-tree ensembles:

1) train a parent tree;  
2) produce several **pruned** variants via multiple pruning methods;  
3) combine them with either **Equal-Average** or **Grad-Weighted** fusion (softmax weights optimized on a validation set);  
4) evaluate on **ACC/F1/AUC** and report mean/variance across seeds.

The goal is to retain **interpretability** while improving stability and predictive performance on tabular data.

---

## 2) Features (brief)

- Multiple pruning variants to generate structural diversity.  
- Two ensemble modes: **Equal-Average** and **Grad-Weighted** (softmax-constrained weights).  
- Single entry point: `run_pmlb.py`; outputs organized under `outputs/`.

*For theory and extended method details, please refer to the paper; this README focuses on running the code.*

---

## 3) Installation

Create an environment (recommended) and install minimal dependencies.

```bash
# clone
git clone <your-repo-url>
cd MultiPruneEnsemble

# minimal requirements
# (put this in requirements.txt if you want a file)
pip install numpy>=1.20 scikit-learn>=1.2 scipy>=1.8 pmlb>=1.0
```

**Data:** On the first run, datasets are fetched from **PMLB** and cached into `data/`.  
On later runs the code first checks `data/` and skips downloading if the dataset already exists.

---

## 4) Quick Start

Run with defaults:

```bash
python run_pmlb.py
```

Specify a dataset (example: Adult):

```bash
python run_pmlb.py --dataset adult
```

See all available CLI flags:

```bash
python run_pmlb.py --help
```

**Outputs:** results and figures are written under `outputs/{dataset}/`. Cached datasets live in `data/`.

---

## 5) Reproduction

- **Seed range:** use integer seeds from **41 to 50** (inclusive).  
- **Splits:** train / **pruning-validation** / test (see `run_pmlb.py` for the exact implementation).  
- **Metrics:** computed with `scikit-learn`; report **mean and variance** across seeds.

Typical workflow:

```bash
# 1) run default config
python run_pmlb.py --dataset adult

# 2) adjust key hyperparameters in run_pmlb.py (or via CLI flags, see --help)
#    dmax / nmin / M / K, then re-run
python run_pmlb.py --dataset adult --dmax 8 --nmin 5 --M 128 --K 32
```

---

## 🔧 Key Parameters

All core parameters are defined at the top of **`run_pmlb.py`** and can be edited directly or passed via CLI.

| Name            | Default  | Meaning |
|:----------------|:--------:|:--------|
| `CACHE_DIR`     | `r""`    | Cache directory for datasets. Empty string means use `data/` under the repo root. |
| `DATASETS`      | `[]`     | List of PMLB dataset names to run (e.g., `"adult"`, `"mushroom"`, `"sonar"`). Leave empty to use script defaults. |
| `N_ESTIMATORS`  | `128`    | Number of candidate subtrees produced from the parent model before selection. |
| `K_KEEP`        | `32`     | Number of pruned subtrees kept in the ensemble. |
| `D_MAX`         | `None`   | Maximum depth of trees; `None` means no limit (scikit-learn default). |
| `N_MIN`         | `1`      | Minimum samples per leaf (`min_samples_leaf`). |
| `SEED`          | `42`     | Random seed for reproducibility. Use the range **41–50** for reported results. |

**Tips**
- Keep dataset and seed fixed when comparing hyperparameters.  
- If you prefer not to pass CLI flags, modify the defaults in `run_pmlb.py` and re-run.

---

## 6) Repository Structure

```
.
├── data/            # PMLB cache (created on first download)
├── methods/         # Pruning methods (e.g., MRMR / DISC / AUC-Greedy / EPIC / UMEP / Hybrid)
├── ensemble/        # Ensemble strategies (Equal-Average, Grad-Weighted)
├── scripts/         # Batch runs / plotting utilities (if provided)
├── outputs/         # Results and figures per dataset
├── checkpoints/     # Optional: saved models/weights (if used)
├── run_pmlb.py      # ★ Main entry point (supports --help)
├── utils.py         # Splits, metrics, calibration, misc helpers
└── __pycache__/     # Python bytecode cache
```

---

## 7) Citation

If this code is helpful in your research, please cite:

```bibtex
@misc{MultiPruneEnsemble2025,
  title        = {MultiPruneEnsemble: Pruning-ensemble with gradient-weighted integration},
  author       = {Yin, Hanyang},
  year         = {2025},
  note         = {Project Bellerophon},
  howpublished = {\url{https://github.com/suonion/MultiPruneEnsemble}}
}
