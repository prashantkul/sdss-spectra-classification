# Getting up to speed

Short path for a new contributor or your future self. Read this once, then use the **notebook** as the source of truth for experiments on real data.

## What this repo does

We predict **spectroscopic object class** (STAR, GALAXY, QSO) from **photometry and metadata** similar to SDSS: magnitudes `u, g, r, i, z`, sky position, redshift, and survey IDs. The canonical dataset is on Kaggle ([Stellar Classification Dataset — SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data)); the checked-in CSV lives at `dataset/star_classification.csv`.

This is standard **supervised multiclass classification** with a **Random Forest** + **StandardScaler** baseline.

## Prerequisites

- **Python:** 3.10+ (see `pyproject.toml`).
- **Disk:** ~20 MB for the CSV plus space for a virtual environment.

## Environment setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you use **uv**, `uv sync` from the project root will match `pyproject.toml` / `uv.lock`.

For the Jupyter notebook, ensure you can run a kernel in that environment (e.g. `pip install jupyter` if you do not already have it).

## What to run first

1. **Open `sdss_analysis.ipynb`** — This is the full pipeline on **real** data: EDA, preprocessing, train/test split, metrics, confusion matrices, feature importance, and a dedicated **misclassification** section (§11c). Paths assume `./dataset/star_classification.csv`.

2. **Skim `README.md`** — Dataset citation, high-level results, and links to exported figures under `docs/plots/`.

3. **Browse `lab-notes/`** — Numbered notes on lessons learned, future work, and why errors happen (`01`–`03`).

## How the Python files fit together

| File | Role |
|------|------|
| `sdss_analysis.ipynb` | **Main reference** — experiments on the Kaggle CSV. |
| `load_data.py` | Helpers to load and validate the CSV column schema. |
| `config.py` | Shared configuration/constants (if used by other modules). |
| `analysis.py` | Extra EDA / PCA-style utilities (notebook and scripts may overlap). |
| `main.py` | **Demo script** that trains on **synthetic random data** for a quick smoke test. It does **not** load `dataset/star_classification.csv` by default. For paper-ready numbers, use the notebook instead—or wire `main.py` to `load_sdss_data()` if you want one CLI entrypoint on real rows. |

## If something fails

- **`FileNotFoundError` for the CSV** — Download the Kaggle dataset, place `star_classification.csv` under `dataset/`, or change the path in the first data-loading cell of the notebook.
- **Class names** — The real CSV uses `GALAXY`, `STAR`, `QSO` (see notebook / `LabelEncoder`). `main.py` uses lowercase fake labels only for its toy generator.
- **Replicating README plots** — Run the notebook; figures checked into `docs/plots/` are snapshots from prior notebook runs.

## Suggested reading order

1. This file  
2. `README.md` (results + layout)  
3. `sdss_analysis.ipynb` (run top to bottom once)  
4. `lab-notes/01-lessons-learned.md` … `03-why-misclassifications-happen.md`  

When you change the model or data pipeline, update the notebook first; refresh `docs/plots/` and the README metrics only after you are happy with the new outputs.
