GlobalBrine-LithiumModel
==============================

Global ML model for lithium extraction metrics from brine chemistry and irradiance.

Versioning & Release
------------

- Current version: `0.2.1`
- Changelog: `CHANGELOG.md`
- Tagging checklist: `RELEASING.md`

Reproducibility (Quickstart)
------------

Reproduce predictions using the included pipeline and checkpoints:

```bash
make create_environment
conda activate global-brine-lithium-model
make requirements

python src/models/predict_brines.py \
  --processed-dir data/processed \
  --out-dir data/predictions
```

Tutorials:

- `docs/REPRODUCIBILITY.md`
- `docs/INFERENCE.md`

Web Platform
------------

An end-to-end web UI lives in `web/` (FastAPI backend + React/Tailwind frontend).

```bash
# Backend (from repo root)
python -m pip install -r requirements.txt
uvicorn web.backend.main:app --reload

# Frontend (new terminal)
cd web/app && npm install && npm run dev  # proxies /api to localhost:8000
```

**Public site**

- Production (DigitalOcean): https://globalbrine-web-gwvzz.ondigitalocean.app/

Homepage preview:

![GlobalBrine homepage](web/app/public/homepage.png)

**Model pipeline overview**

![GlobalBrine pipeline](assets/figures/Enhanced_Global_Lithium_Pipeline_Diagram.svg)

Environment Setup
------------

### Prerequisites

- Python >= 3.10 + `pip`
- Optional: `conda`/`mamba` (for environment management)
- Optional: `make` (to use the included `Makefile` shortcuts)

### Option A: Conda/Mamba (recommended)

```bash
# defaults to PYTHON_VERSION=3.10 (override: make create_environment PYTHON_VERSION=3.11)
make create_environment  
conda activate global-brine-lithium-model
make requirements
```

### Option B: `venv` (no conda)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

### Verify the environment

```bash
make test_environment
```

### Environment variables

Create a local `.env` file at the repo root for secrets/config (it is gitignored). Python scripts load it via `python-dotenv` (see `src/data/make_dataset.py`).

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
