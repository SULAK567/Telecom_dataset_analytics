# Telecom Network Coverage Analytics & Regional Clustering (India)

A data pipeline and interactive dashboard that turns raw, carrier-agnostic cell-tower
records into a regional (state and district level) classification of mobile network
coverage across India — with clustering, a supervised coverage-tier model, and an
interactive Streamlit dashboard for exploration.

B.Tech Computer Engineering mini project — Institute of Advanced Research (IAR), Gandhinagar.

## Overview

Mobile coverage in India is reported separately by each carrier (Airtel, Vi, Jio/Reliance,
BSNL), with no independent, carrier-agnostic way to compare regions. This project ingests
raw tower-level data, cleans and aggregates it to state and district level, engineers
coverage-related features, classifies regions using both a rule-based score and unsupervised
clustering (K-Means, DBSCAN), and trains a supervised Random Forest model (with SHAP
explanations) to predict coverage tier — all surfaced through an interactive dashboard.

## Tech stack

- **Language:** Python 3.10
- **Data processing:** pandas, numpy
- **Geospatial:** geopandas, shapely
- **Machine learning:** scikit-learn (K-Means, DBSCAN, PCA, Random Forest), shap
- **Dashboard / visualization:** Streamlit, Plotly, matplotlib
- **Storage:** flat files only — CSV, GeoJSON, Parquet (no database server; this is an
  offline analytical pipeline, not a transactional application)
- **Model persistence:** joblib

## Project structure

    Telecom_dataset_analytics/
    ├── common/                  # shared config and viz-style helpers, used across folders
    │   ├── config.py
    │   └── viz_style.py
    │
    ├── datasets/                # all data + the code that builds it
    │   ├── raw/                 # source data, exactly as downloaded
    │   ├── processed/           # everything the pipeline produces (aggregates, features, boundaries)
    │   └── scripts/             # ingestion, cleaning, and state/district aggregation
    │
    ├── backend/                 # feature engineering, EDA, clustering, supervised modelling
    │   └── run_all.py           # runs the full pipeline end to end
    │
    ├── frontend/                # the Streamlit dashboard
    │   └── dashboard.py
    │
    ├── models/                  # trained model artifacts (.joblib)
    ├── outputs/                 # generated figures and stage reports
    └── docs/                    # project write-ups and blueprint

Large dataset files (`datasets/raw/404.csv`, `datasets/processed/tower_table_clean.csv`,
and others) are tracked with **Git LFS** — see setup below.

## Getting started

**1. Clone the repo** (Git LFS must be installed first — see [git-lfs.com](https://git-lfs.com)
if you don't have it):

    git lfs install
    git clone https://github.com/SULAK567/Telecom_dataset_analytics.git
    cd Telecom_dataset_analytics

**2. Set up a virtual environment and install dependencies:**

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

**3. Confirm the large files pulled correctly** (they should show their full size, not a
few hundred bytes):

    ls -lh datasets/raw/404.csv datasets/processed/tower_table_clean.csv

If either looks tiny, Git LFS didn't pull automatically — run `git lfs pull` and check again.

## Running the pipeline

The full pipeline (feature engineering through model training) runs from the repo root:

    python3 backend/run_all.py

By default this starts from step 02 — step 01 (the raw tower-table build) is skipped since
its output (`tower_table_clean.csv`) already ships with the repo. Resume from a specific
step with `--from`, e.g. `python3 backend/run_all.py --from 06`.

## Running the dashboard

    streamlit run frontend/dashboard.py

Opens an interactive dashboard for exploring regional coverage: view the coverage map,
select and compare regions, filter by carrier or network generation, inspect the
clustering-based and rule-based coverage classifications, and export the underlying data.

## Data sources

- **Cell-tower data:** a public, crowd-sourced dataset for India (via Kaggle, originally
  [OpenCelliD](https://opencellid.org/)) — tower location, radio type/generation, and
  carrier identifiers.
- **District boundaries:** [DataMeet's](https://github.com/datameet) open India-districts
  GeoJSON repository (Census 2011).

## Models

Trained models are stored in `models/`:

- `cluster_model_state.joblib`, `cluster_model_district.joblib` — unsupervised clustering
- `rf_coverage_tier.joblib` — supervised Random Forest coverage-tier classifier

## License

MIT — see [LICENSE](LICENSE).

## Team

- Abhishek Rathod
- Sulaksh Patanaik
- Harshil Darji

**Guide:** Ms. Niral Jadav
**Institute:** Institute of Advanced Research (IAR), Gandhinagar
