"""
Central configuration for the Telecom Network Coverage Analytics pipeline.

Every script imports paths from here so the pipeline can be relocated by
editing one file.
"""
from pathlib import Path

# Project root = parent of common/
ROOT = Path(__file__).resolve().parent.parent

# ---- Raw inputs (as downloaded) ----
RAW_DIR = ROOT / "datasets" / "raw"
RAW_404 = RAW_DIR / "404.csv"
RAW_405 = RAW_DIR / "405.csv"
OPERATOR_LOOKUP = RAW_DIR / "india_mcc_mnc_operator_circle.csv"
STATES_GEOJSON = RAW_DIR / "states_india.geojson"

# ---- Intermediate / processed data ----
DATA = ROOT / "datasets" / "processed"
TOWER_TABLE_V1 = DATA / "tower_table_clean.csv"          # output of 01_build_tower_table.py
TOWER_TABLE_V2 = DATA / "tower_table_v2.csv"             # after QC, bbox filter, dedupe
DISTRICTS_GEOJSON = DATA / "districts_india.geojson"
TOWERS_WITH_DISTRICT = DATA / "tower_district_lookup.parquet"
STATE_AGG = DATA / "state_aggregated.csv"
DISTRICT_AGG = DATA / "district_aggregated.csv"
STATE_FEATURES = DATA / "state_features.csv"
DISTRICT_FEATURES = DATA / "district_features.csv"
STATE_CLUSTERED = DATA / "state_clustered.csv"
DISTRICT_CLUSTERED = DATA / "district_clustered.csv"

# ---- Outputs ----
OUT = ROOT / "outputs"
FIGURES = OUT / "figures"
MODELS = ROOT / "models"
REPORTS = OUT / "reports"

for _d in (DATA, FIGURES, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Analysis constants ----

# India bounding box (generous, includes Andaman & Nicobar and Lakshadweep).
INDIA_BBOX = dict(lon_min=68.0, lon_max=97.5, lat_min=6.0, lat_max=37.7)

# Carriers still operating (used for the "active carrier diversity" feature).
ACTIVE_CARRIERS = ["Airtel", "Vi", "Jio/Reliance", "BSNL"]

RADIO_TYPES = ["GSM", "UMTS", "LTE", "NR", "CDMA"]

# Human-readable generation labels for the radio field.
RADIO_TO_GEN = {
    "GSM": "2G",
    "CDMA": "2G",
    "UMTS": "3G",
    "LTE": "4G",
    "NR": "5G",
}

RANDOM_STATE = 42

# Memory-efficient dtypes for the 2.5M-row tower table.
TOWER_DTYPES = {
    "radio": "category",
    "mcc": "int16",
    "mnc": "int16",
    "lac": "int32",
    "cid": "int64",
    "long": "float32",
    "lat": "float32",
    "range": "int32",
    "sample": "int32",
    "created": "int64",
    "updated": "int64",
    "carrier": "category",
    "circle": "category",
    "st_nm": "category",
}


# ---------------------------------------------------------------------------
# Safe geospatial write.
#
# GDAL/pyogrio DELETES an existing file before rewriting it. On filesystems
# where unlink is not permitted (sandboxed or read-restricted mounts, some
# network shares) that raises PermissionError even though the file is
# perfectly writable. Writing to a temporary file and then copying the bytes
# over the destination truncates in place instead of unlinking, so re-running
# the pipeline never fails on an existing output.
# ---------------------------------------------------------------------------
def write_geojson(gdf, path):
    """Write a GeoDataFrame to GeoJSON, overwriting without unlinking."""
    import shutil
    import tempfile
    from pathlib import Path as _P

    path = _P(path)
    with tempfile.TemporaryDirectory() as td:
        tmp = _P(td) / "out.geojson"
        gdf.to_file(tmp, driver="GeoJSON")
        if path.exists():
            shutil.copyfile(tmp, path)      # truncate + write, no unlink
        else:
            shutil.move(str(tmp), str(path))
    return path


# ---------------------------------------------------------------------------
# Human-readable feature names.
# The modelling code uses snake_case column names; a dashboard aimed at
# newcomers should not show them "log_sample_mean" and expect it to mean
# anything. Charts and controls look names up here and fall back to the raw
# column when a label is missing.
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "log_tower_density": "Tower density (log)",
    "tower_density": "Towers per km²",
    "tower_count": "Total towers",
    "pct_2g": "2G share (%)",
    "pct_3g": "3G share (%)",
    "pct_4g": "4G share (%)",
    "pct_5g": "5G share (%)",
    "pct_modern": "4G + 5G share (%)",
    "active_carrier_count": "Operators present",
    "carrier_count": "Distinct carriers",
    "carrier_hhi": "Market concentration (HHI)",
    "range_mean": "Mean cell range (m)",
    "range_median": "Median cell range (m)",
    "sample_mean": "Samples per tower",
    "log_sample_mean": "Measurement density (log)",
    "samples_per_sqkm": "Samples per km²",
    "gen_score": "Generation score (2-5)",
    "area_sqkm": "Area (km²)",
    "towers_per_carrier": "Towers per operator",
    "updated_year_mean": "Mean year last updated",
    "created_year_mean": "Mean year first recorded",
    "pct_airtel": "Airtel share (%)",
    "pct_vi": "Vi share (%)",
    "pct_jio_reliance": "Jio/Reliance share (%)",
    "pct_bsnl": "BSNL share (%)",
    "rule_score": "Rule-based score",
}


def label_of(col):
    """Friendly name for a column, falling back to the raw name."""
    return FEATURE_LABELS.get(col, col)
