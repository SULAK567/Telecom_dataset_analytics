"""
Phase 2 — Regional enrichment at STATE level.

Input : data_processed/tower_table_v2.csv, states_india.geojson
Output: data_processed/state_aggregated.csv
        outputs/reports/02_state_aggregation_report.txt
"""

# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
# --- end shim ---

import geopandas as gpd
import pandas as pd

import config as C
import aggregate_lib as A

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def main():
    log("=" * 78)
    log("PHASE 2 — STATE-LEVEL AGGREGATION")
    log("=" * 78)

    dtypes = dict(C.TOWER_DTYPES)
    dtypes["generation"] = "category"

    df = pd.read_csv(C.TOWER_TABLE_V2, dtype=dtypes)
    log(f"\nLoaded tower table: {len(df):,} rows")

    states = gpd.read_file(C.STATES_GEOJSON)
    log(f"Loaded state boundaries: {len(states)} polygons")
    log(f"Boundary columns: {states.columns.tolist()}")

    # ---- Aggregate ----
    agg = A.aggregate_towers(df, key="st_nm")
    log(f"\nAggregated to {len(agg)} states")

    # ---- Areas + density ----
    areas = A.compute_region_areas(states, "st_nm")
    log(f"Computed areas for {len(areas)} states (EPSG:6933 equal-area)")

    agg = A.attach_area_and_density(agg, areas, "st_nm")

    missing_area = agg["area_sqkm"].isna().sum()
    log(f"States with no area matched: {missing_area}")
    if missing_area:
        log(agg.loc[agg["area_sqkm"].isna(), "region"].tolist())

    agg["level"] = "state"
    agg = agg.sort_values("tower_count", ascending=False).reset_index(drop=True)

    agg.to_csv(C.STATE_AGG, index=False)
    log(f"\nSAVED: {C.STATE_AGG}")
    log(f"       {len(agg)} rows x {agg.shape[1]} columns")

    log("\nColumns produced:")
    for c in agg.columns:
        log(f"  {c}")

    log("\n" + "-" * 78)
    log("TOP 10 STATES BY TOWER COUNT")
    log("-" * 78)
    cols = ["region", "tower_count", "tower_density", "pct_2g", "pct_3g",
            "pct_4g", "active_carrier_count", "range_mean"]
    log(agg[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    log("\n" + "-" * 78)
    log("BOTTOM 10 STATES BY TOWER DENSITY")
    log("-" * 78)
    log(agg.nsmallest(10, "tower_density")[cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))

    log("\n" + "-" * 78)
    log("SUMMARY STATISTICS")
    log("-" * 78)
    num = agg.select_dtypes("number")
    log(num.describe().T.to_string(float_format=lambda x: f"{x:,.3f}"))

    (C.REPORTS / "02_state_aggregation_report.txt").write_text("\n".join(LINES))
    print(f"\nReport written: {C.REPORTS / '02_state_aggregation_report.txt'}")


if __name__ == "__main__":
    main()
