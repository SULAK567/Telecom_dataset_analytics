# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
# --- end shim ---

"""
Phase 3b — Regional enrichment at DISTRICT level.

Spatially joins the 2.49M-row tower table against 641 Census-2011 district
polygons (chunked, to bound peak memory), then applies the SAME aggregation
function used at state level.

Input : data_processed/tower_table_v2.csv
        data_processed/districts_india.geojson
Output: data_processed/tower_district_lookup.parquet   (tower -> district)
        data_processed/district_aggregated.csv
        outputs/reports/03_district_aggregation_report.txt
"""
import geopandas as gpd
import pandas as pd

import config as C
import aggregate_lib as A

LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def main():
    log("=" * 78)
    log("PHASE 3 — DISTRICT-LEVEL AGGREGATION")
    log("=" * 78)

    districts = gpd.read_file(C.DISTRICTS_GEOJSON)
    log(f"\nLoaded {len(districts)} district polygons")

    dtypes = dict(C.TOWER_DTYPES)
    dtypes["generation"] = "category"

    log("\nChunked spatial join (tower point -> district polygon):")
    joined = A.chunked_spatial_join(
        C.TOWER_TABLE_V2, districts, key="district", dtypes=dtypes, chunksize=300_000
    )

    total = len(joined)
    matched = joined["district"].notna().sum()
    log(f"\nJoin complete: {matched:,}/{total:,} towers matched a district "
        f"({matched/total:.3%})")
    log(f"Unmatched: {total - matched:,} "
        f"(coastline precision / offshore / post-2011 boundary changes)")

    joined["district"] = joined["district"].astype("category")

    # Persist the tower->district assignment (parquet: far smaller than CSV)
    slim = joined[["radio", "mcc", "mnc", "lac", "cid", "long", "lat", "range",
                   "sample", "created", "updated", "carrier", "circle",
                   "st_nm", "generation", "district"]]
    slim.to_parquet(C.DATA / "tower_district_lookup.parquet", index=False)
    log(f"\nSAVED tower-level district assignment: "
        f"{C.DATA / 'tower_district_lookup.parquet'}")

    # ---- Aggregate with the SAME function used for states ----
    agg = A.aggregate_towers(joined, key="district")
    log(f"\nAggregated to {len(agg)} districts with tower data")

    areas = A.compute_region_areas(districts, "district")
    agg = A.attach_area_and_density(agg, areas, "district")

    # Carry the parent state through for dashboard filtering
    meta = districts[["district", "district_id", "state_name"]].drop_duplicates()
    agg = agg.merge(meta, left_on="region", right_on="district", how="left")
    agg = agg.drop(columns=["district"])

    agg["level"] = "district"
    agg = agg.sort_values("tower_count", ascending=False).reset_index(drop=True)

    agg.to_csv(C.DISTRICT_AGG, index=False)
    log(f"\nSAVED: {C.DISTRICT_AGG}")
    log(f"       {len(agg)} rows x {agg.shape[1]} columns")

    # ---- Coverage of the district universe ----
    all_d = set(districts["district"])
    have_d = set(agg["region"])
    missing = sorted(all_d - have_d)
    log(f"\nDistricts with ZERO towers in the dataset: {len(missing)}")
    if missing:
        log("  " + "; ".join(missing[:30]) + ("..." if len(missing) > 30 else ""))
    log("(These are genuine coverage gaps / crowd-sourcing blind spots and are")
    log(" reported rather than dropped.)")

    log("\n" + "-" * 78)
    log("TOP 15 DISTRICTS BY TOWER COUNT")
    log("-" * 78)
    cols = ["region", "tower_count", "tower_density", "pct_2g", "pct_3g",
            "pct_4g", "active_carrier_count"]
    log(agg[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    log("\n" + "-" * 78)
    log("BOTTOM 15 DISTRICTS BY TOWER DENSITY")
    log("-" * 78)
    log(agg.nsmallest(15, "tower_density")[cols].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))

    log("\n" + "-" * 78)
    log("SUMMARY STATISTICS")
    log("-" * 78)
    log(agg.select_dtypes("number").describe().T.to_string(
        float_format=lambda x: f"{x:,.3f}"))

    (C.REPORTS / "03_district_aggregation_report.txt").write_text("\n".join(LINES))
    log(f"\nReport written: {C.REPORTS / '03_district_aggregation_report.txt'}")


if __name__ == "__main__":
    main()
