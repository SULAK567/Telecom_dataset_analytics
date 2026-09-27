# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
# --- end shim ---

"""
Phase 1 (completion) — Data quality audit + cleaning.

Input : tower_table_clean.csv  (output of 01_build_tower_table.py)
Output: data_processed/tower_table_v2.csv
        outputs/reports/01_data_quality_report.txt

Steps
-----
1. Structural audit of the RAW files (column schema, constant columns).
2. Load the merged tower table; .info() / null audit / dtype audit.
3. Drop rows with invalid or out-of-India latitude/longitude.
4. Deduplicate on the true cell identity key (radio, mcc, mnc, lac, cid).
5. Audit unmatched states (towers that fell outside every state polygon).
6. Save the cleaned table + a written quality report.

IMPORTANT SCHEMA FINDING (documented here because it changes the feature set)
---------------------------------------------------------------------------
The Kaggle CSV headers are MISALIGNED against the canonical OpenCelliD schema:

    OpenCelliD : radio, mcc, net, area, cell, unit, lon, lat, range,
                 samples, changeable, created, updated, averageSignal
    Kaggle hdr : radio, mcc, mnc, lac, cid, changeable_0, long, lat, range,
                 sample, changeable_1, created, updated, avgsignal

So the column labelled `changeable_0` is really `unit` (the UMTS PSC /
LTE physical cell ID), and `changeable_1` is the real `changeable` flag.

Verified on the data:
  * changeable_1 (= changeable) == 1 for 100% of rows
  * avgsignal    (= averageSignal) == 0 for 100% of rows
  * changeable_0 (= unit) is 0 / -1 / small ints in the PSC range -> not a flag

CONSEQUENCE: the originally-planned feature "% of towers directly obtained
from the telecom firm (changeable == 0)" CANNOT be built from this dataset —
every record is a crowd-sourced estimate. This is reported as a finding, not
silently dropped.
"""
import sys
import pandas as pd
import numpy as np
import config as C

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def audit_raw_schema():
    log("=" * 78)
    log("SECTION 1 — RAW FILE SCHEMA AUDIT")
    log("=" * 78)

    for name, path in [("404.csv", C.RAW_404), ("405.csv", C.RAW_405)]:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        log(f"\n{name} columns ({len(cols)}): {cols}")

    log("\nCanonical OpenCelliD schema:")
    log("  radio, mcc, net, area, cell, unit, lon, lat, range, samples,")
    log("  changeable, created, updated, averageSignal")
    log("\n-> Kaggle 'changeable_0' maps to OpenCelliD 'unit'")
    log("-> Kaggle 'changeable_1' maps to OpenCelliD 'changeable'")

    log("\nConstant-column verification across BOTH raw files:")
    frames = []
    for path in (C.RAW_404, C.RAW_405):
        frames.append(pd.read_csv(path, usecols=["changeable_0", "changeable_1", "avgsignal"]))
    raw = pd.concat(frames, ignore_index=True)
    n = len(raw)

    ch1 = raw["changeable_1"]
    sig = raw["avgsignal"]
    unit = raw["changeable_0"]

    log(f"  rows inspected                      : {n:,}")
    log(f"  changeable_1 (= changeable) == 1    : {(ch1 == 1).sum():,} / {n:,} "
        f"({(ch1 == 1).mean():.2%})  -> CONSTANT, zero variance")
    log(f"  avgsignal    (= averageSignal) == 0 : {(sig == 0).sum():,} / {n:,} "
        f"({(sig == 0).mean():.2%})  -> CONSTANT, unusable")
    log(f"  changeable_0 (= unit) distinct vals : {unit.nunique():,}  "
        f"min={unit.min()} max={unit.max()}  -> identifier, not a flag")
    log(f"  changeable_0 (= unit) == 0          : {(unit == 0).mean():.2%}")
    log(f"  changeable_0 (= unit) == -1         : {(unit == -1).mean():.2%}")

    log("\nCONCLUSION: dropping avgsignal / changeable_0 / changeable_1 is correct.")
    log("The planned 'verified vs crowd-estimated' feature is NOT constructible:")
    log("100% of records are crowd-sourced (changeable == 1).")
    del raw, frames


def load_tower_table():
    log("\n" + "=" * 78)
    log("SECTION 2 — MERGED TOWER TABLE AUDIT (tower_table_clean.csv)")
    log("=" * 78)

    df = pd.read_csv(C.TOWER_TABLE_V1, dtype=C.TOWER_DTYPES)
    log(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:,.1f} MB")

    log("\nDtypes:")
    for col, dt in df.dtypes.items():
        log(f"  {col:<10s} {str(dt)}")

    log("\nNull counts:")
    nulls = df.isnull().sum()
    for col, cnt in nulls.items():
        log(f"  {col:<10s} {cnt:>12,}  ({cnt/len(df):>7.3%})")

    log("\nNumeric describe():")
    log(df[["long", "lat", "range", "sample"]].describe().to_string())

    log("\nradio value counts:")
    log(df["radio"].value_counts(dropna=False).to_string())

    log("\ncarrier value counts:")
    log(df["carrier"].value_counts(dropna=False).to_string())

    return df


def filter_bbox(df):
    log("\n" + "=" * 78)
    log("SECTION 3 — COORDINATE VALIDITY FILTER")
    log("=" * 78)

    before = len(df)
    bb = C.INDIA_BBOX

    null_coord = df["long"].isna() | df["lat"].isna()
    zero_coord = (df["long"] == 0) & (df["lat"] == 0)
    in_box = (
        df["long"].between(bb["lon_min"], bb["lon_max"])
        & df["lat"].between(bb["lat_min"], bb["lat_max"])
    )
    keep = in_box & ~null_coord & ~zero_coord

    log(f"\nIndia bounding box: lon [{bb['lon_min']}, {bb['lon_max']}], "
        f"lat [{bb['lat_min']}, {bb['lat_max']}]")
    log(f"  null coordinates            : {null_coord.sum():,}")
    log(f"  (0, 0) null-island coords   : {zero_coord.sum():,}")
    log(f"  outside bounding box        : {(~in_box).sum():,}")
    log(f"  ---")
    log(f"  rows before                 : {before:,}")
    log(f"  rows dropped                : {(~keep).sum():,} ({(~keep).mean():.4%})")

    df = df[keep].copy()
    log(f"  rows after                  : {len(df):,}")
    return df


def deduplicate(df):
    log("\n" + "=" * 78)
    log("SECTION 4 — DEDUPLICATION")
    log("=" * 78)

    before = len(df)

    exact = df.duplicated().sum()
    log(f"\nFully identical rows                     : {exact:,}")

    key = ["radio", "mcc", "mnc", "lac", "cid"]
    dup_key = df.duplicated(subset=key).sum()
    log(f"Duplicate cell identities (radio+mcc+mnc+lac+cid): {dup_key:,}")

    # Keep the most recently updated record for each cell identity.
    df = df.sort_values("updated", ascending=False)
    df = df.drop_duplicates(subset=key, keep="first")
    df = df.sort_index()

    log(f"\nStrategy: keep the most recently 'updated' record per cell identity.")
    log(f"  rows before : {before:,}")
    log(f"  rows removed: {before - len(df):,} ({(before-len(df))/before:.4%})")
    log(f"  rows after  : {len(df):,}")
    return df


def audit_states(df):
    log("\n" + "=" * 78)
    log("SECTION 5 — STATE ASSIGNMENT AUDIT")
    log("=" * 78)

    unmatched = df["st_nm"].isna()
    log(f"\nTowers with no state assigned: {unmatched.sum():,} ({unmatched.mean():.4%})")
    log("(These fall inside the bounding box but outside every state polygon —")
    log(" typically offshore / just across a land border / coastline precision.)")

    if unmatched.sum():
        log("\nBounding box of unmatched towers:")
        u = df.loc[unmatched, ["long", "lat"]]
        log(f"  lon: {u['long'].min():.3f} .. {u['long'].max():.3f}")
        log(f"  lat: {u['lat'].min():.3f} .. {u['lat'].max():.3f}")

    log(f"\nDistinct states matched: {df['st_nm'].nunique()}")
    log("\nTowers per state:")
    log(df["st_nm"].value_counts(dropna=False).to_string())

    # Drop unmatched: regional aggregation is the whole point of the project.
    df = df[~unmatched].copy()
    df["st_nm"] = df["st_nm"].cat.remove_unused_categories()
    log(f"\nDropping unmatched -> final row count: {len(df):,}")
    return df


def add_generation(df):
    log("\n" + "=" * 78)
    log("SECTION 6 — DERIVED COLUMN: network generation")
    log("=" * 78)
    df["generation"] = (
        df["radio"].astype(str).map(C.RADIO_TO_GEN).fillna("Other").astype("category")
    )
    log("\nGeneration distribution:")
    vc = df["generation"].value_counts()
    for g, c in vc.items():
        log(f"  {g:<6s} {c:>12,}  ({c/len(df):6.2%})")
    return df


def main():
    audit_raw_schema()
    df = load_tower_table()
    df = filter_bbox(df)
    df = deduplicate(df)
    df = audit_states(df)
    df = add_generation(df)

    df.to_csv(C.TOWER_TABLE_V2, index=False)
    log("\n" + "=" * 78)
    log(f"SAVED: {C.TOWER_TABLE_V2}")
    log(f"       {len(df):,} rows x {df.shape[1]} columns")
    log("=" * 78)

    report = C.REPORTS / "01_data_quality_report.txt"
    report.write_text("\n".join(LINES))
    print(f"\nReport written: {report}")


if __name__ == "__main__":
    main()
