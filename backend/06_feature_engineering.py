"""
Phase 4 — Feature engineering + rule-based coverage tier.

Runs for BOTH granularities using identical logic.

Input : data_processed/state_aggregated.csv, district_aggregated.csv
Output: data_processed/state_features.csv, district_features.csv
        outputs/reports/04_feature_engineering_report.txt

The rule-based tier is deliberately built WITHOUT any model. It exists to
sanity-check the clustering later: if unsupervised clusters broadly agree
with a transparent rule, the clusters are measuring something real.
"""

# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

import numpy as np
import pandas as pd

import config as C

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


# Features fed to the clustering model.
# Chosen to avoid redundancy: pct_3g/pct_5g are omitted because the four
# generation shares sum to 100 (perfect collinearity); tower_count is omitted
# in favour of density so clusters are not driven by raw region size.
CLUSTER_FEATURES = [
    "log_tower_density",   # coverage intensity, area-normalised
    "pct_2g",              # legacy-network burden
    "pct_4g",              # modern-network penetration
    "active_carrier_count",# operator competition
    "carrier_hhi",         # market concentration
    "range_mean",          # mean cell radius (large => sparse/rural cells)
    "log_sample_mean",     # measurement density / confidence
]


def engineer(df, level):
    log("\n" + "=" * 78)
    log(f"FEATURE ENGINEERING — {level.upper()} LEVEL ({len(df)} regions)")
    log("=" * 78)

    df = df.copy()

    # ---- Network modernity: mean generation on a 2..5 scale ----
    gen_n = df[["n_2g", "n_3g", "n_4g", "n_5g"]].to_numpy(dtype=float)
    weights = np.array([2, 3, 4, 5], dtype=float)
    totals = gen_n.sum(axis=1)
    df["gen_score"] = np.where(totals > 0, (gen_n * weights).sum(axis=1) / totals, np.nan)

    # ---- Modern share (4G + 5G) ----
    df["pct_modern"] = df["pct_4g"] + df["pct_5g"]

    # ---- Measurement activity ----
    df["log_sample_mean"] = np.log10(df["sample_mean"].clip(lower=0.01))
    df["samples_per_sqkm"] = df["sample_total"] / df["area_sqkm"]

    # ---- Load per carrier ----
    df["towers_per_carrier"] = df["tower_count"] / df["active_carrier_count"].clip(lower=1)

    # ---- Urban/rural proxy (density-based, as the blueprint specifies) ----
    q = df["log_tower_density"].quantile([0.33, 0.67])
    df["settlement_proxy"] = pd.cut(
        df["log_tower_density"],
        bins=[-np.inf, q.iloc[0], q.iloc[1], np.inf],
        labels=["Rural-like", "Mixed", "Urban-like"],
    )

    log("\nDerived features added:")
    for c in ["gen_score", "pct_modern", "log_sample_mean", "samples_per_sqkm",
              "towers_per_carrier", "settlement_proxy"]:
        log(f"  {c}")

    # ---- Rule-based coverage tier (NO model involved) ----
    log("\n" + "-" * 78)
    log("RULE-BASED COVERAGE TIER")
    log("-" * 78)
    log("Transparent composite of three percentile ranks, equally weighted:")
    log("  r1 = percentile rank of log_tower_density   (how much coverage)")
    log("  r2 = percentile rank of pct_modern (4G+5G)  (how modern)")
    log("  r3 = percentile rank of active_carrier_count(how competitive)")
    log("  score = (r1 + r2 + r3) / 3, then split at the 33rd/67th percentile.")

    r1 = df["log_tower_density"].rank(pct=True)
    r2 = df["pct_modern"].rank(pct=True)
    r3 = df["active_carrier_count"].rank(pct=True)
    df["rule_score"] = (r1 + r2 + r3) / 3

    cuts = df["rule_score"].quantile([1 / 3, 2 / 3])
    df["coverage_tier"] = pd.cut(
        df["rule_score"],
        bins=[-np.inf, cuts.iloc[0], cuts.iloc[1], np.inf],
        labels=["Underserved", "Moderate", "Well-served"],
    )

    log("\nTier distribution:")
    vc = df["coverage_tier"].value_counts().reindex(
        ["Underserved", "Moderate", "Well-served"])
    for t, n in vc.items():
        log(f"  {t:<13s} {n:>4}  ({n/len(df):6.2%})")

    log("\nMean feature values by tier:")
    prof = df.groupby("coverage_tier", observed=True)[
        ["tower_density", "pct_2g", "pct_4g", "gen_score",
         "active_carrier_count", "range_mean"]].mean()
    log(prof.to_string(float_format=lambda x: f"{x:,.3f}"))

    # ---- Missing-value audit on clustering features ----
    log("\nClustering feature null check:")
    for f in CLUSTER_FEATURES:
        n = df[f].isna().sum()
        log(f"  {f:<22s} nulls={n}  var={df[f].var():.6f}")

    # Guard against zero-variance features silently entering the model
    zero_var = [f for f in CLUSTER_FEATURES if df[f].var() == 0]
    if zero_var:
        log(f"\nWARNING zero-variance features present: {zero_var}")
    else:
        log("\nAll clustering features have non-zero variance. OK.")

    return df


def main():
    log("=" * 78)
    log("PHASE 4 — FEATURE ENGINEERING")
    log("=" * 78)
    log("\nNOTE: 'range_median' is excluded from all modelling — it is exactly")
    log("1000 for every region (OpenCelliD's default fill), i.e. zero variance.")

    state = pd.read_csv(C.STATE_AGG)
    dist = pd.read_csv(C.DISTRICT_AGG)

    state = engineer(state, "state")
    dist = engineer(dist, "district")

    state.to_csv(C.STATE_FEATURES, index=False)
    dist.to_csv(C.DISTRICT_FEATURES, index=False)

    log("\n" + "=" * 78)
    log(f"SAVED: {C.STATE_FEATURES}    ({len(state)} rows x {state.shape[1]} cols)")
    log(f"SAVED: {C.DISTRICT_FEATURES} ({len(dist)} rows x {dist.shape[1]} cols)")
    log("=" * 78)

    log("\nFeature set used for clustering:")
    for f in CLUSTER_FEATURES:
        log(f"  - {f}")

    (C.REPORTS / "04_feature_engineering_report.txt").write_text("\n".join(LINES))
    print(f"\nReport written: {C.REPORTS / '04_feature_engineering_report.txt'}")


if __name__ == "__main__":
    main()
