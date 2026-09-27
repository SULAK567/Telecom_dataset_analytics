# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Phase 5 — Exploratory Data Analysis.

Produces the figure set in outputs/figures/ plus a written EDA report.
Every figure follows the shared visual system in viz_style.py.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import viz_style as V

V.apply_style()
LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def save(fig, name):
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    log(f"  saved figure: {name}")


# ---------------------------------------------------------------- maps
def fig_density_maps(state_f, dist_f):
    states = gpd.read_file(C.STATES_GEOJSON)[["st_nm", "geometry"]]
    dists = gpd.read_file(C.DISTRICTS_GEOJSON)[["district", "geometry"]]

    sm = states.merge(state_f[["region", "tower_density", "tower_count"]],
                      left_on="st_nm", right_on="region", how="left")
    dm = dists.merge(dist_f[["region", "tower_density", "tower_count"]],
                     left_on="district", right_on="region", how="left")

    for gdf, label, fname in [
        (sm, "state", "fig01_tower_density_state_map.png"),
        (dm, "district", "fig02_tower_density_district_map.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 9))
        # log scale: density spans four orders of magnitude
        vals = gdf["tower_density"].replace(0, np.nan)
        gdf = gdf.assign(_plot=np.log10(vals))
        gdf.plot(column="_plot", cmap=V.CMAP_SEQ, ax=ax, linewidth=0.25,
                 edgecolor=V.SURFACE, missing_kwds={
                     "color": "#ececea", "edgecolor": V.SURFACE,
                     "linewidth": 0.25, "label": "No towers recorded"})
        V.map_axes(ax)
        V.fig_title(fig, f"Cell-tower density by {label}",
                    "Towers per square kilometre, log scale")

        sm_ = plt.cm.ScalarMappable(
            cmap=V.CMAP_SEQ,
            norm=plt.Normalize(vmin=gdf["_plot"].min(), vmax=gdf["_plot"].max()))
        cbar = fig.colorbar(sm_, ax=ax, fraction=0.03, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(color=V.INK_MUTED, labelcolor=V.INK_MUTED, length=0)
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{10**t:,.3g}" for t in ticks])
        cbar.ax.yaxis.get_offset_text().set_visible(False)  # kill the stray 1e+03
        cbar.set_label("towers / km²", color=V.INK_SECONDARY, fontsize=9)

        n_missing = gdf["tower_density"].isna().sum()
        fig.text(0.02, 0.025, f"Grey = no towers recorded ({n_missing} {label}s). "
                 "Source: OpenCelliD via Kaggle; boundaries DataMeet / Census 2011.",
                 fontsize=8, color=V.INK_MUTED, va="bottom")
        save(fig, fname)


# ------------------------------------------------- national generation mix
def fig_generation_national(state_f):
    tot = state_f[["n_2g", "n_3g", "n_4g", "n_5g"]].sum()
    tot.index = ["2G", "3G", "4G", "5G"]
    pct = tot / tot.sum() * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(tot.index, tot.values,
                  color=[V.GENERATION_COLORS[g] for g in tot.index],
                  width=0.62, zorder=3)
    for b, v, p in zip(bars, tot.values, pct.values):
        # direct labels: required relief for sub-3:1 fills, and removes the legend
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\n{p:.1f}%",
                ha="center", va="bottom", fontsize=9.5, color=V.INK_PRIMARY,
                linespacing=1.35)
    V.strip_spines(ax, keep=("bottom",))
    ax.grid(axis="x", visible=False)
    ax.set_yticks([])
    ax.set_ylim(0, tot.max() * 1.22)
    ax.set_xlabel("")
    V.title_block(ax, "India is still a 2G-majority tower estate",
                  "Towers by network generation, all carriers, 2.49M records")
    ax.text(0.0, -0.16, "5G is 136 towers nationally — 0.01% — this dataset predates the 5G rollout.",
            transform=ax.transAxes, fontsize=8.5, color=V.INK_MUTED)
    save(fig, "fig03_generation_mix_national.png")

    log("\nNational generation mix:")
    for g in tot.index:
        log(f"  {g}: {tot[g]:>10,}  ({pct[g]:5.2f}%)")


# --------------------------------------------- generation mix by state
def fig_generation_by_state(state_f):
    d = state_f.nlargest(20, "tower_count").sort_values("pct_4g")
    fig, ax = plt.subplots(figsize=(8.4, 8))

    left = np.zeros(len(d))
    for gen in ["2G", "3G", "4G", "5G"]:
        vals = d[f"pct_{gen.lower()}"].to_numpy()
        ax.barh(d["region"], vals, left=left, height=0.68,
                color=V.GENERATION_COLORS[gen], label=gen,
                edgecolor=V.SURFACE, linewidth=1.6, zorder=3)  # 2px surface gap
        left += vals

    # selective direct labels: only the 4G segment, the series that carries the story
    for i, (_, row) in enumerate(d.iterrows()):
        x = row["pct_2g"] + row["pct_3g"] + row["pct_4g"] / 2
        if row["pct_4g"] > 6:
            ax.text(x, i, f"{row['pct_4g']:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="#ffffff", weight="600")

    V.strip_spines(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of towers (%)")
    V.legend_below(ax, ncol=4, y=-0.075)
    V.title_block(ax, "4G penetration varies three-fold across the biggest states",
                  "Generation mix, 20 states with the most towers, ordered by 4G share")
    save(fig, "fig04_generation_mix_by_state.png")


# ------------------------------------------------------- carrier comparison
def fig_carriers(state_f):
    counts = {c: state_f[f"n_{c.lower().replace('/', '_')}"].sum()
              for c in C.ACTIVE_CARRIERS}
    s = pd.Series(counts).sort_values(ascending=True)

    # carrier x generation, computed once from the tower-level table
    tl = pd.read_parquet(C.DATA / "tower_district_lookup.parquet",
                         columns=["carrier", "generation"])
    tl = tl[tl["carrier"].isin(C.ACTIVE_CARRIERS)]
    mix = pd.crosstab(tl["carrier"], tl["generation"], normalize="index") * 100
    mix = mix.reindex(columns=["2G", "3G", "4G", "5G"], fill_value=0.0)
    mix = mix.reindex(s.index[::-1])
    del tl

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax = axes[0]
    bars = ax.barh(s.index, s.values,
                   color=[V.CARRIER_COLORS[c] for c in s.index],
                   height=0.62, zorder=3)
    for b, v in zip(bars, s.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,.0f}",
                va="center", ha="left", fontsize=9.5, color=V.INK_PRIMARY)
    V.strip_spines(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)
    ax.set_xticks([])
    ax.set_xlim(0, s.max() * 1.18)
    V.title_block(ax, "Airtel and Vi dominate the recorded estate",
                  "Total towers per carrier")

    ax = axes[1]
    left = np.zeros(len(mix))
    for gen in ["2G", "3G", "4G", "5G"]:
        vals = mix[gen].to_numpy()
        ax.barh(mix.index, vals, left=left, height=0.62,
                color=V.GENERATION_COLORS[gen], label=gen,
                edgecolor=V.SURFACE, linewidth=1.6, zorder=3)
        left += vals
    for i, c in enumerate(mix.index):
        v = mix.loc[c, "4G"]
        if v > 5:
            ax.text(mix.loc[c, "2G"] + mix.loc[c, "3G"] + v / 2, i, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5, color="#ffffff",
                    weight="600")
    V.strip_spines(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that carrier's towers (%)")
    V.legend_below(ax, ncol=4, y=-0.22)
    V.title_block(ax, "Carrier network modernity differs sharply",
                  "Generation mix within each carrier")

    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    save(fig, "fig05_carrier_comparison.png")

    log("\nCarrier totals:")
    for c, v in s.sort_values(ascending=False).items():
        log(f"  {c:<14s} {v:>10,}")
    log("\nGeneration mix within each carrier (%):")
    log(mix.to_string(float_format=lambda x: f"{x:6.2f}"))


# ------------------------------------------------ top / bottom districts
def fig_top_bottom(dist_f):
    top = dist_f.nlargest(15, "tower_density")
    bot = dist_f[dist_f["tower_count"] >= 50].nsmallest(15, "tower_density")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax, d, title, sub in [
        (axes[0], top.sort_values("tower_density"),
         "Densest districts", "Towers per km², top 15"),
        (axes[1], bot.sort_values("tower_density", ascending=False),
         "Sparsest districts", "Towers per km², bottom 15 (min. 50 towers)"),
    ]:
        bars = ax.barh(d["region"], d["tower_density"],
                       color=V.CATEGORICAL[0], height=0.62, zorder=3)
        for b, v in zip(bars, d["tower_density"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,.2f}",
                    va="center", ha="left", fontsize=9, color=V.INK_PRIMARY)
        V.strip_spines(ax, keep=("bottom",))
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.set_xlim(0, d["tower_density"].max() * 1.25)
        V.title_block(ax, title, sub)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig06_density_extremes_district.png")

    log("\nDensest 5 districts:")
    log(top.head(5)[["region", "tower_density", "tower_count"]].to_string(index=False))
    log("\nSparsest 5 districts (>=50 towers):")
    log(bot.head(5)[["region", "tower_density", "tower_count"]].to_string(index=False))


# ---------------------------------------------------- correlation heatmap
def fig_correlation(dist_f, features):
    corr = dist_f[features].corr()

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    im = ax.imshow(corr, cmap=V.CMAP_DIV, vmin=-1, vmax=1)
    labels = [C.label_of(f) for f in features]
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(labels, rotation=34, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(len(features)):
        for j in range(len(features)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.55 else V.INK_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=V.INK_MUTED, labelcolor=V.INK_MUTED, length=0)
    V.title_block(ax, "Clustering features are complementary, not redundant",
                  "Pearson correlation, 632 districts")
    save(fig, "fig07_feature_correlation_district.png")

    log("\nDistrict feature correlation matrix:")
    log(corr.to_string(float_format=lambda x: f"{x:6.3f}"))

    high = [(features[i], features[j], corr.iloc[i, j])
            for i in range(len(features)) for j in range(i + 1, len(features))
            if abs(corr.iloc[i, j]) > 0.8]
    log(f"\nFeature pairs with |r| > 0.8: {len(high)}")
    for a, b, r in high:
        log(f"  {a} <-> {b}: {r:.3f}")


# ------------------------------------------- density vs modernity scatter
def fig_density_vs_modern(dist_f):
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for tier in ["Underserved", "Moderate", "Well-served"]:
        d = dist_f[dist_f["coverage_tier"] == tier]
        ax.scatter(d["tower_density"], d["pct_modern"], s=34,
                   color=V.TIER_COLORS[tier], label=tier, alpha=0.85,
                   edgecolor=V.SURFACE, linewidth=1.2, zorder=3)
    ax.set_xscale("log")
    V.strip_spines(ax)
    ax.set_xlabel("tower density (towers / km², log scale)")
    ax.set_ylabel("4G + 5G share of towers (%)")
    V.legend_below(ax, ncol=3, y=-0.17, title="Rule-based tier")
    V.title_block(ax, "Denser districts are also more modern",
                  "Each point is one district; tiers are the rule-based reference labels")
    save(fig, "fig08_density_vs_modernity_district.png")

    r = np.corrcoef(np.log10(dist_f["tower_density"].clip(lower=1e-6)),
                    dist_f["pct_modern"])[0, 1]
    log(f"\nCorrelation log10(density) vs 4G+5G share: r = {r:.3f}")


def main():
    log("=" * 78)
    log("PHASE 5 — EXPLORATORY DATA ANALYSIS")
    log("=" * 78)

    state_f = pd.read_csv(C.STATE_FEATURES)
    dist_f = pd.read_csv(C.DISTRICT_FEATURES)
    log(f"\nStates: {len(state_f)}   Districts: {len(dist_f)}")

    import importlib
    fe = importlib.import_module("06_feature_engineering")
    features = fe.CLUSTER_FEATURES

    log("\nGenerating figures:")
    fig_density_maps(state_f, dist_f)
    fig_generation_national(state_f)
    fig_generation_by_state(state_f)
    fig_carriers(state_f)
    fig_top_bottom(dist_f)
    fig_correlation(dist_f, features)
    fig_density_vs_modern(dist_f)

    (C.REPORTS / "05_eda_report.txt").write_text("\n".join(LINES))
    log(f"\nReport written: {C.REPORTS / '05_eda_report.txt'}")


if __name__ == "__main__":
    main()
