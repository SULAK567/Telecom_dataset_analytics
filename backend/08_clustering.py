# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Phase 6 — Clustering (K-Means + DBSCAN), for BOTH granularities.

K is chosen with the elbow method AND silhouette score, not picked by hand.
DBSCAN is run as an independent comparison method, with eps selected from the
k-distance curve rather than guessed.

Outputs: data_processed/state_clustered.csv, district_clustered.csv
         outputs/models/*.joblib
         outputs/figures/fig09..fig15
         outputs/reports/06_clustering_report.txt
"""
import importlib
import json

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config as C
import viz_style as V

V.apply_style()
FE = importlib.import_module("06_feature_engineering")
FEATURES = FE.CLUSTER_FEATURES

LINES = []
K_RANGE = range(2, 11)

# A cluster smaller than this is an outlier pocket, not a coverage regime.
# DBSCAN is the method used to surface those instead.
MIN_CLUSTER_FRAC = 0.03
MIN_CLUSTER_ABS = 3

# Data-sufficiency gate.
# Regional percentages (radio mix, carrier share) computed from a handful of
# crowd-sourced points are noise, not measurement: some districts hold a single
# tower, and Lakshadweep holds 21 for an entire state. Such regions are extreme
# multivariate outliers that K-Means isolates into singleton clusters at EVERY
# k, which destroys the partition. They are therefore held out of model fitting
# and reported honestly as "Insufficient data" rather than given a coverage
# label the evidence cannot support.
MIN_TOWERS = 50
INSUFFICIENT = -1
INSUFFICIENT_LABEL = f"Insufficient data (<{MIN_TOWERS} towers)"


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def save(fig, name):
    fig.savefig(C.FIGURES / name)
    plt.close(fig)
    log(f"  saved figure: {name}")


# ------------------------------------------------------------ model selection
def choose_k(X, level, fig_name):
    """
    Select k using the elbow method AND silhouette, subject to a minimum
    cluster-size constraint.

    WHY THE CONSTRAINT: silhouette is maximised by isolating extreme outliers.
    At state level the unconstrained optimum is k=2 with a silhouette of 0.75 —
    but one cluster contains a SINGLE region (Lakshadweep: 21 towers, 100% 2G).
    That is a high score for a useless partition: it describes one island, not a
    coverage structure, and it scores an Adjusted Rand Index of 0.00 against the
    independent rule-based tiers. Isolated extremes are exactly what DBSCAN is
    run to surface, so K-Means is restricted to partitions where every cluster
    is large enough to represent a real coverage regime.
    """
    n = X.shape[0]
    min_size = max(MIN_CLUSTER_ABS, int(np.ceil(MIN_CLUSTER_FRAC * n)))

    inertia, sil, ch, db, smallest = [], [], [], [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=25, random_state=C.RANDOM_STATE)
        lab = km.fit_predict(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, lab))
        ch.append(calinski_harabasz_score(X, lab))
        db.append(davies_bouldin_score(X, lab))
        smallest.append(int(np.bincount(lab).min()))

    tbl = pd.DataFrame({"k": list(K_RANGE), "inertia": inertia,
                        "silhouette": sil, "calinski_harabasz": ch,
                        "davies_bouldin": db, "smallest_cluster": smallest})
    tbl["valid"] = tbl["smallest_cluster"] >= min_size

    log(f"\nModel-selection scan ({level}):")
    log(tbl.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    log(f"\n  Minimum acceptable cluster size: {min_size} regions "
        f"(max of {MIN_CLUSTER_ABS}, {MIN_CLUSTER_FRAC:.0%} of {n})")

    unconstrained = int(tbl.loc[tbl["silhouette"].idxmax(), "k"])
    valid = tbl[tbl["valid"]]
    rejected = tbl.loc[~tbl["valid"], "k"].tolist()
    if rejected:
        log(f"  Rejected k values (cluster too small to be a coverage regime): {rejected}")

    if valid.empty:
        log("  WARNING no k satisfies the size constraint; falling back to unconstrained.")
        valid = tbl

    best_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    log(f"  Unconstrained silhouette optimum : k = {unconstrained} "
        f"({tbl['silhouette'].max():.4f})")
    log(f"  SELECTED k (size-constrained)    : k = {best_k} "
        f"({valid['silhouette'].max():.4f})")
    log(f"  Best Calinski-Harabasz at k = {valid.loc[valid['calinski_harabasz'].idxmax(), 'k']:.0f}")
    log(f"  Best Davies-Bouldin  at k = {valid.loc[valid['davies_bouldin'].idxmin(), 'k']:.0f} (lower is better)")

    # ---- elbow + silhouette figure (two charts, never a dual axis) ----
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ax = axes[0]
    ax.plot(tbl["k"], tbl["inertia"], color=V.CATEGORICAL[0], linewidth=2,
            marker="o", markersize=7, markeredgecolor=V.SURFACE,
            markeredgewidth=1.6, zorder=3)
    V.strip_spines(ax)
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("within-cluster sum of squares")
    V.title_block(ax, "Elbow method", "Inertia falls steeply, then flattens")

    ax = axes[1]
    ax.plot(tbl["k"], tbl["silhouette"], color=V.CATEGORICAL[0], linewidth=2,
            marker="o", markersize=7, markeredgecolor=V.SURFACE,
            markeredgewidth=1.6, zorder=3)
    bad = tbl[~tbl["valid"]]
    if len(bad):
        ax.scatter(bad["k"], bad["silhouette"], s=46, color=V.INK_MUTED,
                   marker="x", linewidth=1.8, zorder=4,
                   label="rejected: cluster too small")
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    sel_sil = float(tbl.loc[tbl["k"] == best_k, "silhouette"].iloc[0])
    ax.scatter([best_k], [sel_sil], s=150, facecolor="none",
               edgecolor=V.CATEGORICAL[1], linewidth=2.4, zorder=5)
    ax.annotate(f"k = {best_k}\n{sel_sil:.3f}", (best_k, sel_sil),
                textcoords="offset points", xytext=(12, 6), fontsize=9.5,
                color=V.INK_PRIMARY)
    V.strip_spines(ax)
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("silhouette score")
    V.title_block(ax, "Silhouette score", "Higher is better; peak selects k")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, fig_name)
    return best_k, tbl


# ------------------------------------------------------------------- DBSCAN
def run_dbscan(X, level, fig_name):
    min_samples = max(4, X.shape[1] + 1)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    kdist = np.sort(dists[:, -1])

    # eps from the knee of the k-distance curve.
    # The k-distance curve is convex-increasing, so it lies BELOW the chord
    # from (0,0) to (1,1); the knee is the point of maximum distance BELOW
    # that chord, i.e. argmax(x_n - y_n). (Taking argmax(y_n - x_n) puts the
    # knee at index 0 and collapses DBSCAN into all-noise.)
    x = np.arange(len(kdist), dtype=float)
    x_n, y_n = x / x[-1], (kdist - kdist.min()) / (kdist.max() - kdist.min() + 1e-12)
    knee = int(np.argmax(x_n - y_n))
    eps = float(kdist[knee])

    log(f"\nDBSCAN ({level}): min_samples={min_samples}, "
        f"eps={eps:.4f} (knee of k-distance curve at index {knee})")

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log(f"  clusters found: {n_clusters}   noise/outlier regions: {n_noise} "
        f"({n_noise/len(labels):.1%})")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(x, kdist, color=V.CATEGORICAL[0], linewidth=2, zorder=3)
    ax.axhline(eps, color=V.CATEGORICAL[1], linewidth=1.6, linestyle="--", zorder=4)
    ax.annotate(f"eps = {eps:.3f}", (0.02, eps), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 6), fontsize=9.5,
                color=V.INK_PRIMARY)
    V.strip_spines(ax)
    ax.set_xlabel(f"regions sorted by distance to {min_samples}th neighbour")
    ax.set_ylabel("distance")
    V.title_block(ax, f"DBSCAN eps selection — {level}",
                  "eps read from the knee of the k-distance curve, not guessed")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, fig_name)

    return labels, eps, min_samples, n_clusters, n_noise


# --------------------------------------------------------------- interpretation
def name_clusters(df, labels, features):
    """
    Order clusters by a coverage-quality composite and give each a descriptive
    name derived from its own profile (never a hand-written guess).
    """
    tmp = df.copy()
    tmp["_c"] = labels
    prof = tmp.groupby("_c")[["log_tower_density", "pct_modern", "pct_2g",
                              "active_carrier_count", "range_mean"]].mean()

    # rank composite: density + modernity + competition
    comp = (prof["log_tower_density"].rank(pct=True)
            + prof["pct_modern"].rank(pct=True)
            + prof["active_carrier_count"].rank(pct=True)) / 3
    order = comp.sort_values().index.tolist()

    names = {}
    for rank, cid in enumerate(order):
        dens = prof.loc[cid, "log_tower_density"]
        modern = prof.loc[cid, "pct_modern"]
        carr = prof.loc[cid, "active_carrier_count"]
        rng = prof.loc[cid, "range_mean"]

        dens_word = "Sparse" if dens < -0.8 else ("Mid-density" if dens < 0.4 else "Dense")
        mod_word = "legacy" if modern < 8 else ("mixed" if modern < 18 else "modern")
        if carr < 2.5:
            comp_word = "single-operator"
        elif carr < 3.6:
            comp_word = "partly-served"
        else:
            comp_word = "competitive"
        parts = [dens_word, comp_word, mod_word]
        if rng > 2500:                      # unusually wide cells => rural macro
            parts.insert(1, "wide-cell")
        names[cid] = " · ".join(parts)

    # Names must be unique: they are used as dashboard labels and chart keys.
    seen = {}
    for cid in order:
        base = names[cid]
        if base in seen.values():
            names[cid] = f"{base} (grp {list(order).index(cid)})"
        seen[cid] = names[cid]

    # remap ids so 0 = weakest coverage, K-1 = strongest (ordinal colour ramp)
    remap = {cid: i for i, cid in enumerate(order)}
    new_labels = np.array([remap[c] for c in labels])
    new_names = {remap[cid]: nm for cid, nm in names.items()}
    return new_labels, new_names, prof.rename(index=remap).sort_index()


def profile_figure(df, features, names, level, fig_name):
    z = df.groupby("cluster")[features].mean()
    zs = (z - df[features].mean()) / df[features].std()

    fig, ax = plt.subplots(figsize=(10.4, 1.3 + 0.8 * len(zs)))
    im = ax.imshow(zs.values, cmap=V.CMAP_DIV, vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(features)))
    # Friendly names: "log_sample_mean" means nothing to a reader.
    ax.set_xticklabels([C.label_of(f) for f in features], rotation=30, ha="right")
    ax.set_yticks(range(len(zs)))
    ax.set_yticklabels([f"C{i} · {names[i]}" for i in zs.index])
    ax.grid(False)
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            v = zs.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > 1.2 else V.INK_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=V.INK_MUTED, labelcolor=V.INK_MUTED, length=0)
    # Short label: the long form overflowed the figure edge even with a tight
    # bounding box, because a rotated colorbar label sits outside the axes.
    cbar.set_label("σ from mean", color=V.INK_SECONDARY, fontsize=8.5,
                   labelpad=6)
    V.title_block(ax, f"What defines each cluster — {level}",
                  "Cluster mean per feature, z-scored; red = above average, blue = below")
    fig.tight_layout(rect=[0, 0, 0.97, 0.84])
    save(fig, fig_name)
    return zs


def cluster_map(df, level, names, fig_name):
    if level == "state":
        gdf = gpd.read_file(C.STATES_GEOJSON)[["st_nm", "geometry"]]
        gdf = gdf.merge(df[["region", "cluster"]], left_on="st_nm",
                        right_on="region", how="left")
    else:
        gdf = gpd.read_file(C.DISTRICTS_GEOJSON)[["district", "geometry"]]
        gdf = gdf.merge(df[["region", "cluster"]], left_on="district",
                        right_on="region", how="left")

    k = int(df.loc[df["cluster"] >= 0, "cluster"].max()) + 1
    ramp = {3: V.ORDINAL_3, 4: V.ORDINAL_4,
            2: [V.ORDINAL_3[0], V.ORDINAL_3[2]],
            5: ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
            6: ["#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]}
    colors = ramp.get(k, V.SEQ_BLUE[:: max(1, len(V.SEQ_BLUE) // k)][:k])

    fig, ax = plt.subplots(figsize=(8.2, 9.4))
    for cid in range(k):
        sub = gdf[gdf["cluster"] == cid]
        if len(sub):
            sub.plot(ax=ax, color=colors[cid], linewidth=0.25,
                     edgecolor=V.SURFACE, label=f"C{cid} · {names[cid]}")
    insuf = gdf[gdf["cluster"] == INSUFFICIENT]
    if len(insuf):
        insuf.plot(ax=ax, color="#d7d6d1", linewidth=0.25, edgecolor=V.SURFACE)
    miss = gdf[gdf["cluster"].isna()]
    if len(miss):
        miss.plot(ax=ax, color="#ececea", linewidth=0.25, edgecolor=V.SURFACE)
    V.map_axes(ax)
    V.fig_title(fig, f"Coverage clusters — {level} level",
                "K-Means clusters, ordered from weakest to strongest coverage")
    # geopandas draws a PatchCollection, which legend() cannot build handles
    # from — construct proxy handles explicitly.
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=colors[cid], edgecolor=V.SURFACE,
                              label=f"C{cid} · {names[cid]}")
               for cid in range(k) if (gdf["cluster"] == cid).any()]
    if len(insuf):
        handles.append(mpatches.Patch(facecolor="#d7d6d1", edgecolor=V.SURFACE,
                                      label=INSUFFICIENT_LABEL))
    if len(miss):
        handles.append(mpatches.Patch(facecolor="#ececea", edgecolor=V.SURFACE,
                                      label="No tower data"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.52, 0.02),
              frameon=False, fontsize=8.5, title="Cluster", title_fontsize=9)
    fig.text(0.02, 0.025, "Grey = no tower data. Clusters are relative, "
             "data-driven groupings — not official coverage assessments.",
             fontsize=8, color=V.INK_MUTED)
    save(fig, fig_name)


def pca_scatter(X, df, names, level, fig_name):
    p = PCA(n_components=2, random_state=C.RANDOM_STATE)
    xy = p.fit_transform(X)
    k = int(df["cluster"].max()) + 1
    colors = {3: V.ORDINAL_3, 4: V.ORDINAL_4}.get(k, V.SEQ_BLUE[:k])

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for cid in range(k):
        m = df["cluster"] == cid
        ax.scatter(xy[m, 0], xy[m, 1], s=38, color=colors[cid],
                   label=f"C{cid} · {names[cid]}", alpha=0.85,
                   edgecolor=V.SURFACE, linewidth=1.2, zorder=3)
    V.strip_spines(ax)
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%} of variance)")
    V.legend_below(ax, ncol=2, y=-0.16)
    V.title_block(ax, f"Cluster separation in feature space — {level}",
                  "Regions projected onto the first two principal components")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save(fig, fig_name)
    log(f"  PCA variance explained by PC1+PC2: "
        f"{p.explained_variance_ratio_[:2].sum():.1%}")


# ------------------------------------------------------------------- pipeline
def run_level(level, path, out_path, figs):
    log("\n" + "=" * 78)
    log(f"CLUSTERING — {level.upper()} LEVEL")
    log("=" * 78)

    full = pd.read_csv(path)
    log(f"\nRegions loaded: {len(full)}   Features: {len(FEATURES)}")
    log(f"Features: {FEATURES}")

    # ---- Data-sufficiency gate ----
    eligible = full["tower_count"] >= MIN_TOWERS
    df = full[eligible].reset_index(drop=True).copy()
    held_out = full[~eligible].reset_index(drop=True).copy()
    log(f"\nData-sufficiency gate: >= {MIN_TOWERS} towers required to be clustered")
    log(f"  clustered      : {len(df)} regions")
    log(f"  held out       : {len(held_out)} regions ({len(held_out)/len(full):.1%})")
    if len(held_out):
        log("  held-out regions: "
            + "; ".join(f"{r} ({int(n)})" for r, n in
                        held_out.nsmallest(min(15, len(held_out)), "tower_count")
                        [["region", "tower_count"]].to_numpy())
            + ("..." if len(held_out) > 15 else ""))

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    best_k, scan = choose_k(X, level, figs["elbow"])

    km = KMeans(n_clusters=best_k, n_init=50, random_state=C.RANDOM_STATE)
    raw_labels = km.fit_predict(X)

    labels, names, _ = name_clusters(df, raw_labels, FEATURES)
    df["cluster"] = labels
    df["cluster_name"] = [names[c] for c in labels]

    log(f"\nFinal K-Means model: k={best_k}")
    log(f"  silhouette      : {silhouette_score(X, labels):.4f}")
    log(f"  davies-bouldin  : {davies_bouldin_score(X, labels):.4f}")
    log(f"  calinski-harabasz: {calinski_harabasz_score(X, labels):,.1f}")

    log("\nCluster sizes and names (0 = weakest coverage):")
    for cid in range(best_k):
        n = int((labels == cid).sum())
        log(f"  C{cid} · {names[cid]:<28s} {n:>4} regions ({n/len(df):6.2%})")

    log("\nCluster profiles (raw means):")
    prof = df.groupby("cluster")[
        ["tower_density", "pct_2g", "pct_4g", "pct_modern", "gen_score",
         "active_carrier_count", "carrier_hhi", "range_mean", "tower_count"]].mean()
    log(prof.to_string(float_format=lambda x: f"{x:,.3f}"))

    if len(held_out):
        held_out["cluster"] = INSUFFICIENT
        held_out["cluster_name"] = INSUFFICIENT_LABEL
    map_frame = pd.concat([df, held_out], ignore_index=True) if len(held_out) else df

    zs = profile_figure(df, FEATURES, names, level, figs["profile"])
    cluster_map(map_frame, level, names, figs["map"])
    pca_scatter(X, df, names, level, figs["pca"])

    # ---- DBSCAN comparison ----
    db_labels, eps, min_samples, n_db, n_noise = run_dbscan(X, level, figs["kdist"])
    df["dbscan"] = db_labels
    outliers = df.loc[db_labels == -1, "region"].tolist()
    log(f"  DBSCAN flagged as outliers/gap zones ({len(outliers)}):")
    log("    " + ("; ".join(outliers[:25]) + ("..." if len(outliers) > 25 else "")
                  if outliers else "(none)"))

    if n_db >= 2:
        mask = db_labels != -1
        log(f"  DBSCAN silhouette (non-noise only): "
            f"{silhouette_score(X[mask], db_labels[mask]):.4f}")
    log(f"  Agreement K-Means vs DBSCAN (Adjusted Rand Index): "
        f"{adjusted_rand_score(labels, db_labels):.4f}")

    # ---- Validation against the rule-based tier ----
    log("\n" + "-" * 78)
    log("VALIDATION: clusters vs the rule-based coverage tier")
    log("-" * 78)
    ct = pd.crosstab(df["cluster_name"], df["coverage_tier"])
    ct = ct.reindex(columns=["Underserved", "Moderate", "Well-served"], fill_value=0)
    log("\n" + ct.to_string())

    tier_code = df["coverage_tier"].map(
        {"Underserved": 0, "Moderate": 1, "Well-served": 2}).to_numpy()
    ari = adjusted_rand_score(tier_code, labels)
    log(f"\nAdjusted Rand Index (clusters vs rule-based tiers): {ari:.4f}")
    log("Interpretation: ARI 0 = random agreement, 1 = identical partitions.")
    log("A clearly positive ARI means the unsupervised clusters independently")
    log("recovered the same coverage structure the transparent rule encodes,")
    log("without ever seeing that rule.")

    # crosstab figure
    fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.62 * len(ct)))
    im = ax.imshow(ct.values, cmap=V.CMAP_SEQ, aspect="auto")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.grid(False)
    vmax = ct.values.max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > vmax * 0.55 else V.INK_PRIMARY)
    V.title_block(ax, f"Clusters agree with the rule-based tiers — {level}",
                  f"Region counts; Adjusted Rand Index = {ari:.3f}")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, figs["cross"])

    if len(held_out):
        held_out["dbscan"] = INSUFFICIENT
    combined = pd.concat([df, held_out], ignore_index=True) if len(held_out) else df
    combined.to_csv(out_path, index=False)
    log(f"\nSAVED: {out_path}  ({len(combined)} regions, "
        f"{len(held_out)} marked insufficient-data)")

    joblib.dump({"scaler": scaler, "kmeans": km, "features": FEATURES,
                 "cluster_names": names, "k": best_k,
                 "dbscan_eps": eps, "dbscan_min_samples": min_samples},
                C.MODELS / f"cluster_model_{level}.joblib")
    log(f"SAVED: {C.MODELS / f'cluster_model_{level}.joblib'}")

    return dict(level=level, k=best_k, silhouette=float(silhouette_score(X, labels)),
                ari_vs_rule=float(ari), dbscan_clusters=int(n_db),
                dbscan_noise=int(n_noise), names=names)


def main():
    log("=" * 78)
    log("PHASE 6 — CLUSTERING")
    log("=" * 78)

    summary = []
    summary.append(run_level(
        "state", C.STATE_FEATURES, C.STATE_CLUSTERED,
        dict(elbow="fig09_model_selection_state.png",
             profile="fig11_cluster_profile_state.png",
             map="fig13_cluster_map_state.png",
             pca="fig15_cluster_pca_state.png",
             kdist="fig17_dbscan_kdistance_state.png",
             cross="fig19_cluster_vs_rule_state.png")))

    summary.append(run_level(
        "district", C.DISTRICT_FEATURES, C.DISTRICT_CLUSTERED,
        dict(elbow="fig10_model_selection_district.png",
             profile="fig12_cluster_profile_district.png",
             map="fig14_cluster_map_district.png",
             pca="fig16_cluster_pca_district.png",
             kdist="fig18_dbscan_kdistance_district.png",
             cross="fig20_cluster_vs_rule_district.png")))

    (C.REPORTS / "06_clustering_summary.json").write_text(json.dumps(summary, indent=2))
    (C.REPORTS / "06_clustering_report.txt").write_text("\n".join(LINES))
    print(f"\nReport written: {C.REPORTS / '06_clustering_report.txt'}")


if __name__ == "__main__":
    main()
