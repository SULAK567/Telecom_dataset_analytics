# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Phase 7 — Supervised extension: Random Forest + SHAP.

Task: predict a district's coverage tier from regional features.

METHODOLOGICAL NOTE (this is the point of the phase, not a footnote)
--------------------------------------------------------------------
The rule-based coverage tier is *constructed from* log_tower_density,
pct_modern and active_carrier_count. Training a classifier on those same
three inputs and reporting 99% accuracy would be circular: the model would
simply be re-deriving the arithmetic that produced its own label.

So two models are trained:

  Model A — full feature set (INCLUDES the rule's own inputs).
            Expected to be near-perfect. Reported as a sanity check that the
            pipeline is wired correctly, and explicitly labelled circular.

  Model B — leakage-free feature set (EXCLUDES every rule input).
            Trained only on carrier market structure, cell geometry and
            measurement characteristics. This is the real experiment: is
            coverage tier recoverable from signals the rule never saw?

SHAP is run on Model B, because explaining Model A would only explain the rule.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import config as C
import viz_style as V

V.apply_style()
LINES = []
TIERS = ["Underserved", "Moderate", "Well-served"]

# Model A: the rule's own inputs are present -> circular by construction
FEATURES_A = ["log_tower_density", "pct_2g", "pct_4g", "active_carrier_count",
              "carrier_hhi", "range_mean", "log_sample_mean"]

# Model B: every rule input removed. Nothing here encodes density, 4G share
# or carrier COUNT; only market shares, cell geometry and measurement traits.
FEATURES_B = ["carrier_hhi", "range_mean", "log_sample_mean",
              "pct_airtel", "pct_vi", "pct_jio_reliance", "pct_bsnl",
              "updated_year_mean", "created_year_mean", "area_sqkm"]


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def save(fig, name):
    fig.savefig(C.FIGURES / name)
    plt.close(fig)
    log(f"  saved figure: {name}")


def confusion_figure(y_true, y_pred, title, subtitle, fname):
    cm = confusion_matrix(y_true, y_pred, labels=TIERS)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap=V.CMAP_SEQ)
    ax.set_xticks(range(3)); ax.set_xticklabels(TIERS, rotation=18, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(TIERS)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.grid(False)
    vmax = cm.max()
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="#ffffff" if cm[i, j] > vmax * 0.55 else V.INK_PRIMARY)
    V.title_block(ax, title, subtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, fname)
    return cm


def train_model(df, features, name, fname_cm):
    log("\n" + "=" * 78)
    log(f"{name}")
    log("=" * 78)
    log(f"Features ({len(features)}): {features}")

    X = df[features]
    y = df["coverage_tier"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=C.RANDOM_STATE, stratify=y)
    log(f"\nTrain: {len(X_tr)}   Test: {len(X_te)} (stratified 70/30)")

    clf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        random_state=C.RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    log(f"\nHeld-out accuracy      : {acc:.4f}")
    log(f"Held-out macro F1      : {f1m:.4f}")

    cv = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True,
                         random_state=C.RANDOM_STATE), scoring="accuracy")
    log(f"5-fold CV accuracy     : {cv.mean():.4f} +/- {cv.std():.4f}  "
        f"({np.round(cv, 4).tolist()})")

    log("\nPer-class report (held-out test set):")
    log(classification_report(y_te, pred, labels=TIERS, zero_division=0))

    confusion_figure(y_te, pred, f"Confusion matrix — {name}",
                     f"Held-out test set, n={len(y_te)}, accuracy {acc:.1%}",
                     fname_cm)
    return clf, X_tr, X_te, y_te, pred, dict(accuracy=float(acc),
                                             macro_f1=float(f1m),
                                             cv_mean=float(cv.mean()),
                                             cv_std=float(cv.std()))


def importance_figure(clf, X_te, y_te, features, fname):
    imp = pd.Series(clf.feature_importances_, index=features)
    perm = permutation_importance(clf, X_te, y_te, n_repeats=30,
                                  random_state=C.RANDOM_STATE, n_jobs=-1)
    permi = pd.Series(perm.importances_mean, index=features)

    order = permi.sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    for ax, ser, title, sub in [
        (axes[0], imp.reindex(order), "Impurity-based importance",
         "Gini decrease across 500 trees — biased toward high-cardinality features"),
        (axes[1], permi.reindex(order), "Permutation importance",
         "Accuracy drop when a feature is shuffled — measured on held-out data"),
    ]:
        bars = ax.barh(ser.index, ser.values, color=V.CATEGORICAL[0],
                       height=0.62, zorder=3)
        for b, v in zip(bars, ser.values):
            ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.3f}",
                    va="center", ha="left", fontsize=9, color=V.INK_PRIMARY)
        V.strip_spines(ax, keep=("bottom",))
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.set_xlim(min(0, ser.min() * 1.2), max(ser.max() * 1.28, 1e-6))
        V.title_block(ax, title, sub)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, fname)

    log("\nPermutation importance (held-out, 30 repeats):")
    log(permi.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    return permi


def shap_figures(clf, X_tr, features):
    try:
        import shap
    except ImportError:
        log("\nSHAP not installed — skipping SHAP explanations.")
        return

    log("\n" + "-" * 78)
    log("SHAP EXPLANATIONS (Model B)")
    log("-" * 78)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_tr)

    # normalise across shap versions: want a list of (n, f) arrays, one per class
    if isinstance(sv, list):
        per_class = sv
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        per_class = [sv[:, :, i] for i in range(sv.shape[2])]
    else:
        per_class = [sv]
    log(f"SHAP values computed: {len(per_class)} class(es), "
        f"{per_class[0].shape[0]} samples x {per_class[0].shape[1]} features")

    classes = list(clf.classes_)

    # ---- mean |SHAP| per feature, per class ----
    mean_abs = pd.DataFrame(
        {classes[i]: np.abs(per_class[i]).mean(axis=0) for i in range(len(per_class))},
        index=features)
    mean_abs["overall"] = mean_abs.mean(axis=1)
    mean_abs = mean_abs.sort_values("overall")

    log("\nMean |SHAP| per feature (higher = more influence on the prediction):")
    log(mean_abs.sort_values("overall", ascending=False).to_string(
        float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    left = np.zeros(len(mean_abs))
    for i, cls in enumerate([c for c in TIERS if c in mean_abs.columns]):
        vals = mean_abs[cls].to_numpy()
        ax.barh(mean_abs.index, vals, left=left, height=0.62,
                color=V.ORDINAL_3[i], label=cls, edgecolor=V.SURFACE,
                linewidth=1.6, zorder=3)
        left += vals
    V.strip_spines(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP value|")
    V.legend_below(ax, ncol=3, y=-0.13)
    V.title_block(ax, "What drives the coverage-tier prediction",
                  "Mean absolute SHAP contribution per feature, split by predicted class")
    fig.tight_layout(rect=[0, 0.02, 1, 0.89])
    save(fig, "fig24_shap_importance_modelB.png")

    # ---- beeswarm for the Well-served class ----
    try:
        idx = classes.index("Well-served")
        plt.figure(figsize=(8.4, 5.6))
        shap.summary_plot(per_class[idx], X_tr, feature_names=features,
                          show=False, plot_size=None, color_bar_label="feature value")
        fig = plt.gcf()
        fig.suptitle("SHAP value distribution — 'Well-served' class",
                     x=0.02, ha="left", fontsize=13, fontweight="600",
                     color=V.INK_PRIMARY)
        fig.patch.set_facecolor(V.SURFACE)
        for a in fig.axes:
            a.set_facecolor(V.SURFACE)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save(fig, "fig25_shap_beeswarm_wellserved.png")
    except Exception as e:
        log(f"  beeswarm skipped: {e}")


def main():
    log("=" * 78)
    log("PHASE 7 — SUPERVISED EXTENSION (Random Forest + SHAP)")
    log("=" * 78)

    df = pd.read_csv(C.DISTRICT_CLUSTERED)
    df = df[df["cluster"] >= 0].copy()          # exclude insufficient-data regions
    log(f"\nDistricts available for supervised learning: {len(df)}")
    log("(Regions held out by the data-sufficiency gate are excluded — a label")
    log(" built on <50 towers is not a trustworthy training target.)")

    log("\nClass balance:")
    for t in TIERS:
        n = int((df["coverage_tier"] == t).sum())
        log(f"  {t:<13s} {n:>4}  ({n/len(df):6.2%})")

    log("\nState level is NOT modelled: 35 regions cannot support a train/test")
    log("split without the estimate being dominated by split noise.")

    results = {}

    clf_a, _, _, _, _, res_a = train_model(
        df, FEATURES_A, "MODEL A — full feature set (CIRCULAR, includes rule inputs)",
        "fig21_confusion_modelA.png")
    results["model_a_circular"] = res_a
    log("\n>>> Model A's accuracy is NOT evidence of anything: three of its")
    log(">>> features are the exact quantities the label was computed from.")

    clf_b, X_tr_b, X_te_b, y_te_b, _, res_b = train_model(
        df, FEATURES_B, "MODEL B — leakage-free feature set (THE REAL TEST)",
        "fig22_confusion_modelB.png")
    results["model_b_leakage_free"] = res_b

    baseline = df["coverage_tier"].value_counts(normalize=True).max()
    log(f"\nMajority-class baseline accuracy: {baseline:.4f}")
    log(f"Model B lift over baseline      : {res_b['accuracy'] - baseline:+.4f}")

    importance_figure(clf_b, X_te_b, y_te_b, FEATURES_B,
                      "fig23_feature_importance_modelB.png")
    shap_figures(clf_b, X_tr_b, FEATURES_B)

    joblib.dump({"model_a": clf_a, "features_a": FEATURES_A,
                 "model_b": clf_b, "features_b": FEATURES_B,
                 "classes": TIERS},
                C.MODELS / "rf_coverage_tier.joblib")
    log(f"\nSAVED: {C.MODELS / 'rf_coverage_tier.joblib'}")

    results["majority_baseline"] = float(baseline)
    (C.REPORTS / "07_supervised_summary.json").write_text(json.dumps(results, indent=2))
    (C.REPORTS / "07_supervised_report.txt").write_text("\n".join(LINES))
    print(f"\nReport written: {C.REPORTS / '07_supervised_report.txt'}")


if __name__ == "__main__":
    main()
