# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Automated insight generation.

Every number in here is computed from the aggregated tables at request time —
nothing is hard-coded. Each function returns a list of findings shaped as
{"title", "body", "kind"}, where `kind` is one of
neutral / good / watch / concern and drives the callout's severity rule.

The point of this module: a chart shows *what* the data is, an insight states
*what it means*. Without this layer the dashboard makes the reader do all the
interpretive work.
"""
import numpy as np
import pandas as pd

import config as C

ACTIVE = C.ACTIVE_CARRIERS
NCOL = {c: "n_" + c.lower().replace("/", "_") for c in ACTIVE}
PCOL = {c: "pct_" + c.lower().replace("/", "_") for c in ACTIVE}


# ----------------------------------------------------------------- helpers
def ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def num(x, nd=0):
    return f'<span class="tc-n">{x:,.{nd}f}</span>'


def _f(x, nd=1):
    return "—" if pd.isna(x) else f"{x:,.{nd}f}"


def clustered(df):
    return df[df["cluster"] >= 0] if "cluster" in df.columns else df


# ------------------------------------------------------- region insights
def region_insights(df, region, level):
    """Findings about one selected region, in context of the current scope."""
    out = []
    pool = clustered(df)
    row = df[df["region"] == region]
    if row.empty:
        return out
    r = row.iloc[0]
    n = len(pool)
    unit = "state" if level == "State" else "district"

    # --- data confidence gate first: it qualifies everything below ---
    if r.get("cluster", 0) < 0:
        out.append(dict(
            kind="concern",
            title="Treat these numbers as indicative only",
            body=(f"Only {num(r['tower_count'])} towers are recorded here, below the "
                  f"50-tower threshold this project requires. Percentages computed "
                  f"from so few crowd-sourced points are unstable, so {region} was "
                  f"held out of the clustering rather than given a coverage label "
                  f"the evidence cannot support.")))
        return out

    # --- density rank ---
    rank_d = int((pool["tower_density"] > r["tower_density"]).sum()) + 1
    pctile = 100 * (1 - (rank_d - 1) / max(n - 1, 1))
    med = pool["tower_density"].median()
    ratio = r["tower_density"] / med if med else np.nan
    out.append(dict(
        kind="good" if pctile >= 66 else ("watch" if pctile >= 33 else "concern"),
        title=f"{ordinal(rank_d)} densest of {n} {unit}s in view",
        body=(f"{region} carries {num(r['tower_density'], 2)} towers per km², "
              f"which is {num(ratio, 1)}× the median {unit} "
              f"({_f(med, 2)}/km²). That places it in the "
              f"{num(pctile, 0)}th percentile for raw coverage intensity.")))

    # --- modernity vs the pool ---
    med_m = pool["pct_modern"].median()
    diff = r["pct_modern"] - med_m
    out.append(dict(
        kind="good" if diff > 3 else ("concern" if diff < -3 else "neutral"),
        title=("Network is more modern than typical" if diff > 3 else
               "Network lags on modernisation" if diff < -3 else
               "Network modernity is about average"),
        body=(f"{num(r['pct_modern'], 1)}% of towers here are 4G or 5G, against a "
              f"median of {_f(med_m, 1)}% — {num(abs(diff), 1)} points "
              f"{'above' if diff >= 0 else 'below'}. "
              f"{num(r['pct_2g'], 1)}% of the estate is still 2G.")))

    # --- the interesting cross-signal: dense but legacy ---
    if r["tower_density"] > med and r["pct_modern"] < med_m:
        out.append(dict(
            kind="watch",
            title="Modernisation gap: plenty of towers, dated technology",
            body=(f"{region} has above-median tower density but below-median 4G/5G "
                  f"share. The infrastructure footprint exists; the generation mix "
                  f"has not caught up. Regions in this position are upgrade "
                  f"candidates rather than new-build candidates — the sites are "
                  f"already there.")))
    elif r["tower_density"] < med and r["pct_modern"] > med_m:
        out.append(dict(
            kind="neutral",
            title="Thin but modern: quality over quantity",
            body=(f"Coverage is sparse ({num(r['tower_density'], 2)}/km²) yet the "
                  f"towers that exist are relatively modern "
                  f"({num(r['pct_modern'], 1)}% 4G/5G). This pattern usually means "
                  f"newer selective deployment rather than a legacy build-out.")))

    # --- competition ---
    leader = max(ACTIVE, key=lambda c: r.get(PCOL[c], 0))
    lead_share = r.get(PCOL[leader], np.nan)
    hhi = r.get("carrier_hhi", np.nan)
    if hhi > 0.45:
        kind, verdict = "concern", "highly concentrated"
    elif hhi > 0.32:
        kind, verdict = "watch", "moderately concentrated"
    else:
        kind, verdict = "good", "competitive"
    out.append(dict(
        kind=kind,
        title=f"{leader} leads a {verdict} market",
        body=(f"{leader} holds {num(lead_share, 1)}% of towers here, with "
              f"{int(r['active_carrier_count'])} of 4 major operators present. "
              f"The concentration index is {num(hhi, 3)} "
              f"(0.25 = four equal operators, 1.0 = monopoly), which reads as "
              f"{verdict}.")))

    # --- district vs parent state ---
    if level == "District" and "state_name" in df.columns and pd.notna(r.get("state_name")):
        sib = pool[pool["state_name"] == r["state_name"]]
        if len(sib) > 2:
            rk = int((sib["tower_density"] > r["tower_density"]).sum()) + 1
            out.append(dict(
                kind="neutral",
                title=f"{ordinal(rk)} of {len(sib)} districts within {r['state_name']}",
                body=(f"Against its own state rather than all of India, {region} sits "
                      f"at {num(r['tower_density'], 2)}/km² versus a state-median "
                      f"{_f(sib['tower_density'].median(), 2)}/km². Comparing within a "
                      f"state controls for how much of the difference is simply "
                      f"regional economics.")))

    # --- outlier flag ---
    if r.get("dbscan", 0) == -1:
        out.append(dict(
            kind="watch",
            title="Flagged as a statistical outlier by DBSCAN",
            body=(f"{region} does not sit inside the main density manifold — its "
                  f"combination of features is unusual enough that a "
                  f"density-based method treats it as its own case. Worth a look "
                  f"before drawing conclusions from it.")))

    # --- measurement confidence ---
    if r["sample_mean"] < 3:
        out.append(dict(
            kind="watch",
            title="Thin measurement evidence per tower",
            body=(f"Each tower here is backed by only {num(r['sample_mean'], 1)} "
                  f"crowd-sourced observations on average. Low sample counts mean "
                  f"the tower locations themselves are less precisely established.")))
    return out


# ----------------------------------------------------- overview insights
def overview_insights(df, level, scope="all of India"):
    out = []
    pool = clustered(df)
    if len(pool) < 3:
        return out
    unit = "state" if level == "State" else "district"
    n = len(pool)

    # --- inequality of coverage ---
    top = pool.nlargest(1, "tower_density").iloc[0]
    bot = pool.nsmallest(1, "tower_density").iloc[0]
    spread = top["tower_density"] / max(bot["tower_density"], 1e-9)
    out.append(dict(
        kind="concern" if spread > 1000 else "watch",
        title=f"Coverage is wildly unequal across {scope}",
        body=(f"{top['region']} carries {num(top['tower_density'], 1)} towers per km²; "
              f"{bot['region']} carries {num(bot['tower_density'], 4)}. That is a "
              f"{num(spread, 0)}-fold spread between the densest and sparsest "
              f"{unit}. Any national average hides this completely.")))

    # --- concentration of the estate ---
    tot = pool["tower_count"].sum()
    k = max(3, int(round(n * 0.10)))
    share = pool.nlargest(k, "tower_count")["tower_count"].sum() / tot * 100
    out.append(dict(
        kind="watch",
        title=f"The top {k} {unit}s hold {_f(share, 1)}% of all towers",
        body=(f"{num(share, 1)}% of the {num(tot)} towers in view sit in just "
              f"{k} of {n} {unit}s ({num(100 * k / n, 0)}% of regions). Coverage "
              f"follows population and economic density far more closely than it "
              f"follows land area.")))

    # --- modernisation priority: dense but legacy ---
    med_d, med_m = pool["tower_density"].median(), pool["pct_modern"].median()
    gap = pool[(pool["tower_density"] > med_d) & (pool["pct_modern"] < med_m)]
    if len(gap):
        worst = gap.nsmallest(3, "pct_modern")["region"].tolist()
        out.append(dict(
            kind="watch",
            title=f"{len(gap)} {unit}s are upgrade candidates, not build candidates",
            body=(f"These have above-median tower density but below-median 4G/5G "
                  f"share — the sites exist, the technology is dated. The clearest "
                  f"cases are {', '.join(worst)}. This is the cheapest kind of "
                  f"coverage improvement to deliver.")))

    # --- competition gaps ---
    thin = pool[pool["active_carrier_count"] <= 2]
    if len(thin):
        out.append(dict(
            kind="concern",
            title=f"{len(thin)} {unit}s have two or fewer major operators",
            body=(f"Limited operator presence means limited price competition and no "
                  f"fallback when one network fails. Examples: "
                  f"{', '.join(thin.nsmallest(3, 'active_carrier_count')['region'].tolist())}.")))

    # --- 2G burden ---
    heavy = pool[pool["pct_2g"] > 75]
    if len(heavy):
        out.append(dict(
            kind="concern",
            title=f"{len(heavy)} {unit}s remain more than three-quarters 2G",
            body=(f"Across {scope} the median 2G share is {_f(pool['pct_2g'].median(), 1)}%. "
                  f"In {len(heavy)} {unit}s it exceeds 75%, meaning mobile data there "
                  f"is effectively unusable for modern applications even where a "
                  f"signal exists.")))

    # --- tier split ---
    if "coverage_tier" in pool.columns:
        vc = pool["coverage_tier"].value_counts()
        und = int(vc.get("Underserved", 0))
        und_towers = pool[pool["coverage_tier"] == "Underserved"]["tower_count"].sum()
        out.append(dict(
            kind="neutral",
            title=f"{und} {unit}s classify as underserved, holding only "
                  f"{_f(und_towers / tot * 100, 1)}% of towers",
            body=(f"The underserved third of {unit}s accounts for "
                  f"{num(und_towers / tot * 100, 1)}% of the national tower estate. "
                  f"The imbalance is the finding: being in the bottom tier by rank "
                  f"understates how little infrastructure is actually there.")))
    return out


# ------------------------------------------------------ carrier insights
def carrier_insights(df, mix=None):
    out = []
    pool = clustered(df)
    tot = {c: pool[NCOL[c]].sum() for c in ACTIVE}
    grand = sum(tot.values())
    if grand == 0:
        return out

    lead = max(tot, key=tot.get)
    out.append(dict(
        kind="neutral",
        title=f"{lead} operates the largest recorded estate",
        body=" · ".join(f"{c}: {num(tot[c])} ({num(100 * tot[c] / grand, 1)}%)"
                        for c in sorted(ACTIVE, key=lambda x: -tot[x])) +
             ". Shares are of towers recorded in this dataset, not of subscribers.")

    )
    if mix is not None and not mix.empty:
        m = mix.copy()
        if "4G" in m.columns:
            best = m["4G"].idxmax()
            worst = m["4G"].idxmin()
            out.append(dict(
                kind="watch",
                title=f"{best} is the most modern network; {worst} the least",
                body=(f"{best} runs {num(m.loc[best, '4G'], 1)}% of its towers on 4G, "
                      f"against {num(m.loc[worst, '4G'], 1)}% for {worst} — a "
                      f"{num(m.loc[best, '4G'] - m.loc[worst, '4G'], 1)}-point gap. "
                      f"Operator choice, not just location, determines what "
                      f"generation of service a user actually gets.")))
            legacy = m[m["2G"] > 60].index.tolist()
            if legacy:
                out.append(dict(
                    kind="concern",
                    title=f"{', '.join(legacy)} remain majority-2G networks",
                    body=("More than 60% of their towers are 2G. For subscribers "
                          "this is the practical difference between a phone that "
                          "makes calls and one that carries usable data.")))

    # where each carrier is uniquely strong
    strong = []
    for c in ACTIVE:
        sub = pool[pool[PCOL[c]] == pool[[PCOL[x] for x in ACTIVE]].max(axis=1)]
        if len(sub):
            strong.append(f"{c} leads in {len(sub)} regions")
    if strong:
        out.append(dict(kind="neutral", title="Regional leadership is split",
                        body=" · ".join(strong) +
                             ". No single operator dominates everywhere, which is why "
                             "carrier-level coverage differs from aggregate coverage."))
    return out


# --------------------------------------------------- generation insights
def generation_insights(df, level):
    out = []
    pool = clustered(df)
    tot = {g: pool[f"n_{g}"].sum() for g in ["2g", "3g", "4g", "5g"]}
    grand = sum(tot.values())
    if grand == 0:
        return out

    out.append(dict(
        kind="concern" if tot["2g"] / grand > 0.5 else "watch",
        title=f"{_f(100 * tot['2g'] / grand, 1)}% of towers in view are still 2G",
        body=(f"2G {num(tot['2g'])} · 3G {num(tot['3g'])} · 4G {num(tot['4g'])} · "
              f"5G {num(tot['5g'])}. A 2G-majority estate means the ceiling on "
              f"mobile service for most of this area is voice and SMS, not data.")))

    if tot["5g"] < grand * 0.001:
        out.append(dict(
            kind="neutral",
            title="5G is effectively absent from this dataset",
            body=(f"Only {num(tot['5g'])} towers are 5G "
                  f"({num(100 * tot['5g'] / grand, 4)}%). This snapshot predates "
                  f"India's commercial 5G rollout, so the absence reflects the "
                  f"vintage of the data, not the state of the network today. Any "
                  f"conclusion about 5G from this project would be invalid.")))

    unit = "state" if level == "State" else "district"
    spread = pool["pct_4g"].max() - pool["pct_4g"].min()
    hi = pool.nlargest(1, "pct_4g").iloc[0]
    lo = pool.nsmallest(1, "pct_4g").iloc[0]
    out.append(dict(
        kind="watch",
        title=f"4G share varies {num(spread, 0)} points between {unit}s",
        body=(f"{hi['region']} reaches {num(hi['pct_4g'], 1)}% 4G while "
              f"{lo['region']} sits at {num(lo['pct_4g'], 1)}%. Where you are "
              f"matters more for service quality than the national average implies.")))
    return out


# ----------------------------------------------------- cluster insights
def cluster_insights(df, level):
    out = []
    pool = clustered(df)
    if pool.empty or "cluster_name" not in pool.columns:
        return out

    sizes = pool["cluster_name"].value_counts()
    prof = pool.groupby("cluster_name").agg(
        density=("tower_density", "mean"), modern=("pct_modern", "mean"),
        legacy=("pct_2g", "mean"), carriers=("active_carrier_count", "mean"),
        towers=("tower_count", "sum"))

    unit = "state" if level == "State" else "district"
    for name, sz in sizes.items():
        p = prof.loc[name]
        out.append(dict(
            kind="neutral",
            title=f"{name} — {sz} {unit}s ({num(100 * sz / len(pool), 1)}% of regions)",
            body=(f"Averages {num(p['density'], 2)} towers/km², "
                  f"{num(p['modern'], 1)}% 4G/5G, {num(p['legacy'], 1)}% 2G and "
                  f"{num(p['carriers'], 1)} operators. Together these hold "
                  f"{num(p['towers'])} towers "
                  f"({num(100 * p['towers'] / pool['tower_count'].sum(), 1)}% of "
                  f"the estate in view).")))

    if "coverage_tier" in pool.columns:
        ct = pd.crosstab(pool["cluster_name"], pool["coverage_tier"])
        agree = sum(ct.loc[i].max() for i in ct.index) / len(pool) * 100
        out.append(dict(
            kind="good" if agree > 60 else "watch",
            title=f"Clusters and the independent rule agree {num(agree, 0)}% of the time",
            body=("The clustering never saw the rule-based tier — it was built only "
                  "from raw features. That the two land in the same place this "
                  "often is the main evidence these groups describe real coverage "
                  "structure rather than arbitrary partitions.")))
    return out


# ============================================================================
# Question-driven explorer: the user picks what they want to know.
# ============================================================================
def _bar_answer(pool, col, label, ascending, headline, detail, nd=2, kind="neutral"):
    sub = (pool.nsmallest(12, col) if ascending else pool.nlargest(12, col))
    # De-duplicate: the ranked column is often already one of the context
    # columns (e.g. ranking BY pct_4g), and selecting it twice makes pandas
    # emit a duplicate-label frame that Streamlit refuses to render.
    wanted = ["region", col, "tower_count", "pct_2g", "pct_4g",
              "active_carrier_count"]
    cols, seen = [], set()
    for c in wanted:
        if c not in seen and c in sub.columns:
            cols.append(c); seen.add(c)
    table = sub[cols].copy()
    return dict(headline=headline, detail=detail, kind=kind, table=table,
                value_col=col, value_label=label, nd=nd, ascending=ascending)


QUESTIONS = [
    "Which regions are the most underserved?",
    "Which regions have the best coverage?",
    "Where is 4G weakest?",
    "Which regions still run mostly on 2G?",
    "Where is there the least operator competition?",
    "Which regions have lots of towers but old technology?",
    "Where does each carrier dominate?",
    "Which regions are unusual compared to everywhere else?",
    "Compare two regions side by side",
]


def answer(question, df, level, region_a=None, region_b=None):
    pool = clustered(df)
    unit = "state" if level == "State" else "district"
    if pool.empty:
        return dict(headline="Not enough data in this selection.", detail="",
                    table=None, kind="neutral")

    if question == "Which regions are the most underserved?":
        p = pool.copy()
        p["_score"] = (p["tower_density"].rank(pct=True)
                       + p["pct_modern"].rank(pct=True)
                       + p["active_carrier_count"].rank(pct=True)) / 3
        sub = p.nsmallest(12, "_score")
        return dict(
            headline=f"{sub.iloc[0]['region']} is the most underserved {unit} in view",
            detail=(f"Ranked on a combination of tower density, 4G/5G share and "
                    f"operator presence. The bottom {len(sub)} {unit}s shown here "
                    f"average {_f(sub['tower_density'].mean(), 3)} towers/km² and "
                    f"{_f(sub['pct_modern'].mean(), 1)}% modern towers, against "
                    f"{_f(pool['tower_density'].median(), 2)}/km² and "
                    f"{_f(pool['pct_modern'].median(), 1)}% for the median {unit}."),
            kind="concern", table=sub[["region", "tower_density", "pct_modern",
                                       "active_carrier_count", "tower_count"]],
            value_col="tower_density", value_label="towers / km²", nd=4,
            ascending=True)

    if question == "Which regions have the best coverage?":
        sub = pool.nlargest(12, "tower_density")
        return _bar_answer(
            pool, "tower_density", "towers / km²", False,
            f"{sub.iloc[0]['region']} has the densest coverage in view",
            (f"Density is towers per km², so small urban {unit}s dominate this "
             f"ranking by construction — the metric rewards compactness. "
             f"{sub.iloc[0]['region']} reaches {_f(sub.iloc[0]['tower_density'], 1)}/km², "
             f"{_f(sub.iloc[0]['tower_density'] / pool['tower_density'].median(), 0)}× "
             f"the median."), nd=2, kind="good")

    if question == "Where is 4G weakest?":
        return _bar_answer(
            pool, "pct_4g", "% of towers on 4G", True,
            "4G penetration is lowest in these regions",
            (f"The median {unit} runs {_f(pool['pct_4g'].median(), 1)}% of towers on "
             f"4G. The regions below sit far under that, meaning even where a "
             f"signal exists, usable mobile data often does not."),
            nd=2, kind="concern")

    if question == "Which regions still run mostly on 2G?":
        return _bar_answer(
            pool, "pct_2g", "% of towers on 2G", False,
            f"{pool.nlargest(1, 'pct_2g').iloc[0]['region']} has the heaviest 2G reliance",
            (f"2G carries voice and SMS but not meaningful data. "
             f"{int((pool['pct_2g'] > 75).sum())} {unit}s in view are more than "
             f"75% 2G; the median is {_f(pool['pct_2g'].median(), 1)}%."),
            nd=2, kind="concern")

    if question == "Where is there the least operator competition?":
        return _bar_answer(
            pool, "carrier_hhi", "market concentration (HHI)", False,
            "These regions have the most concentrated operator markets",
            (f"HHI runs from 0.25 (four equally-sized operators) to 1.0 (a single "
             f"operator). {int((pool['active_carrier_count'] <= 2).sum())} {unit}s "
             f"here have two or fewer major operators present, which limits both "
             f"price competition and network redundancy."),
            nd=3, kind="concern")

    if question == "Which regions have lots of towers but old technology?":
        med_d, med_m = pool["tower_density"].median(), pool["pct_modern"].median()
        sub = pool[(pool["tower_density"] > med_d) & (pool["pct_modern"] < med_m)]
        sub = sub.nsmallest(12, "pct_modern")
        return dict(
            headline=f"{len(pool[(pool['tower_density'] > med_d) & (pool['pct_modern'] < med_m)])} "
                     f"{unit}s have the towers but not the technology",
            detail=("These sit above the median on tower density and below it on "
                    "4G/5G share. The infrastructure is already built, so improving "
                    "service here is an upgrade problem rather than a construction "
                    "problem — usually the cheaper and faster intervention."),
            kind="watch",
            table=sub[["region", "tower_density", "pct_modern", "pct_2g",
                       "tower_count"]],
            value_col="pct_modern", value_label="% 4G + 5G", nd=2, ascending=True)

    if question == "Where does each carrier dominate?":
        rows = []
        for c in ACTIVE:
            lead = pool[pool[[PCOL[x] for x in ACTIVE]].idxmax(axis=1) == PCOL[c]]
            rows.append(dict(Carrier=c, regions_led=len(lead),
                             mean_share=lead[PCOL[c]].mean() if len(lead) else np.nan,
                             towers=int(pool[NCOL[c]].sum())))
        t = pd.DataFrame(rows).sort_values("regions_led", ascending=False)
        top = t.iloc[0]
        return dict(
            headline=f"{top['Carrier']} leads in the most regions ({int(top['regions_led'])})",
            detail=("A carrier can hold the largest national estate yet lead in "
                    "fewer regions, because leadership is about local share. "
                    "'Regions led' counts where that operator has the highest "
                    "tower share; 'mean share' is its average share where it leads."),
            kind="neutral", table=t, value_col="regions_led",
            value_label=f"{unit}s led", nd=0, ascending=False, label_col="Carrier")

    if question == "Which regions are unusual compared to everywhere else?":
        if "dbscan" not in pool.columns:
            return dict(headline="Outlier detection not available.", detail="",
                        table=None, kind="neutral")
        sub = pool[pool["dbscan"] == -1]
        if sub.empty:
            return dict(headline="No regions were flagged as outliers here.",
                        detail="DBSCAN found every region in this selection to sit "
                               "inside the main density manifold.",
                        table=None, kind="good")
        return dict(
            headline=f"DBSCAN flags {len(sub)} {unit}s as statistical outliers",
            detail=("These do not fit the main pattern — their mix of density, "
                    "technology and operator presence is unusual enough that a "
                    "density-based method treats each as its own case. They are "
                    "where the national story breaks down, so they are worth "
                    "inspecting individually."),
            kind="watch",
            table=sub[["region", "tower_density", "pct_modern", "pct_2g",
                       "active_carrier_count", "tower_count"]].nlargest(
                           12, "tower_count"),
            value_col="tower_density", value_label="towers / km²", nd=3,
            ascending=False)

    if question == "Compare two regions side by side":
        if not region_a or not region_b or region_a == region_b:
            return dict(headline="Pick two different regions to compare.",
                        detail="", table=None, kind="neutral")
        a = pool[pool["region"] == region_a]
        b = pool[pool["region"] == region_b]
        if a.empty or b.empty:
            return dict(headline="One of those regions has no clustered data.",
                        detail="", table=None, kind="neutral")
        a, b = a.iloc[0], b.iloc[0]
        metrics = [("Towers", "tower_count", 0), ("Towers per km²", "tower_density", 3),
                   ("Area km²", "area_sqkm", 0), ("2G %", "pct_2g", 1),
                   ("3G %", "pct_3g", 1), ("4G %", "pct_4g", 1),
                   ("4G+5G %", "pct_modern", 1),
                   ("Operators", "active_carrier_count", 0),
                   ("Concentration HHI", "carrier_hhi", 3),
                   ("Mean cell range m", "range_mean", 0),
                   ("Samples per tower", "sample_mean", 1)]
        t = pd.DataFrame([{"Metric": lab, region_a: round(a[c], nd),
                           region_b: round(b[c], nd)} for lab, c, nd in metrics])
        dd = a["tower_density"] / max(b["tower_density"], 1e-9)
        return dict(
            headline=f"{region_a} vs {region_b}",
            detail=(f"{region_a} is {_f(dd, 1)}× as dense as {region_b} "
                    f"({_f(a['tower_density'], 3)} vs {_f(b['tower_density'], 3)} "
                    f"towers/km²) and "
                    f"{_f(abs(a['pct_modern'] - b['pct_modern']), 1)} points "
                    f"{'ahead' if a['pct_modern'] > b['pct_modern'] else 'behind'} "
                    f"on 4G/5G share. "
                    f"{region_a} is grouped as “{a.get('cluster_name', '—')}”, "
                    f"{region_b} as “{b.get('cluster_name', '—')}”."),
            kind="neutral", table=t, value_col=None, value_label=None, nd=2,
            ascending=False)

    return dict(headline="Question not recognised.", detail="", table=None,
                kind="neutral")
