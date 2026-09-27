"""
India Telecom Network Coverage Analytics — Streamlit dashboard.

Run from the project root:
    streamlit run src/dashboard.py

Design notes
------------
* Every view pairs a chart with COMPUTED insights (insights.py) so the reader
  is told what the data means, not just shown what it is.
* All charts come from chart_kit.py, which fixes label contrast, clipped
  value labels and Streamlit's chart-theme override.
* First-time visitors land on "Start here", which carries a guided tour and a
  plain-language glossary.
"""

# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import chart_kit as K
import config as C
import insights as I
import ui_kit as U
import viz_style as V

st.set_page_config(page_title="India Telecom Coverage Analytics",
                   page_icon="📶", layout="wide",
                   initial_sidebar_state="expanded")
U.inject_theme()

TIERS = ["Underserved", "Moderate", "Well-served"]
INSUFFICIENT_COLOR = "#d7d6d1"
NCOL = {c: "n_" + c.lower().replace("/", "_") for c in C.ACTIVE_CARRIERS}
PCOL = {c: "pct_" + c.lower().replace("/", "_") for c in C.ACTIVE_CARRIERS}


# ------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_regions(level):
    df = pd.read_csv(C.STATE_CLUSTERED if level == "State" else C.DISTRICT_CLUSTERED)
    df["is_clustered"] = df["cluster"] >= 0
    return df


@st.cache_data(show_spinner=False)
def load_geo(level):
    path = (C.DATA / "states_simplified.geojson" if level == "State"
            else C.DATA / "districts_simplified.geojson")
    key = "st_nm" if level == "State" else "district"
    gj = json.loads(path.read_text())
    for f in gj["features"]:
        f["id"] = f["properties"][key]
    return gj


@st.cache_data(show_spinner=False)
def load_tower_level():
    return pd.read_parquet(C.DATA / "tower_district_lookup.parquet",
                           columns=["carrier", "generation", "st_nm", "district"])


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for lvl in ("state", "district"):
        p = C.MODELS / f"cluster_model_{lvl}.joblib"
        if p.exists():
            out[lvl] = joblib.load(p)
    p = C.MODELS / "rf_coverage_tier.joblib"
    out["rf"] = joblib.load(p) if p.exists() else None
    return out


def cluster_palette(n):
    return {1: [V.ORDINAL_3[1]], 2: [V.ORDINAL_3[0], V.ORDINAL_3[2]],
            3: V.ORDINAL_3, 4: V.ORDINAL_4}.get(n, V.SEQ_BLUE[:n])


def fmt(v, nd=2):
    return "—" if pd.isna(v) else f"{v:,.{nd}f}"


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("seen_welcome", False)
ss.setdefault("tour_step", 0)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📶 Coverage Analytics")
    st.caption("Clustering 2.49M Indian cell-tower records")

    if not ss.seen_welcome:
        st.info("New here? The **Start here** tab walks you through it.")

    st.divider()
    level = st.radio(
        "1 · Map granularity", ["State", "District"], index=0,
        help="State = 36 big regions, easier to read. District = 632 regions, "
             "much more detail. Both are produced by the same pipeline.")

    df_all = load_regions(level)

    color_by = st.selectbox(
        "2 · Colour the map by",
        ["Coverage cluster (K-Means)", "Rule-based tier", "Tower density",
         "4G + 5G share", "2G share", "Active carriers"],
        help="'Coverage cluster' is what the machine-learning model found on "
             "its own. 'Rule-based tier' is a transparent formula used to "
             "check the model. The rest are single raw measures.")

    if level == "District" and "state_name" in df_all.columns:
        states = ["All India"] + sorted(df_all["state_name"].dropna().unique())
        state_filter = st.selectbox("3 · Zoom to one state", states,
                                    help="Narrows every tab, not just the map.")
    else:
        state_filter = "All India"

    df = (df_all if state_filter == "All India"
          else df_all[df_all["state_name"] == state_filter])
    scope = "all of India" if state_filter == "All India" else state_filter

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Regions", f"{len(df):,}")
    c2.metric("Towers", f"{int(df['tower_count'].sum()):,}")
    n_insuf = int((~df["is_clustered"]).sum())
    if n_insuf:
        st.caption(f"⚠️ {n_insuf} region(s) held out of clustering "
                   f"(under 50 towers) — shown in grey.")

    st.divider()
    st.caption("Clusters are **relative, data-driven groupings** built from "
               "crowd-sourced OpenCelliD data — not official or "
               "carrier-provided coverage assessments.")


# ----------------------------------------------------------------- header
st.title("India Telecom Network Coverage Analytics")
st.markdown(
    f'<div class="tc-kicker">{level} view · {scope} · '
    f'{int(df["tower_count"].sum()):,} towers</div>', unsafe_allow_html=True)

if not ss.seen_welcome:
    with st.container():
        st.markdown(
            '<div class="tc-card"><b>👋 First time here?</b><br>'
            '<span style="color:var(--ink-2);font-size:0.92rem">'
            'This dashboard groups Indian states and districts by how good their '
            'mobile coverage is, using 2.49 million real cell-tower records. '
            'Open <b>Start here</b> for a 6-step tour, or jump to '
            '<b>Ask the data</b> and pick a question in plain English.'
            '</span></div>', unsafe_allow_html=True)
        if st.button("Got it — hide this", type="primary"):
            ss.seen_welcome = True
            st.rerun()

tabs = st.tabs(["🧭 Start here", "🗺️ Coverage map", "❓ Ask the data",
                "📡 Carriers", "📶 Generations", "🔬 Clusters",
                "🤖 Classifier", "📋 Data & about"])
(t_start, t_map, t_ask, t_carrier, t_radio, t_cluster, t_model, t_data) = tabs


# =========================================================== START HERE
with t_start:
    st.subheader("What this dashboard is")
    U.card(
        '<span style="color:var(--ink-2);font-size:0.94rem;line-height:1.6">'
        'Every mobile call and megabyte in India runs through a physical cell '
        'tower. This project takes <b>2,489,618 tower records</b>, works out '
        'features for each region — how many towers per square kilometre, how '
        'modern they are, how many operators compete there — and then lets a '
        'machine-learning model group regions that look alike. The result is a '
        'map of which parts of India are well served and which are not, built '
        'from evidence rather than assertion.</span>')

    st.subheader("A 6-step tour")
    TOUR = [
        ("Choose how detailed a view you want",
         "In the left sidebar, <b>Map granularity</b> switches between 36 states "
         "and 632 districts. Start with <b>State</b> — it is easier to read. "
         "Switch to <b>District</b> once you know what you are looking for; "
         "that view is where the interesting variation lives."),
        ("Decide what the colours should mean",
         "<b>Colour the map by</b> changes the question the map answers. "
         "<b>Coverage cluster</b> shows the groups the model discovered by "
         "itself. <b>Tower density</b> shows one raw measure. They are different "
         "questions — the cluster combines several measures at once, a single "
         "measure does not."),
        ("Read the map, then click it",
         "On the <b>Coverage map</b> tab, darker blue always means stronger "
         "coverage. Grey means there was not enough data to judge, which is "
         "reported honestly rather than hidden. <b>Click any region</b> and the "
         "right-hand panel fills with its numbers plus written findings about it."),
        ("Let the dashboard tell you what it means",
         "Every tab has an <b>Insights</b> section. These sentences are computed "
         "live from whatever you have selected — rankings, comparisons against "
         "the median, gaps worth noticing. Change the filter and they change."),
        ("Ask your own question",
         "The <b>Ask the data</b> tab takes a plain-English question — 'where is "
         "4G weakest?', 'which regions have towers but old technology?' — and "
         "answers it with a ranked chart, a table and an explanation. Use this "
         "if you are not sure where to start."),
        ("Check the modelling, not just the map",
         "<b>Clusters</b> shows what actually defines each group and how well "
         "the model agrees with an independent formula. <b>Classifier</b> lets "
         "you move sliders and watch a trained model predict a coverage tier. "
         "Both exist so the analysis can be questioned, not just admired."),
    ]

    step = ss.tour_step
    U.step(step + 1, TOUR[step][0], TOUR[step][1])

    nav = st.columns([1, 1, 5, 2])
    if nav[0].button("← Back", disabled=step == 0, key="tour_back"):
        ss.tour_step = max(0, step - 1); st.rerun()
    if nav[1].button("Next →", disabled=step == len(TOUR) - 1, key="tour_next"):
        ss.tour_step = min(len(TOUR) - 1, step + 1); st.rerun()
    nav[2].caption(f"Step {step + 1} of {len(TOUR)}")
    if nav[3].button("Show all steps", key="tour_all"):
        ss.tour_step = 0
    st.progress((step + 1) / len(TOUR))

    with st.expander("Show every step at once"):
        for i, (t, b) in enumerate(TOUR, 1):
            U.step(i, t, b)

    st.subheader("What each tab does")
    U.pdataframe(pd.DataFrame([
        ["🗺️ Coverage map", "The main view. Map plus a detail panel for whichever region you click."],
        ["❓ Ask the data", "Pick a plain-English question, get a ranked answer. Best starting point."],
        ["📡 Carriers", "Airtel vs Vi vs Jio vs BSNL — size of estate and how modern each network is."],
        ["📶 Generations", "How much of the network is 2G, 3G, 4G, 5G, nationally and by region."],
        ["🔬 Clusters", "What defines each discovered group, and whether the model can be trusted."],
        ["🤖 Classifier", "A trained model you can experiment with, plus what drives its predictions."],
        ["📋 Data & about", "The full table to sort, filter and download, plus method and limitations."],
    ], columns=["Tab", "What you'll find"]), hide_index=True)

    st.subheader("Glossary — plain English")
    U.pdataframe(pd.DataFrame([
        ["Cell tower", "The physical mast that carries mobile signal. One record here is one tower sector."],
        ["2G / 3G / 4G / 5G", "Generations of mobile technology, oldest to newest. 2G carries calls and texts; 4G carries usable internet."],
        ["Tower density", "Towers per square kilometre. Higher means more infrastructure packed into the area."],
        ["Coverage tier", "A transparent formula labelling regions Underserved / Moderate / Well-served. Used to check the model."],
        ["Cluster", "A group the machine-learning model found on its own, without being told what to look for."],
        ["K-Means", "The algorithm that forms those groups by repeatedly finding group centres."],
        ["DBSCAN", "A second algorithm used as a cross-check. Good at spotting oddities rather than forming groups."],
        ["HHI", "Market concentration. 0.25 means four equal operators competing; 1.0 means a single operator monopoly."],
        ["Operators", "How many of the four majors (Airtel, Vi, Jio, BSNL) have towers in the region."],
        ["Cell range", "Roughly how far a tower reaches, in metres. Bigger usually means rural."],
        ["Samples", "How many crowd-sourced readings back a tower. Low numbers mean a less certain location."],
        ["Crowd-sourced", "The data comes from volunteers' phones, not from the operators. Quiet areas get under-recorded."],
    ], columns=["Term", "What it means"]), hide_index=True)

    st.subheader("Two things to keep in mind")
    U.insight("This data is crowd-sourced, so sparse can mean unmeasured",
              "Tower records come from volunteers running apps, not from "
              "operators. A region with few recorded towers may genuinely have "
              "few towers — or simply few people contributing readings. Treat "
              "low density as a flag to investigate, not proof of neglect.",
              "watch")
    U.insight("This snapshot predates India's 5G rollout",
              "5G is 136 towers out of 2.49 million here — 0.01%. Any "
              "conclusion about 5G drawn from this dataset would be wrong. "
              "Everything else, 2G through 4G, is solid.", "neutral")


# =========================================================== COVERAGE MAP
with t_map:
    st.caption("Darker blue = stronger coverage · grey = not enough data to "
               "judge · **click any region** to inspect it")
    left, right = st.columns([2.0, 1])

    with left:
        if color_by == "Coverage cluster (K-Means)":
            d = df.copy()
            d["_label"] = np.where(d["is_clustered"], d["cluster_name"],
                                   "Insufficient data (<50 towers)")
            names = (d.loc[d["is_clustered"]].sort_values("cluster")
                     ["cluster_name"].unique().tolist())
            cmap = dict(zip(names, cluster_palette(len(names))))
            cmap["Insufficient data (<50 towers)"] = INSUFFICIENT_COLOR
            fig = px.choropleth(
                d, geojson=load_geo(level), locations="region",
                color="_label", color_discrete_map=cmap,
                category_orders={"_label": names + ["Insufficient data (<50 towers)"]},
                hover_name="region",
                hover_data={"_label": True, "tower_count": ":,",
                            "tower_density": ":.3f", "pct_modern": ":.1f",
                            "region": False})
            fig.update_layout(legend_title_text="Discovered cluster")
            show_legend = True
        elif color_by == "Rule-based tier":
            fig = px.choropleth(
                df, geojson=load_geo(level), locations="region",
                color="coverage_tier",
                color_discrete_map=dict(zip(TIERS, V.ORDINAL_3)),
                category_orders={"coverage_tier": TIERS}, hover_name="region",
                hover_data={"tower_count": ":,", "tower_density": ":.3f",
                            "region": False})
            fig.update_layout(legend_title_text="Tier")
            show_legend = True
        else:
            col, label, logscale = {
                "Tower density": ("tower_density", "towers / km²", True),
                "4G + 5G share": ("pct_modern", "% of towers", False),
                "2G share": ("pct_2g", "% of towers", False),
                "Active carriers": ("active_carrier_count", "operators", False),
            }[color_by]
            d = df.copy()
            d["_v"] = np.log10(d[col].clip(lower=1e-4)) if logscale else d[col]
            fig = px.choropleth(
                d, geojson=load_geo(level), locations="region", color="_v",
                color_continuous_scale=V.SEQ_BLUE, hover_name="region",
                hover_data={"_v": False, col: ":,.3f", "region": False})
            fig.update_layout(coloraxis_colorbar=dict(
                title=(f"log₁₀ {label}" if logscale else label),
                outlinewidth=0,
                tickfont=dict(color=V.INK_SECONDARY, size=10.5)))
            show_legend = False

        fig.update_geos(fitbounds="locations", visible=False,
                        projection_type="mercator", bgcolor="rgba(0,0,0,0)")
        fig.update_traces(marker_line_color=V.SURFACE, marker_line_width=0.4)
        fig.update_layout(
            height=650, margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=K.FONT, color=V.INK_SECONDARY, size=12),
            hoverlabel=dict(bgcolor=V.SURFACE, bordercolor=V.BASELINE,
                            font=dict(color=V.INK_PRIMARY, size=12)),
            showlegend=show_legend,
            legend=dict(orientation="v", yanchor="bottom", y=0.01,
                        xanchor="right", x=1.0,
                        bgcolor="rgba(252,252,251,0.86)",
                        bordercolor=V.GRIDLINE, borderwidth=1,
                        font=dict(color=V.INK_PRIMARY, size=11),
                        title_font=dict(color=V.INK_SECONDARY, size=11)))

        event = U.pchart(fig, key=f"map_{level}_{state_filter}",
                         on_select="rerun")
        clicked = None
        try:
            pts = (event.get("selection", {}) or {}).get("points", [])
            if pts:
                p = pts[0]
                clicked = p.get("location") or p.get("hovertext") or p.get("label")
        except Exception:
            clicked = None
        if clicked and clicked in set(df["region"]):
            ss[f"sel_{level}"] = clicked

    with right:
        options = sorted(df["region"].tolist())
        default = ss.get(f"sel_{level}")
        idx = options.index(default) if default in options else 0
        region = st.selectbox("Region — click the map or choose here", options,
                              index=idx, key=f"sel_{level}")
        r = df[df["region"] == region].iloc[0]

        if r["is_clustered"]:
            st.markdown(f"**Model group:** {r['cluster_name']}")
        st.markdown(f"**Rule-based tier:** {r['coverage_tier']}")

        m = st.columns(2)
        m[0].metric("Towers", f"{int(r['tower_count']):,}")
        m[1].metric("Towers / km²", fmt(r["tower_density"], 3))
        m[0].metric("4G + 5G share", f"{fmt(r['pct_modern'], 1)}%")
        m[1].metric("2G share", f"{fmt(r['pct_2g'], 1)}%")
        m[0].metric("Operators", f"{int(r['active_carrier_count'])} of 4")
        m[1].metric("Area km²", f"{r['area_sqkm']:,.0f}")

        gens = ["2G", "3G", "4G", "5G"]
        U.pchart(K.vbar(
            gens, [int(r[f"n_{g.lower()}"]) for g in gens],
            [V.GENERATION_COLORS[g] for g in gens],
            title="Generation mix", height=252,
            secondary_text=[f"{r[f'pct_{g.lower()}']:.1f}%" for g in gens]),
            key=f"gmix_{region}")

        U.pchart(K.vbar(
            C.ACTIVE_CARRIERS,
            [int(r[NCOL[c]]) for c in C.ACTIVE_CARRIERS],
            [V.CARRIER_COLORS[c] for c in C.ACTIVE_CARRIERS],
            title="Carrier mix", height=252),
            key=f"cmix_{region}")

    st.divider()
    st.subheader(f"Insights — {region}")
    U.insight_list(I.region_insights(df, region, level))

    st.subheader(f"Insights — {scope}")
    U.insight_list(I.overview_insights(df, level, scope))


# ============================================================ ASK THE DATA
with t_ask:
    st.subheader("Ask a question in plain English")
    st.caption("Pick what you want to know. The answer is computed from "
               "whatever scope you have selected in the sidebar.")

    q = st.selectbox("What do you want to know?", I.QUESTIONS,
                     label_visibility="collapsed")

    ra = rb = None
    if q == "Compare two regions side by side":
        pool = df[df["is_clustered"]]
        opts = sorted(pool["region"].tolist())
        if len(opts) >= 2:
            cc = st.columns(2)
            ra = cc[0].selectbox("First region", opts, index=0)
            rb = cc[1].selectbox("Second region", opts,
                                 index=min(1, len(opts) - 1))

    ans = I.answer(q, df, level, ra, rb)
    U.insight(ans["headline"], ans.get("detail", ""), ans.get("kind", "neutral"))

    table = ans.get("table")
    if table is not None and len(table):
        vc = ans.get("value_col")
        if vc and vc in table.columns:
            lab = ans.get("label_col", "region")
            t = table.sort_values(vc, ascending=not ans.get("ascending", False))
            U.pchart(K.hbar(
                t[lab].tolist(), t[vc].tolist(),
                color=V.CATEGORICAL[0],
                value_fmt="{:,." + str(ans.get("nd", 2)) + "f}",
                xtitle=ans.get("value_label"),
                median_line=(float(df[df["is_clustered"]][vc].median())
                             if vc in df.columns else None),
                median_label="median of all regions"),
                key=f"ask_chart_{q}")
        st.markdown("**The numbers behind that answer**")
        U.pdataframe(table.reset_index(drop=True), hide_index=True)
        st.download_button("Download this answer as CSV",
                           table.to_csv(index=False).encode("utf-8"),
                           file_name="answer.csv", mime="text/csv",
                           key=f"dl_{q}")

    st.divider()
    st.subheader("Two measures at once")
    st.caption("A ranking answers one question. This answers a better one: "
               "which regions have the towers but not the technology?")
    pool = df[df["is_clustered"]].copy()
    if len(pool) > 4:
        U.pchart(K.quadrant_scatter(
            pool, "tower_density", "pct_modern", "region",
            color_col="coverage_tier",
            color_map=dict(zip(TIERS, V.ORDINAL_3)),
            title="Coverage intensity vs network modernity",
            subtitle="Dotted lines are the medians. Each point is one region.",
            xtitle="towers per km² (log scale)", ytitle="% of towers 4G or 5G",
            quad_labels={
                (0.015, 0.97, "left", "top"): "Sparse but modern",
                (0.985, 0.97, "right", "top"): "Well served",
                (0.015, 0.04, "left", "bottom"): "Underserved",
                (0.985, 0.04, "right", "bottom"): "Towers exist, tech is dated",
            }), key="ask_quad")


# ================================================================ CARRIERS
with t_carrier:
    st.subheader(f"Carrier comparison — {scope}")
    pool = df[df["is_clustered"]]
    totals = pd.Series({c: int(pool[NCOL[c]].sum()) for c in C.ACTIVE_CARRIERS})
    totals = totals.sort_values()

    tl = load_tower_level()
    if level == "District" and state_filter != "All India":
        tl = tl[tl["district"].isin(set(pool["region"]))]
    elif level == "State" and state_filter != "All India":
        tl = tl[tl["st_nm"] == state_filter]
    tl = tl[tl["carrier"].isin(C.ACTIVE_CARRIERS)]
    mix = (pd.crosstab(tl["carrier"], tl["generation"], normalize="index") * 100
           ).reindex(columns=["2G", "3G", "4G", "5G"], fill_value=0.0)
    mix = mix.reindex(totals.index).dropna(how="all")

    cc = st.columns(2)
    with cc[0]:
        U.pchart(K.hbar(
            totals.index.tolist(), totals.values.tolist(),
            colors=[V.CARRIER_COLORS[c] for c in totals.index],
            title="Size of each recorded estate",
            subtitle="Total towers attributed to each operator",
            value_fmt="{:,.0f}", xtitle="towers", height=330),
            key="car_totals")
    with cc[1]:
        if not mix.empty:
            U.pchart(K.stacked_hbar(
                mix.index.tolist(),
                {g: mix[g].tolist() for g in ["2G", "3G", "4G", "5G"]},
                V.GENERATION_COLORS,
                title="How modern each network is",
                subtitle="Share of that operator's own towers by generation",
                xtitle="% of that carrier's towers", height=330,
                label_series={"2G", "4G"}), key="car_mix")

    st.markdown("**Carrier share by region** — top 20 by tower count")
    top = pool.nlargest(20, "tower_count")
    U.pchart(K.stacked_hbar(
        top.sort_values("tower_count")["region"].tolist(),
        {c: top.sort_values("tower_count")[PCOL[c]].tolist()
         for c in C.ACTIVE_CARRIERS},
        V.CARRIER_COLORS, xtitle="% of that region's towers",
        height=max(340, 30 * len(top) + 60), label_series=None,
        min_label_width=12), key="car_region")

    st.divider()
    st.subheader("Insights")
    U.insight_list(I.carrier_insights(df, mix if not mix.empty else None))


# ============================================================= GENERATIONS
with t_radio:
    st.subheader(f"Network generations — {scope}")
    pool = df[df["is_clustered"]]
    tot = pd.Series({g.upper(): int(pool[f"n_{g}"].sum())
                     for g in ["2g", "3g", "4g", "5g"]})
    pct = tot / max(tot.sum(), 1) * 100

    cols = st.columns(4)
    for col, g in zip(cols, tot.index):
        col.metric(g, f"{tot[g]:,}", f"{pct[g]:.2f}% of towers",
                   delta_color="off")

    cc = st.columns([1, 1])
    with cc[0]:
        U.pchart(K.vbar(
            tot.index.tolist(), tot.values.tolist(),
            [V.GENERATION_COLORS[g] for g in tot.index],
            title="Towers by generation",
            subtitle="Absolute counts across the current scope",
            height=340, secondary_text=[f"{v:,}<br>{p:.1f}%"
                                        for v, p in zip(tot.values, pct.values)]),
            key="gen_nat")
    with cc[1]:
        U.pchart(K.quadrant_scatter(
            pool, "tower_density", "pct_2g", "region",
            title="Does density mean modernity?",
            subtitle="Dotted lines are medians. Bottom-right is the good corner.",
            xtitle="towers per km² (log scale)",
            ytitle="% of towers still on 2G", height=340),
            key="gen_quad")

    st.markdown("**Generation mix by region** — top 25 by tower count, "
                "ordered by 4G share")
    top = pool.nlargest(25, "tower_count").sort_values("pct_4g")
    U.pchart(K.stacked_hbar(
        top["region"].tolist(),
        {g: top[f"pct_{g.lower()}"].tolist() for g in ["2G", "3G", "4G", "5G"]},
        V.GENERATION_COLORS, xtitle="% of that region's towers",
        height=max(400, 30 * len(top) + 70), label_series={"2G", "4G"}),
        key="gen_region")

    st.divider()
    st.subheader("Insights")
    U.insight_list(I.generation_insights(df, level))


# ================================================================ CLUSTERS
with t_cluster:
    st.subheader(f"What the model discovered — {scope}")
    pool = df[df["is_clustered"]]

    if pool.empty:
        st.info("No clustered regions in this selection.")
    else:
        st.caption("These groups were formed without being told what to look "
                   "for. The panels below show what distinguishes them and "
                   "whether they can be trusted.")
        sizes = pool["cluster_name"].value_counts()
        U.pchart(K.vbar(
            sizes.index.tolist(), sizes.values.tolist(),
            cluster_palette(len(sizes))[:len(sizes)],
            title="How many regions fall in each group", height=320,
            ytitle="regions"), key="cl_sizes")

        mdl = load_models().get(level.lower())
        feats = mdl["features"] if mdl else [
            "log_tower_density", "pct_2g", "pct_4g", "active_carrier_count",
            "carrier_hhi", "range_mean", "log_sample_mean"]

        st.markdown("**What defines each group** — distance from the average, "
                    "in standard deviations. Red is above average, blue below.")
        z = pool.groupby("cluster_name")[feats].mean()
        z = (z - pool[feats].mean()) / pool[feats].std()
        U.pchart(K.heatmap(z.values, [C.label_of(f) for f in feats],
                           z.index.tolist(), zmid=0,
                           cbar_title="σ", text_fmt="{:+.1f}",
                           height=140 + 62 * len(z)), key="cl_heat")

        st.markdown("**Average values per group** (raw units, not standardised)")
        prof = pool.groupby("cluster_name")[
            ["tower_count", "tower_density", "pct_2g", "pct_3g", "pct_4g",
             "pct_modern", "active_carrier_count", "carrier_hhi",
             "range_mean", "sample_mean"]].mean().round(3)
        U.pdataframe(prof)

        st.markdown("**Does the model agree with an independent formula?**")
        ct = pd.crosstab(pool["cluster_name"], pool["coverage_tier"]
                         ).reindex(columns=TIERS, fill_value=0)
        U.pchart(K.heatmap(
            ct.values, TIERS, ct.index.tolist(), text_fmt="{:.0f}",
            colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
            cbar_title="regions", height=130 + 62 * len(ct)), key="cl_cross")
        st.caption("The clustering never saw the rule-based tier. Agreement "
                   "between them is the main evidence the groups are real.")

        if "dbscan" in pool.columns:
            out = pool[pool["dbscan"] == -1]["region"].tolist()
            with st.expander(f"Regions DBSCAN flags as unusual ({len(out)})"):
                st.write(", ".join(out) if out else "None flagged.")
                st.caption("DBSCAN finds one continuous density manifold rather "
                           "than separate coverage regimes — consistent with "
                           "coverage being a gradient. Its value here is "
                           "flagging the extremes of that gradient.")

        st.divider()
        st.subheader("Insights")
        U.insight_list(I.cluster_insights(df, level))


# ============================================================== CLASSIFIER
with t_model:
    st.subheader("Coverage-tier classifier")
    rf = load_models().get("rf")

    if rf is None:
        st.info("Run `python3 src/09_supervised.py` to train the classifier.")
    else:
        U.insight(
            "Why there are two models, and why only one of them counts",
            "The rule-based tier is calculated FROM tower density, 4G share and "
            "operator count. A model trained on those same three inputs scores "
            "93% — but that is circular: it is just re-deriving its own label's "
            "arithmetic. Model B removes every input the rule used and predicts "
            "the tier from market structure and measurement traits alone. It "
            "reaches 67% against a 35% baseline, which is a real result.",
            "neutral")

        feats, model = rf["features_b"], rf["model_b"]
        base = pd.read_csv(C.DISTRICT_CLUSTERED)
        base = base[base["cluster"] >= 0]

        pick = st.selectbox("Load values from a real district",
                            ["(manual entry)"] + sorted(base["region"].tolist()))
        seed = (base[base["region"] == pick].iloc[0] if pick != "(manual entry)"
                else base[feats].median())

        with st.expander("Adjust the inputs yourself", expanded=False):
            st.caption("Move a slider and watch the prediction change.")
            vals, cc = {}, st.columns(2)
            for i, f in enumerate(feats):
                lo, hi = float(base[f].min()), float(base[f].max())
                dflt = float(np.clip(float(seed[f]), lo, hi))
                vals[f] = cc[i % 2].slider(C.label_of(f), lo, hi, dflt,
                                           step=(hi - lo) / 200 or 0.01,
                                           key=f"rf_{f}")
        if not vals:
            vals = {f: float(seed[f]) for f in feats}

        Xq = pd.DataFrame([vals])[feats]
        pred = model.predict(Xq)[0]
        proba = model.predict_proba(Xq)[0]

        cc = st.columns([1, 1])
        cc[0].metric("Predicted tier", pred)
        if pick != "(manual entry)":
            actual = seed["coverage_tier"]
            cc[1].metric("Actual rule-based tier", actual,
                         "match" if actual == pred else "differs",
                         delta_color="normal" if actual == pred else "inverse")

        pf = (pd.DataFrame({"Tier": model.classes_, "p": proba})
              .set_index("Tier").reindex(TIERS).fillna(0).reset_index())
        U.pchart(K.vbar(
            pf["Tier"].tolist(), (pf["p"] * 100).tolist(), V.ORDINAL_3,
            title="How confident the model is",
            subtitle="Probability assigned to each tier", height=310,
            value_fmt="{:,.1f}", ytitle="probability (%)",
            secondary_text=[f"{v:.1%}" for v in pf["p"]]), key="rf_proba")

        summ = C.REPORTS / "07_supervised_summary.json"
        if summ.exists():
            s = json.loads(summ.read_text())
            a, b, c3 = st.columns(3)
            a.metric("Model B accuracy",
                     f"{s['model_b_leakage_free']['accuracy']:.1%}")
            b.metric("Majority baseline", f"{s['majority_baseline']:.1%}")
            c3.metric("Model A (circular)",
                      f"{s['model_a_circular']['accuracy']:.1%}",
                      help="Trained on the rule's own inputs — inflated by "
                           "construction, reported only as a wiring check.")

        st.divider()
        st.markdown("**What the model actually relies on**")
        for name, cap in [
            ("fig23_feature_importance_modelB.png",
             "Left: how often the trees split on a feature. Right: how much "
             "accuracy drops when that feature is shuffled — the more honest "
             "measure, computed on held-out data."),
            ("fig24_shap_importance_modelB.png",
             "SHAP attributes each prediction to its inputs and averages the "
             "magnitudes. Taller bars drive more of the decision."),
            ("fig25_shap_beeswarm_wellserved.png",
             "Each dot is one district. Position shows whether that feature "
             "pushed the prediction toward 'Well-served'; colour shows whether "
             "the feature value was high or low."),
        ]:
            p = C.FIGURES / name
            if p.exists():
                U.pimage(p)
                st.caption(cap)


# ============================================================ DATA & ABOUT
with t_data:
    st.subheader(f"Full region table — {scope}")
    show = [c for c in ["region", "cluster_name", "coverage_tier", "tower_count",
                        "tower_density", "area_sqkm", "pct_2g", "pct_3g",
                        "pct_4g", "pct_modern", "active_carrier_count",
                        "carrier_hhi", "range_mean", "sample_mean"]
            if c in df.columns]
    tbl = (df[show].sort_values("tower_count", ascending=False)
           .reset_index(drop=True)
           .rename(columns={c: C.label_of(c) for c in show}))
    tbl = tbl.rename(columns={"region": "Region",
                              "cluster_name": "Model group",
                              "coverage_tier": "Coverage tier"})
    U.pdataframe(tbl, hide_index=True, height=520)
    st.download_button("Download this table as CSV",
                       df[show].to_csv(index=False).encode("utf-8"),
                       file_name=f"coverage_{level.lower()}.csv",
                       mime="text/csv")

    st.divider()
    st.subheader("How this was built")
    U.card(
        '<span style="color:var(--ink-2);font-size:0.92rem;line-height:1.65">'
        '2,489,618 tower records were merged from two raw files, decoded to '
        'carriers via MCC/MNC codes, and spatially joined to 36 states and 641 '
        'Census-2011 districts (100.0% match rate). Each region got features for '
        'density, generation mix, operator competition and measurement quality. '
        'K-Means grouped the regions, with k chosen by elbow and silhouette '
        'under a minimum-cluster-size constraint; DBSCAN ran as an independent '
        'cross-check. A Random Forest then tested whether the resulting tiers '
        'are predictable from signals the labelling rule never used.'
        '</span>')

    st.subheader("Limitations, stated plainly")
    for t, b in [
        ("The data is crowd-sourced, not operator-reported",
         "Density partly reflects where volunteers run measurement apps. Sparse "
         "regions may be under-measured rather than under-served."),
        ("Every record is an estimate",
         "All 2.49M rows carry changeable=1, meaning crowd-derived. None are "
         "operator-verified, so the planned 'verified coverage' feature could "
         "not be built at all."),
        ("Tiers are relative, not official",
         "'Underserved' means underserved compared to the rest of India in this "
         "dataset — not a regulatory or carrier assessment."),
        ("The two map levels are not a strict hierarchy",
         "State boundaries are current (Telangana separate); district "
         "boundaries are Census 2011 (Telangana still inside Andhra Pradesh). "
         "Districts do not roll up exactly into states."),
        ("This is a batch snapshot, not a live monitor",
         "The data predates India's 5G rollout and is not refreshed."),
    ]:
        U.insight(t, b, "watch")

    st.subheader("Data sources")
    U.pdataframe(pd.DataFrame([
        ["404.csv / 405.csv", "Kaggle 'Mobile Network Coverage India' (from OpenCelliD)"],
        ["india_mcc_mnc_operator_circle.csv", "Same Kaggle dataset — carrier lookup"],
        ["states_india.geojson", "Same Kaggle dataset — state boundaries"],
        ["districts_india.geojson", "DataMeet maps, Census 2011 districts (CC-BY / CC0)"],
    ], columns=["File", "Source"]), hide_index=True)

    with st.expander("Environment diagnostics"):
        st.json(U.diagnostics())
        st.caption("The dashboard detects which argument names your Streamlit "
                   "version supports, so it stays quiet instead of emitting a "
                   "deprecation warning for every chart.")
