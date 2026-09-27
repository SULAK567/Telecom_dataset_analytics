# Project Status — Telecom Coverage Clustering

Build complete: **Phases 1–8 (all technical work).** Report writing is the only
remaining task and is deliberately out of scope here.

Full pipeline verified end to end: `python3 src/run_all.py` → 10/10 steps pass
in ~50 s (excluding the slow raw rebuild, step 01).

---

## Part A — Build Roadmap

### Phase 1 — Data Preparation — COMPLETE
- [x] Merge raw tower data (404.csv + 405.csv → 2,494,859 rows)
- [x] Decode carriers via MCC-MNC lookup
- [x] Data quality check — `.info()`, nulls, dtypes, constant-column audit
- [x] Clean invalid/missing lat-long — **0 found**, all inside India's bbox
- [x] Deduplicate cell IDs — **0 duplicates found**
- [x] Verified AverageSignal unusable (100% zero) and corrected the schema
      misalignment (see Findings)
- Final cleaned table: **2,489,618 towers** (5,241 dropped: no state polygon)

### Phase 2 — Regional Enrichment (State) — COMPLETE
- [x] Load `states_india.geojson`
- [x] Spatial join tower → state
- [x] Aggregate to **36 states**: tower count, radio mix, carrier count,
      range/sample means, HHI, area, density
- [ ] ~~Changeable=0 ratio~~ — **not constructible**, see Findings

### Phase 3 — Regional Enrichment (District) — COMPLETE
- [x] Source district boundaries (DataMeet / Census 2011, 641 districts)
- [x] Spatial join tower → district — **100.000% match rate**, chunked
- [x] Aggregate to **632 districts** with the identical feature set
      (9 districts have zero towers)

### Phase 4 — Feature Engineering — COMPLETE
- [x] Finalise the clustering feature set (7 features, no |r| > 0.8)
- [x] StandardScaler, fitted separately per granularity
- [x] Rule-based coverage tier (Underserved / Moderate / Well-served)
- [x] Excluded `range_median` — zero variance (always 1000)

### Phase 5 — Clustering — COMPLETE
- [x] K-Means on state data — k chosen by elbow + silhouette → **k=2**
- [x] K-Means on district data → **k=3**
- [x] DBSCAN on both, eps from the k-distance knee
- [x] Interpret and label every cluster from its own profile
- [x] Cross-check against rule-based tiers — **ARI 0.45 (state), 0.44 (district)**

### Phase 6 — Supervised Extension — COMPLETE
- [x] Random Forest predicting coverage tier
- [x] Accuracy / precision / recall / F1 / confusion matrix / 5-fold CV
- [x] Permutation importance + SHAP explanations
- [x] Two models trained to eliminate label leakage (see Findings)

### Phase 7 — Dashboard — COMPLETE
- [x] Streamlit app (`src/dashboard.py`)
- [x] State-level map coloured by cluster
- [x] District-level map + toggle between the two
- [x] Click-to-inspect region detail panel
- [x] Carrier comparison charts
- [x] Radio-generation distribution charts
- [x] Cluster explanation panel
- [x] Classifier prediction interface + importance panels
- [x] Verified: 0 exceptions across 8 tested state/granularity/colouring paths;
      server boots and serves HTTP 200

### Phase 8b — Dashboard rework (review feedback) — COMPLETE
- [x] **Insights engine** (`insights.py`) — every tab states what the data
      means, computed live from the current selection
- [x] **"Ask the data" tab** — 9 plain-English questions, each answered with a
      ranked chart, table and CSV download
- [x] **Label contrast fixed** — white-on-light-blue measured 2.11:1
      (unreadable); label colour is now chosen from the fill's own luminance
- [x] **Washed-out chart text fixed** — Streamlit's default `theme="streamlit"`
      was overriding explicit font colours; now `theme=None`
- [x] **Clipped value labels fixed** — outside-labelled axes get explicit headroom
- [x] **Charts upgraded** — median reference lines, quadrant scatters with
      labelled quadrants, adaptive in-bar labels
- [x] **Deprecation-warning spam removed** — version-safe wrappers detect which
      argument spelling the installed Streamlit supports
- [x] **Beginner onboarding** — "Start here" tab with a 6-step guided tour,
      tab guide and plain-English glossary
- [x] **Telecom-themed styling** — honeycomb lattice + broadcast-arc background,
      card and callout components
- [x] **Clipped labels fixed globally** — fixed pixel margins were cropping
      y-axis category names ("Jio/Reliance" rendered as "e", "BSNL" as "L")
      and letting axis titles collide with tick labels. Every axis now uses
      `automargin=True`, so Plotly measures the rendered text and grows the
      margin to fit; `title_standoff` separates titles from ticks; stacked-bar
      and scatter legends moved to the top-right, clear of the x-axis title;
      the median reference note moved into the axis title, the only position
      that never overlaps a bar or the subtitle
- [x] **Friendly feature names** — `config.FEATURE_LABELS` maps modelling
      columns to readable names ("log_sample_mean" → "Measurement density
      (log)") across the heatmap, classifier sliders and region table
- [x] **Report figures fixed too** — `fig07` and `fig11/12` had a colorbar
      label running off the canvas and raw snake_case axis names
- [x] Re-verified: **23 scenarios, 0 exceptions, 0 deprecation warnings**;
      every chart type rendered to PNG and visually inspected for clipping

---

## Part B — Report (NOT STARTED, by request)

| Section | Status | Material already available |
|---|---|---|
| 1.1 Problem Definition | pending | README "Key findings" |
| 1.2 Requirement Specs | pending | this checklist + README |
| 1.3 Tools & Technology | pending | `requirements.txt` |
| 2.1 Data Flow Diagrams | pending | pipeline table in README |
| 2.2 Use Case Diagram | pending | dashboard tab structure |
| 2.3 Class Diagram | pending | `src/` module layout |
| 2.4 Sequence Diagram | pending | dashboard interaction flow |
| 3. Data Dictionary | pending | `outputs/reports/02..04` column lists |
| 4. Implementation | pending | all scripts + 25 figures |
| 5. Conclusion | pending | README findings + limitations |
| 6. References | pending | README data sources table |

---

## Findings that changed the plan

1. **Kaggle headers are misaligned against OpenCelliD.** `changeable_0` is
   really `unit`; `changeable_1` is the real `changeable` flag. Verified on all
   2.49M rows: `changeable == 1` for 100%, `averageSignal == 0` for 100%.
   → The planned "% verified by telecom firm" feature **cannot be built**.

2. **Silhouette alone selects a degenerate clustering.** Unconstrained, the
   state-level optimum isolates Lakshadweep (21 towers) alone — silhouette
   0.75, ARI 0.00. A minimum cluster-size constraint plus a 50-tower
   data-sufficiency gate was added.

3. **The supervised phase needed two models.** The rule-based tier is computed
   from three of the features, so Model A (93%) is circular by construction.
   Model B removes every rule input and reaches **66.7% vs a 35.3% baseline**.

4. **DBSCAN finds one cluster, not several.** Coverage is a gradient, not
   density-separated groups. DBSCAN's value here is flagging the 45 extreme
   districts, not partitioning.

---

## Output inventory

| Location | Contents |
|---|---|
| `src/` (14 files) | full pipeline + dashboard + shared modules |
| `data_processed/` (13 files) | cleaned table, aggregates, features, clustered outputs, boundaries |
| `outputs/figures/` (25 files) | fig01–fig25 |
| `outputs/models/` (3 files) | cluster models (state, district) + Random Forest |
| `outputs/reports/` (9 files) | a written report per pipeline stage |
| `README.md` | overview, findings, methodology defence, limitations |
| `requirements.txt` | pinned dependencies |

Note: `data_processed/_to_delete/` holds one stray 14-byte scratch file that
could not be removed automatically — safe to delete manually.
