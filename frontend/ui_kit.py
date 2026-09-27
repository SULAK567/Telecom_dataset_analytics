# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
UI helpers for the dashboard: version-safe Streamlit wrappers, theming,
and small presentation components.

WHY THE WRAPPERS EXIST
----------------------
Streamlit renamed the "fill the container" argument from
`use_container_width=True` to `width="stretch"`. Both spellings exist in the
wild, and passing the wrong one does not raise — it falls through to
`**kwargs`, which makes Streamlit emit:

    "The keyword arguments have been deprecated and will be removed in a
     future release. Use config instead to specify Plotly configuration
     options."

once per chart, flooding the page. These wrappers inspect the installed
signature once and pass whichever spelling that version actually supports,
so the app is silent and correct on any Streamlit release.
"""
import inspect
from functools import lru_cache

import streamlit as st

# Plotly toolbar config: keep the chrome minimal but leave the image export.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d",
                               "hoverClosestGeo", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


@lru_cache(maxsize=None)
def _width_kwarg(func_name):
    """Return the kwarg dict this Streamlit version uses for full width."""
    func = getattr(st, func_name, None)
    if func is None:
        return {}
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def pchart(fig, key=None, on_select=None, config=None):
    """
    st.plotly_chart with the correct width kwarg, an explicit config, and
    theme=None.

    theme defaults to "streamlit", which RE-STYLES the figure using
    Streamlit's own chart theme and overrides explicitly set font colours.
    That is what turns carefully chosen dark ink into washed-out pale grey
    that is hard to read. theme=None hands Plotly our styling untouched.
    """
    kwargs = dict(_width_kwarg("plotly_chart"))
    kwargs["config"] = {**PLOTLY_CONFIG, **(config or {})}
    try:
        if "theme" in inspect.signature(st.plotly_chart).parameters:
            kwargs["theme"] = None
    except (TypeError, ValueError):
        pass
    if key is not None:
        kwargs["key"] = key
    if on_select is not None:
        kwargs["on_select"] = on_select
    return st.plotly_chart(fig, **kwargs)


def pdataframe(df, **extra):
    kwargs = dict(_width_kwarg("dataframe"))
    kwargs.update(extra)
    return st.dataframe(df, **kwargs)


def pimage(path, **extra):
    func_params = {}
    try:
        func_params = inspect.signature(st.image).parameters
    except (TypeError, ValueError):
        pass
    kwargs = {}
    if "width" in func_params:
        kwargs["width"] = "stretch"
    elif "use_container_width" in func_params:
        kwargs["use_container_width"] = True
    kwargs.update(extra)
    return st.image(str(path), **kwargs)


def diagnostics():
    """What the wrappers resolved to — surfaced in the About tab."""
    return {
        "streamlit": st.__version__,
        "plotly_chart width kwarg": _width_kwarg("plotly_chart") or "(none)",
        "dataframe width kwarg": _width_kwarg("dataframe") or "(none)",
    }


# ---------------------------------------------------------------------------
# Theming
# ---------------------------------------------------------------------------
# Two layered SVG patterns evoke the subject without competing with the data:
#   * a hexagonal lattice  -> the classic cellular coverage honeycomb
#   * concentric arcs      -> signal broadcasting from a mast
# Both sit at very low opacity and are fixed to the viewport, so they read as
# texture rather than as marks that could be mistaken for data.
_HEX = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='96' viewBox='0 0 56 96'%3E"
    "%3Cg fill='none' stroke='%232a78d6' stroke-opacity='0.055' stroke-width='1.1'%3E"
    "%3Cpath d='M28 0 L56 16 L56 48 L28 64 L0 48 L0 16 Z'/%3E"
    "%3Cpath d='M28 32 L56 48 L56 80 L28 96 L0 80 L0 48 Z'/%3E"
    "%3C/g%3E%3C/svg%3E"
)
_WAVES = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='640' viewBox='0 0 640 640'%3E"
    "%3Cg fill='none' stroke='%232a78d6' stroke-opacity='0.07'%3E"
    "%3Ccircle cx='640' cy='640' r='140' stroke-width='1.6'/%3E"
    "%3Ccircle cx='640' cy='640' r='240' stroke-width='1.5'/%3E"
    "%3Ccircle cx='640' cy='640' r='340' stroke-width='1.3'/%3E"
    "%3Ccircle cx='640' cy='640' r='440' stroke-width='1.1'/%3E"
    "%3Ccircle cx='640' cy='640' r='540' stroke-width='0.9'/%3E"
    "%3C/g%3E%3C/svg%3E"
)

THEME_CSS = f"""
<style>
  :root {{
    --ink-1: #0b0b0b;
    --ink-2: #52514e;
    --ink-3: #898781;
    --surface: #fcfcfb;
    --rule: #e1e0d9;
    --brand: #2a78d6;
    --brand-deep: #0d366b;
  }}

  /* Layered telecom texture: honeycomb lattice + broadcast arcs. */
  [data-testid="stAppViewContainer"] {{
    background-color: var(--surface);
    background-image: url("{_WAVES}"), url("{_HEX}");
    background-position: right -80px bottom -80px, top left;
    background-repeat: no-repeat, repeat;
    background-attachment: fixed, fixed;
    background-size: 720px 720px, 56px 96px;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}

  /* Readable, deliberate typography. */
  h1, h2, h3, h4 {{ color: var(--ink-1); letter-spacing: -0.012em; }}
  h1 {{ font-weight: 680; }}
  p, li, label, .stMarkdown {{ color: var(--ink-1); }}

  /* Cards: give panels a surface so the background never sits behind text. */
  .tc-card {{
    background: rgba(252,252,251,0.93);
    border: 1px solid var(--rule);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(11,11,11,0.035);
  }}

  /* Insight callouts. The left rule carries severity; an icon+label pairing
     in the title keeps meaning off colour alone. */
  .tc-insight {{
    background: rgba(252,252,251,0.95);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--brand);
    border-radius: 10px;
    padding: 13px 16px;
    margin-bottom: 10px;
  }}
  .tc-insight.good     {{ border-left-color: #0ca30c; }}
  .tc-insight.watch    {{ border-left-color: #fab219; }}
  .tc-insight.concern  {{ border-left-color: #d03b3b; }}
  .tc-insight.neutral  {{ border-left-color: var(--brand); }}
  .tc-insight .tc-t {{
    font-weight: 640; font-size: 0.95rem; color: var(--ink-1);
    margin-bottom: 3px; display: block;
  }}
  .tc-insight .tc-b {{ font-size: 0.89rem; color: var(--ink-2); line-height: 1.5; }}
  .tc-insight .tc-n {{ font-variant-numeric: tabular-nums; font-weight: 640;
                        color: var(--brand-deep); }}

  /* Step chips for the guided tour. */
  .tc-step {{
    display: inline-block; background: var(--brand); color: #fff;
    width: 22px; height: 22px; line-height: 22px; text-align: center;
    border-radius: 50%; font-size: 0.78rem; font-weight: 700;
    margin-right: 8px;
  }}

  .tc-kicker {{
    text-transform: uppercase; letter-spacing: 0.085em; font-size: 0.7rem;
    color: var(--ink-3); font-weight: 680; margin-bottom: 2px;
  }}

  [data-testid="stMetricValue"] {{ color: var(--ink-1); font-weight: 660; }}
  [data-testid="stMetricLabel"] {{ color: var(--ink-2); }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--rule); }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 560; color: var(--ink-2); }}
  .stTabs [aria-selected="true"] {{ color: var(--brand-deep); }}

  [data-testid="stSidebar"] {{
    background: rgba(246,246,243,0.97);
    border-right: 1px solid var(--rule);
  }}
</style>
"""


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def insight(title, body, kind="neutral"):
    """Render one computed finding as a callout."""
    st.markdown(
        f'<div class="tc-insight {kind}"><span class="tc-t">{title}</span>'
        f'<span class="tc-b">{body}</span></div>',
        unsafe_allow_html=True)


def insight_list(findings, empty="No notable patterns for this selection."):
    if not findings:
        st.caption(empty)
        return
    for f in findings:
        insight(f["title"], f["body"], f.get("kind", "neutral"))


def kicker(text):
    st.markdown(f'<div class="tc-kicker">{text}</div>', unsafe_allow_html=True)


def card(body_md):
    st.markdown(f'<div class="tc-card">{body_md}</div>', unsafe_allow_html=True)


def step(n, title, body):
    st.markdown(
        f'<div class="tc-card"><span class="tc-step">{n}</span>'
        f'<b>{title}</b><div class="tc-b" style="margin-top:7px;'
        f'color:var(--ink-2);font-size:0.9rem;line-height:1.55">{body}</div></div>',
        unsafe_allow_html=True)
