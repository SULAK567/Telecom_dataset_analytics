"""
Reusable Plotly chart builders with the contrast and layout defects fixed.

Three defects this module exists to prevent:

1. WASHED-OUT TEXT — Streamlit's default chart theme overrides explicitly set
   font colours. Handled in ui_kit.pchart via theme=None; here we also set ink
   colours explicitly on every layout so nothing relies on a default.

2. INVISIBLE IN-BAR LABELS — white text on the light end of a colour ramp
   measures ~2:1 contrast. Every in-bar label colour is chosen from the fill's
   own luminance (viz_style.label_on), so labels stay readable on any step.

3. CLIPPED VALUE LABELS — `textposition="outside"` writes past the plot area
   and Plotly crops it. Every outside-labelled axis gets explicit headroom.
"""

# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

import numpy as np
import plotly.graph_objects as go

import viz_style as V

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
HEADROOM = 1.26          # multiplier giving outside labels room to render


def _base(fig, height, title=None, subtitle=None, top=None, legend=None):
    """
    Base layout.

    MARGINS ARE MINIMUMS, NOT FIXED VALUES. A hard-coded left margin crops
    long category names — "Jio/Reliance" renders as "e", "BSNL" as "L" — and a
    tight bottom margin lets the axis title collide with the tick labels.
    Every axis therefore sets automargin=True (see _axes), which makes Plotly
    grow these margins to fit whatever text is actually there. The values
    below are floors that automargin expands from.
    """
    pad_top = top if top is not None else (78 if title else 16)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=30, t=pad_top, b=20, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=V.INK_SECONDARY),
        hoverlabel=dict(bgcolor=V.SURFACE, bordercolor=V.BASELINE,
                        font=dict(family=FONT, size=12, color=V.INK_PRIMARY)),
        showlegend=legend if legend is not None else False,
    )
    if title:
        fig.add_annotation(
            text=f"<b>{title}</b>", xref="paper", yref="paper", x=0, y=1.0,
            xanchor="left", yanchor="bottom", showarrow=False,
            font=dict(family=FONT, size=15, color=V.INK_PRIMARY),
            yshift=30 if subtitle else 14)
        if subtitle:
            fig.add_annotation(
                text=subtitle, xref="paper", yref="paper", x=0, y=1.0,
                xanchor="left", yanchor="bottom", showarrow=False,
                font=dict(family=FONT, size=11.5, color=V.INK_SECONDARY), yshift=10)
    return fig


def _axes(fig, xgrid=True, ygrid=False):
    """
    automargin=True is the fix for clipped labels: Plotly measures the rendered
    tick text and axis title, then expands the margin to fit. title_standoff
    keeps the axis title clear of the tick labels rather than overlapping them.
    """
    fig.update_xaxes(gridcolor=V.GRIDLINE, zeroline=False,
                     showgrid=xgrid, linecolor=V.BASELINE,
                     automargin=True, title_standoff=14,
                     tickfont=dict(color=V.INK_SECONDARY, size=11))
    fig.update_yaxes(gridcolor=V.GRIDLINE, zeroline=False,
                     showgrid=ygrid, linecolor=V.BASELINE,
                     automargin=True, title_standoff=14,
                     tickfont=dict(color=V.INK_SECONDARY, size=11))
    return fig


def hbar(labels, values, color=None, colors=None, title=None, subtitle=None,
         value_fmt="{:,.0f}", xtitle=None, height=None, median_line=None,
         median_label="median", hovertemplate=None):
    """
    Horizontal bar chart with outside value labels that cannot clip and
    an optional median reference line.
    """
    labels, values = list(labels), list(values)
    fills = colors or [color or V.CATEGORICAL[0]] * len(labels)
    h = height or max(240, 34 * len(labels) + (80 if title else 30))

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker=dict(color=fills),
        text=[value_fmt.format(v) for v in values],
        textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT, size=11.5, color=V.INK_PRIMARY),
        hovertemplate=hovertemplate or "%{y}<br>%{x:,.3f}<extra></extra>"))

    vmax = max(values) if len(values) else 1
    vmin = min(min(values), 0) if len(values) else 0

    # The median reference is documented in the AXIS TITLE rather than as a
    # floating annotation beside the line. Any text placed next to the line
    # collides with something: above the plot it hits the subtitle, inside it
    # covers the nearest bar's value label. The axis title is the one place
    # that is always free, and it reads as a caption rather than clutter.
    x_title = xtitle
    if median_line is not None and np.isfinite(median_line):
        fig.add_vline(x=median_line, line=dict(color=V.INK_MUTED, width=1.4,
                                               dash="dot"))
        note = f"dotted line = {median_label} ({median_line:,.2f})"
        x_title = f"{xtitle}  ·  {note}" if xtitle else note

    fig.update_xaxes(range=[vmin, vmax * HEADROOM if vmax > 0 else 1],
                     title_text=x_title,
                     title_font=dict(size=11.5, color=V.INK_SECONDARY))
    _base(fig, h, title, subtitle)
    return _axes(fig, xgrid=True, ygrid=False)


def stacked_hbar(index, series, colors, title=None, subtitle=None,
                 xtitle="% of towers", height=None, label_series=None,
                 min_label_width=7):
    """
    Stacked horizontal bars with per-segment labels whose colour is chosen
    from the segment fill's luminance, so no label is ever invisible.

    `label_series` limits labels to the series that carries the story;
    segments narrower than `min_label_width` are left unlabelled rather than
    rendering text that overflows its own segment.
    """
    index = list(index)
    h = height or max(250, 36 * len(index) + (86 if title else 34))
    fig = go.Figure()

    for name, vals in series.items():
        vals = list(vals)
        fill = colors[name]
        show = (label_series is None) or (name in label_series)
        text, tcol = [], V.label_on(fill)
        for v in vals:
            text.append(f"{v:.0f}%" if (show and v >= min_label_width) else "")
        fig.add_bar(
            y=index, x=vals, orientation="h", name=name,
            marker=dict(color=fill, line=dict(color=V.SURFACE, width=2)),
            text=text, textposition="inside", insidetextanchor="middle",
            textfont=dict(family=FONT, size=11, color=tcol),
            hovertemplate=f"%{{y}}<br>{name}: %{{x:.1f}}%<extra></extra>")

    # Legend sits top-right, on the same band as the left-aligned title.
    # Below the plot it collided with the x-axis title once automargin gave
    # that title its proper room.
    fig.update_layout(barmode="stack",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="right", x=1.0, traceorder="normal",
                                  font=dict(color=V.INK_SECONDARY, size=11.5)))
    fig.update_xaxes(range=[0, 100], title_text=xtitle,
                     title_font=dict(size=11.5, color=V.INK_SECONDARY))
    _base(fig, h, title, subtitle, legend=True)
    return _axes(fig, xgrid=True, ygrid=False)


def quadrant_scatter(df, x, y, label, color_map=None, color_col=None,
                     title=None, subtitle=None, height=520, logx=True,
                     xtitle=None, ytitle=None, quad_labels=None):
    """
    Scatter with median crosshairs dividing the field into four labelled
    quadrants. Far more informative than a bar chart for two-variable
    questions: it names the *combinations* rather than ranking one measure.
    """
    fig = go.Figure()
    xv = df[x].clip(lower=1e-4) if logx else df[x]
    mx, my = float(np.median(xv)), float(df[y].median())

    if color_col and color_map:
        for key, col in color_map.items():
            sub = df[df[color_col] == key]
            if sub.empty:
                continue
            sx = sub[x].clip(lower=1e-4) if logx else sub[x]
            fig.add_trace(go.Scatter(
                x=sx, y=sub[y], mode="markers", name=str(key),
                marker=dict(size=9, color=col, opacity=0.86,
                            line=dict(color=V.SURFACE, width=1.4)),
                customdata=sub[[label]].to_numpy(),
                hovertemplate=("%{customdata[0]}<br>"
                               f"{xtitle or x}: %{{x:,.3f}}<br>"
                               f"{ytitle or y}: %{{y:,.1f}}<extra></extra>")))
        show_legend = True
    else:
        fig.add_trace(go.Scatter(
            x=xv, y=df[y], mode="markers", name="",
            marker=dict(size=9, color=V.CATEGORICAL[0], opacity=0.86,
                        line=dict(color=V.SURFACE, width=1.4)),
            customdata=df[[label]].to_numpy(),
            hovertemplate=("%{customdata[0]}<br>"
                           f"{xtitle or x}: %{{x:,.3f}}<br>"
                           f"{ytitle or y}: %{{y:,.1f}}<extra></extra>")))
        show_legend = False

    fig.add_vline(x=mx, line=dict(color=V.INK_MUTED, width=1.2, dash="dot"))
    fig.add_hline(y=my, line=dict(color=V.INK_MUTED, width=1.2, dash="dot"))

    if quad_labels:
        for (xa, ya, xanc, yanc), txt in quad_labels.items():
            fig.add_annotation(x=xa, y=ya, xref="paper", yref="paper", text=txt,
                               showarrow=False, xanchor=xanc, yanchor=yanc,
                               font=dict(size=10.5, color=V.INK_SECONDARY),
                               bgcolor="rgba(252,252,251,0.82)")

    if logx:
        fig.update_xaxes(type="log")
    fig.update_xaxes(title_text=xtitle or x,
                     title_font=dict(size=11.5, color=V.INK_SECONDARY))
    fig.update_yaxes(title_text=ytitle or y,
                     title_font=dict(size=11.5, color=V.INK_SECONDARY))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="right", x=1.0,
                                  font=dict(color=V.INK_SECONDARY, size=11.5)))
    _base(fig, height, title, subtitle, legend=show_legend)
    return _axes(fig, xgrid=True, ygrid=True)


def vbar(labels, values, colors, title=None, subtitle=None, value_fmt="{:,.0f}",
         height=300, ytitle=None, secondary_text=None):
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values), marker=dict(color=colors),
        text=[value_fmt.format(v) if secondary_text is None else secondary_text[i]
              for i, v in enumerate(values)],
        textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT, size=11.5, color=V.INK_PRIMARY),
        hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
    vmax = max(values) if len(values) else 1
    fig.update_yaxes(range=[0, vmax * HEADROOM], title_text=ytitle,
                     title_font=dict(size=11.5, color=V.INK_SECONDARY))
    _base(fig, height, title, subtitle)
    return _axes(fig, xgrid=False, ygrid=True)


def heatmap(z, x, y, title=None, subtitle=None, height=None, zmid=None,
            colorscale=None, text_fmt="{:.2f}", cbar_title=None):
    zz = np.asarray(z, dtype=float)
    txt = [[text_fmt.format(v) for v in row] for row in zz]
    # label colour per cell, from that cell's own position on the ramp
    fig = go.Figure(go.Heatmap(
        z=zz, x=list(x), y=list(y), text=txt, texttemplate="%{text}",
        textfont=dict(family=FONT, size=11),
        colorscale=colorscale or [[0, "#184f95"], [0.5, "#f0efec"], [1, "#a62625"]],
        zmid=zmid, colorbar=dict(title=cbar_title, outlinewidth=0,
                                 tickfont=dict(color=V.INK_SECONDARY, size=10.5))))
    _base(fig, height or (120 + 60 * len(list(y))), title, subtitle)
    fig.update_xaxes(showgrid=False, automargin=True,
                     tickfont=dict(color=V.INK_SECONDARY, size=11))
    fig.update_yaxes(showgrid=False, automargin=True,
                     tickfont=dict(color=V.INK_SECONDARY, size=11))
    return fig
