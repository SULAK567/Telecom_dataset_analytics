"""
Shared visual system for every figure and for the dashboard.

Colours are a validated palette (checked for colour-vision-deficiency
separation, lightness band, chroma floor and surface contrast) rather than
matplotlib defaults, so all outputs read as one system.

Encoding rules applied throughout:
  * continuous magnitude (density)      -> single-hue sequential blue ramp
  * ordered categories (2G/3G/4G/5G,
    coverage tiers, clusters)           -> ordinal blue ramp, light -> dark
  * unordered identity (carriers)       -> categorical slots in fixed order
  * correlation (signed)                -> diverging blue <-> red, grey midpoint
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Surfaces & ink ----
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# ---- Categorical (fixed order, never cycled) ----
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Carrier identity is fixed so a filter never repaints the survivors.
CARRIER_COLORS = {
    "Airtel": "#2a78d6",
    "Vi": "#eb6834",
    "Jio/Reliance": "#1baf7a",
    "BSNL": "#eda100",
}

# ---- Sequential blue ramp (100 -> 700) for continuous magnitude ----
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# ---- Ordinal ramp (validated: light end clears 2:1 on the light surface) ----
ORDINAL_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#0d366b"]

GENERATION_COLORS = dict(zip(["2G", "3G", "4G", "5G"], ORDINAL_4))
TIER_COLORS = dict(zip(["Underserved", "Moderate", "Well-served"], ORDINAL_3))

# ---- Diverging (blue <-> red with a neutral grey midpoint) ----
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#b7d3f6", "#f0efec",
                     "#f2b0af", "#e34948", "#a62625"]
)


def apply_style():
    """Global matplotlib defaults: recessive chrome, system sans, no junk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK_PRIMARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def strip_spines(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


def title_block(ax, title, subtitle=None):
    """
    Title + optional deck line placed ABOVE the axes with explicit vertical
    separation. matplotlib's own set_title() sits at the same height as a
    manually placed deck line, which collides; placing both by hand in axes
    coordinates guarantees they never overlap each other or a legend.
    """
    if subtitle:
        ax.text(0.0, 1.155, title, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=13, fontweight="600", color=INK_PRIMARY)
        ax.text(0.0, 1.055, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9.5, color=INK_SECONDARY)
    else:
        ax.text(0.0, 1.055, title, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=13, fontweight="600", color=INK_PRIMARY)


def fig_title(fig, title, subtitle=None):
    """Figure-level title, for map panels where the axes fill the canvas."""
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=14,
             fontweight="600", color=INK_PRIMARY)
    if subtitle:
        fig.text(0.02, 0.938, subtitle, ha="left", va="top", fontsize=9.5,
                 color=INK_SECONDARY)


def legend_below(ax, ncol=4, y=-0.14, title=None):
    """
    Legends go BELOW the plot. Anchoring a legend above the axes puts it in the
    same band as the title block, which collides on wide figures.
    """
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, y),
              frameon=False, title=title, handlelength=1.0,
              handleheight=1.0, columnspacing=1.8, borderpad=0.0)


def map_axes(ax):
    """Geographic axes: no grid, no ticks, no frame."""
    ax.set_axis_off()
    ax.grid(False)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Contrast-aware label colour.
#
# Hard-coding white text inside coloured bars fails on light fills: white on
# the light end of a blue ramp is close to invisible. Choosing the label
# colour from the fill's own relative luminance (WCAG definition) means a
# label is always readable, whatever step of whatever ramp it lands on.
# ---------------------------------------------------------------------------
def _srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def contrast_ratio(fg, bg):
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def label_on(fill, light="#ffffff", dark="#0b0b0b"):
    """Return whichever of light/dark text has more contrast on `fill`."""
    return light if contrast_ratio(light, fill) >= contrast_ratio(dark, fill) else dark


def readable_labels(fills, light="#ffffff", dark="#0b0b0b"):
    return [label_on(f, light, dark) for f in fills]
