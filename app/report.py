"""The result screen: a size-distribution histogram, optionally narrowed to one
pattern class, with a composition band underneath showing what each size bin is
made of (English, light theme). Rendered inline in the app AND exported; the
app keeps the numbers in native widgets beside it. Export writes only images
(annotated micrograph(s) + this report).
"""
from __future__ import annotations

import io
import os
import re
import numpy as np
from PIL import Image

from viz import render, pick_spread
from fonts import matplotlib_family, matplotlib_ui_family, ROBOTO

# export type -> filename keyword
TYPE_WORDS = {
    "line": "line",                # measurement lines only
    "undercooled": "undercooled",  # solid/undercooled colouring
    "pattern": "pattern",          # per-pattern (janus/stripe/…) colouring
}

# light palette. BG is the card grey the figure blends into (the plot used to be
# a white slab on grey, which read as a separate object); PLOT is the panel the
# bars stand on — a shade lighter so the data still lifts off the background.
BG, PANEL, INK, MUT = "#f4f6f8", "#f1f4f7", "#1a2129", "#6b7580"
PLOT = BG                 # the plot panel IS the card grey — no white slab
BLUE, RED, LINE = "#2b6fff", "#e5484d", "#e2e7ec"
# the All histogram is a neutral TEAL/turquoise — deliberately not the green of
# the undercooled class nor the blue of stripe, so the overall distribution never
# reads as one particular class. Range highlight is a deeper shade of the same.
HIST_FILL, HIST_EDGE = "#a9dcd8", "#4fb3ac"
RANGE_FILL, RANGE_EDGE = "#6cc4bd", "#2f918a"

# ---- per-class identity, shared by the composition band, the class-filtered
# histogram and the GUI's filter chips. Same muted family as the Pattern × Size
# window (softer than the bright overlay colours) so everything reads as one
# system. Crystalline particles the net gave no pattern to are left out entirely
# (the user doesn't work with them), so the shares are of the classified set —
# the same convention as the Pattern × Size window.
CLASS_ORDER = ["undercooled", "janus", "stripe", "lamellar", "composite"]
CLASS_LABELS = {"undercooled": "Undercooled", "janus": "Janus", "stripe": "Stripe",
                "lamellar": "Lamellar", "composite": "Composite"}
CLASS_COLORS = {
    "undercooled": "#6dbf8b",
    "janus":       "#e3a857",
    "stripe":      "#6e9bd6",
    "lamellar":    "#d081ae",
    "composite":   "#a78bd0",
}
# deeper sibling of each fill, used for bar edges and the filtered histogram
CLASS_EDGES = {
    "undercooled": "#4a9d69",
    "janus":       "#c1873a",
    "stripe":      "#4d79b3",
    "lamellar":    "#ac628c",
    "composite":   "#856bad",
}


def _mix(a, b, f):
    """Blend two '#rrggbb' colours; f=0 -> a, f=1 -> b."""
    ra, ga, ba = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    rb, gb, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * f),
                              round(ga + (gb - ga) * f),
                              round(ba + (bb - ba) * f))


# a size-range highlight one notch richer than the bar it sits on — noticeably
# picked out, but not the near-black jump a full CLASS_EDGE fill used to be
CLASS_RANGE_FILL = {k: _mix(CLASS_COLORS[k], CLASS_EDGES[k], 0.45) for k in CLASS_ORDER}

# "nice" bin widths, preferring ones that divide 100 so round ticks land on edges
_NICE_W = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]


def _nice_bin_width(dmin: float, dmax: float, target_bins: int = 20) -> float:
    raw = max(1.0, (dmax - dmin)) / target_bins
    for w in _NICE_W:
        if w >= raw:
            return float(w)
    return float(_NICE_W[-1])


def _bin_edges(diams, target_bins: int = 20):
    """Round bin edges starting from a round base, so 0/100/200… fall on edges."""
    dmin, dmax = float(np.min(diams)), float(np.max(diams))
    w = _nice_bin_width(dmin, dmax, target_bins)
    lo = np.floor(dmin / w) * w
    hi = np.ceil(dmax / w) * w
    if hi <= lo:
        hi = lo + w
    return np.arange(lo, hi + w * 0.5, w), w


def _tick_step(span: float, binw: float) -> float:
    """A round tick spacing that is a multiple of the bin width (so ticks sit on
    bar edges) and yields roughly 5–8 labels across the axis."""
    step = _nice_bin_width(0, span, target_bins=6)
    if step % binw:
        step = np.ceil(step / binw) * binw
    return float(step)


def _quiet_corner(counts):
    """Which top corner of the histogram is emptier — where a label can sit."""
    if counts is None or len(counts) < 4:
        return "right"
    k = max(1, int(len(counts) * 0.4))
    return "left" if counts[:k].max() < counts[-k:].max() else "right"


def class_diams(target):
    """{class: array of diameters} over exactly the particles the size statistics
    are built from (measurable, not excluded) — so a class filter and the overall
    histogram always add up. Empty dict when nothing is classified (ETD, or an
    analysis run without the pattern step)."""
    from analyze import measurable
    out = {k: [] for k in CLASS_ORDER}
    analyses = getattr(target, "analyses", None) or [target]
    for a in analyses:
        if not getattr(a, "classifiable", False):
            continue
        for p in a.particles:
            if not getattr(p, "diam_nm", 0) or not measurable(p):
                continue
            if not p.is_solid:
                out["undercooled"].append(p.diam_nm)
            elif p.pattern in out:
                out[p.pattern].append(p.diam_nm)
            # solid without a pattern: not a class the user works with -> skip
    out = {k: np.asarray(v, float) for k, v in out.items() if v}
    # a single bucket (e.g. everything undercooled) is not a composition
    return out if len(out) > 1 else {}


# the stat tiles shown under the chart in the app. Exports draw the SAME set, so
# a saved figure carries the numbers the user was reading on screen.
STAT_SOLID_COLOR = "#e0685f"
STAT_LABELS = {**CLASS_LABELS, "solid": "Solid"}
STAT_COLORS = {**CLASS_COLORS, "solid": STAT_SOLID_COLOR}
STAT_CLASSES = ["undercooled", "solid", "janus", "stripe", "composite", "lamellar"]


def stat_tiles(target, cls_filter=None, size_range=None):
    """Rows of (caption, value, sub, accent) exactly mirroring the app's tiles."""
    s = target.stats()
    groups = class_diams(target)
    if cls_filter not in groups:
        cls_filter = None
    measured = int(target.diam_array(whole_only=True).size)
    d = groups[cls_filter] if cls_filter else target.diam_array(whole_only=True)
    if size_range:
        lo, hi = size_range
        m = np.ones(d.shape, bool)
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
        d = d[m]
    n = int(d.size)
    mean = (f"{d.mean():.0f}", "nm") if n else ("—", "")
    rng = (f"{d.min():.0f} – {d.max():.0f}", "nm") if n else ("—", "")

    if cls_filter:
        tot = sum(int(x.size) for x in groups.values())
        pct = f"{100.0 * n / tot:.0f}%" if tot else ""
        return [[("TOTAL PARTICLES", f"{int(s['count_total'])}", "", None),
                 (CLASS_LABELS[cls_filter].upper(), f"{n}", pct,
                  CLASS_COLORS[cls_filter])],
                [("MEAN SIZE", *mean, None), ("RANGE", *rng, None)]]
    if groups:
        gsz = {k: int(v.size) for k, v in groups.items()}
        und = gsz.get("undercooled", 0)
        counts = dict(gsz)
        counts["solid"] = max(0, measured - und)
        patterned = und + sum(gsz.get(k, 0) for k in
                              ("janus", "stripe", "composite", "lamellar"))
        cls_row = []
        for k in STAT_CLASSES:
            c = counts.get(k, 0)
            pc = f"{100.0 * c / measured:.0f}%" if measured else "—"
            cls_row.append((STAT_LABELS[k].upper(), pc, f"{c}", STAT_COLORS[k]))
        return [[("MEAN SIZE", *mean, None), ("MEASURED", f"{measured}", "", None),
                 ("PATTERNED", f"{patterned}", "", None)],
                cls_row[:3], cls_row[3:]]
    return [[("TOTAL PARTICLES", f"{int(s['count_total'])}", "", None),
             ("MEASURED", f"{measured}", "", None)],
            [("MEAN SIZE", *mean, None), ("RANGE", *rng, None)]]


# the app's own tile styling (gui.py QSS #tile / #tilecap / #tileval / #tilesub),
# reproduced here so an exported figure looks like the panel it came from
TILE_BG, TILE_EDGE = "#ffffff", "#e6ebf0"
TILE_CAP_FG, TILE_VAL_FG, TILE_SUB_FG = "#8a95a1", "#1a2129", "#6b7580"


def _draw_tiles(fig, rect, rows):
    """Paint the tile block into `rect` ([x,y,w,h] in figure fractions).

    Everything is laid out in PIXELS (the axes is given a pixel coordinate
    system) so the rounded corners stay circular and the proportions are the
    app's own: a 133x47 tile with an 8px radius, scaled up to export size.
    """
    import matplotlib.patches as mpatches
    ui = matplotlib_ui_family()
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_facecolor(BG)
    w_px = rect[2] * fig.get_size_inches()[0] * fig.dpi
    h_px = rect[3] * fig.get_size_inches()[1] * fig.dpi
    ax.set_xlim(0, w_px); ax.set_ylim(0, h_px)

    nrow = len(rows)
    gap = 0.10 * h_px / nrow                    # between tiles, both directions
    th = (h_px - gap * (nrow - 1)) / nrow
    scale = th / 47.0                           # the app's tile is 47px tall
    pt = 72.0 / fig.dpi                         # px -> points
    for ri, row in enumerate(rows):
        ncol = len(row)
        tw = (w_px - gap * (ncol - 1)) / ncol
        y0 = h_px - (ri + 1) * th - ri * gap
        for ci, (cap, val, sub, accent) in enumerate(row):
            x0 = ci * (tw + gap)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y0), tw, th,
                boxstyle=f"round,pad=0,rounding_size={8.0 * scale}",
                linewidth=1.0, edgecolor=TILE_EDGE, facecolor=TILE_BG,
                clip_on=False))
            tx = x0 + 10.0 * scale
            ax.text(tx, y0 + th * 0.70, cap, va="center", ha="left",
                    fontsize=10.0 * scale * pt, family=ui, weight="bold",
                    color=TILE_CAP_FG)
            t = ax.text(tx, y0 + th * 0.30, val, va="center", ha="left",
                        fontsize=15.0 * scale * pt, family=ui, weight="bold",
                        color=accent or TILE_VAL_FG)
            if sub:
                fig.canvas.draw()
                bb = t.get_window_extent(fig.canvas.get_renderer())
                dx = bb.width * (w_px / ax.get_window_extent().width)
                ax.text(tx + dx + 7.0 * scale, y0 + th * 0.30, sub,
                        va="center", ha="left", fontsize=11.5 * scale * pt,
                        family=ui, weight="bold", color=TILE_SUB_FG)


def _draw_title(fig, title, H, top, plot_top):
    """The caption the user typed in the Save menu, sitting just above the plot
    (not floating at the top of the sheet)."""
    fig.text(0.5, plot_top + 0.10 / H, title, ha="center", va="bottom",
             fontsize=16, family=matplotlib_family(ROBOTO, "Roboto Slab"),
             weight="bold", color=INK)


def render_pattern_size(target, title=None, thresh=None) -> Image.Image:
    """The Pattern × Size view as a standalone figure: the 100%-stacked
    composition strip over the size axis, plus its legend. Deliberately just the
    coloured chart — no stat tiles — since that is what it is read for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.family"] = matplotlib_family()
    slab = matplotlib_family(ROBOTO, "Roboto Slab")
    groups = class_diams(target)
    present = [k for k in CLASS_ORDER if k in groups]
    if not present:
        raise ValueError("nothing classified — no Pattern × Size chart")
    n_classified = int(sum(groups[k].size for k in present))
    diams = np.concatenate([groups[k] for k in present])
    edges, binw = _bin_edges(diams)
    centres = edges[:-1] + binw / 2

    W = 6.8
    lm, rm, top, bot = 0.80, 0.40, 0.26, 0.30
    title_h = 0.34 if title else 0.0
    band_h, axlab_h = 2.10, 0.78
    leg_rows = int(np.ceil(len(present) / 3))
    leg_h = 0.34 * leg_rows + 0.10
    H = top + title_h + band_h + axlab_h + leg_h + bot

    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax_x, ax_w = lm / W, (W - lm - rm) / W
    y = bot
    leg_rect = [ax_x, y / H, ax_w, leg_h / H]
    y += leg_h + axlab_h
    band_rect = [ax_x, y / H, ax_w, band_h / H]

    axc = fig.add_axes(band_rect); axc.set_facecolor(PLOT)
    totals = np.zeros(len(centres))
    shares = {}
    for k in present:
        h, _ = np.histogram(groups[k], bins=edges)
        shares[k] = h.astype(float)
        totals += h
    safe = np.where(totals > 0, totals, 1.0)
    bottom = np.zeros(len(centres))
    for k in present:
        frac = shares[k] / safe
        axc.bar(centres, frac, width=binw, bottom=bottom, color=CLASS_COLORS[k],
                edgecolor="none", align="center")
        bottom += frac
    for c, t in zip(centres, totals):        # empty bins stay blank
        if t == 0:
            axc.bar([c], [1.0], width=binw, color=PLOT, edgecolor="none",
                    align="center", zorder=3)
    axc.set_xlim(edges[0], edges[-1]); axc.set_ylim(0, 1); axc.set_yticks([])
    axc.xaxis.set_major_locator(MultipleLocator(_tick_step(edges[-1] - edges[0], binw)))
    axc.tick_params(colors="#4a5560", labelsize=13, length=3)
    for lab in axc.get_xticklabels():
        lab.set_fontweight("semibold")
    xl = axc.get_xticklabels()
    if xl:
        xl[-1].set_horizontalalignment("right")
    for sp in axc.spines.values():
        sp.set_color("#c7cfd8")
    axc.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold",
                   labelpad=6)

    axl = fig.add_axes(leg_rect); axl.axis("off"); axl.set_facecolor(BG)
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    slot = 1.0 / 3
    chip_w, chip_h = 0.014, 0.34 / max(1, leg_rows)
    for i, k in enumerate(present):
        r, c = divmod(i, 3)
        x0 = c * slot
        y0 = 1.0 - (r + 0.5) / leg_rows
        pct = 100.0 * groups[k].size / n_classified if n_classified else 0.0
        axl.add_patch(plt.Rectangle((x0, y0 - chip_h / 2), chip_w, chip_h,
                                    transform=axl.transAxes,
                                    facecolor=CLASS_COLORS[k], edgecolor="none",
                                    clip_on=False))
        axl.text(x0 + 0.030, y0, CLASS_LABELS[k], transform=axl.transAxes,
                 va="center", ha="left", fontsize=12.5, family=slab,
                 weight="semibold", color="#576270")
        axl.text(x0 + slot - 0.045, y0, f"{pct:.0f}%", transform=axl.transAxes,
                 va="center", ha="right", fontsize=13, family=slab,
                 weight="bold", color=INK)

    if title:
        _draw_title(fig, title, H, top, band_rect[1] + band_rect[3])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_report(target, aspect: float = 6.8 / 8.9, size_range=None,
                  cls_filter=None, standalone=False, stats=False,
                  title=None) -> Image.Image:
    """Return the result-screen figure as a PIL image: a size histogram, and —
    when no class filter is active — the composition band underneath it.

    `aspect` is width/height; the caller passes the panel's aspect so the figure
    fills its card with an even border. There is no title (the panel is narrow;
    the experiment name is already in the file tree) and no data table: the
    numbers live next to the figure as native widgets in the app, and are drawn
    into the plot on export.

    `cls_filter` gives one class its OWN chart — the janus particles' size
    distribution on their own, in the janus colour, with nothing else behind it,
    and on its OWN size axis (a class that stops at 900 nm doesn't get an axis
    running to 2400, so its shape fills the plot).

    `standalone=True` writes the label an exported image needs to be read on its
    own (which class this is); on screen that lives in the panel beside it.

    `stats=True` draws the app's tile block (mean size, measured, patterned and
    the class breakdown) under the chart, so an exported figure carries the same
    numbers the user was reading beside it on screen.

    `size_range` is an optional (lo, hi) in nm; matching bars are highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.family"] = matplotlib_family()
    s = target.stats()
    diams_all = target.diam_array(whole_only=True)

    # per-class diameters drive both the filter and the composition band
    groups = class_diams(target)
    if cls_filter not in groups:
        cls_filter = None
    present = [k for k in CLASS_ORDER if k in groups]
    n_classified = int(sum(groups[k].size for k in present))
    diams = groups[cls_filter] if cls_filter else diams_all

    slab = matplotlib_family(ROBOTO, "Roboto Slab")

    # ---- geometry in inches. The figure ALWAYS fills the card (its height
    # follows the card's aspect) so only the card's own even margin frames it.
    # The histogram takes everything the band and its legend don't need. ----
    W = 6.8
    # the right margin is tuned so the plot's right edge sits as far from the
    # card edge as the "Count" label does on the left (visually balanced)
    # rm must clear HALF the last x tick label ("2400"), or it gets cut off at
    # the figure edge; the label is also right-anchored below for good measure
    lm, rm = 0.80, 0.40
    top, bot = 0.26, 0.30
    # The composition band that used to sit under the All histogram has moved out
    # to its own view (the breakdown now lives in the tiles beside the figure), so
    # every chart here is just the histogram + its own x-axis.
    band = False
    leg_cols = 3
    leg_rows = 0
    band_h, band_gap, axlab_h = 0.60, 0.16, 0.78
    leg_h = 0.0

    # the tile block under the chart (exports only): one strip per row of tiles
    tile_rows = stat_tiles(target, cls_filter, size_range) if stats else []
    tiles_h = (0.62 * len(tile_rows) + 0.22) if tile_rows else 0.0
    title_h = 0.34 if title else 0.0      # the user's own caption, above the plot

    # cap how tall the figure may get: a very tall card would otherwise stretch
    # the plot into a narrow column (the leftover simply frames the figure)
    H = max(4.6, min(9.6 if band else 7.6, W / max(0.45, min(1.6, aspect))))
    H += tiles_h + title_h                # these ADD height, never squeeze the plot
    band_block = band_gap + band_h + axlab_h + leg_h if band else 0.0
    hist_h = H - top - bot - band_block - (0.0 if band else axlab_h) - tiles_h - title_h
    if band and hist_h < 2.2:            # short card: the plot comes first
        band, band_block = False, 0.0
        hist_h = H - top - bot - axlab_h - tiles_h - title_h
    hist_h = max(1.4, hist_h)

    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax_x, ax_w = lm / W, (W - lm - rm) / W
    # stack upwards from the bottom: tiles · legend · band (+ its axis) · plot
    y = bot
    tiles_rect = None
    if tile_rows:
        tiles_rect = [rm / W, y / H, (W - 2 * rm) / W, (tiles_h - 0.22) / H]
        y += tiles_h
    leg_rect = band_rect = None
    if band:
        leg_rect = [ax_x, y / H, ax_w, leg_h / H]
        y += leg_h + axlab_h
        band_rect = [ax_x, y / H, ax_w, band_h / H]
        y += band_h + band_gap
    else:
        y += axlab_h
    hist_rect = [ax_x, y / H, ax_w, hist_h / H]

    # ---- histogram: All = the familiar green; a class = its own colour ----
    ax = fig.add_axes(hist_rect); ax.set_facecolor(PLOT)
    edges = binw = counts = None
    if diams.size:
        # bins follow whatever is being shown, so a class fills its own axis
        edges, binw = _bin_edges(diams)
        fill = CLASS_COLORS[cls_filter] if cls_filter else HIST_FILL
        edge = CLASS_EDGES[cls_filter] if cls_filter else HIST_EDGE
        # the size-range highlight is a richer shade of whatever the bars are —
        # noticeable, but not the near-black jump a full CLASS_EDGE fill gave
        hi_fill = CLASS_RANGE_FILL[cls_filter] if cls_filter else RANGE_FILL
        hi_edge = CLASS_EDGES[cls_filter] if cls_filter else RANGE_EDGE
        counts, _, patches = ax.hist(diams, bins=edges, color=fill,
                                     edgecolor=edge, linewidth=0.9)
        # highlight bars whose centre falls inside the selected size range
        if size_range is not None:
            lo, hi = size_range
            for patch, left in zip(patches, edges[:-1]):
                c = left + binw / 2
                if (lo is None or c >= lo) and (hi is None or c < hi):
                    patch.set_facecolor(hi_fill)
                    patch.set_edgecolor(hi_edge)
        ax.xaxis.set_major_locator(MultipleLocator(_tick_step(edges[-1] - edges[0], binw)))
        ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Count", color=INK, fontsize=15, weight="semibold")
    for sp in ax.spines.values():
        sp.set_color("#c7cfd8")
    ax.tick_params(colors="#4a5560", labelsize=13)
    if band:
        ax.tick_params(axis="x", labelbottom=False, length=0)
    else:
        ax.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("semibold")
    # the right-most x label used to run off the edge ("2400" showing as "240(");
    # anchoring its right side keeps it inside the figure
    xlabs = ax.get_xticklabels()
    if xlabs:
        xlabs[-1].set_horizontalalignment("right")

    # annotations go in whichever top corner the bars leave empty, on a soft
    # white pad so they stay readable if a tall bar reaches up anyway
    quiet = _quiet_corner(counts)
    pad = dict(facecolor=PLOT, edgecolor="none", alpha=0.85,
               boxstyle="round,pad=0.28")

    # (a filtered chart used to caption itself in the top corner; the tiles below
    # now say which class and how many, so the plot area stays clean)

    # ---- composition band: what each size bin is made of (100% stacked) ----
    if band:
        axc = fig.add_axes(band_rect); axc.set_facecolor(PLOT)
        centres = edges[:-1] + binw / 2
        totals = np.zeros(len(centres))
        shares = {}
        for k in present:
            h, _ = np.histogram(groups[k], bins=edges)
            shares[k] = h.astype(float)
            totals += h
        safe = np.where(totals > 0, totals, 1.0)
        bottom = np.zeros(len(centres))
        for k in present:
            frac = shares[k] / safe
            axc.bar(centres, frac, width=binw, bottom=bottom,
                    color=CLASS_COLORS[k], edgecolor="none", align="center")
            bottom += frac
        # bins with no particle stay blank instead of showing a stray block
        for c, t in zip(centres, totals):
            if t == 0:
                axc.bar([c], [1.0], width=binw, color=PLOT,
                        edgecolor="none", align="center", zorder=3)
        axc.set_xlim(edges[0], edges[-1])
        axc.set_ylim(0, 1)
        axc.set_yticks([])
        axc.xaxis.set_major_locator(MultipleLocator(_tick_step(edges[-1] - edges[0], binw)))
        axc.tick_params(colors="#4a5560", labelsize=13, length=3)
        for lab in axc.get_xticklabels():
            lab.set_fontweight("semibold")
        for sp in axc.spines.values():
            sp.set_color("#c7cfd8")
        axc.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold",
                       labelpad=6)

        # ---- legend: colour chip · class · share of the classified set,
        # three per row, the share right-aligned so the numbers line up ----
        axl = fig.add_axes(leg_rect); axl.axis("off"); axl.set_facecolor(BG)
        axl.set_xlim(0, 1); axl.set_ylim(0, 1)
        slot = 1.0 / leg_cols
        chip_w, chip_h = 0.014, 0.34 / max(1, leg_rows)
        for i, k in enumerate(present):
            r, c = divmod(i, leg_cols)
            x0 = c * slot
            y0 = 1.0 - (r + 0.5) / leg_rows          # row centre
            pct = 100.0 * groups[k].size / n_classified if n_classified else 0.0
            axl.add_patch(plt.Rectangle((x0, y0 - chip_h / 2), chip_w, chip_h,
                                        transform=axl.transAxes,
                                        facecolor=CLASS_COLORS[k], edgecolor="none",
                                        clip_on=False))
            axl.text(x0 + 0.030, y0, CLASS_LABELS[k], transform=axl.transAxes,
                     va="center", ha="left", fontsize=12.5, family=slab,
                     weight="semibold", color="#576270")
            axl.text(x0 + slot - 0.045, y0, f"{pct:.0f}%", transform=axl.transAxes,
                     va="center", ha="right", fontsize=13, family=slab,
                     weight="bold", color=INK)

    if tiles_rect is not None:
        _draw_tiles(fig, tiles_rect, tile_rows)
    if title:
        _draw_title(fig, title, H, top, hist_rect[1] + hist_rect[3])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _export_stem(analysis, typeword):
    """'UÖ - 15 Karışık 3'  +  'line'  ->  'UÖ - 15 Karışık line.3'."""
    title = analysis.title()
    stem = os.path.splitext(analysis.image)[0]
    suffix = stem[len(title):] if stem.startswith(title) else stem
    m = re.search(r"(\d+)", suffix)
    return f"{title} {typeword}.{m.group(1)}" if m else f"{title} {typeword}"


def _combine_both(lines_img, report_img, gap=24, bg=None):
    """Place the lined micrograph (left) and the result screen (right) side by
    side, scaled to a common height. The backdrop is the report's own grey, so
    the chart doesn't sit on a mismatched white slab."""
    if bg is None:
        bg = tuple(int(BG[i:i + 2], 16) for i in (1, 3, 5))
    h = max(lines_img.height, report_img.height)
    def scaled(im):
        w = int(im.width * h / im.height)
        return im.resize((w, h), Image.LANCZOS)
    a, b = scaled(lines_img), scaled(report_img)
    out = Image.new("RGB", (a.width + gap + b.width, h), bg)
    out.paste(a, (0, 0)); out.paste(b, (a.width + gap, 0))
    return out


def render_variant(analysis, typ, chosen=None, k=5):
    """Render one per-image SEM variant as a PIL image."""
    if typ == "line":
        return render(analysis, show_measurements=True, chosen=chosen, k=k)
    if typ == "undercooled":
        return render(analysis, show_measurements=False,
                      show_classification=analysis.classifiable)
    if typ == "pattern":
        return render(analysis, show_measurements=False,
                      show_pattern=analysis.classifiable)
    raise ValueError(typ)


def _safe(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "export"


def export(analyses, out_dir, types, facet_thresh=None, chosen_map=None, k=5,
           size_range=None, cls_filters=None, title=None):
    """Export into a sub-folder named after the experiment.

    analyses    : list of Analysis
    types       : SEM-image variants, a subset of {'line','undercooled','pattern'};
                  written once per selected image
    cls_filters : which CHARTS to write — "all" is the whole distribution, a class
                  key ("janus", …) that class alone, "patternsize" the Pattern ×
                  Size composition chart. One file each.
    title       : optional caption drawn above every chart
    size_range  : optional (lo, hi) nm range echoed into the charts

    Charts are always POOLED: with several images selected you get ONE chart per
    entry, over their combined particles — never one chart per image.
    """
    cls_filters = list(cls_filters or [])
    titles = [a.title() for a in analyses]
    folder = titles[0] if len(set(titles)) == 1 else \
        (os.path.commonprefix(titles).rstrip(" -_") or "SEM export")
    out_dir = os.path.join(out_dir, _safe(folder))
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    chosen_map = chosen_map or {}

    for a in analyses:
        if facet_thresh is not None:
            a.reclassify(facet_thresh)
        chosen = chosen_map.get(getattr(a, "path", None)) or pick_spread(a.particles, k)
        for typ in types:
            img = render_variant(a, typ, chosen=chosen, k=k)
            name = _export_stem(a, TYPE_WORDS[typ]) + ".png"
            p = os.path.join(out_dir, name)
            img.save(p)
            outputs[name] = p

    if not cls_filters:
        return outputs

    if len(analyses) == 1:
        target, stem = analyses[0], None
    else:
        from analyze import Aggregate
        target = Aggregate(list(analyses))
        stem = _safe(target.title()) + " combined"
    present = class_diams(target)

    for cf in cls_filters:
        if cf == "patternsize":
            if not present:
                continue
            img = render_pattern_size(target, title=title)
            word = "pattern x size"
        else:
            cf_key = None if cf in (None, "all") else cf
            # a class the selection doesn't contain would silently fall back to
            # the whole distribution and write a duplicate file — skip it
            if cf_key is not None and cf_key not in present:
                continue
            img = render_report(target, size_range=size_range, cls_filter=cf_key,
                                stats=True, title=title)
            word = "results" if cf_key is None else f"results {cf_key}"
        name = (f"{stem} {word}.png" if stem
                else _export_stem(analyses[0], word) + ".png")
        p = os.path.join(out_dir, name)
        img.save(p)
        outputs[name] = p
    return outputs
