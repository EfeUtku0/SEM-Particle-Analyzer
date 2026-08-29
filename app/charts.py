"""Drawing the figures: histogram, cumulative curve, Solid / Liquid, Pattern x Size.

Every function here returns a PIL image and is used by BOTH the on-screen panel
and the file export, which is what keeps the two identical. The numbers come
from chart_data; nothing in this file decides what a count means.

Matplotlib is imported inside the functions, not at module scope: it costs
about a second to load and the app must not pay that on startup for a user who
never opens a chart.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from fonts import matplotlib_family, matplotlib_ui_family, ROBOTO
from chart_data import (BG, INK, MUT, PLOT, HIST_FILL, HIST_EDGE,
                        RANGE_FILL, RANGE_EDGE,
                        CLASS_ORDER, CLASS_LABELS, CLASS_COLORS, CLASS_EDGES,
                        CLASS_RANGE_FILL, PANEL_AXIS_MAX, PANEL_BINS,
                        SPLIT_ORDER, SPLIT_COLORS, SPLIT_EDGES, SPLIT_LABELS,
                        _bin_edges, _tick_step, _quiet_corner,
                        class_diams, stat_tiles, d_values, cumulative_tiles,
                        solid_split, split_tiles)

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


# ---------------------------------------------------------------------------
# Shared chart parts. The composition band and its legend appear both under the
# main histogram and as the whole of the Pattern x Size figure; they were
# written out twice and had already begun to drift (one elided its last tick
# label, the other did not). One copy each, so the two views cannot disagree
# about what a bin is made of.
# ---------------------------------------------------------------------------


def _draw_composition_band(fig, rect, groups, present, edges, binw,
                           elide_last_tick=False, last_tick_label=None):
    """100%-stacked bars: what each size bin is made of, class by class.

    Bins holding no particle are painted over in the panel colour rather than
    left as a stray full-height block — an empty bin has no composition, and
    showing one was reading as "100% of the last class".

    `last_tick_label` renames the right-hand tick ("1500+"), for callers whose
    axis is capped and whose final bin absorbs the tail.
    """
    from matplotlib.ticker import MultipleLocator

    axc = fig.add_axes(rect); axc.set_facecolor(PLOT)
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
        axc.bar(centres, frac, width=binw, bottom=bottom, color=CLASS_COLORS[k],
                edgecolor="none", align="center")
        bottom += frac
    for c, t in zip(centres, totals):
        if t == 0:
            axc.bar([c], [1.0], width=binw, color=PLOT, edgecolor="none",
                    align="center", zorder=3)
    axc.set_xlim(edges[0], edges[-1])
    axc.set_ylim(0, 1)
    axc.set_yticks([])
    step = _tick_step(edges[-1] - edges[0], binw)
    if last_tick_label is not None:
        ticks = np.arange(edges[0], edges[-1] + step * 0.5, step)
        if ticks[-1] < edges[-1] - 1e-9:     # the cap always gets its own tick
            ticks = np.append(ticks, edges[-1])
        axc.set_xticks(ticks)
        axc.set_xticklabels([f"{t:g}" for t in ticks[:-1]] + [last_tick_label])
    else:
        axc.xaxis.set_major_locator(MultipleLocator(step))
    axc.tick_params(colors="#4a5560", labelsize=13, length=3)
    for lab in axc.get_xticklabels():
        lab.set_fontweight("semibold")
    if elide_last_tick:
        xl = axc.get_xticklabels()
        if xl:
            xl[-1].set_horizontalalignment("right")
    for sp in axc.spines.values():
        sp.set_color("#c7cfd8")
    axc.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold",
                   labelpad=6)
    return axc


def _draw_class_legend(fig, rect, groups, present, n_classified, cols, rows, slab):
    """Colour chip, class name, and its share of the classified set — `cols` per
    row, the share right-aligned so the numbers line up down the column."""
    import matplotlib.pyplot as plt

    axl = fig.add_axes(rect); axl.axis("off"); axl.set_facecolor(BG)
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    slot = 1.0 / cols
    chip_w, chip_h = 0.014, 0.34 / max(1, rows)
    for i, k in enumerate(present):
        r, c = divmod(i, cols)
        x0 = c * slot
        y0 = 1.0 - (r + 0.5) / rows              # row centre
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
    return axl


def _figure_image(fig):
    """Render a finished figure to a PIL image and release it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)          # matplotlib keeps every open figure alive forever
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_cumulative(target, aspect: float = 6.8 / 4.6, title=None,
                      stats=False) -> Image.Image:
    """The cumulative size distribution: % of particles at or below a diameter.

    One dark curve for everything measured, a thin coloured curve per class when
    the selection holds more than one (that is where "are the janus ones bigger
    than the undercooled ones?" gets answered at a glance), and D10/D50/D90
    marked on the main curve. `stats=True` draws the D-value tiles underneath,
    the way render_report does — used for the export, while the app shows those
    numbers as native tiles instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = matplotlib_family()
    slab = matplotlib_family(ROBOTO, "Roboto Slab")
    d_all = np.asarray(target.diam_array(whole_only=True), float)
    d_all = np.sort(d_all[np.isfinite(d_all)])
    if d_all.size == 0:
        raise ValueError("nothing measured — no cumulative chart")
    groups = class_diams(target)

    W = 6.8
    tiles = cumulative_tiles(target) if stats else []
    tile_h = (0.62 * len(tiles) + 0.12) if tiles else 0.0
    lm, rm, top, bot = 0.92, 0.40, 0.26, 0.30
    title_h = 0.34 if title else 0.0
    axlab_h = 0.78
    # SAME height rule as render_solid_split (clamped both ways rather than
    # solved from `aspect` directly): the two are exported side by side with the
    # same `aspect` value, and solving straight from it here gave the plot area
    # a fraction of the height the tile block ate into — Solid/Liquid came out
    # visibly bigger than Cumulative for no reason a reader could see.
    H = max(4.6, min(7.6, W / max(0.45, min(1.6, aspect)))) + tile_h + title_h
    plot_h = max(1.4, H - top - bot - axlab_h - tile_h - title_h)

    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax_x, ax_w = lm / W, (W - lm - rm) / W
    y = bot
    tile_rect = [ax_x, y / H, ax_w, tile_h / H] if tiles else None
    y += tile_h + axlab_h
    ax = fig.add_axes([ax_x, y / H, ax_w, plot_h / H])
    ax.set_facecolor(PLOT)

    def curve(d):
        """Step-free empirical CDF: sorted diameters against 0…100%."""
        d = np.sort(np.asarray(d, float))
        return d, np.arange(1, d.size + 1) / d.size * 100.0

    drawn = []
    if len(groups) > 1:
        for k in CLASS_ORDER:
            arr = groups.get(k)
            if arr is None or arr.size < 3:
                continue
            gx, gy = curve(arr)
            ax.plot(gx, gy, color=CLASS_COLORS[k], lw=1.4, alpha=0.75,
                    solid_capstyle="round", zorder=2, label=CLASS_LABELS[k])
            drawn.append(k)
    x, yv = curve(d_all)
    ax.plot(x, yv, color=INK, lw=2.4, solid_capstyle="round", zorder=4)

    dv = d_values(d_all)
    for key, pct in (("d10", 10), ("d50", 50), ("d90", 90)):
        val = dv[key]
        ax.plot([x[0], val], [pct, pct], color=MUT, lw=1.0, ls=(0, (3, 3)),
                zorder=3)
        ax.plot([val, val], [0, pct], color=MUT, lw=1.0, ls=(0, (3, 3)), zorder=3)
        ax.plot([val], [pct], "o", ms=5.5, color=BG, mec=INK, mew=1.8, zorder=5)
        ax.annotate(f"{key.upper()}  {val:.0f}", (val, pct),
                    textcoords="offset points", xytext=(7, -12),
                    fontsize=11.5, family=slab, weight="bold", color=INK,
                    zorder=6)

    ax.set_xlim(max(0.0, x[0] - (x[-1] - x[0]) * 0.03), x[-1] + (x[-1] - x[0]) * 0.03)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(True, axis="y", color="#dfe4ea", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4a5560", labelsize=13, length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("semibold")
    xl = ax.get_xticklabels()
    if xl:
        xl[-1].set_horizontalalignment("right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cfd8")
    ax.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold",
                  labelpad=6)
    ax.set_ylabel("Cumulative", color=INK, fontsize=14, weight="semibold",
                  labelpad=6)
    if drawn:
        # bottom-right is the one corner an S-shaped CDF always leaves empty
        leg = ax.legend(loc="lower right", frameon=False, fontsize=11,
                        handlelength=1.4, labelspacing=0.25, borderpad=0.2)
        for t in leg.get_texts():
            t.set_color("#576270")
            t.set_fontweight("semibold")

    if tiles:
        _draw_tiles(fig, tile_rect, tiles)
    if title:
        _draw_title(fig, title, H, top, (y + plot_h) / H)
    return _figure_image(fig)


def render_solid_split(target, aspect: float = 6.8 / 4.6, size_range=None,
                       title=None, stats=False) -> Image.Image:
    """The size histogram split by state: solid stacked under undercooled.

    The "All" histogram answers how big the particles are; this one answers the
    question the experiment is actually about — WHERE on the size axis the
    particles stop being undercooled and start being crystalline. Same bins as
    everywhere else (`_bin_edges`), so the x-axis follows the data of the
    selection instead of a fixed scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.family"] = matplotlib_family()
    groups = solid_split(target)
    if not groups:
        raise ValueError("nothing to split — need both solid and undercooled")
    d_all = np.concatenate([groups[k] for k in SPLIT_ORDER])

    edges, binw = _bin_edges(d_all)
    centres = edges[:-1] + binw / 2
    counts = {k: np.histogram(groups[k], bins=edges)[0].astype(float)
              for k in SPLIT_ORDER}
    total = sum(counts.values())

    W = 6.8
    tiles = split_tiles(target, size_range) if stats else []
    tiles_h = (0.62 * len(tiles) + 0.22) if tiles else 0.0
    title_h = 0.34 if title else 0.0
    # The height follows the SAME rule as render_report (capped both ways), so a
    # tall panel doesn't stretch this figure into a narrow column and the two
    # charts end up with type of the same size when the tab is switched.
    lm, rm, top, bot, axlab_h = 0.80, 0.40, 0.26, 0.30, 0.78
    H = max(4.6, min(7.6, W / max(0.45, min(1.6, aspect)))) + tiles_h + title_h
    plot_h = max(1.4, H - top - bot - axlab_h - tiles_h - title_h)

    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax_x, ax_w = lm / W, (W - lm - rm) / W
    y = bot
    tiles_rect = None
    if tiles:
        tiles_rect = [rm / W, y / H, (W - 2 * rm) / W, (tiles_h - 0.22) / H]
        y += tiles_h
    y += axlab_h
    ax = fig.add_axes([ax_x, y / H, ax_w, plot_h / H])
    ax.set_facecolor(PLOT)

    # the typed size range is a band across the whole plot rather than a
    # re-colouring of the bars: a stacked bar has two fills already, and a third
    # shade on top of them stops being readable
    if size_range is not None:
        lo, hi = size_range
        ax.axvspan(edges[0] if lo is None else lo, edges[-1] if hi is None else hi,
                   color="#2b6fff", alpha=0.07, zorder=0)
        for v in (lo, hi):
            if v is not None and edges[0] <= v <= edges[-1]:
                ax.axvline(v, color="#2b6fff", lw=1.1, ls=(0, (4, 3)),
                           alpha=0.55, zorder=1)

    bottom = np.zeros(len(centres))
    for k in SPLIT_ORDER:
        ax.bar(centres, counts[k], width=binw, bottom=bottom,
               color=SPLIT_COLORS[k], edgecolor=SPLIT_EDGES[k], linewidth=0.9,
               align="center", zorder=2, label=SPLIT_LABELS[k])
        bottom += counts[k]

    ax.set_xlim(edges[0], edges[-1])
    ax.xaxis.set_major_locator(MultipleLocator(_tick_step(edges[-1] - edges[0], binw)))
    ax.set_ylabel("Count", color=INK, fontsize=15, weight="semibold")
    ax.set_xlabel("Diameter (nm)", color=INK, fontsize=15, weight="semibold")
    for sp in ax.spines.values():
        sp.set_color("#c7cfd8")
    ax.tick_params(colors="#4a5560", labelsize=13)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("semibold")
    xlabs = ax.get_xticklabels()
    if xlabs:
        xlabs[-1].set_horizontalalignment("right")

    # A per-bin "solid share" curve on a second axis was tried here and removed
    # at the user's request (2026-08-02): the stack already shows where the
    # colours change over, and the extra line plus its right-hand axis made the
    # panel busy for a number the Size range box below answers exactly.

    # the legend goes wherever the bars leave room (a size distribution is
    # right-skewed, so that is normally the right-hand side)
    loc = "upper right" if _quiet_corner(total) == "right" else "upper left"
    leg = ax.legend(loc=loc, frameon=False, fontsize=11.5, handlelength=1.4,
                    labelspacing=0.3, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color("#576270")
        t.set_fontweight("semibold")
    leg.set_zorder(6)

    if tiles_rect is not None:
        _draw_tiles(fig, tiles_rect, tiles)
    if title:
        _draw_title(fig, title, H, top, (y + plot_h) / H)
    return _figure_image(fig)


def render_pattern_size(target, title=None, thresh=None) -> Image.Image:
    """The Pattern × Size view as a standalone figure: the 100%-stacked
    composition strip over the size axis, plus its legend. Deliberately just the
    coloured chart — no stat tiles — since that is what it is read for.

    Geometry and size axis follow the on-screen panel (window_pattern_size):
    a nearly square band, and the axis capped at 1500 nm with the last bin
    absorbing everything above it ("1500+"). Letting the axis run to the largest
    particle stretched the export to 2000+ nm, where the last few hundred
    nanometres are a handful of particles and the part that is actually read —
    where the colours change over, under ~1000 nm — got squeezed flat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = matplotlib_family()
    slab = matplotlib_family(ROBOTO, "Roboto Slab")
    groups = class_diams(target)
    present = [k for k in CLASS_ORDER if k in groups]
    if not present:
        raise ValueError("nothing classified — no Pattern × Size chart")
    n_classified = int(sum(groups[k].size for k in present))

    # 50 nm bins over 0–1500, exactly as the panel bins them; particles above the
    # cap are pulled into the final bin rather than dropped (the legend keeps the
    # unclipped groups, so the percentages still count every particle)
    binw = PANEL_AXIS_MAX / PANEL_BINS
    edges = np.arange(0.0, PANEL_AXIS_MAX + binw * 0.5, binw)
    capped = {k: np.minimum(groups[k], PANEL_AXIS_MAX - binw * 0.5)
              for k in present}

    W = 6.8
    lm, rm, top, bot = 0.42, 0.42, 0.26, 0.30
    title_h = 0.34 if title else 0.0
    # the band is as tall as it is wide (the panel's proportions); a flat strip
    # was unreadable once exported
    band_h, axlab_h = (W - lm - rm) / 1.15, 0.78
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

    _draw_composition_band(fig, band_rect, capped, present, edges, binw,
                           elide_last_tick=True,
                           last_tick_label=f"{PANEL_AXIS_MAX:g}+")
    _draw_class_legend(fig, leg_rect, groups, present, n_classified,
                       cols=3, rows=leg_rows, slab=slab)

    if title:
        _draw_title(fig, title, H, top, band_rect[1] + band_rect[3])
    return _figure_image(fig)


def render_report(target, aspect: float = 6.8 / 8.9, size_range=None,
                  cls_filter=None, stats=False,
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

    `stats=True` draws the app's tile block (mean size, measured, patterned and
    the class breakdown) under the chart, so an exported figure carries the same
    numbers the user was reading beside it on screen.

    `size_range` is an optional (lo, hi) in nm; matching bars are highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.family"] = matplotlib_family()
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

    # (a filtered chart used to caption itself in whichever top corner the bars
    # left empty; the tiles below now say which class and how many, so the plot
    # area stays clean)

    # ---- composition band + its legend: what each size bin is made of ----
    if band:
        _draw_composition_band(fig, band_rect, groups, present, edges, binw)
        _draw_class_legend(fig, leg_rect, groups, present, n_classified,
                           cols=leg_cols, rows=leg_rows, slab=slab)

    if tiles_rect is not None:
        _draw_tiles(fig, tiles_rect, tile_rows)
    if title:
        _draw_title(fig, title, H, top, hist_rect[1] + hist_rect[3])

    return _figure_image(fig)


# ---------------------------------------------------------------------------
# Data saturation. Unlike every other figure in this file, this one is not about
# the particles on a micrograph — it is about the training set itself: how the
# model's accuracy responded to being given more of it. It lives here anyway
# because "one place draws figures" is worth more than the taxonomy, and because
# it must look like it came from the same app as the rest.
# ---------------------------------------------------------------------------

# reading order: the measured curve is the teal the app already uses for "all
# particles"; the fit is a quiet grey extrapolation, deliberately weaker than
# the data it is drawn through so it can never be mistaken for a measurement.
SAT_CURVE, SAT_FIT = RANGE_EDGE, "#96a1ad"
INSTRUMENT_COLORS = ("#4d79b3", "#c1873a", "#856bad", "#4a9d69")


def render_saturation(record, fit=None, verdict=None, history=None,
                      title=None) -> Image.Image:
    """The learning curve: accuracy against how much labelled data it was given.

    Read left to right — if the last stretch is still rising, more labelled
    photos are still buying accuracy; if it has gone flat, they are not. The
    dashed vertical marks where the real training set sits today, so "where are
    we on this curve" is answered by looking at one line.

    The lower strip splits the SAME held-out particles by microscope. A curve
    that flattens only because a second instrument was added to the mix looks
    identical to a real plateau up top, and that strip is the only place the
    difference is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import model_pattern_curve as curve

    plt.rcParams["font.family"] = matplotlib_family()
    slab = matplotlib_family(ROBOTO, "Roboto Slab")

    rows = curve._agg(record["points"])
    if len(rows) < 2:
        raise ValueError("not enough measured points for a saturation chart")
    fit = fit if fit is not None else curve.fit(record["points"])
    verdict = verdict or curve.verdict(record, fit)
    insts = sorted({g for p in record["points"]
                    for g in (p.get("by_instrument") or {})})
    multi = len(insts) > 1

    import textwrap
    # wrap to the figure's own text column: 5.4 in of it, and a character of
    # this face averages about half its point size
    head_t = "\n".join(textwrap.wrap(verdict.get("text", ""), 76))
    note_raw = "\n\n".join(x["text"] for x in
                           (verdict.get("capacity"), verdict.get("instrument")) if x)
    note_t = "\n\n".join("\n".join(textwrap.wrap(par, 88))
                         for par in note_raw.split("\n\n")) if note_raw else ""

    W = 6.8
    lm, rm, top, bot = 0.95, 0.45, 0.26, 0.30
    title_h = 0.34 if title else 0.0
    plot_h, axlab_h, tick_h = 2.60, 0.62, 0.42
    inst_h = 1.35 if multi else 0.0
    # the caption grows with what it has to say — a fixed box either clipped the
    # instrument note or left a hand's width of grey under a one-line verdict
    # measured line pitch of this face at these sizes, in inches — the nominal
    # point size understates it (the app's Roboto carries generous leading)
    HEAD_LINE, NOTE_LINE = 0.30, 0.25
    cap_h = (0.34 + HEAD_LINE * (head_t.count("\n") + 1)
             + (0.14 + NOTE_LINE * (note_t.count("\n") + 1) if note_t else 0.0))
    H = top + title_h + plot_h + tick_h + inst_h + axlab_h + cap_h + bot

    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax_x, ax_w = lm / W, (W - lm - rm) / W
    y = bot
    cap_rect = [ax_x, y / H, ax_w, cap_h / H]
    y += cap_h + axlab_h              # room for an x label UNDER the lower strip
    inst_rect = [ax_x, y / H, ax_w, inst_h / H] if multi else None
    y += inst_h + tick_h              # …and for the main plot's own tick labels
    ax = fig.add_axes([ax_x, y / H, ax_w, plot_h / H])
    ax.set_facecolor(PLOT)

    n = np.array([r["n"] for r in rows], float)
    acc = np.array([r["y"] for r in rows], float) * 100.0
    n_now = float(record["n_total"])
    x_max = max(n_now * 2.1, n.max() * 1.15)

    # the fit, extrapolated past the data it was measured on — drawn first and
    # faintly, so the eye lands on the measured points
    if fit:
        xs = np.linspace(max(n.min() * 0.6, 50), x_max, 200)
        ys = np.array([curve.project(fit, v) for v in xs]) * 100.0
        ax.plot(xs, ys, color=SAT_FIT, lw=1.6, ls=(0, (5, 4)), zorder=2)
        # the ceiling is drawn only when the data actually pins one down; on a
        # curve still climbing steeply the fit puts it at 100%, which is the
        # search's bound and not a measurement (see model_pattern_curve.fit)
        if fit.get("ceiling_identified"):
            ax.axhline(fit["a"] * 100.0, color=SAT_FIT, lw=1.0, ls=(0, (2, 3)),
                       zorder=1)
            ax.annotate(f"fitted ceiling {fit['a'] * 100:.0f}%",
                        (x_max, fit["a"] * 100.0), textcoords="offset points",
                        xytext=(-2, 5), ha="right", fontsize=10.5,
                        color="#7d8894", weight="semibold", zorder=6)
        else:
            ax.annotate("projection — no ceiling identifiable yet",
                        (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(-4, -16), ha="right", fontsize=9.5,
                        color="#8b96a2", weight="semibold", zorder=6)

    # spread between repeated draws at the same size: with few photos, WHICH
    # photos you drew matters, and hiding that would oversell the shape
    for r in rows:
        if r["reps"] > 1 and r["hi"] > r["lo"]:
            ax.plot([r["n"], r["n"]], [r["lo"] * 100, r["hi"] * 100],
                    color=_mix_hex(SAT_CURVE, BG, 0.55), lw=3.2,
                    solid_capstyle="round", zorder=3)
    # how well each model fit the data it was trained ON — the distance between
    # the two curves is the part of the error that more data can still remove
    tr_rows = curve._agg(record["points"], "acc_train")
    if len(tr_rows) >= 2:
        tn = [r["n"] for r in tr_rows]
        ty = [r["y"] * 100 for r in tr_rows]
        ax.plot(tn, ty, color="#b9c3cd", lw=1.6, zorder=2)
        ax.annotate("on its own training data", (tn[-1], ty[-1]),
                    textcoords="offset points", xytext=(8, -3), fontsize=9.5,
                    color="#8b96a2", weight="semibold", zorder=6)

    ax.plot(n, acc, color=SAT_CURVE, lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot(n, acc, "o", ms=6.5, color=BG, mec=SAT_CURVE, mew=2.2, zorder=5)
    # below the last point: above it is where the projection runs
    ax.annotate("held out", (n[-1], acc[-1]), textcoords="offset points",
                xytext=(4, -17), fontsize=9.5, color=SAT_CURVE,
                weight="semibold", zorder=6)

    # where the real training set sits today
    ax.axvline(n_now, color=MUT, lw=1.2, ls=(0, (3, 3)), zorder=3)
    ax.annotate(f"you are here\n{int(n_now):,} labelled".replace(",", " "),
                (n_now, ax.get_ylim()[0]), textcoords="offset points",
                xytext=(6, 8), fontsize=10.5, color=MUT, weight="semibold",
                zorder=6)

    # Past training runs used to be drawn here as a second series, back when
    # each run's headline was a cross-validation over the same pattern classes
    # this curve is measured in. Since 2026-08-05 a run's accuracy comes from
    # the golden set and is a FIVE-class figure over every measured particle —
    # a different quantity, which sharing this axis would invite reading as the
    # same one. Comparing runs is the model report's job now (its tabs), so the
    # series is gone rather than relabelled.

    seen = list(acc) + ([fit["a"] * 100]
                        if fit and fit.get("ceiling_identified") else [])
    seen += [r["y"] * 100 for r in tr_rows]
    lo, hi = min(seen) - 6, max(seen) + 6
    ax.set_ylim(max(0, lo), min(100, hi))
    ax.set_xlim(0, x_max)
    ax.grid(True, axis="y", color="#dfe4ea", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4a5560", labelsize=12.5, length=3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: f"{v:.0f}%"))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("semibold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cfd8")
    ax.set_ylabel("Held-out accuracy", color=INK, fontsize=13.5,
                  weight="semibold", labelpad=6)
    if not multi:
        ax.set_xlabel("Labelled particles used for training", color=INK,
                      fontsize=14, weight="semibold", labelpad=6)

    # ---- per-instrument strip -------------------------------------------
    if multi:
        ax2 = fig.add_axes(inst_rect)
        ax2.set_facecolor(PLOT)
        ends = []
        for i, g in enumerate(insts):
            gx, gy = [], []
            for r in curve._agg_instrument(record["points"], g):
                gx.append(r["n"]); gy.append(r["y"] * 100.0)
            if len(gx) < 2:
                continue
            c = INSTRUMENT_COLORS[i % len(INSTRUMENT_COLORS)]
            ax2.plot(gx, gy, color=c, lw=1.9, solid_capstyle="round", zorder=3)
            ax2.plot(gx, gy, "o", ms=4.5, color=BG, mec=c, mew=1.8, zorder=4)
            ends.append((gy[-1], gx[-1], c, _instrument_label(g, record)))
        ax2.set_xlim(0, x_max)
        ax2.axvline(n_now, color=MUT, lw=1.2, ls=(0, (3, 3)), zorder=2)
        ax2.grid(True, axis="y", color="#dfe4ea", lw=0.9)
        ax2.set_axisbelow(True)
        ax2.tick_params(colors="#4a5560", labelsize=11.5, length=3)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _p: f"{v:.0f}%"))
        for lab in ax2.get_xticklabels() + ax2.get_yticklabels():
            lab.set_fontweight("semibold")
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax2.spines[side].set_color("#c7cfd8")
        ax2.set_xlabel("Labelled particles used for training", color=INK,
                       fontsize=14, weight="semibold", labelpad=6)
        ax2.set_ylabel("By microscope", color=INK, fontsize=12,
                       weight="semibold", labelpad=6)
        # the names are annotated on the lines themselves, so the strip needs
        # elbow room on the right rather than a legend
        ax2.set_xlim(0, x_max * 1.16)
        # …and when the two machines end up at the same accuracy — which is the
        # good outcome, and so the likeliest one — their labels would print on
        # top of each other. Push them apart, keeping each next to its own line.
        lo2, hi2 = ax2.get_ylim()
        sep = 0.13 * (hi2 - lo2)
        placed = []
        for yv_, xv_, c, lab in sorted(ends, reverse=True):
            while any(abs(yv_ - q) < sep for q in placed):
                yv_ -= sep
            placed.append(yv_)
            ax2.annotate(lab, (xv_, yv_), textcoords="offset points",
                         xytext=(9, -4), fontsize=10, color=c,
                         weight="semibold", zorder=6)

    # ---- caption: the sentence, and the conditions it was measured under --
    axc = fig.add_axes(cap_rect); axc.axis("off")
    axc.set_xlim(0, 1); axc.set_ylim(0, cap_h)          # y in inches: no guessing
    yc = cap_h - 0.04
    axc.text(0, yc, head_t, ha="left", va="top",
             fontsize=10.5, color=INK, linespacing=1.4)
    yc -= HEAD_LINE * (head_t.count("\n") + 1) + 0.08
    if note_t:
        axc.text(0, yc, note_t, ha="left", va="top", fontsize=9,
                 color="#576270", linespacing=1.4)
    hold = record.get("holdout", {})
    sub = (f"{record['n_photos']} photos · {record['n_total']} labelled "
           f"particles · held out {hold.get('photos', 0)} photos "
           f"({hold.get('particles', 0)} particles) · {record.get('epochs')} "
           f"epochs per point")
    axc.text(0, 0.02, sub, ha="left", va="bottom", fontsize=9.5, color=MUT)

    if title:
        _draw_title(fig, title, H, top, y / H + plot_h / H)
    return _figure_image(fig)


def _instrument_label(g, record):
    n = (record.get("instruments") or {}).get(g)
    name = "METU-METE" if g in ("no nameplate", "unknown") else g
    return f"{name}" + (f"  ({n} photos)" if n else "")


def _mix_hex(a, b, f):
    ra, ga, ba = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    rb, gb, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * f),
                              round(ga + (gb - ga) * f),
                              round(ba + (bb - ba) * f))
