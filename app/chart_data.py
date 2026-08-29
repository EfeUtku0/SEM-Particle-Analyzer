"""The numbers behind every chart, and the colour identity of a class.

Separated from the drawing so a count can be checked without a figure. This is
also the single source the panel and the exported image both read — they used
to compute Solid two different ways, and a saved figure could disagree with
what was on screen beside it.

One rule runs through all of it: counts and distributions are taken over the
MEASURED set. A particle too occluded for a trustworthy diameter is also too
occluded to say what it is, so it counts towards the total and nothing else.
"""
from __future__ import annotations

import numpy as np

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

# the Pattern × Size size axis: capped, with the final bin absorbing the tail.
# Lives here because both the on-screen panel (window_pattern_size) and the
# exported figure (charts.render_pattern_size) draw the same axis, and they must
# not drift apart — a chart the user reads on screen and then exports has to be
# the same chart.
PANEL_AXIS_MAX = 1500.0
PANEL_BINS = 30                 # 50 nm bins


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
            if getattr(p, "unclassified", False):
                continue          # measured, but the user says its class is unclear
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


def n_unclassified(target):
    """Measured particles the user marked "class unclear" (no class, real size)."""
    from analyze import measurable
    analyses = getattr(target, "analyses", None) or [target]
    return sum(1 for a in analyses for p in a.particles
               if getattr(p, "unclassified", False) and measurable(p)
               and getattr(p, "diam_nm", 0))


def class_counts(target):
    """The counts behind the tile block — ONE definition, shared by the panel and
    the exported figure.

    They used to be computed twice (here and in gui._update_stats) and had already
    drifted: this one subtracted the "class unclear" particles from Solid and the
    app's one did not, so the figure you saved could disagree with the numbers you
    were reading beside it. Nothing derives a class count by subtraction any more
    than it has to, and both callers get the same dict.

    Everything is counted over the MEASURED set, which is the user's rule: a
    particle too occluded for a trustworthy diameter is also too occluded to state
    what it is, so the greyed-out ones count towards nothing but the total. Hence
    `classified` (undercooled + the four patterns) can be less than `measured` —
    the difference is crystalline particles with no readable pattern plus the ones
    hand-marked "class unclear", and the tiles show it rather than hiding it.
    """
    groups = class_diams(target)
    measured = int(target.diam_array(whole_only=True).size)
    gsz = {k: int(v.size) for k, v in groups.items()}
    unclassified = n_unclassified(target)
    counts = dict(gsz)
    # "Solid" is the crystalline parent category: every measured particle that is
    # neither undercooled nor of unstated class — so it INCLUDES the four pattern
    # classes listed beside it, and the class row deliberately sums past 100%.
    counts["solid"] = max(0, measured - gsz.get("undercooled", 0) - unclassified)
    return dict(counts=counts, groups=groups, measured=measured,
                classified=int(sum(gsz.values())), unclassified=unclassified)


def sphericity(target, cls=None, state=None):
    """How round the particles are: dict(mean, raw, n, total), or None.

    `mean` is the average of the per-particle scores on the app's stretched
    reading scale (analyze.sphericity_score) — the number shown on the tile.
    `raw` is the same average in plain ISO/ImageJ circularity, kept because that
    is the quantity with a name outside this app and the one to quote in a paper.
    `n` is how many particles could be scored and `total` how many are in the
    same measured set, so the panel can say what share the number rests on.

    Membership follows the rest of this module exactly — `cls` picks one of the
    class_diams() buckets, `state` one side of solid_split(), neither means every
    measured particle — with ONE subtraction: particles touching the frame are
    scored by nobody (see analyze.spherical_measurable). They stay in `total`, so
    the share on screen shows what was left out rather than hiding it.
    """
    from analyze import (measurable, spherical_measurable, circularity,
                         sphericity_score)
    if target is None:
        return None
    vals, raws, total = [], [], 0
    analyses = getattr(target, "analyses", None) or [target]
    for a in analyses:
        if (cls or state) and not getattr(a, "classifiable", False):
            continue
        for p in a.particles:
            if not getattr(p, "diam_nm", 0) or not measurable(p):
                continue
            if cls or state:
                if getattr(p, "unclassified", False):
                    continue
                if state and ("solid" if p.is_solid else "undercooled") != state:
                    continue
                if cls and (p.pattern if p.is_solid else "undercooled") != cls:
                    continue
            total += 1
            if spherical_measurable(p):
                vals.append(sphericity_score(p))
                raws.append(circularity(p))
    if not total:
        return None
    return dict(mean=(float(np.mean(vals)) if vals else None),
                raw=(float(np.mean(raws)) if raws else None),
                n=len(vals), total=total)


def sphericity_tile(target, cls=None, state=None, cap="SPHERICITY", accent=None):
    """The sphericity number as a tile row entry, shared by the panel and the
    exported figure so the two can never quote it differently."""
    s = sphericity(target, cls=cls, state=state)
    if not s or s["mean"] is None:
        return (cap, "—", "", accent)
    share = f"{100.0 * s['n'] / s['total']:.0f}%" if s["total"] else ""
    return (cap, f"{s['mean']:.2f}", share, accent)


def stat_tiles(target, cls_filter=None, size_range=None):
    """Rows of (caption, value, sub, accent) exactly mirroring the app's tiles."""
    s = target.stats()
    cc = class_counts(target)
    groups = cc["groups"]
    if cls_filter not in groups:
        cls_filter = None
    measured = cc["measured"]
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
        # RANGE gave way to SPHERICITY (user request, 2026-08-08): the min/max of
        # a class is two outliers, while how round that class is, is a property
        # of the class. The range is still readable off the histogram beside it.
        return [[("TOTAL PARTICLES", f"{int(s['count_total'])}", "", None),
                 (CLASS_LABELS[cls_filter].upper(), f"{n}", pct,
                  CLASS_COLORS[cls_filter])],
                [("MEAN SIZE", *mean, None),
                 sphericity_tile(target, cls=cls_filter)]]
    # MEASURED's own share is of every particle DETECTED in the image (not of
    # the classified ones) — "how much of what I segmented actually got a
    # trustworthy size", the question the occlusion/edge/shape gate answers.
    total = int(s["count_total"])
    mshare = f"{100.0 * measured / total:.0f}%" if total else ""
    if groups:
        counts = cc["counts"]
        cls_row = []
        for k in STAT_CLASSES:
            c = counts.get(k, 0)
            pc = f"{100.0 * c / measured:.0f}%" if measured else "—"
            cls_row.append((STAT_LABELS[k].upper(), pc, f"{c}", STAT_COLORS[k]))
        # The third tile used to be CLASSIFIED (undercooled + the four patterns).
        # It was replaced by SPHERICITY on the user's request (2026-08-08): the
        # class row underneath already shows every class with its share, so
        # CLASSIFIED was a sum of the numbers directly below it, while how round
        # the particles are was nowhere in the app.
        return [[("MEAN SIZE", *mean, None), ("MEASURED", f"{measured}", mshare, None),
                 sphericity_tile(target)],
                cls_row[:3], cls_row[3:]]
    # Nothing classified (ETD, or a selection that is all one state): sphericity
    # still applies — it is geometry, not a class — so it takes the same place in
    # the top row as it does in the classified view, and RANGE keeps its own.
    return [[("TOTAL PARTICLES", f"{int(s['count_total'])}", "", None),
             ("MEASURED", f"{measured}", mshare, None),
             sphericity_tile(target)],
            [("MEAN SIZE", *mean, None), ("RANGE", *rng, None)]]


def d_values(diams):
    """D10 / D50 / D90 and the span, the standard way a particle-size
    distribution is quoted.

    Dx = the diameter below which x% of the measured particles fall; span =
    (D90-D10)/D50, a scale-free width of the distribution. Returns None when
    there is nothing measured.

    These are NUMBER-based percentiles — every particle counts once. That is the
    only thing image analysis can give (we count particles, not scattered light),
    and it is what this app reports on purpose. It is NOT the same quantity a
    Zetasizer or a laser-diffraction instrument prints: those default to
    VOLUME-weighted Dv values, where each particle counts as d³, which shifts the
    percentiles up enormously on a polydisperse sample — measured on this user's
    43 photos, Dv50 came out 1216 nm against Dn50 259 nm, a factor of 4.7. So do
    not put these side by side with a Zetasizer D50 as if they were comparable.
    """
    d = np.asarray(diams, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    d10, d50, d90 = (float(x) for x in np.percentile(d, [10, 50, 90]))
    return dict(n=int(d.size), d10=d10, d50=d50, d90=d90,
                span=(d90 - d10) / d50 if d50 else 0.0)


def cumulative_tiles(target):
    """The tile rows for the Cumulative view — same shape as stat_tiles()."""
    dv = d_values(target.diam_array(whole_only=True)) if target is not None else None
    if dv is None:
        return [[("D10", "—", "", None), ("D50", "—", "", None),
                 ("D90", "—", "", None)],
                [("SPAN", "—", "", None), ("MEASURED", "0", "", None)]]
    return [[("D10", f"{dv['d10']:.0f}", "nm", None),
             ("D50", f"{dv['d50']:.0f}", "nm", None),
             ("D90", f"{dv['d90']:.0f}", "nm", None)],
            [("SPAN", f"{dv['span']:.2f}", "(D90−D10)/D50", None),
             ("MEASURED", f"{dv['n']}", "particles", None)]]


def solid_split(target):
    """{"solid": diameters, "undercooled": diameters} over the measured set.

    This is the PARENT split of class_diams(): "solid" here is every measured
    crystalline particle, whether or not a pattern could be read on it — the same
    definition the "Solid" tile uses in class_counts(), so the two never
    disagree. Particles the user marked "class unclear" belong to neither and are
    left out, exactly as they are everywhere else.

    Returns {} unless BOTH states are present: one bucket is not a split, and a
    "solid vs undercooled" chart of a single state would only mislead.
    """
    from analyze import measurable
    out = {"solid": [], "undercooled": []}
    analyses = getattr(target, "analyses", None) or [target]
    for a in analyses:
        if not getattr(a, "classifiable", False):
            continue
        for p in a.particles:
            if not getattr(p, "diam_nm", 0) or not measurable(p):
                continue
            if getattr(p, "unclassified", False):
                continue
            out["solid" if p.is_solid else "undercooled"].append(p.diam_nm)
    out = {k: np.asarray(v, float) for k, v in out.items() if v}
    return out if len(out) == 2 else {}


SPLIT_COLORS = {"solid": STAT_SOLID_COLOR, "undercooled": CLASS_COLORS["undercooled"]}
SPLIT_EDGES = {"solid": "#b34a42", "undercooled": CLASS_EDGES["undercooled"]}
SPLIT_LABELS = {"solid": "Solid", "undercooled": "Undercooled"}
# Solid is drawn at the BOTTOM of each stack, the way the user's own Excel charts
# have always had it, so the two can be read side by side.
SPLIT_ORDER = ["solid", "undercooled"]


def solid_share(groups, lo=None, hi=None):
    """(share of solids, n solid, n total) inside [lo, hi) — the question the
    size-range box on the Solid / Liquid page asks. Bounds are the same
    half-open convention the rest of the panel uses (lo inclusive, hi exclusive),
    and either may be None for open-ended."""
    def cut(d):
        d = np.asarray(d, float)
        m = np.ones(d.shape, bool)
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
        return int(m.sum())
    ns = cut(groups.get("solid", np.array([])))
    nu = cut(groups.get("undercooled", np.array([])))
    tot = ns + nu
    return (ns / tot if tot else None), ns, tot


def majority_solid_above(groups, min_n=20):
    """The smallest diameter from which the particles are MOSTLY solid.

    Read as: "take every particle at or above d — at least half of them are
    solid". Defined on the tail rather than per bin because a per-bin fraction
    flickers around 50% wherever the counts are thin; the tail share is monotone
    enough to quote as one number. The tail must still hold `min_n` particles, so
    a couple of stray big crystals can't set it. None when the solids never take
    over (or there is too little data).
    """
    s = np.asarray(groups.get("solid", np.array([])), float)
    u = np.asarray(groups.get("undercooled", np.array([])), float)
    if s.size == 0 or u.size == 0:
        return None
    d = np.concatenate([s, u])
    is_s = np.concatenate([np.ones(s.size, bool), np.zeros(u.size, bool)])
    order = np.argsort(d, kind="stable")
    d, is_s = d[order], is_s[order]
    n = d.size
    suf_solid = np.cumsum(is_s[::-1])[::-1]          # solids at or above each d
    suf_n = np.arange(n, 0, -1)
    ok = (suf_n >= min(min_n, n)) & (suf_solid * 2 >= suf_n)
    idx = np.nonzero(ok)[0]
    return float(d[idx[0]]) if idx.size else None


def split_tiles(target, size_range=None):
    """The tile rows under the Solid / Liquid chart — same shape as
    stat_tiles(), so the app and an exported figure show the same numbers."""
    groups = solid_split(target) if target is not None else {}
    if not groups:
        return [[("SOLID", "—", "", STAT_SOLID_COLOR),
                 ("UNDERCOOLED", "—", "", CLASS_COLORS["undercooled"])]]
    ns = int(groups["solid"].size)
    nu = int(groups["undercooled"].size)
    tot = ns + nu
    ms = groups["solid"].mean()
    mu = groups["undercooled"].mean()
    frac, k, kt = solid_share(groups, *(size_range or (None, None)))
    # two per row, mirroring the panel: solid on the left, undercooled on the
    # right, so the pairs are read across
    # The last row was MOSTLY SOLID ABOVE + CLASSIFIED; it now carries the
    # sphericity of each state (user request, 2026-08-08), which keeps the whole
    # block reading as solid-vs-undercooled pairs — share, mean size, roundness.
    # (majority_solid_above() is kept: the crossover diameter is still the point
    # of the chart above, it is simply not quoted as a tile any more.)
    rows = [[("SOLID", f"{100.0 * ns / tot:.0f}%", f"{ns}", STAT_SOLID_COLOR),
             ("UNDERCOOLED", f"{100.0 * nu / tot:.0f}%", f"{nu}",
              CLASS_COLORS["undercooled"])],
            [("MEAN SOLID", f"{ms:.0f}", "nm", None),
             ("MEAN UNDERCOOLED", f"{mu:.0f}", "nm", None)],
            [sphericity_tile(target, state="solid", cap="SPHERICITY SOLID"),
             sphericity_tile(target, state="undercooled",
                             cap="SPHERICITY UNDERCOOLED")]]
    if size_range:
        rows.append([("SOLID IN RANGE",
                      (f"{100.0 * frac:.0f}%" if frac is not None else "—"),
                      f"{k} / {kt} particles", STAT_SOLID_COLOR)])
    return rows
