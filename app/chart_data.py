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
    # `size_range` is NOT applied here. It used to be, which made an exported
    # MEAN SIZE the mean of the typed range while the app's tile (the one this
    # mirrors) showed the whole distribution. What the range holds is stated
    # on its own, by range_sentence(), under the chart.
    d = groups[cls_filter] if cls_filter else target.diam_array(whole_only=True)
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


def range_share(d, lo=None, hi=None):
    """(k, n, in-range diameters): how many of `d` fall in [lo, hi).

    ONE definition for the readout beside the Size range boxes and the sentence
    drawn under an exported chart. Either end may be None (open)."""
    d = np.asarray(d, float)
    m = np.ones(d.shape, bool)
    if lo is not None:
        m &= d >= lo
    if hi is not None:
        m &= d < hi
    return int(m.sum()), int(d.size), d[m]


def share_text(k, n):
    """'23%', but '<1%' rather than a misleading '0%' when something is there,
    and '>99%' rather than '100%' when something is missing."""
    if not n:
        return "\u2014"
    pct = 100.0 * k / n
    if 0 < pct < 0.5:
        return "<1%"
    if 99.5 <= pct < 100:
        return ">99%"
    return f"{pct:.0f}%"


def range_sentence(d, lo=None, hi=None, cls=None):
    """The size range as a sentence for the exported report, split into runs of
    (text, emphasis) so the drawing can set the share and the range in bold:

        23% of the measured particles (492 of 2138) are between 100 and
        150 nm, with a mean size of 124 nm.

    `cls` names the class when the chart is one class on its own. Returns None
    when there is no range or nothing to count."""
    if lo is None and hi is None:
        return None
    k, n, inside = range_share(d, lo, hi)
    if not n:
        return None
    # at most one decimal, none when whole: "120", "120.5", never "485.402"
    f = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
    if lo is not None and hi is not None:
        where = f"between {f(lo)} and {f(hi)} nm"
    elif lo is not None:
        where = f"{f(lo)} nm or larger"
    else:
        where = f"smaller than {f(hi)} nm"
    # non-breaking spaces: the range is one phrase and must not wrap mid-way
    where = where.replace(" ", "\u00a0")
    who = f"{CLASS_LABELS[cls].lower()} particles" if cls else "measured particles"
    are = "is" if k == 1 else "are"
    runs = [(share_text(k, n), True),
            (f" of the {who} ({k} of {n}) {are} ", False),
            (where, True)]
    if k:
        runs.append((f", with a mean size of {inside.mean():.0f} nm.", False))
    else:
        runs.append((".", False))
    return runs


def d_values(diams):
    """D10 / D50 / D90 and the span, the standard way a particle-size
    distribution is quoted.

    Dx = the diameter below which x% of the measured particles fall; span =
    (D90-D10)/D50, a scale-free width of the distribution. Returns None when
    there is nothing measured.

    These are NUMBER-based percentiles — every particle counts once, which is
    what image analysis measures directly. It is NOT the same quantity a
    Zetasizer or a laser-diffraction instrument prints: those default to
    VOLUME-weighted Dv values, where each particle counts as d³, which shifts the
    percentiles up enormously on a polydisperse sample — measured on this user's
    43 photos, Dv50 came out 1216 nm against Dn50 259 nm, a factor of 4.7. So do
    not put these side by side with a Zetasizer D50 as if they were comparable.
    (The Volume tab does the d³ weighting for the CLASS shares; see
    volume_shares.)
    """
    d = np.asarray(diams, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    d10, d50, d90 = (float(x) for x in np.percentile(d, [10, 50, 90]))
    return dict(n=int(d.size), d10=d10, d50=d50, d90=d90,
                span=(d90 - d10) / d50 if d50 else 0.0)


# ---- the Volume tab: class shares by material, not by particle count -------
# (user request, 2026-11-14). Every other share in the app is a NUMBER share:
# each particle counts once. Small particles are mostly undercooled, so a photo
# that looks almost all janus can still read "60% undercooled" — true as a count
# of particles, but not what the photo is made of. Here each particle counts by
# its volume, (π/6)·d³, taking it as a sphere: its projected outline is the
# sphere's great circle, so the equivalent diameter IS the sphere's diameter.
# (π/6 cancels in every share and is never applied.) For equal densities this
# is also the MASS share — the quantity a DSC enthalpy is proportional to.
#
# Measured before it was built (2026-11-14, read-only over the session):
#   163 classifiable photos, 100415 particles: undercooled is 60% by number and
#   16% by volume (median photo); 59 photos flip from a majority to a minority.
#   On the golden set the model's volume shares sit within 2-6 points of the
#   hand-labelled ones per class, and volume-weighted accuracy (85.3%) equals
#   the number-weighted one (85.4%).
#   Frame-edge (Horvitz-Thompson) correction for big particles being cut by the
#   frame moved a photo's volume share by 0.5 points (median), 7.5 at worst —
#   smaller than the sampling interval below, so it is not applied.
#
# The catch is how few particles carry the answer: in the median photo half the
# volume sits in 46 particles. So every volume share comes with its 95%
# interval, from the linearised (delta-method) variance of a ratio estimator —
# deterministic, 0.08 s on 100k particles, and checked against a 400-sample
# bootstrap on 157 photos: the half-widths differ by 0.1 points (median), 1.4
# at worst.

# the bar segments, in stacking order. "nopattern" is crystalline with no
# readable pattern: it belongs to Solid (so the bar still sums to the measured
# set) but to none of the four named patterns.
VOLUME_SEGMENTS = ["undercooled", "janus", "stripe", "lamellar", "composite",
                   "nopattern"]
VOLUME_LABELS = {**CLASS_LABELS, "nopattern": "Solid, no pattern"}
VOLUME_COLORS = {**CLASS_COLORS, "nopattern": "#e7a39d"}
VOLUME_EDGES = {**CLASS_EDGES, "nopattern": "#c97a73"}
VOLUME_TILE_KEYS = ["undercooled", "solid", "janus", "stripe", "composite",
                    "lamellar"]           # the same six, in the same order, as
                                          # the class row of the All tab


def volume_rows(target):
    """(diameters, segment keys) over the measured, classified set — the same
    particles class_counts() counts: measured, not hand-marked "class unclear".
    Unlike class_diams() the crystalline particles with no pattern are KEPT (as
    "nopattern"): leaving them out would hand their volume to the other classes.
    Empty arrays when nothing in the selection is classifiable."""
    from analyze import measurable
    d, k = [], []
    analyses = getattr(target, "analyses", None) or [target]
    for a in analyses:
        if not getattr(a, "classifiable", False):
            continue
        for p in a.particles:
            if not getattr(p, "diam_nm", 0) or not measurable(p):
                continue
            if getattr(p, "unclassified", False):
                continue
            d.append(p.diam_nm)
            if not p.is_solid:
                k.append("undercooled")
            else:
                k.append(p.pattern if p.pattern in CLASS_COLORS else "nopattern")
    return np.asarray(d, float), np.asarray(k, dtype=object)


def _ratio_ci(w, z):
    """Share Σw·z / Σw and the half-width of its 95% interval (delta method)."""
    tot = w.sum()
    p = float((w * z).sum() / tot)
    se = float(np.sqrt(((w * (z - p)) ** 2).sum()) / tot)
    return p, 1.96 * se


def volume_shares(target):
    """Number and volume share of each segment, plus Solid (= not undercooled),
    each volume share with its 95% half-width. None when nothing is classified
    or the selection is all one state (one bucket is not a composition)."""
    if target is None:
        return None
    d, k = volume_rows(target)
    if d.size == 0 or len(set(k)) < 2:
        return None
    v = d ** 3
    keys = VOLUME_SEGMENTS + ["solid"]
    by_n, by_v, ci = {}, {}, {}
    for key in keys:
        z = (k != "undercooled") if key == "solid" else (k == key)
        z = z.astype(float)
        by_n[key] = float(z.mean())
        by_v[key], ci[key] = _ratio_ci(v, z)
    vs = np.sort(v)[::-1]
    half = int(np.searchsorted(np.cumsum(vs), vs.sum() / 2)) + 1
    return dict(n=int(d.size), by_number=by_n, by_volume=by_v, ci=ci,
                half_n=half)


def volume_tiles(target):
    """The tile rows under the Volume figure — same shape as stat_tiles(): each
    class's share of the volume, with its ± (95%) beside it."""
    vs = volume_shares(target)
    caps = [(STAT_LABELS[key].upper(), STAT_COLORS[key])
            for key in VOLUME_TILE_KEYS]
    if vs is None:
        cells = [(cap, "—", "", col) for cap, col in caps]
    else:
        cells = []
        for (cap, col), key in zip(caps, VOLUME_TILE_KEYS):
            p, h = vs["by_volume"][key], vs["ci"][key]
            val = share_text(p, 1.0)
            # nothing of the class at all is not "0 ± 0", it is just absent
            sub = ("" if p == 0 else f"±{100 * h:.0f}" if h >= 0.005 else "±<1")
            cells.append((cap, val, sub, col))
    return [cells[:3], cells[3:]]


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


def split_tiles(target):
    """The tile rows under the Solid / Liquid chart — same shape as
    stat_tiles(), so the app and an exported figure show the same numbers.

    2026-09-27 (user request — colleagues ask for the ratio and the size, not
    a per-state breakdown): the share of each state, then ONE mean and the
    D-values of the whole measured set. The mean and the D-values are taken
    over exactly the particles the All and Cumulative tabs use
    (diam_array(whole_only=True)), so the three tabs quote the same numbers.
    The per-state means and the per-state sphericity are gone from this page."""
    groups = solid_split(target) if target is not None else {}
    d = target.diam_array(whole_only=True) if target is not None else np.array([])
    dv = d_values(d)
    mean = (f"{np.mean(d):.0f}", "nm") if d.size else ("—", "")
    drow = ([("D10", f"{dv['d10']:.0f}", "nm", None),
             ("D50", f"{dv['d50']:.0f}", "nm", None),
             ("D90", f"{dv['d90']:.0f}", "nm", None)] if dv else
            [("D10", "—", "", None), ("D50", "—", "", None), ("D90", "—", "", None)])
    if not groups:
        return [[("SOLID", "—", "", STAT_SOLID_COLOR),
                 ("UNDERCOOLED", "—", "", CLASS_COLORS["undercooled"]),
                 ("MEAN", *mean, None)], drow]
    ns = int(groups["solid"].size)
    nu = int(groups["undercooled"].size)
    tot = ns + nu
    # share_text: 14 solids in 2995 read "<1%", never a "0%" beside the 14
    rows = [[("SOLID", share_text(ns, tot), f"{ns}", STAT_SOLID_COLOR),
             ("UNDERCOOLED", share_text(nu, tot), f"{nu}",
              CLASS_COLORS["undercooled"]),
             ("MEAN", *mean, None)],
            drow]
    return rows

