"""Writing the results to disk: which files an export produces and what they
are called.

The naming carries meaning — an exported file has to be identifiable months
later from its name alone, so it keeps the experiment title, says which view it
is, and (for a per-image export) which image it came from.

Charts over a multi-image selection are POOLED into one figure rather than
written per image: the user selects several photos to get a combined
distribution, and writing eight near-identical charts was never what that meant.
"""
from __future__ import annotations

import os
import re

from overlay_draw import render, pick_spread
from chart_data import class_diams
from charts import (render_report, render_cumulative, render_solid_split,
                    render_pattern_size)

# export type -> filename keyword
TYPE_WORDS = {
    "line": "line",                # measurement lines only
    "undercooled": "undercooled",  # solid/undercooled colouring
    "pattern": "pattern",          # per-pattern (janus/stripe/…) colouring
}


def _image_no(analysis):
    """'.3' — which image of the experiment this is, or '' when it has no
    number. It is what keeps eight per-image exports apart."""
    title = analysis.title()
    stem = os.path.splitext(analysis.image)[0]
    suffix = stem[len(title):] if stem.startswith(title) else stem
    m = re.search(r"(\d+)", suffix)
    return f".{m.group(1)}" if m else ""


def _export_stem(analysis, typeword):
    """'UÖ - 15 Karışık 3'  +  'line'  ->  'UÖ - 15 Karışık line.3'."""
    return f"{analysis.title()} {typeword}{_image_no(analysis)}"


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


def _free_name(out_dir, name, taken):
    """`name`, or the first ' 2', ' 3', … variant of it that is not on disk and
    not already claimed by this same export.

    An export NEVER overwrites. Saving the stripe chart and then the janus one
    used to put the second file on top of the first — same view, same auto-name —
    and the first was simply gone. Silently destroying a file the user asked us
    to write is the worst thing this function could do, so a collision costs a
    suffix instead."""
    stem, ext = os.path.splitext(name)
    i, cand = 1, name
    while cand in taken or os.path.exists(os.path.join(out_dir, cand)):
        i += 1
        cand = f"{stem} {i}{ext}"
    taken.add(cand)
    return cand


def export(analyses, out_dir, types, facet_thresh=None, chosen_map=None, k=5,
           size_range=None, cls_filters=None, title=None):
    """Export straight into `out_dir` — the folder the user picked in the
    dialog, not a new sub-folder inside it. A sub-folder buried the files one
    level below where the user just told us to put them, and named it after the
    experiment title rather than anything they chose.

    analyses    : list of Analysis
    types       : SEM-image variants, a subset of {'line','undercooled','pattern'};
                  written once per selected image
    cls_filters : which CHARTS to write — "all" is the whole distribution, a class
                  key ("janus", …) that class alone, "patternsize" the Pattern ×
                  Size composition chart, "cumulative" the CDF with its D-values,
                  "solidsplit" the stacked solid/undercooled histogram.
                  One file each.
    title       : optional caption drawn above every chart, AND the name the
                  files are given — the user named this export, so that is what
                  it should be called on disk. When the export writes several
                  files they keep their distinguishing word after the name
                  ("My run pattern x size"), since one name cannot identify five
                  different views.
    size_range  : optional (lo, hi) nm range echoed into the charts

    Nothing is ever overwritten: a name already on disk gets ' 2', ' 3', …

    Charts are always POOLED: with several images selected you get ONE chart per
    entry, over their combined particles — never one chart per image.
    """
    cls_filters = list(cls_filters or [])
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    taken = set()
    chosen_map = chosen_map or {}

    # an upper bound on what this run writes; only a lone file can carry the
    # user's name and nothing else
    n_planned = len(analyses) * len(types) + len(cls_filters)
    named = _safe(title) if title else None

    def save(img, auto_name, word):
        """Write one output under the user's name when they gave one."""
        if named:
            base = named if n_planned == 1 else f"{named} {word}"
            auto_name = _safe(base) + ".png"
        name = _free_name(out_dir, auto_name, taken)
        p = os.path.join(out_dir, name)
        img.save(p)
        outputs[name] = p

    for a in analyses:
        if facet_thresh is not None:
            a.reclassify(facet_thresh)
        chosen = chosen_map.get(getattr(a, "path", None)) or pick_spread(a.particles, k)
        for typ in types:
            img = render_variant(a, typ, chosen=chosen, k=k)
            word = TYPE_WORDS[typ]
            # the image number rides along with the word: several images under
            # one user name must not all be called the same thing
            save(img, _export_stem(a, word) + ".png", word + _image_no(a))

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
        elif cf == "cumulative":
            try:
                img = render_cumulative(target, aspect=6.8 / 6.0, title=title,
                                        stats=True)
            except ValueError:
                continue                     # nothing measured in this selection
            word = "cumulative"
        elif cf == "solidsplit":
            try:
                img = render_solid_split(target, aspect=6.8 / 6.0,
                                         size_range=size_range, title=title,
                                         stats=True)
            except ValueError:
                continue                     # only one state here — nothing to split
            word = "solid vs liquid"
        else:
            cf_key = None if cf in (None, "all") else cf
            # a class the selection doesn't contain would silently fall back to
            # the whole distribution and write a duplicate file — skip it
            if cf_key is not None and cf_key not in present:
                continue
            img = render_report(target, size_range=size_range, cls_filter=cf_key,
                                stats=True, title=title)
            word = "results" if cf_key is None else f"results {cf_key}"
        save(img, (f"{stem} {word}.png" if stem
                   else _export_stem(analyses[0], word) + ".png"), word)
    return outputs
