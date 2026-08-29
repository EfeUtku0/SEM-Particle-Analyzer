"""Score the trained model on the golden set, and keep the history of scores.

HOW IT SCORES. It runs the app's real pipeline on each golden photo —
`analyze.analyze_image` — and compares the answer the user would see on screen
with the label the user placed. Not the network's raw argmax: the solid/liquid
gate, the size prior, the janus solidity floor, the brightness cut and the
measurement gate all run, because a pattern the pipeline never lets through is
not a pattern the user gets. Duplicating that rule chain here to save a few
seconds would mean two copies of it that drift apart, and the copy in this file
would be the one nobody notices is stale.

WHICH PARTICLES COUNT. Only the ones `analyze.measurable()` passes — the same
population every number in the app is computed over. A particle the app
deliberately refuses to measure counting as a miss would report an error the
user never sees (the question that prompted the measured/unmeasured split,
2026-07-31). The all-labelled figure is stored alongside so the gap stays
visible rather than being quietly chosen.

WHAT IT COSTS. One forward pass per golden photo, on cached Cellpose masks:
seconds each. Compare with the image-grouped cross-validation it replaces, which
retrained the model K times and cost three quarters of every training run.

MATCHING. Labels are matched to today's segmentation by CENTROID, never by
particle id: ids are assigned per segmentation run, so a photo re-analysed under
a newer pipeline renumbers every particle and an id-keyed match would silently
score the wrong particle. Anything further than `MATCH_PX` from a centroid is
reported as unmatched rather than guessed at.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import golden_store
from golden_store import PATTERN_CLASSES, NOPAT, REPORT_CLASSES

MATCH_PX = 3.0          # fallback radius when the label lands on no particle
HISTORY_KEEP = 40


def _history_path():
    from paths import data_dir
    return os.path.join(data_dir(), "training_runs.json")


# ----------------------------------------------------------------- metrics

def _metrics(yt, yp):
    """Per-class recall / precision / F1 plus the three headline numbers.

    Macro-F1 averages only over the classes the truth actually contains: a set
    with no composite in it must not be scored as though it had failed to find
    any, which would drag every model's number down by the same wrong amount and
    hide the differences between them.
    """
    yt, yp = np.asarray(yt), np.asarray(yp)
    per, f1s = {}, []
    for c in REPORT_CLASSES:
        t, p = yt == c, yp == c
        n = int(t.sum())
        if not n:
            per[c] = dict(n=0, recall=None, prec=None, f1=None)
            continue
        r = float((yp[t] == c).mean())
        pr = float((yt[p] == c).mean()) if p.any() else 0.0
        f1 = 2 * r * pr / (r + pr) if (r + pr) > 0 else 0.0
        per[c] = dict(n=n, recall=r, prec=pr, f1=f1)
        f1s.append(f1)
    solid_t, solid_p = yt != NOPAT, yp != NOPAT
    return dict(n=int(len(yt)),
                acc=float((yt == yp).mean()) if len(yt) else None,
                macro_f1=float(np.mean(f1s)) if f1s else None,
                solid_acc=float((solid_t == solid_p).mean()) if len(yt) else None,
                per_class=per,
                confusion=[[int(((yt == a) & (yp == b)).sum())
                            for b in REPORT_CLASSES] for a in REPORT_CLASSES])


# ------------------------------------------------------------------ scoring

def _answer(p):
    """What the app shows for this particle, as one of REPORT_CLASSES."""
    return p.pattern if p.pattern in PATTERN_CLASSES else NOPAT


def score_photo(stem, path):
    """(truth, pred, measurable, n_unmatched) for one golden photo.

    A label is matched to the particle whose MASK its centroid falls inside, and
    only falls back to the nearest centroid within `MATCH_PX` when it lands on
    background. Matching by distance alone looked fine and was not: a photo
    re-segmented under a newer pipeline shifts every centroid by a pixel or
    two, and a fixed radius then threw away 5% of the labels (measured: 211 of
    4299) — silently, since an unscored particle simply vanishes from the
    denominator. Containment does not drift with the boundary, and it cannot
    pick a neighbour.
    """
    import analyze
    a = analyze.analyze_image(path)
    parts = list(a.particles)
    if not parts:
        return [], [], [], len(golden_store.truth_of(stem))
    by_id = {p.id: p for p in parts}
    mask = a.label_mask
    h, w = (mask.shape if mask is not None else (0, 0))
    pts = np.array([[p.cx, p.cy] for p in parts], float)
    yt, yp, meas, miss = [], [], [], 0
    for _pid, cls, cx, cy in golden_store.truth_of(stem):
        p = None
        if h and 0 <= int(round(cy)) < h and 0 <= int(round(cx)) < w:
            p = by_id.get(int(mask[int(round(cy)), int(round(cx))]))
        if p is None:
            j = int(np.argmin(((pts - [cx, cy]) ** 2).sum(1)))
            if ((pts[j] - [cx, cy]) ** 2).sum() <= MATCH_PX ** 2:
                p = parts[j]
        if p is None:
            miss += 1
            continue
        yt.append(golden_store.report_class(cls))
        yp.append(_answer(p))
        meas.append(bool(analyze.measurable(p)))
    return yt, yp, meas, miss


def evaluate(progress=None):
    """Score the model currently in force over the whole golden set.

    Returns the `golden` block of a run record, or None when there is no golden
    set yet — in which case training still succeeds and the report says how to
    make one, because a missing ruler is not a failed training run.
    """
    photos = golden_store.photos()
    if not photos:
        return None
    t0 = time.time()
    rows = []                       # (instrument, truth, pred, measurable)
    per_photo, unmatched = [], 0
    for i, (stem, path, ins) in enumerate(photos):
        if progress:
            progress(i, len(photos), stem)
        try:
            yt, yp, meas, miss = score_photo(stem, path)
        except Exception as exc:    # one unreadable photo must not lose the rest
            per_photo.append(dict(stem=stem, instrument=ins, error=str(exc)))
            continue
        unmatched += miss
        rows += [(ins, t, p, m) for t, p, m in zip(yt, yp, meas)]
        sel = [k for k, m in enumerate(meas) if m]
        pm = _metrics([yt[k] for k in sel], [yp[k] for k in sel])
        # accuracy AS WELL as macro-F1 per photo: one photo often holds a class
        # only two or three times, and macro-F1 turns those into a 0 that swings
        # the photo's figure by twenty points. Accuracy is the stable one to
        # rank by; macro-F1 is the one that shows a class being missed at all.
        per_photo.append(dict(
            stem=stem, instrument=ins, n=len(yt), n_measured=len(sel),
            unmatched=miss, macro_f1=pm["macro_f1"], acc=pm["acc"]))
    if progress:
        progress(len(photos), len(photos), "")

    def block(sub):
        m_yt = [r[1] for r in sub if r[3]]
        m_yp = [r[2] for r in sub if r[3]]
        out = _metrics(m_yt, m_yp)
        out["all_labelled"] = _metrics([r[1] for r in sub], [r[2] for r in sub])
        return out

    instruments = sorted({r[0] for r in rows})
    return dict(
        photos=len(photos), particles=len(rows), unmatched=unmatched,
        secs=round(time.time() - t0, 1),
        combined=block(rows) if rows else None,
        by_instrument={ins: block([r for r in rows if r[0] == ins])
                       for ins in instruments},
        per_photo=per_photo,
        classes=list(REPORT_CLASSES))


# ------------------------------------------------------------------ history

def append_run(record):
    """Add a run to the history, newest last, oldest pruned. Never raises: the
    model is already saved by the time this is called and losing the bookkeeping
    must not turn a good training run into a failed one."""
    runs = load_history()
    runs.append(record)
    runs = runs[-HISTORY_KEEP:]
    try:
        p = _history_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(runs=runs), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except OSError:
        pass
    return runs


def load_history():
    try:
        with open(_history_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        return list(d.get("runs") or [])
    except (OSError, ValueError):
        return []


def latest():
    h = load_history()
    return h[-1] if h else None
