"""Re-derive the probability-space thresholds from the user's own labels.

WHY THIS EXISTS: those thresholds are cuts through a trained network's output,
so every retrain moves the thing they are cutting. Hand-tuning them once, as the
app did until 2026-08-04, means they are correct for exactly one model and then
quietly drift — which is how the second microscope ended up with a solid/liquid
gate wanting 0.88 while the first wanted 0.54 (see thresholds.py).

WHAT IT COSTS: nothing to speak of. Every input it needs is already stored on
the particles from the last analysis — P(solid) in `facet_frac`, the pattern
softmax in `pattern_probs` — so there is no inference to run, only counting.

HOW IT STAYS HONEST, three rules:

  * STALE ANALYSES ARE REFUSED. A particle's facet_frac came from whichever
    model analysed that photo. Fitting a threshold for the CURRENT model against
    an OLD model's numbers is the exact mistake this module exists to prevent,
    so photos whose pipeline stamp is not the current one are dropped, and if
    too few survive the calibration does not run at all.
  * THE SCORE IS OUT-OF-PHOTO. Each candidate value is judged leave-one-photo-
    out: fitted on the other photos, scored on the held-out one. A threshold
    fitted and scored on the same particles always looks better than it is.
  * A NEW VALUE HAS TO WIN. If the shipped value scores as well out-of-photo,
    it is kept. Movement is not improvement.

The criterion is BALANCED accuracy, not plain accuracy: the new instrument's
photos run 80-95% liquid, and a plain-accuracy fit on those would buy its score
by giving up every solid particle there is.
"""
from __future__ import annotations

import numpy as np

import thresholds

SOLID_CLASSES = ("janus", "stripe", "lamellar", "composite")
MIN_PHOTOS = 4
MIN_PARTICLES = 300
GRID = np.round(np.linspace(0.05, 0.95, 91), 3)


def _balanced(pred, y):
    """Mean of the two per-class accuracies. Undefined if a class is missing."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())


def _fit_cut(p, y, grid=GRID):
    """The cut that maximises balanced accuracy on these particles."""
    best, bt = -1.0, None
    for t in grid:
        s = _balanced((p >= t).astype(int), y)
        if s is not None and s > best:
            best, bt = s, float(t)
    return bt, best


def _lopo(p, y, photo, current):
    """(score of a fitted cut, score of `current`) — both out-of-photo."""
    photos = sorted(set(photo.tolist()))
    if len(photos) < MIN_PHOTOS:
        return None
    fit_s, cur_s, n = 0.0, 0.0, 0
    for ph in photos:
        te = photo == ph
        if not te.any() or te.all():
            continue
        t, _ = _fit_cut(p[~te], y[~te])
        if t is None:
            continue
        s_new = _balanced((p[te] >= t).astype(int), y[te])
        s_cur = _balanced((p[te] >= current).astype(int), y[te])
        if s_new is None or s_cur is None:
            continue                       # a one-class photo scores nothing
        w = int(te.sum())
        fit_s += s_new * w; cur_s += s_cur * w; n += w
    if not n:
        return None
    return fit_s / n, cur_s / n


def collect(results, labels_of, stamp):
    """Gather (P(solid), pattern probs, size, user class, photo) per labelled
    particle from FRESH analyses only.

    `labels_of(path)` returns {particle_id: class} for a photo — the user's
    confirmed labels; `stamp` is analyze.pipeline_stamp().
    """
    rows = []
    for path, a in (results or {}).items():
        if getattr(a, "pipeline", "") != stamp or not getattr(a, "classifiable", False):
            continue
        truth = labels_of(path) or {}
        if not truth:
            continue
        for p in a.particles:
            c = truth.get(p.id)
            if c not in SOLID_CLASSES and c != "undercooled":
                continue
            rows.append(dict(photo=path, cls=c, ff=float(p.facet_frac),
                             diam_nm=float(p.diam_nm),
                             probs=tuple(getattr(p, "pattern_probs", ()) or ())))
    return rows


def fit(rows):
    """Fit every calibratable threshold. Returns (values, report).

    `values` holds only the ones that actually beat what is in force; `report`
    explains each decision, including the ones that were left alone.
    """
    import analyze

    report, values = [], {}
    if not rows:
        return values, ["No fresh analyses with labels — nothing to calibrate."]
    photo = np.array([r["photo"] for r in rows])
    n_ph = len(set(photo.tolist()))
    if len(rows) < MIN_PARTICLES or n_ph < MIN_PHOTOS:
        return values, [f"Not enough fresh labelled data yet ({len(rows)} "
                        f"particles over {n_ph} photos; need {MIN_PARTICLES} "
                        f"over {MIN_PHOTOS})."]

    ff = np.array([r["ff"] for r in rows], float)
    solid = np.array([1 if r["cls"] in SOLID_CLASSES else 0 for r in rows])

    def consider(name, p, y, ph, label):
        cur = thresholds.get(name)
        sc = _lopo(p, y, ph, cur)
        if sc is None:
            report.append(f"{label}: not enough spread to judge — kept {cur:.2f}")
            return
        new_s, cur_s = sc
        t, _ = _fit_cut(p, y)
        if t is None:
            return
        if new_s > cur_s + 0.002:          # a real gain, not a rounding wobble
            values[name] = t
            report.append(f"{label}: {cur:.2f} → {t:.2f}  "
                          f"(balanced accuracy {cur_s:.3f} → {new_s:.3f}, "
                          f"leave-one-photo-out, n={len(y)})")
        else:
            report.append(f"{label}: kept {cur:.2f} — refitting would not beat it "
                          f"({cur_s:.3f} vs {new_s:.3f})")

    # 1. the solid / liquid cut, over every labelled particle
    consider("facet_thresh", ff, solid, photo, "Solid / undercooled cut")

    # 2. the extra solidity a JANUS call needs. Judged only on the particles the
    #    pattern step would actually name janus — that is the population the
    #    rule governs, and scoring it on the rest would drown the signal.
    jan, jy, jp = [], [], []
    for r, f in zip(rows, ff):
        if not r["probs"] or len(r["probs"]) < 4:
            continue
        if analyze.size_aware_pattern(r["diam_nm"], r["probs"]) != "janus":
            continue
        jan.append(f); jy.append(1 if r["cls"] == "janus" else 0); jp.append(r["photo"])
    if len(jan) >= 100:
        consider("janus_min_solid", np.array(jan), np.array(jy), np.array(jp),
                 "Janus solidity floor")
    else:
        report.append(f"Janus solidity floor: kept "
                      f"{thresholds.get('janus_min_solid'):.2f} — only {len(jan)} "
                      f"janus calls to judge it on")

    # 3. "no pattern could be named, but the gate is sure it is solid". Governs
    #    only the particles in that corner, so it is fitted there.
    npc, npy, npp = [], [], []
    for r, f in zip(rows, ff):
        if not r["probs"] or len(r["probs"]) < 4:
            continue
        if analyze.size_aware_pattern(r["diam_nm"], r["probs"]):
            continue                       # a pattern WAS named — different rule
        npc.append(f); npy.append(1 if r["cls"] in SOLID_CLASSES else 0)
        npp.append(r["photo"])
    if len(npc) >= 100:
        consider("nopattern_solid_conf", np.array(npc), np.array(npy),
                 np.array(npp), "Solid-without-pattern bar")
    else:
        report.append(f"Solid-without-pattern bar: kept "
                      f"{thresholds.get('nopattern_solid_conf'):.2f} — only "
                      f"{len(npc)} particles in that corner")

    # 4. the small-particle confidence bars, each on the small particles whose
    #    class it decides
    for name, cls, idx, label in (
            ("stripe_small_min_conf", "stripe", 1, "Small-particle stripe bar"),
            ("janus_small_min_conf", "janus", 0, "Small-particle janus bar")):
        pv, yv, ph = [], [], []
        for r in rows:
            if not r["probs"] or len(r["probs"]) < 4:
                continue
            if r["diam_nm"] >= analyze.PATTERN_SMALL_NM:
                continue
            pv.append(float(r["probs"][idx]))
            yv.append(1 if r["cls"] == cls else 0)
            ph.append(r["photo"])
        if len(pv) >= 100 and 0 < sum(yv) < len(yv):
            consider(name, np.array(pv), np.array(yv), np.array(ph), label)
        else:
            report.append(f"{label}: kept {thresholds.get(name):.2f} — "
                          f"{len(pv)} small particles, not enough to refit")
    return values, report


def run(results, labels_of, stamp):
    """Fit and store. Returns (values_written, report_lines)."""
    rows = collect(results, labels_of, stamp)
    values, report = fit(rows)
    if values:
        thresholds.save(values, meta=dict(
            particles=len(rows), photos=len(set(r["photo"] for r in rows)),
            pipeline=stamp))
        import analyze
        analyze.reload_thresholds()
    return values, report
