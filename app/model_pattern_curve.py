"""Data saturation: does the pattern model still get better when you label more?

The question this answers is the one that decides where the next hour of work
should go. If accuracy is still climbing with dataset size, more labelled photos
are the cheapest win available. If it has flattened, more of the same data buys
nothing and the lever moves to the model itself (a bigger backbone, higher
resolution, a different architecture) or to the labels' ceiling.

METHOD — a learning curve, measured honestly:

  * One fold of PHOTOS is held out as a fixed test set and never trained on, so
    every point on the curve is scored against the same particles. Grouping by
    photo (not by particle) is what keeps a crop from being tested against its
    own siblings, which would flatter every point equally and hide the shape.
  * The remaining photos are the pool. Subsets of the pool — 20/40/60/80/100% of
    its PHOTOS, again never split within a photo — each train a model from
    scratch and are scored on the held-out fold.
  * The small subsets are measured twice with different random photos and
    averaged: with few photos, WHICH photos you drew matters more than how many,
    and a single draw there can invent a slope that is not real.

WHAT THE CURVE IS NOT: these models are trained for `CURVE_EPOCHS` epochs, fewer
than the real one, and on at most two thirds of the data. Their absolute
accuracy therefore sits BELOW the number "Train model" reports — that is
expected and does not mean the model got worse. The curve is read for its SHAPE.
The full cross-validated accuracy is carried in the same record and drawn as a
separate marker, so the offset is visible instead of being quietly ignored.

COST: proportional to the data, like everything here. The whole curve costs
about 3.6 pool-sized training passes of `CURVE_EPOCHS` epochs — measured at
6049 particles / 36 photos: ~12 minutes on MPS, against ~30 for the training run
it rides along with. That is why it is a checkbox and not unconditional.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import paths
import training_store
from training_store import CLASSES

CURVE_EPOCHS = 14
# (fraction of the training pool's photos, how many random draws to average)
FRACTIONS = ((0.2, 2), (0.4, 2), (0.6, 1), (0.8, 1), (1.0, 1))
HISTORY_KEEP = 40


def instrument_of(stem):
    """Which microscope a training photo came from.

    The two machines in use write different info bars, and the app already reads
    that: BIOMATEN stamps a detector nameplate ("CBS"), the METU-METE exports
    write none. The name is stored in the photo's labels.json when it is
    confirmed, so this costs a small JSON read and no image work.

    This matters here more than anywhere else in the app. A learning curve that
    mixes two instruments answers the wrong question: adding photos from a
    machine the model has barely seen can leave accuracy flat — or push it down —
    while the data is still far from saturated. That looks exactly like a
    plateau on an undifferentiated curve, and the conclusion drawn from it
    ("stop labelling, buy a bigger model") would be the opposite of right.
    """
    try:
        p = os.path.join(training_store.train_dir(), stem + ".labels.json")
        with open(p, "r", encoding="utf-8") as fh:
            det = (json.load(fh).get("detector") or "").strip()
        return det or "no nameplate"
    except (OSError, ValueError):
        return "unknown"


def instruments_for(stems):
    """{stem: instrument} for a list of photo stems."""
    return {s: instrument_of(s) for s in stems}


def _stratified(order, groups, take):
    """Pick `take(k)` photos from each instrument group, preserving the mix.

    Every subset on the curve holds the same proportion of each machine as the
    whole set does, so a point moving up or down is about HOW MUCH data there
    is, not about which machine happened to be drawn.
    """
    out = []
    for g in sorted(set(groups.values())):        # by MACHINE, not by photo
        members = [s for s in order if groups.get(s) == g]
        out.extend(take(members))
    return out


def _cost_units():
    """Training cost is linear in the number of particles, so the fractions
    themselves are the cost weights — used to make the progress bar move at a
    roughly even rate instead of crawling through the big subsets."""
    return sum(f * r for f, r in FRACTIONS)


def measure(X, y, keys, device, epochs=CURVE_EPOCHS, progress=None):
    """Run the learning curve. Returns the record dict (without the fit).

    `progress(done, total)` is called in cost units (see _cost_units), not
    epochs: an epoch over 20% of the pool is a fifth of the work of a full one.
    """
    import model_pattern_training as T

    stems = np.array(sorted(set(keys.tolist())))
    if len(stems) < 6:
        raise RuntimeError(
            f"Data saturation needs at least 6 training photos ({len(stems)} so far).")

    rng = np.random.default_rng(0)
    order = list(rng.permutation(stems))
    groups = instruments_for(order)              # stem -> instrument
    # hold out every 3rd photo OF EACH MACHINE, so the test set carries both
    hold = set(_stratified(order, groups, lambda m: m[::3]))
    pool = [s for s in order if s not in hold]
    test_idx = np.where(np.isin(keys, list(hold)))[0]
    pool_rows = np.where(np.isin(keys, pool))[0]
    if not len(test_idx) or len(pool) < 3:
        raise RuntimeError("Not enough photos to hold a test set aside.")
    test_inst = np.array([groups.get(k, "unknown") for k in keys[test_idx]])

    total = _cost_units() * epochs
    done = 0.0

    points = []
    for frac, reps in FRACTIONS:
        for rep in range(reps if frac < 1.0 else 1):
            if frac >= 1.0:
                sub = list(pool)
            else:
                r = np.random.default_rng(100 * rep + int(frac * 100))
                sub = _stratified(pool, groups, lambda m: list(
                    r.choice(m, max(1, int(round(frac * len(m)))), replace=False)))
            tr = pool_rows[np.isin(keys[pool_rows], sub)]
            if len(tr) < 20:
                continue
            base = done

            def cb(ep, base=base, frac=frac):
                if progress:
                    progress(base + ep * frac, total)

            net = T._train(X, y, tr, epochs, device, cb)
            yp = T._predict(net, X[test_idx], device).argmax(1)
            yt = y[test_idx]
            # …and how it does on the data it was just trained ON. One extra
            # forward pass, and it separates the two reasons a curve can be low.
            # Near-perfect here and poor on the held-out set = variance, which
            # more data fixes. Poor in BOTH = the model is underfitting, and no
            # amount of labelling fixes that — that is the case where a bigger
            # backbone is the right answer.
            fit_idx = tr if len(tr) <= 3000 else np.random.default_rng(1).choice(
                tr, 3000, replace=False)
            yp_tr = T._predict(net, X[fit_idx], device).argmax(1)
            acc_train = float((y[fit_idx] == yp_tr).mean())
            rec = {CLASSES[c]: float((yp[yt == c] == c).mean())
                   for c in range(len(CLASSES)) if (yt == c).any()}
            # the same score, machine by machine: the number that tells a
            # domain shift apart from a plateau
            per_inst = {}
            for g in sorted(set(test_inst.tolist())):
                m = test_inst == g
                per_inst[g] = dict(acc=float((yt[m] == yp[m]).mean()),
                                   n=int(m.sum()))
            sub_inst = {}
            for g in sorted(set(groups.values())):
                sub_inst[g] = int(sum(1 for s in sub if groups.get(s) == g))
            points.append(dict(
                frac=float(frac), rep=rep, n_train=int(len(tr)),
                photos_train=int(len(sub)), acc=float((yt == yp).mean()),
                macro_recall=float(np.mean(list(rec.values()))) if rec else None,
                recalls=rec, acc_train=acc_train,
                by_instrument=per_inst, photos_by_instrument=sub_inst))
            done = base + epochs * frac
            del net
    if progress:
        progress(total, total)

    inst_photos = {}
    for s in stems.tolist():
        g = groups.get(s, "unknown")
        inst_photos[g] = inst_photos.get(g, 0) + 1
    return dict(
        ts=time.time(), epochs=int(epochs),
        n_total=int(len(y)), n_photos=int(len(stems)),
        holdout=dict(photos=int(len(hold)), particles=int(len(test_idx)),
                     by_instrument={g: int((test_inst == g).sum())
                                    for g in sorted(set(test_inst.tolist()))}),
        pool=dict(photos=int(len(pool)), particles=int(len(pool_rows))),
        instruments=inst_photos,
        counts={CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))},
        points=points)


# ---------------------------------------------------------------- the fit

def _agg(points, key="acc"):
    """Repeats of the same fraction collapse to their mean, keeping the spread
    so the chart can show how much the draw mattered."""
    out = {}
    for p in points:
        v = p.get(key)
        if v is None:
            continue
        out.setdefault(p["frac"], []).append((p["n_train"], v))
    rows = []
    for frac in sorted(out):
        ns, vs = zip(*out[frac])
        rows.append(dict(frac=frac, n=float(np.mean(ns)), y=float(np.mean(vs)),
                         lo=float(min(vs)), hi=float(max(vs)), reps=len(vs)))
    return rows


def _agg_instrument(points, instrument):
    """The same aggregation, but scoring only the held-out particles from one
    microscope. Same models, same x-axis — only the test subset differs."""
    out = {}
    for p in points:
        v = (p.get("by_instrument") or {}).get(instrument)
        if not v or not v.get("n"):
            continue
        out.setdefault(p["frac"], []).append((p["n_train"], v["acc"]))
    rows = []
    for frac in sorted(out):
        ns, vs = zip(*out[frac])
        rows.append(dict(frac=frac, n=float(np.mean(ns)), y=float(np.mean(vs)),
                         reps=len(vs)))
    return rows


def fit(points, key="acc"):
    """Fit the standard saturating power law  y(n) = a - b * n**(-c).

    `a` is the ceiling this data+model would reach with infinite examples of the
    same kind; `c` says how fast it gets there. Solved by scanning `c` and
    solving the (then linear) least squares for a and b — a dependency-free and
    well-behaved way to do it, where a general optimiser on three coupled
    parameters likes to wander off with five points to chew on.

    Returns None when there is nothing honest to fit (fewer than three
    fractions, or no shape at all in them).
    """
    rows = _agg(points, key)
    if len(rows) < 3:
        return None
    n = np.array([r["n"] for r in rows], float)
    yv = np.array([r["y"] for r in rows], float)
    best = None
    for c in np.linspace(0.05, 2.0, 196):
        A = np.column_stack([np.ones_like(n), -n ** (-c)])
        try:
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b = float(coef[0]), float(coef[1])
        if b <= 0 or a <= yv.max() - 1e-9 or a > 1.0:
            continue                      # a curve that falls, or a silly ceiling
        sse = float(((A @ coef - yv) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, a, b, float(c))
    if best is None:
        return None
    sse, a, b, c = best
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    # A curve still on its steep stretch does not CONTAIN its own ceiling: the
    # power law will happily place it at 100%, which is not a finding, it is the
    # bound the search ran into. Say so rather than printing a number that would
    # be read as a prediction. The near-term slope (what a doubling buys) stays
    # trustworthy either way — it is interpolation, not extrapolation to
    # infinity — and that is what the verdict leads with.
    return dict(a=a, b=b, c=c, sse=sse,
                ceiling_identified=bool(a < 0.99),
                r2=float(1 - sse / ss_tot) if ss_tot > 0 else None,
                n_min=float(n.min()), n_max=float(n.max()))


def project(f, n):
    """The fitted accuracy at a dataset size of `n` training particles."""
    if not f or n <= 0:
        return None
    return float(f["a"] - f["b"] * n ** (-f["c"]))


def instrument_note(record, gap_pts=5.0):
    """The caveat that has to travel with the curve when two microscopes are in
    the set: is one of them dragging the whole thing down?

    Written as a measurement, not a warning — the numbers are the same held-out
    accuracy, split by machine, at the largest training size measured. A machine
    that scores far below the other with only a handful of photos behind it is
    NOT evidence of saturation; its own curve has barely started.
    """
    rows = {}
    for p in record.get("points", []):
        for g, v in (p.get("by_instrument") or {}).items():
            if v.get("n"):
                rows.setdefault(g, []).append((p["n_train"], v["acc"], v["n"]))
    if len(rows) < 2:
        return None
    last = {g: max(v, key=lambda t: t[0]) for g, v in rows.items()}
    best = max(last, key=lambda g: last[g][1])
    worst = min(last, key=lambda g: last[g][1])
    gap = (last[best][1] - last[worst][1]) * 100
    photos = record.get("instruments") or {}

    def name(g):
        return "METU-METE" if g in ("no nameplate", "unknown") else g

    line = ("Per microscope at the largest training size: "
            + ", ".join(f"{name(g)} {last[g][1]:.0%} ({last[g][2]} particles, "
                        f"{photos.get(g, 0)} photos)" for g in sorted(last)))
    if gap >= gap_pts:
        line += (f". {name(worst)} is {gap:.0f} points behind — with only "
                 f"{photos.get(worst, 0)} photos from it, that is a gap in "
                 f"COVERAGE of that machine, not saturation. The mixed curve "
                 f"above will read flatter than it should while those photos "
                 f"are being added; labelling more of them is the fix, and the "
                 f"lower strip is where it will show up first.")
    return dict(gap=gap, weakest=worst, text=line,
                per_instrument={g: dict(acc=last[g][1], n=last[g][2],
                                        photos=photos.get(g, 0))
                                for g in last})


def capacity_note(record, underfit=0.90, gap_pts=15.0):
    """Is the MODEL the limit, or the data?

    Read off the largest point measured: how well it fits the data it trained on
    versus how well it generalises. Two different diagnoses come out of it and
    they lead opposite ways, which is why the saturation window says both:

      * fits its own data badly (< ~90 %) — underfitting. resnet18 at this
        resolution cannot even memorise what it was given, and more labels will
        not change that. Bigger backbone / larger crops is the lever.
      * fits it nearly perfectly, generalises far worse — variance. The model
        has capacity to spare and is using it to memorise; more data (or
        stronger augmentation) is the lever, not a bigger model.
    """
    rows = [p for p in record.get("points", []) if p.get("acc_train") is not None]
    if not rows:
        return None
    import model_pattern_training as T          # for the real run's epoch count
    p = max(rows, key=lambda q: q["n_train"])
    tr, te = p["acc_train"], p["acc"]
    gap = (tr - te) * 100
    if tr < underfit:
        state = "underfit"
        text = (f"At the largest size measured the model gets {tr:.0%} on the "
                f"very data it trained on, against {te:.0%} held out — it is not "
                f"memorising, it cannot even fit what it already has. Two things "
                f"produce that and they are worth separating before acting: the "
                f"curve's models are trained for {record.get('epochs')} epochs "
                f"against the real run's {T.FINAL_EPOCHS}, so part of it may be "
                f"under-training rather than too little model; and labels that "
                f"contradict each other (the same texture called two things on "
                f"two photos) cannot be fitted by any model at all. If neither "
                f"explains it, a bigger backbone or larger crops is the lever.")
    elif gap >= gap_pts:
        state = "variance"
        text = (f"At the largest size measured: {tr:.0%} on its own training "
                f"data against {te:.0%} held out, a {gap:.0f}-point gap. The "
                f"model has capacity to spare and is spending it on memorising; "
                f"more data or stronger augmentation moves this, a bigger model "
                f"would mostly widen the gap.")
    else:
        state = "balanced"
        text = (f"At the largest size measured: {tr:.0%} on its own training "
                f"data against {te:.0%} held out — a {gap:.0f}-point gap, which "
                f"is a model neither starved of capacity nor memorising.")
    return dict(state=state, acc_train=tr, acc_test=te, gap=gap, text=text)


def verdict(record, f=None):
    """Turn the fit into the sentence the user actually asked for: are we on the
    climbing part of the curve or the flat part?

    The measure is the honest one for "should I label more photos": how much
    accuracy a DOUBLING of the dataset is predicted to buy. It is scale-free —
    unlike a slope in points-per-particle, it means the same thing at 500
    particles and at 50000 — and doubling is roughly the effort the user would
    actually be signing up for.
    """
    f = f or fit(record["points"])
    rows = _agg(record["points"])
    if not f or not rows:
        return dict(state="unknown", text=(
            "Not enough points to fit a curve yet — train once more after "
            "adding photos and this will fill in."))
    n_now = float(record["n_total"])             # what the real model trains on
    here = project(f, n_now)
    dbl = project(f, 2 * n_now)
    gain = (dbl - here) * 100 if here is not None and dbl is not None else None
    head = (f["a"] - here) * 100 if here is not None else None

    if gain is None:
        state = "unknown"
    elif gain >= 3.0:
        state = "climbing"
    elif gain >= 1.0:
        state = "flattening"
    else:
        state = "plateau"

    msg = {
        "climbing": ("Still climbing — doubling the labelled set is predicted to "
                     "add about {gain:.1f} points. More photos is the cheapest "
                     "win available; the model is not the bottleneck yet."),
        "flattening": ("Flattening — doubling the labelled set is predicted to add "
                       "only about {gain:.1f} points. Still worth labelling, but "
                       "this is where a bigger model starts to be worth trying "
                       "alongside it."),
        "plateau": ("Plateau — doubling the labelled set is predicted to add about "
                    "{gain:.1f} points, i.e. nothing you would notice. More of the "
                    "same data will not move this; the lever is now the model "
                    "(a larger backbone / higher resolution) or the ceiling of "
                    "the labels themselves."),
    }.get(state, "")
    note = instrument_note(record)
    cap = capacity_note(record)
    text = msg.format(gain=gain) if msg else ""
    if state == "climbing" and not f.get("ceiling_identified"):
        text += (" How far it can go is not yet answerable from this curve — "
                 "the measured points are all on its steep stretch, so any "
                 "ceiling fitted to them is guesswork. The next measurement, "
                 "after more photos, is what narrows it.")
    extra = "\n\n".join(x["text"] for x in (cap, note) if x)
    return dict(state=state, gain_doubling=gain, headroom=head,
                here=here, ceiling=f["a"], instrument=note, capacity=cap,
                text=text, text_full=text + (("\n\n" + extra) if extra else ""))


# ------------------------------------------------------------- persistence

def history_path():
    return os.path.join(paths.data_dir(), "saturation.json")


def load_history():
    """Every saturation run ever measured, oldest first. Never raises: a broken
    file must not take the panel down with it."""
    try:
        with open(history_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("runs", []) if isinstance(data, dict) else []
    except (OSError, ValueError):
        return []


def append_run(record):
    """Store a finished run. The history is the second, slower curve: one point
    per retrain, real full-CV accuracy against the real dataset size, accumulated
    over months. It costs nothing to keep and it is the only measurement here
    that is not a model of the shape but the shape itself."""
    runs = load_history()
    runs.append(record)
    runs = runs[-HISTORY_KEEP:]
    tmp = history_path() + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(runs=runs), fh)
        os.replace(tmp, history_path())
    except OSError:
        pass
    return runs


def latest():
    runs = load_history()
    return runs[-1] if runs else None


def run_saturation(progress=None, epochs=CURVE_EPOCHS):
    """Standalone measurement: load the training folder, featurize, measure the
    curve, store it. Used by the "Measure now" button, where no retraining is
    wanted — the curve says nothing about the weights, only about the data."""
    import torch
    import training_store
    import model_pattern_crops
    import model_pattern_training as T

    imgs, sils, dnm, y, keys = training_store.load_pattern_raw()
    X = model_pattern_crops.featurize_batch(imgs, sils, dnm, size=T.SIZE)
    device = model_pattern_crops.torch_device()
    torch.manual_seed(0)
    np.random.seed(0)
    rec = measure(X, y, keys, device, epochs=epochs, progress=progress)
    rec["device"] = device
    append_run(rec)
    return rec
