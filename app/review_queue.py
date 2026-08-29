"""Contact-sheet review: the cheap way to turn a model's guesses into ground truth.

Labelling a photo particle-by-particle on the micrograph costs one click per
particle, and a dense photo carries a thousand of them. But CONFIRMING is far
cheaper than labelling: forty crops laid out in a grid are taken in at a glance,
and only the handful that are wrong need a click. The same forty particles cost
minutes one at a time.

Two things make that trade honest rather than a shortcut:

* Every particle on a reviewed page is a label the user actually LOOKED at.
  That is the whole difference from `gui._train_effective`, which quietly writes
  the model's own answer as ground truth for particles nobody ever saw — 10% of
  the existing training set on the reference instrument, and far more dangerous
  on a new one where the model is wrong in a systematic direction (see
  [[sem-analyzer-multi-instrument]]: on METU-METE it calls almost everything
  janus). Reviewed labels carry no such bias because a human vetoed each one.

* Pages are ordered by the model's own UNCERTAINTY, least confident first, so
  the errors cluster at the front. The user can stop whenever the corrections
  dry up instead of grinding through pages of obviously-right crops.

This module holds the selection/ordering/crop logic with no Qt in it, so it can
be tested headless; gui.ReviewDialog draws it.
"""
from __future__ import annotations

import numpy as np

from analyze import SOLID_MIN_DIAM_NM, measurable

PATTERN_CLASSES = ["janus", "stripe", "lamellar", "composite"]


def skip_review(p) -> bool:
    """True when there is nothing for a human to decide about this particle.

    The review's whole value is the user's attention, so anything whose class
    was not actually the model's judgement must not spend it. Two cases, both
    the user's own rules (2026-08-04):

    * BELOW THE SIZE FLOOR. `SOLID_MIN_DIAM_NM` forces `is_solid = False`, so
      everything under 200 nm is undercooled by rule — the model never got a
      say. Confirming it confirms arithmetic. This is the big one: 11207 of the
      19531 particles in the undercooled queue, over a third of the whole
      review_queue.

    * NOT MEASURED — the particles the normal view paints slate BLUE. These are
      crescents, slivers and caps buried under a neighbour: not fully in view,
      so the app gives them neither a size nor a place in the statistics. The
      user's rule is simply that the two views agree — "bu review sayfasında da
      hiçbir mavi gözükmesin". `measurable` is the right and only gate for it,
      not `pattern_eligible`: the pattern gate is looser on every axis, so it
      leaves blue particles in the queue, which is what the first attempt at
      this filter got wrong.

    Note the size floor is checked FIRST and independently: a sub-200 nm
    particle is usually measured perfectly well, so measurability says nothing
    about whether its class was decided by rule.
    """
    if float(getattr(p, "diam_nm", 0.0)) < SOLID_MIN_DIAM_NM:
        return True
    return not measurable(p)


def particle_class(p) -> str:
    """The class the model settled on, in the vocabulary the user labels in.

    "" means the model reached no opinion worth reviewing — excluded (too dim
    to read), hand-marked unclassified, or decided by rule rather than by the
    model (see `skip_review`). Those are not offered for review: confirming "I
    also don't know", or confirming a size threshold, teaches nothing.
    """
    if getattr(p, "excluded", False) or getattr(p, "unclassified", False):
        return ""
    if skip_review(p):
        return ""
    if not p.is_solid:
        return "undercooled"
    return p.pattern or "solid"


def confidence(p, facet_thresh: float = 0.50) -> float:
    """How sure the model is about THIS particle's class, in [0, 1].

    Two different questions decide a particle, so two different margins:

    * For a pattern call, the softmax gap between the top two classes. A janus
      at 0.40 against a stripe at 0.38 is a coin flip however high the winner
      looks on its own.
    * For solid-vs-undercooled, the distance of P(solid) from the threshold it
      was compared against. A particle at 0.51 is a coin flip too, and no
      pattern softmax will say so.

    A particle that had to clear both gates is only as certain as its weaker
    one, so take the minimum.
    """
    ff = float(getattr(p, "facet_frac", 0.0))
    gate = min(1.0, abs(ff - facet_thresh) / max(facet_thresh, 1e-6))
    probs = tuple(getattr(p, "pattern_probs", ()) or ())
    if not p.is_solid or not p.pattern or len(probs) < 2:
        return gate
    s = sorted(probs, reverse=True)
    return min(gate, float(s[0] - s[1]))


def collect(analyses, cls, facet_thresh=0.50, skip=None):
    """Every particle the model called `cls`, least-confident first.

    `analyses` is {path: Analysis}. `skip` is {path: {pid}} already labelled by
    hand — those are ground truth already and must not be offered again (nor
    silently overwritten by a page accept).
    """
    skip = skip or {}
    out = []
    for path, a in analyses.items():
        if not getattr(a, "classifiable", False):
            continue
        done = skip.get(path) or set()
        for p in a.particles:
            if p.id in done or particle_class(p) != cls:
                continue
            out.append(dict(path=path, pid=p.id, cls=cls,
                            conf=confidence(p, facet_thresh),
                            diam_nm=float(p.diam_nm)))
    out.sort(key=lambda r: r["conf"])
    return out


def crop_rgb(analysis, pid, size=132, pad=1.55):
    """An RGB crop centred on one particle, with the others dimmed.

    A packed field is ambiguous — the tile has to say WHICH particle is being
    asked about, or the user confirms the wrong thing. Everything outside the
    target's own mask is darkened, which reads instantly and, unlike an outline,
    does not hide the particle's rim (the rim is exactly where janus and stripe
    are told apart).
    """
    g = analysis.micrograph
    m = analysis.label_mask
    if g is None or m is None:
        return None
    p = next((q for q in analysis.particles if q.id == pid), None)
    if p is None:
        return None
    side = max(float(getattr(p, "major_px", 0.0)), float(p.diam_px), 6.0) * pad
    half = max(int(round(side / 2)), 6)
    cy, cx = int(round(p.cy)), int(round(p.cx))
    h, w = g.shape
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    sub = g[y0:y1, x0:x1].astype(np.float32)
    tgt = (m[y0:y1, x0:x1] == pid)
    # Grey levels are passed through UNTOUCHED, and that is a measured choice,
    # not laziness. Per-particle contrast stretching looks like the obvious win
    # — the facet step the user is judging is only a few grey levels wide — but
    # tried two ways (full 2-98 stretch of the interior, and one capped at a
    # fraction of the frame's own range) it came out clearly WORSE: a particle
    # with a genuinely uniform interior has almost no range to stretch, so the
    # transform spends the full 0-255 on its shot noise and every tile turns
    # into a blown-out white blob. The two variants were also indistinguishable
    # from each other. Raw keeps tiles calm, comparable, and honest about how
    # much signal is actually there — which is itself information the user needs
    # when deciding whether a call can be made at all.
    rgb = np.repeat(np.clip(sub, 0, 255)[:, :, None], 3, axis=2)
    rgb[~tgt] *= 0.45                      # push the neighbours back
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    # pad to square so tiles line up regardless of where the particle sat
    ph, pw = out.shape[:2]
    n = max(ph, pw)
    if ph != pw:
        sq = np.zeros((n, n, 3), np.uint8)
        sq[(n - ph) // 2:(n - ph) // 2 + ph, (n - pw) // 2:(n - pw) // 2 + pw] = out
        out = sq
    return out


def pages(items, per_page):
    for i in range(0, len(items), per_page):
        yield items[i:i + per_page]
