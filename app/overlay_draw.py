"""Rendering: draw measurements, mask outlines and solid/flat classification
onto the micrograph. Used by both the GUI (live preview) and export."""
from __future__ import annotations

import math
import random
import numpy as np
from PIL import Image, ImageDraw

from fonts import pil_font

CYAN = (0, 245, 245)
RED = (255, 70, 70)       # solid / patterned (crystalline)
GREEN = (70, 220, 130)    # flat / undercooled
INK = (0, 0, 0)
GREY = (150, 158, 168)    # UI chip colour for "excluded"
# "not measured" tint. A neutral grey is invisible on a greyscale micrograph, so
# the tint is a cool slate that still reads as greyed-out but stands off the image.
UNMEASURED = (70, 95, 140)
UNMEASURED_ALPHA = 0.45

# per-pattern overlay colours (solid particles); undercooled stays GREEN
PATTERN_COLORS = {
    "janus":     (255, 150, 40),    # orange
    "stripe":    (60, 140, 255),    # blue
    "lamellar":  (240, 80, 200),    # magenta
    "composite": (170, 100, 255),   # purple
}

# Every class name render_training can tint. The last three never occur in a
# training LABEL — they are what the review overlay paints, where the fill shows
# the model's own answer and that answer can be "crystalline but I can't read a
# pattern" (RED, as in the normal classification view) or "dropped as too unclear
# to judge" (the panel's grey for excluded).
TRAIN_COLORS = {**PATTERN_COLORS, "undercooled": GREEN,
                "solid": RED, "excluded": GREY, "exclude": GREY}


def pick_spread(particles, k=5, rng=None):
    """Pick k clean particles spanning the size range, one per size band.

    Randomised within each band so pressing "Analyze" again re-rolls a different
    representative set (lets the user re-measure if a pick looked wrong).
    """
    rng = rng or random.Random()
    from analyze import measurable

    def clean(p):
        ar = p.minor_px / p.major_px if p.major_px else 0
        return measurable(p) and ar >= 0.72
    cand = [p for p in particles if clean(p)]
    if len(cand) < k:
        cand = [p for p in particles if measurable(p)] or list(particles)
    cand.sort(key=lambda p: p.diam_nm)
    if len(cand) <= k:
        return cand
    # split the (5th-95th pct) range into k bands, pick one random particle each
    lo, hi = int(len(cand) * 0.05), max(1, int(len(cand) * 0.95))
    band = cand[lo:hi]
    if len(band) < k:
        band = cand
    edges = np.linspace(0, len(band), k + 1).astype(int)
    out = []
    for i in range(k):
        seg = band[edges[i]:max(edges[i] + 1, edges[i + 1])]
        out.append(rng.choice(seg))
    return out


def _draw_pattern_legend(dr, img_w, font_scale=1.0, classes=None):
    """Colour key in the top-left corner for the pattern overlay: a clean white
    rounded card with rounded colour chips, sized to its text. Only the classes
    actually being shown (``classes``; None = all) appear in the key."""
    all_items = [("janus", "Janus", PATTERN_COLORS["janus"]),
                 ("stripe", "Stripe", PATTERN_COLORS["stripe"]),
                 ("composite", "Composite", PATTERN_COLORS["composite"]),
                 ("lamellar", "Lamellar", PATTERN_COLORS["lamellar"]),
                 ("undercooled", "Undercooled", GREEN)]
    items = [(lab, col) for key, lab, col in all_items if classes is None or key in classes]
    if not items:
        return
    fs = int(round(25 * font_scale))
    f = pil_font(fs, "SemiBold")
    pad = int(fs * 0.70)                  # inner padding
    sw = int(fs * 0.80)                   # chip size
    gap = int(fs * 0.55)                  # chip -> text
    rowh = int(fs * 1.55)                 # row pitch
    tw = max(int(dr.textlength(lab, font=f)) for lab, _ in items)
    box_w = pad * 2 + sw + gap + tw
    box_h = pad * 2 + rowh * (len(items) - 1) + sw
    x0, y0 = 16, 16
    dr.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=int(fs * 0.5),
                         fill=(255, 255, 255), outline=(203, 210, 218), width=2)
    for i, (lab, col) in enumerate(items):
        y = y0 + pad + i * rowh
        dr.rounded_rectangle([x0 + pad, y, x0 + pad + sw, y + sw],
                             radius=max(3, sw // 4), fill=col)
        dr.text((x0 + pad + sw + gap, y + sw / 2), lab, fill=INK, font=f,
                anchor="lm")


def render_training(analysis, labels, show_overlay=True, chosen=None,
                    font_scale=1.0, show_classes=None, certainty=None,
                    mark_unmeasured=True):
    """Training-mode view: every particle tinted by its EFFECTIVE label — the
    model's prediction or the user's click-correction. Labelled particles get a
    thin dark ring, exactly like the normal-mode overlay.

    `mark_unmeasured` paints the particles the app does not measure in the same
    slate blue the normal-mode view uses, and that is the whole point of it being
    on by default (user request, 2026-08-04: "eğitim modunda da aynı parçacıklar
    mavi gözüksün, ben aa tamam bu mavi bunun desenine bakmama gerek yok
    diyeyim"). Leaving them merely unlabelled was not enough: bare and
    unclassified look identical, so the user could not tell "the model has no
    answer here" from "this one is not worth judging" and kept correcting
    particles that were never going to be measured.

    A particle the user has labelled ANYWAY keeps its class colour — the tint is
    applied first and the label paints over it. An explicit click outranks the
    gate, and must be visible as such.

    `labels`: {particle_id: (class_name, from_user)} where class_name is a
    PATTERN_COLORS key or "undercooled". Segmented-but-unlabelled particles are
    left completely bare (no outline), so the overlay matches the normal-mode view
    — only labelled particles are tinted and ringed.
    `show_overlay` False returns the plain micrograph (the user flips to it to
    judge fine contrast that the tint would hide).
    `chosen` particles get a cyan diameter line drawn on top — the Measure tool
    works in training mode too."""
    from skimage.segmentation import find_boundaries
    base = Image.open(analysis.path).convert("RGB")
    masks = analysis.label_mask
    if not show_overlay or masks is None:
        im = base
        dr = ImageDraw.Draw(im)
        if chosen:
            _draw_measurements(im, dr, chosen, font_scale)
        if certainty:
            _draw_certainty(im, dr, certainty, font_scale)
        return im
    arr = np.asarray(base).copy()
    nmax = int(masks.max())
    lut = np.zeros((nmax + 1, 3), np.uint8)
    fill = np.zeros(nmax + 1, bool)
    for pid, (cls, from_user) in labels.items():
        if pid <= 0 or pid > nmax:
            continue
        if show_classes is not None and cls not in show_classes:
            continue                     # View checkboxes filter which classes show
        col = TRAIN_COLORS.get(cls)
        if col is None:
            continue
        lut[pid] = col
        fill[pid] = True
    h, w = masks.shape
    region = arr[:h, :w]
    # "not measured" first, so an explicit user label still paints over it
    unmeas = np.zeros(nmax + 1, bool)
    if mark_unmeasured:
        from analyze import measurable
        for p in analysis.particles:
            if 0 < p.id <= nmax and not measurable(p):
                unmeas[p.id] = not fill[p.id]
        mu = (masks > 0) & unmeas[masks]
        a = UNMEASURED_ALPHA
        region[mu] = ((1 - a) * region[mu]
                      + a * np.array(UNMEASURED)).astype(np.uint8)
    m = (masks > 0) & fill[masks]
    col = lut[masks]
    # lighter tint than the results overlay (0.32) so the particle's own
    # texture stays clearly readable while labelling
    region[m] = (0.80 * region[m] + 0.20 * col[m]).astype(np.uint8)
    # Only labelled particles get a ring — the same thin dark outline the
    # normal-mode overlay draws around coloured particles. Detected-but-unlabelled
    # particles are left bare (no grey segmentation frame), so training and normal
    # views match: just tinted fills, no mesh of outlines over everything.
    # The unmeasured ones are ringed too, exactly as the normal view rings them.
    b = find_boundaries(masks, mode="thick")
    region[b & (fill | unmeas)[masks]] = INK
    arr[:h, :w] = region
    # no in-image legend here: the class buttons in the training panel already
    # show the colours, and the legend card would cover clickable particles
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    if chosen:
        _draw_measurements(im, dr, chosen, font_scale)
    if certainty:
        _draw_certainty(im, dr, certainty, font_scale)
    return im


def render(analysis, show_measurements=True, show_classification=False,
           class_filter=None, show_outlines=False, show_pattern=False,
           pattern_classes=None, chosen=None, k=5, font_scale=1.0,
           mark_unmeasured=False, certainty=None):
    """Return a PIL RGB image of the original micrograph with the chosen overlays.

    `chosen` lets the caller pass a fixed representative set (so toggles/redraws
    keep the same 5 particles); otherwise a fresh set is picked.

    `show_pattern` colours each solid particle by its patternnet class (janus/
    stripe/lamellar/composite) and undercooled particles green — so the per-
    particle classification can be eyeballed.

    `mark_unmeasured` tints every particle that is left out of the size
    statistics grey (same way a class is tinted), so with the Measure tool the
    user can see at a glance what is and isn't being measured.
    """
    from skimage.segmentation import find_boundaries
    base = Image.open(analysis.path).convert("RGB")
    arr = np.asarray(base).copy()
    masks = analysis.label_mask
    colour_fill = (show_classification or show_pattern) and analysis.classifiable

    # Only ever draw labels that are actual counted particles. Guards against a
    # mask that still holds filtered-out specks (older analyses) so the Borders
    # overlay never outlines ghost dots inside a large particle.
    valid = None
    if masks is not None:
        valid = np.zeros(int(masks.max()) + 1, bool)
        for p in analysis.particles:
            valid[p.id] = True

    # the caller can restrict which classes get painted — pattern_classes for
    # the per-pattern overlay, class_filter ({"solid","undercooled"}) for the
    # plain classification fill; anything deselected is left untouched.
    fill_ids = None
    if colour_fill and masks is not None:
        from analyze import measurable
        lut = np.zeros((int(masks.max()) + 1, 3), np.uint8)
        fill_ids = np.zeros(int(masks.max()) + 1, bool)
        for p in analysis.particles:
            if getattr(p, "excluded", False):
                continue                           # dim/unreliable -> greyed below
            if getattr(p, "unclassified", False):
                continue                           # user: "class unclear" -> unpainted
            # A particle too buried (or too cut) to have its size measured does
            # not get a class colour either. It used to: reclassify() labels
            # EVERY particle, so a cap peeking out from under a neighbour was
            # painted green while the charts — which do apply this gate — never
            # counted it. On one photo that was 520 particles painted undercooled
            # against 395 in the chart, i.e. the app answering the same question
            # two ways. `measurable` honours the user's own force-in/force-out,
            # so a particle they judged themselves is painted on their say-so.
            if not measurable(p):
                continue
            if show_pattern:
                if p.is_solid:
                    if p.pattern not in PATTERN_COLORS:
                        # Solid with no pattern. Painted red as plain "Solid" when
                        # the user said so themselves, or when the Solid class is
                        # ticked in the View bar; otherwise left plain, the way
                        # the model's unreadable ones have always been.
                        if not (getattr(p, "user_solid", False)
                                or (pattern_classes and "solid" in pattern_classes)):
                            continue
                        cls = "solid"
                    else:
                        cls = p.pattern
                else:
                    cls = "undercooled"
                if (pattern_classes is not None and cls not in pattern_classes
                        and not getattr(p, "user_solid", False)):
                    continue                       # class deselected -> leave plain
                lut[p.id] = RED if cls == "solid" else PATTERN_COLORS.get(cls, GREEN)
            else:
                cls = "solid" if p.is_solid else "undercooled"
                if class_filter is not None and cls not in class_filter:
                    continue
                lut[p.id] = RED if p.is_solid else GREEN
            fill_ids[p.id] = True
        h, w = masks.shape
        m = (masks > 0) & fill_ids[masks]
        col = lut[masks]
        region = arr[:h, :w]
        # keep the tint light so the particle's own morphology stays readable
        region[m] = (0.68 * region[m] + 0.32 * col[m]).astype(np.uint8)
        arr[:h, :w] = region

    # (excluded/dim and unmeasurable particles are NOT tinted in the class/pattern
    # overlay — a slate tint there reads as "stripe" and confuses. They're only
    # greyed in Measure mode below, where the grey clearly means "not measured".
    # Leaving them plain is what makes the painted particles and the charted
    # particles the same set.)

    # particles left out of the size statistics (too buried to expose their real
    # width, or cut by the frame): tint them grey, exactly like a class tint, so
    # "grey = not measured" is readable at a glance
    unmeasured = None
    if mark_unmeasured and masks is not None:
        from analyze import measurable
        unmeasured = np.zeros(int(masks.max()) + 1, bool)
        for p in analysis.particles:
            unmeasured[p.id] = not measurable(p)
        h, w = masks.shape
        m = (masks > 0) & unmeasured[masks]
        region = arr[:h, :w]
        a = UNMEASURED_ALPHA
        region[m] = ((1 - a) * region[m] + a * np.array(UNMEASURED)).astype(np.uint8)
        arr[:h, :w] = region

    if (show_outlines or colour_fill or mark_unmeasured) and masks is not None:
        b = find_boundaries(masks, mode="thick") & valid[masks]
        # when only some classes are filled, outline just those (not every particle)
        if colour_fill and not show_outlines and fill_ids is not None:
            b = b & fill_ids[masks]
        elif mark_unmeasured and not show_outlines and not colour_fill:
            b = b & unmeasured[masks]       # only ring what's greyed out
        h, w = masks.shape
        sub = arr[:h, :w]
        sub[b] = INK if (colour_fill or mark_unmeasured) else (255, 60, 60)
        arr[:h, :w] = sub

    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)

    if show_pattern and colour_fill:
        _draw_pattern_legend(dr, im.width, font_scale, classes=pattern_classes)

    if show_measurements:
        if chosen is None:
            chosen = pick_spread(analysis.particles, k)
        _draw_measurements(im, dr, chosen, font_scale)
    if certainty:
        _draw_certainty(im, dr, certainty, font_scale)
    return im


def _draw_measurements(im, dr, chosen, font_scale=1.0):
    """Draw a cyan diameter line + nm label across each particle in `chosen`.
    Shared by the results overlay and the training overlay so the Measure tool
    looks identical in both modes."""
    fs = int(round(30 * font_scale))
    f = pil_font(fs, "Bold")
    for p in chosen:
        L = p.diam_px / 2.0
        dx, dy = math.sin(p.angle) * L, math.cos(p.angle) * L
        x1, y1, x2, y2 = p.cx - dx, p.cy - dy, p.cx + dx, p.cy + dy
        # line with a dark casing underneath for contrast, cyan on top
        dr.line([(x1, y1), (x2, y2)], fill=INK, width=7)
        dr.line([(x1, y1), (x2, y2)], fill=CYAN, width=3)
        for ex, ey in [(x1, y1), (x2, y2)]:
            dr.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=CYAN, outline=INK, width=2)
        lab = f"{p.diam_nm:.0f} nm"
        lx = min(max(6, x2 + 10), im.width - fs * 5)
        ly = max(2, y2 - fs // 2)
        dr.text((lx, ly), lab, fill=CYAN, font=f, stroke_width=2, stroke_fill=INK)


def _draw_certainty(im, dr, items, font_scale=1.0):
    """Draw a small class-coloured confidence pill centred on each particle.
    `items`: list of (cx, cy, text, rgb). Shared by the results and training
    overlays so the Certainty tool looks identical in both modes."""
    fs = int(round(29 * font_scale))
    f = pil_font(fs, "Bold")
    for cx, cy, text, rgb in items:
        bb = dr.textbbox((cx, cy), text, font=f, anchor="mm")
        pad = int(round(fs * 0.30))
        box = [bb[0] - pad, bb[1] - pad * 0.7, bb[2] + pad, bb[3] + pad * 0.7]
        dr.rounded_rectangle(box, radius=int(round(fs * 0.42)), fill=INK,
                             outline=rgb, width=max(2, int(round(fs * 0.09))))
        dr.text((cx, cy), text, fill=rgb, font=f, anchor="mm")
