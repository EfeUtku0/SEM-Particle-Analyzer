"""Particle segmentation, measurement and classification for BIOMATEN SEM images.

Pipeline per image:
  1. read_scale  -> nm/px calibration (from scale_reader)
  2. crop off the bottom info bar so only the micrograph is segmented
  3. Cellpose (cpsam) -> per-particle masks
  4. per-particle: equivalent diameter (nm), solidity, internal texture
  5. classify each particle solid (patterned/crystalline) vs flat (undercooled)
  6. summary: count, solid/flat split, mean/median diameter, std
"""
from __future__ import annotations

import math
import os
import re
import dataclasses
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from info_bar_reader import read_scale, _find_info_bar_top, ScaleResult

# scikit-image is imported inside the functions that segment, not here: it drags
# in scipy.ndimage and costs ~0.3 s, which is most of the app's startup, and
# nothing on the way to a visible window needs it. Only an actual Analyze does —
# and that already waits ~50 s on Cellpose. Same reason cv2 is imported locally
# below; this file's convention, not an exception to it.

# Category keyword the user cares about; the trailing index number is dropped.
_CATEGORY_RE = re.compile(r"(karışık|karisik|üst|ust|alt)", re.IGNORECASE)


def clean_title(stem: str) -> str:
    """Trim a filename stem down to its category word (Karışık/Alt/Üst), dropping
    the index number and anything after it. 'UÖ - 09 Karışık 7' -> 'UÖ - 09 Karışık'."""
    last = None
    for m in _CATEGORY_RE.finditer(stem):
        last = m
    if last:
        return stem[:last.end()].strip()
    return stem.rstrip(" -_0123456789").strip() or stem


def _diam_summary(d):
    """mean/median/std/min/max with a guard for an empty distribution (an image
    where no particle was detected) so stats() never crashes."""
    if d.size == 0:
        return dict(mean=0.0, median=0.0, std=0.0, dmin=0.0, dmax=0.0)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                std=float(d.std()), dmin=float(d.min()), dmax=float(d.max()))


# --- "was this analysis produced by the rules the app runs today?" -------------
# A saved Analysis is a set of DECISIONS (solid/undercooled, pattern, excluded,
# measured) taken by whatever rules and model weights were in force when Analyze
# ran. Those change often — a threshold the user re-tunes, a retrained pattern
# net — and until now nothing recorded which vintage an analysis was, so a pooled
# result could silently mix them. (Measured in the live session: 15 of 43 saved
# analyses predated the `bright` field entirely, so the dim-particle exclusion
# could not fire on them and they carried no pattern predictions at all, while
# looking exactly like the others in the panel.)
#
# So every analysis now stamps `pipeline`, and the panel marks any row whose stamp
# is not the current one. Re-running Analyze is cheap (the Cellpose masks and the
# scale reading are cached), so the fix for a marked row is one click.
#
# BUMP PIPELINE_VERSION whenever a change would give a different answer on the
# same pixels: a threshold above, a gate, a class rule. The model weights don't
# need a bump — they are hashed into the stamp on their own.
PIPELINE_VERSION = 7   # v2: measurable() gained the aspect-ratio (crescent/buried-
                       # cap) test, 2026-07-30 — same-pixel photos now measure a
                       # slightly smaller, more honest set of particles.
                       # v3: EDGE_MEASURE_MAX 0.60 -> 0.70, same day.
                       # v4: EDGE_MEASURE_MAX -> 0.85 and a separate, lower AR
                       # floor for frame-cut particles (EDGE_MEASURE_MIN_AR) —
                       # the frame was being cropped far harder than intended.
                       # v5: the pattern eligibility gate loosened (solidity
                       # 0.90->0.70, AR 0.60->0.30, edge 0.90->1.10), 2026-08-01.
                       # Classification only — measurable() was fixed in the same
                       # commit so the size statistics stay where they were.
                       # v6: NOPATTERN_SOLID_CONF, 2026-08-02 — a particle the
                       # solid net is >=90% sure of is no longer relabelled
                       # undercooled just because no pattern could be named.
                       # v7: the info bar is now MEASURED instead of assumed to
                       # be 70 px (2026-08-04). BIOMATEN's is 78, so 8 rows of
                       # pure black were part of every micrograph and of the
                       # image median that `bright` is measured against; the
                       # crop, and therefore a few numbers, shift very slightly.
                       # Same commit: the classification gate stopped asking the
                       # detector nameplate alone, and HFW outranks the scale bar.


def pipeline_stamp() -> str:
    """Short identity of the rules + model weights in force right now."""
    import hashlib
    h = hashlib.md5()
    h.update(f"v{PIPELINE_VERSION}".encode())
    try:
        import model_solid_liquid
        import model_pattern
        # _model_path(), not the bundled constant: since the solid/liquid gate
        # became retrainable too, the file actually in force may be the one in
        # the data folder, and a stamp that ignored it would call an analysis
        # made with the OLD gate fresh.
        for p in (model_solid_liquid._model_path(), model_pattern._model_path()):
            try:
                st = os.stat(p)
                h.update(f"|{os.path.basename(p)}:{st.st_mtime_ns}:{st.st_size}".encode())
            except OSError:
                h.update(b"|missing")
    except Exception:
        h.update(b"|nomodels")
    # …and the calibrated thresholds, for exactly the same reason as the weights:
    # they change what the same pixels are called, so an analysis made under the
    # old ones is stale. Without this a recalibration would silently leave every
    # existing analysis looking current.
    try:
        import thresholds
        for k, v in sorted(thresholds.all().items()):
            h.update(f"|{k}={v:.4f}".encode())
    except Exception:
        h.update(b"|nothresholds")
    return f"{PIPELINE_VERSION}.{h.hexdigest()[:10]}"


def reload_thresholds():
    """Re-read the calibrated thresholds after a retrain has rewritten them.

    They are bound to module names at import so every existing caller and test
    keeps working unchanged; this rebinds them in place.
    """
    global DEFAULT_FACET_THRESH, JANUS_MIN_SOLID, NOPATTERN_SOLID_CONF
    global STRIPE_SMALL_MIN_CONF, JANUS_SMALL_MIN_CONF
    _thr.reload()
    v = _thr.all()
    DEFAULT_FACET_THRESH = v["facet_thresh"]
    JANUS_MIN_SOLID = v["janus_min_solid"]
    NOPATTERN_SOLID_CONF = v["nopattern_solid_conf"]
    STRIPE_SMALL_MIN_CONF = v["stripe_small_min_conf"]
    JANUS_SMALL_MIN_CONF = v["janus_small_min_conf"]


_model = None


def _get_model():
    global _model
    if _model is None:
        from cellpose import models
        _model = models.CellposeModel(gpu=True)
    return _model


# ---------------------------------------------------------------------------
# The thresholds that live in a MODEL's units are not written here — they are
# read from thresholds.py, which keeps the shipped value as its default and lets
# calibrate.py re-derive it from the user's labels after a retrain. The comments
# in this file still explain what each one is FOR and what it was originally
# measured at; the number in force may since have been re-measured.
#
# Nothing here is per-instrument, and nothing may become per-instrument: one
# threshold serves every microscope, or the pipeline stops being a pipeline and
# becomes a pile of special cases. (Measured 2026-08-04: after the solid/liquid
# gate was retrained on both machines, the best cut is 0.40 on one and 0.43 on
# the other — a single shared value costs each of them under 0.001 balanced
# accuracy. Before that retrain the same two numbers were 0.54 and 0.88, and no
# shared value existed. The fix for a new instrument is its data, not its own
# constant.)
# ---------------------------------------------------------------------------
import thresholds as _thr

DEFAULT_FACET_THRESH = _thr.get("facet_thresh")   # P(solid) -> crystalline
SOLID_MIN_DIAM_NM = 200       # particles below this are (almost) always undercooled
MIN_COUNT_DIAM_NM = 15.0      # anything this small is a mis-detection, not a real
                              # particle -> dropped entirely (not even counted)
# ...and the same floor in PIXELS, which is the one that actually bites on the
# coarse "Büyükler" photos (user report, 2026-07-30: "büyük katı parçacıkların
# içinde bir sürü minik parçacık var sanıyorsun"). At 13-18 nm/px the 15 nm rule
# is under one pixel, so it filters nothing, and the small-particle recovery's own
# nm floor collapses to its 4-pixel fallback — texture speckle inside a big
# particle then becomes a "30 nm particle", gets measured, and drags the mean size
# down by 20-44% on those photos. Evidence for 4 px (~12.6 px area): of the
# particles the user hand-clicked in the 32 labelled photos, NONE below 12 px sits
# on a visible particle — 48 crops inspected at 6-10 and 13+ nm/px show flat
# texture at the marked spot, and even the 12-26 px band is empty there. On fine
# photos (<6 nm/px) there is not a single hand-clicked particle under 12 px, and
# 4 px is 13-18 nm anyway — below the nm floor. So this only removes noise.
MIN_COUNT_DIAM_PX = 4.0
MIN_COUNT_AREA_PX = np.pi * (MIN_COUNT_DIAM_PX / 2) ** 2      # ~12.6 px

# A particle is pattern-classified only when MOST of it is visible: it may touch
# the image frame a little, but not be cut off by a large chord; and it must be
# reasonably convex (not bitten into by a neighbour) and not a thin sliver.
# `edge_cut` is the length of the particle's run along the image border divided
# by its diameter — ~0 when merely tangent, ~1 when roughly half is off-frame. So
# a particle whose ~60%+ is inside still qualifies (user rule, 2026-07-18: the old
# "touches the edge at all -> reject" was far too aggressive).
# Loosened 2026-08-01 after a full sweep of every post-processing knob over the
# 15122 hand-labelled particles of the 32 clean photos (the user's 759 "exclude"
# marks included as ground truth, so over-classifying junk is paid for). Fitted
# by photo-grouped 4-fold CV — tuned on 24 photos, scored on the 8 unseen — and
# all four folds independently chose these same three values, so the held-out
# score equals the full-fit score exactly (0.7848 both ways: no overfit).
#   PATTERN_MIN_SOLIDITY 0.90 -> 0.70   (+1.7 accuracy alone, the biggest knob)
#   PATTERN_MIN_AR       0.60 -> 0.30   (+1.0)
#   PATTERN_MAX_EDGE_CUT 0.90 -> 1.10   (+0.1 alone, but synergistic with them)
# Together: accuracy 0.750 -> 0.785, macro-recall 0.564 -> 0.622, and EVERY class
# improves (lamellar .567->.684, composite .607->.681, stripe .523->.584, janus
# .681->.728). Why: the old gate refused to read 1137 particles as "solid but no
# pattern" on the assumption that a partly-occluded particle is unreadable — but
# the user could and did label them by eye, and the CNN mostly had them right.
# The gate was swallowing answers the net already knew. That count drops to 145.
# NOTE: this gate is deliberately NOT the measurement gate any more — see
# measurable(), where the "has a pattern -> exposed enough" bypass now keeps the
# old 0.90/0.60 geometry bar, so the size statistics are unchanged by all this.
PATTERN_MAX_EDGE_CUT = 1.10
PATTERN_MIN_SOLIDITY = 0.70
PATTERN_MIN_AR = 0.30         # minor/major axis ratio


def pattern_eligible(edge_cut, solidity, major_px, minor_px) -> bool:
    if edge_cut > PATTERN_MAX_EDGE_CUT:
        return False
    ar = minor_px / major_px if major_px else 0.0
    return solidity >= PATTERN_MIN_SOLIDITY and ar >= PATTERN_MIN_AR


# --- which particles get a SIZE (user rule, evolved 2026-07-25, 2026-07-30) ----
# A particle lying under its neighbours only shows a cap, so its equivalent
# diameter under-reports its true size. The history of this gate:
#   * "measure everything not frame-cut" -> measured buried caps, badly wrong.
#   * pattern_eligible (solidity>=0.90, ar>=0.60) -> too strict; it dropped fully
#     exposed but slightly-concave particles the user wanted measured, while (it
#     turns out) the genuinely bad ones — dim, behind neighbours — have high
#     solidity anyway and slipped through regardless.
# The genuinely unreliable particles are now removed up front by the brightness
# `excluded` flag (see reclassify). So the size gate can be generous on geometry:
#   * a particle the model gave a PATTERN is, by definition, exposed -> measure it;
#   * otherwise measure it if it's reasonably whole (looser than pattern_eligible
#     on solidity), but still drop frame-cut, heavily bitten, or a thin sliver.
# Still gates only the SIZE STATISTICS: the particle stays detected and counted.
#
# The ASPECT-RATIO test (data rule, 2026-07-30, audit finding on the 43-photo
# working session — see [[sem-analyzer-audit]]): a shape this elongated on a
# spherical system is essentially never the particle's own outline — it is the
# crescent-shaped visible sliver of a particle mostly buried under a neighbour,
# which a solidity test alone does not catch (a crescent can be fairly convex).
# Measured: 1696 of 24906 measured particles (6.8%) fell under minor/major<0.60,
# and their mean diameter was 74% of their own photo's median — exactly the
# under-measurement the occlusion gate exists to prevent. Uses the SAME 0.60 bar
# pattern_eligible already applies (so this only tightens the size gate on
# particles the pattern step would have rejected anyway; nothing with a pattern
# is affected — it already passed this test to get one).
# 0.90 (was 0.85, raised 2026-07-30 on the user's call: "arada bir ölçülmemesi
# gereken parçacıklara mavi diyemiyorsun, bu eşiği minik daha yükselt"). Measured
# over 14 photos / 7306 particles: the measured share goes 86.5% -> 80.0% (476
# particles more turn blue) and every one of them is pattern-less — particles the
# model gave a pattern already clear 0.90 through pattern_eligible, so this brings
# the size gate to exactly the same tightness as the pattern gate.
MEASURE_MIN_SOLIDITY = 0.90
# A particle touching / clipped by the frame still has a usable diameter as long
# as most of it is in view. edge_cut = border chord / VISIBLE diameter, so solving
# the circle geometry gives its real meaning:
#   edge_cut  0.44   0.62   0.71   0.86   0.97   1.06   1.24
#   visible    98%    95%    92%    86%    80%    75%    63%
#   diam err  0.3%   1.9%   3.9%   6.2%    11%    15%    19%
# We measure up to 0.85 — ~86% of the particle in view, ~6% under-read on those
# few. History: 0.40 -> 0.60 -> 0.70 -> 0.85, each time because the user found the
# frame being cropped too hard (2026-07-30: "kenara minicik değen her parçacığı
# direkt eliyor, şimdi de fazla eliyor"). Measured on the working session, this
# and the edge AR floor below together take the frame-touching particles that get
# measured from 25% to 40%, while the overall mean moves 417.0 -> 418.3 nm (+0.3%)
# — the loosening is visible at the frame and invisible in the statistics.
EDGE_MEASURE_MAX = 0.85

# Aspect-ratio floors. A particle in the middle of the field that is this flat is
# a crescent — the sliver of something buried under its neighbours (see above).
MEASURE_MIN_AR = PATTERN_MIN_AR       # 0.60, the same bar the pattern gate uses
# ...but a FRAME-cut particle is flattened by the cut itself, and that is already
# measured, directly and better, by edge_cut. Solving the same circle geometry:
# a perfect circle cut down to 86% visible (edge_cut 0.85) has an aspect ratio of
# 0.80, and one cut to 63% has 0.60 — so the 0.60 floor is a second, blunter copy
# of the edge test, and applying both double-penalised the frame. On the working
# session 642 frame particles were failing BOTH tests and 318 the AR test alone.
# Frame particles therefore get a floor low enough to still reject a genuine
# sliver (0.45) but not to re-reject what edge_cut has already judged.
EDGE_MEASURE_MIN_AR = 0.45

# Ellipse fill: the occlusion solidity cannot see (user report, 2026-08-04 —
# "bu parçacıklar gerçekten de kuralları geçiyor mu?", with a review page of
# wedge-shaped fragments).
#
# Solidity asks "is the silhouette BITTEN", i.e. concave. But a particle buried
# under a neighbour usually shows a wedge or a triangular cap, and a wedge is
# CONVEX — solidity 0.90-0.96, straight through the gate. Audited on the working
# session: all 9723 particles in the undercooled review queue passed the gate as
# written, with the lowest solidity in the queue sitting exactly on the 0.900
# bar. Nothing was leaking; the gate simply had no test for this shape.
#
# fill = area / (area of the fitted ellipse) = diam_px^2 / (major_px * minor_px),
# which is 1.0 for any true ellipse of any elongation, and drops as a shape is
# truncated. Verified by eye on two contact sheets drawn from the same queue: the
# lowest-fill 40 are wedges and triangles almost without exception, while 40
# random ones are round and whole at 0.97-1.00.
#
# 0.92 chosen with the user (2026-08-04) from the measured trade over 30299
# measured particles: it drops 438 of them (1.45%) and moves the mean diameter
# 438 -> 440 nm and the median 297 -> 299 nm — i.e. it takes the wedges out of
# the statistics without perceptibly moving the statistics. (0.90 was the
# cautious option at 163 particles; 0.95 was too broad at 6%.)
#
# KNOWN LIMIT, do not oversell it: a particle cut roughly through its middle
# still fills its fitted ellipse to ~0.93, because the fit shrinks with the
# shape. This catches wedges and caps, not every occlusion.
MEASURE_MIN_FILL = 0.92


def ellipse_fill(p) -> float:
    """How much of its own fitted ellipse the silhouette actually fills.

    1.0 for a whole particle (of any elongation), lower for a truncated one.
    Returns 1.0 when the axes are missing, so a particle with no ellipse fit is
    never rejected by a test that could not be run on it.
    """
    e = float(getattr(p, "major_px", 0.0)) * float(getattr(p, "minor_px", 0.0))
    if e <= 0:
        return 1.0
    return float(p.diam_px) ** 2 / e


def measurable(p) -> bool:
    """True when enough of the particle is exposed for its equivalent diameter to
    be a real size rather than the size of a visible cap."""
    if getattr(p, "user_measurable", None) is not None:
        return bool(p.user_measurable)      # the user's judgement overrides
    if getattr(p, "excluded", False):
        return False                        # dim/unreliable -> not a real size
    edge_cut = getattr(p, "edge_cut", 0.0)
    if edge_cut > EDGE_MEASURE_MAX:
        return False                        # a big chord is off-frame -> size wrong
    ar = p.minor_px / p.major_px if p.major_px else 0.0
    # a frame-cut particle is flattened BY the cut, which edge_cut already judged
    if ar < (EDGE_MEASURE_MIN_AR if edge_cut > 0 else MEASURE_MIN_AR):
        return False                        # thin sliver -> likely a buried cap
    # ...and the convex version of the same thing: a wedge cut out from under a
    # neighbour, which passes solidity because it has no bite in it. Applied to
    # frame particles TOO — unlike the AR floor, this is not a blunt second copy
    # of edge_cut. Measured on a synthetic disc cut by a straight frame edge:
    #   width kept   1.00   0.95   0.90   0.86   0.80   0.70   0.60
    #   fill        1.000  0.999  0.995  0.990  0.983  0.970  0.958
    # so even a particle cut down to 60% of its width — far past what edge_cut
    # allows — stays well clear of 0.92. A frame particle below the bar is short
    # of something OTHER than the frame, and exempting them left 222 such wedges
    # in the statistics.
    if ellipse_fill(p) < MEASURE_MIN_FILL:
        return False
    # "has a pattern -> exposed enough" used to be unconditional, which was safe
    # only while pattern_eligible was the STRICTER gate (0.90 solidity / 0.60 AR).
    # Since 2026-08-01 the pattern gate is deliberately looser than the size gate,
    # so an unconditional bypass would quietly widen the statistics along with it:
    # measured share 0.835 -> 0.860 and mean diameter 515 -> 524 nm, i.e. buried
    # caps re-entering the size distribution. Keeping the old geometry bar here
    # leaves the measured set byte-identical to before (0.835 / 515 nm / median
    # 391 nm, verified) while the classification still gets the full gain.
    # (the aspect-ratio bar above already applies to every particle, pattern or
    # not, so solidity is the only test the bypass was skipping.)
    if p.pattern and p.solidity >= MEASURE_MIN_SOLIDITY:
        return True                         # has a pattern -> exposed enough
    return p.solidity >= MEASURE_MIN_SOLIDITY


# ---- how spherical the particles are (user request, 2026-08-08) --------------
# Score per particle, 0..1, 1 = a perfect circle: the ISO/ImageJ CIRCULARITY,
# 4*pi*Area / Perimeter^2. The user came from ImageJ, so this is the definition
# already familiar to them — and it answers the question asked ("how spherical"),
# because it drops both for an elongated outline and for a rough or faceted one.
#
# The perimeter is the CROFTON estimate (4 directions), not the naive boundary
# walk. That choice is measured, not assumed — on 46500 measured particles from
# the working session, mean score per particle-diameter band:
#     diam (px)      4-8   8-16  16-32  32-64  64-96   96+
#     crofton       0.95   0.92   0.92   0.92   0.92   0.91
#     naive walk    1.16   0.96   0.89   0.86   0.85   0.83
# i.e. the naive perimeter makes sphericity a function of MAGNIFICATION (small
# particles score 0.3 higher than large ones), which would make the number
# meaningless across a photo set. The Crofton estimate is flat across the whole
# range. It is also correctly scaled: a synthetic digitised disc scores 1.00 at
# every radius from 2 px up (mean over sub-pixel placements), so 1.0 really is
# "a perfect circle" rather than an unreachable ceiling.
#
# Single-particle noise is real (a 20 px disc scores 0.94-1.03 depending on where
# its centre falls between pixels), which is why this is only ever reported as a
# MEAN over a population, never per particle.
SPHERICITY_MIN_PERIM_PX = 4.0     # below this the estimate is not a shape at all

# ...and then the score is STRETCHED onto 0..1 from the band real particles
# actually occupy (user, 2026-08-08: "her görselim 90 üstü geliyor, bu bana
# hiçbir şey söylemiyor"). Raw circularity is a terrible READING scale here: on
# the working session every photo landed between 0.87 and 0.97, all of it looking
# like "97% round" whatever the sample.
#
# BE HONEST ABOUT WHAT THIS IS: a display scale, not extra evidence. Measured
# over 97 photos, the ratio between the spread across photos and the sampling
# error within one photo is ~6 for the plain mean, ~6 for a harsh power (circ^8,
# ^16, ^32), ~5 for "share of particles above a threshold" and ~6 for this
# rescale — no monotone transform can add discrimination that the measurement
# does not hold. The ORDER of the photos is identical. What changes is that the
# numbers stop piling up against the top of the scale: photo scores now run
# 0.32-0.79 (median 0.53) where they used to run 0.87-0.97.
#
# The floor is where an outline stops reading as round at all, taken from
# synthetic shapes measured with this same estimator: a rippled circle 0.835, a
# square 0.88, a 4:1 ellipse 0.53. At 0.85, 14% of the particles sit on the floor
# and 5% reach the ceiling — harsh, without collapsing either end.
# The ceiling stays a mathematically perfect circle: real round particles score
# ~0.9, and 1.00 is reserved for what digitisation can only reach by luck.
# Raising the floor makes the app harsher; chart_data keeps the raw circularity
# alongside it, which is the number to quote in a paper.
SPHERICITY_FLOOR = 0.85


def sphericity_score(p):
    """A particle's roundness on the app's 0..1 reading scale (see above)."""
    c = circularity(p)
    if c is None:
        return None
    return min(1.0, max(0.0, (c - SPHERICITY_FLOOR) / (1.0 - SPHERICITY_FLOOR)))


def circularity(p):
    """A particle's sphericity score in [0, 1], or None if it has no perimeter.

    Clipped at 1: the perimeter estimate scatters a few percent either way on a
    small particle, and a score above "perfectly round" is noise, not a shape.
    """
    per = float(getattr(p, "perim_px", 0.0) or 0.0)
    if per < SPHERICITY_MIN_PERIM_PX:
        return None
    area = math.pi * 0.25 * float(p.diam_px) ** 2   # diam_px IS sqrt(4A/pi)
    return min(1.0, 4.0 * math.pi * area / (per * per))


def spherical_measurable(p) -> bool:
    """Which particles get a sphericity score (user rule, 2026-08-08).

    Everything the size statistics are built on — so the greyed-out particles the
    Measure tool skips are skipped here too — MINUS every particle touching the
    frame. A frame-cut particle has a straight edge that is the photo's, not the
    particle's; it scores low for a reason that says nothing about the sample,
    and the user asked specifically that those not drag the number down. Measured
    on the working session: they are 4.7% of the measured set and average 0.87
    against 0.92 for the rest.
    """
    return (measurable(p) and not getattr(p, "touches_edge", False)
            and circularity(p) is not None)


def ensure_perimeter(analysis) -> int:
    """Fill in perim_px for an analysis saved before sphericity existed.

    Like apply_count_floors, this is pure post-segmentation measurement off the
    stored label mask, so it gives exactly what a fresh run would — no
    re-segmenting. ~50 ms for a 1000-particle photo, done once per analysis (the
    field is then saved with the session).
    """
    a = analysis
    if a.label_mask is None or not a.particles:
        return 0
    todo = [p for p in a.particles if not getattr(p, "perim_px", 0.0)]
    if not todo:
        return 0
    from skimage.measure import regionprops
    props = {r.label: r for r in regionprops(np.asarray(a.label_mask, np.int32))}
    n = 0
    for p in todo:
        r = props.get(p.id)
        if r is None:
            continue
        p.perim_px = float(r.perimeter_crofton)
        n += 1
    return n


# Size prior on the pattern decision (user rule, 2026-07-17): a ~250-300 nm
# particle is almost always undercooled (liquid). composite/lamellar are
# physically impossible at that size, stripe is unlikely, janus is possible.
# So below PATTERN_SMALL_NM we forbid composite/lamellar, demand strong evidence
# for stripe, allow janus with moderate evidence, and otherwise fall back to
# undercooled. Above it, the plain argmax over all four classes is used.
PATTERN_SMALL_NM = 350.0
STRIPE_SMALL_MIN_CONF = _thr.get("stripe_small_min_conf")   # softmax prob a small particle needs to be stripe
JANUS_SMALL_MIN_CONF = _thr.get("janus_small_min_conf")    # ...to be janus
# ...and janus is physically impossible below this (user rule, 2026-07-25):
JANUS_MIN_NM = 300.0

# Janus-specific solidity floor (data rule, 2026-07-26, UÖ-03 Karışık set): janus
# is patternnet's default guess for marginally-solid / liquid particles. On 1883
# hand-labelled particles the undercooled-mislabelled-as-janus group clustered at
# P(solid) ~ 0.68 while true janus sat at ~ 0.96, and the janus SOFTMAX prob did
# NOT separate them (errors 0.92 vs true 0.81 — the net is confidently wrong). So
# a janus call needs clearer solidity than the general 0.50 solid bar; below this
# the particle is treated as undercooled. Leaves stripe/lamellar/composite alone.
# Measured on the 32 clean photos (2026-07-26): the total janus-boundary error is
# 545 at floor 0.50, drops to 494 at 0.55 (fixes 97 undercooled->janus, costs 46
# true janus = the free 2:1 improvement), then stays FLAT (~494) for higher floors
# — 0.55->0.60 is a 1:1 wash (74 fixed, 74 true janus lost) with no net gain, so
# 0.55 is the efficient knee. (An earlier 3-photo estimate looked 8:1 but was
# unrepresentative.) Retraining the whole solidnet gate was also tried and
# rejected — it undercounts every pattern class, not just janus.
JANUS_MIN_SOLID = _thr.get("janus_min_solid")

# When the pattern step can name NO pattern for a small particle, the particle
# used to be turned undercooled outright — however sure the solid/undercooled net
# was about it. That let a "WHICH pattern?" classifier overturn the "solid or
# liquid?" one, and it produced a state the app openly contradicted itself in:
# the Certainty tool read "5% undercooled" (i.e. P(solid) = 0.95) on a particle
# painted undercooled — the case that prompted this rule (2026-08-02, a 261 nm
# particle at P(solid) 0.95: below JANUS_MIN_NM janus is impossible and stripe
# needs STRIPE_SMALL_MIN_CONF, so nothing was nameable and it flipped).
#
# Above this floor the particle now stays SOLID WITH NO PATTERN instead — the
# same outcome the manual "Solid" tool already produces (see solidify_pattern:
# "an unreadable pattern must not cost the particle its class"). Below it the
# old behaviour stands, because patternnet's silence IS evidence of liquid.
#
# HONEST NUMBERS (held-out golden set, 4514 particles). solidnet is well
# calibrated on its own — P(solid) >= 0.95 is 99% truly crystalline overall, 92%
# inside the 200-350 nm band this rule governs. But conditioned on "and no
# pattern could be named" the sample collapses: only 4 golden particles sit above
# 0.90 (2 undercooled, 1 janus, 1 exclude), so the data can neither prove nor
# refute the change — macro-F1 0.7399 -> 0.7399, solid/undercooled accuracy
# 0.9214 -> 0.9212, i.e. a wash. It is adopted for CONSISTENCY, not for a
# measured gain: on the user's 44-photo working set it moves 150 of 25837
# measured particles (0.6%), all in 200-350 nm. Lowering the floor is NOT
# supported — at >= 0.80 the golden corner is only ~42% truly solid.
NOPATTERN_SOLID_CONF = _thr.get("nopattern_solid_conf")


def size_aware_pattern(diam_nm, pattern_probs) -> str:
    """Pattern label for an eligible solid particle, applying the size prior.
    Returns "" when a small particle has no credible pattern -> the caller then
    treats it as undercooled."""
    probs = np.asarray(pattern_probs, float)
    if diam_nm >= PATTERN_SMALL_NM:
        p = PATTERN_CLASSES[int(np.argmax(probs))]
        return p
    j = probs[PATTERN_CLASSES.index("janus")]
    s = probs[PATTERN_CLASSES.index("stripe")]
    if s >= STRIPE_SMALL_MIN_CONF and s >= j:
        return "stripe"
    if j >= JANUS_SMALL_MIN_CONF and diam_nm >= JANUS_MIN_NM:
        return "janus"
    return ""


def solidify_pattern(p) -> str:
    """User forces particle `p` to be solid (the manual 'Solid' tool): re-run the
    normal pattern assignment on its already-computed patternnet probabilities and
    return the class the model would give a solid particle here — one of
    PATTERN_CLASSES — or plain "solid" when there is no credible / reliable
    pattern (small with no confident call, too dim to trust, or edge/occluded).
    Never returns "undercooled" (the user's click already says this isn't liquid)
    and never "exclude" (user rule, 2026-07-30: "solid dediysem o soliddir" — an
    unreadable pattern must not cost the particle its class or its size; the
    particle simply stays solid-with-no-pattern and the user can name it)."""
    if (len(p.pattern_probs)
            and pattern_eligible(getattr(p, "edge_cut", 0.0), p.solidity,
                                 p.major_px, p.minor_px)):
        pat = size_aware_pattern(p.diam_nm, p.pattern_probs)
        if pat and getattr(p, "bright", 0.0) >= PATTERN_MIN_BRIGHT:
            return pat
    return "solid"


# Dim-particle exclusion (user rule, 2026-07-25): the pattern CNN over-calls
# "stripe" (and other patterns) on barely-visible, dim, background/occluded
# particles. On the golden set those wrong calls sit far below the image's median
# grey (median brightness ≈ -52 vs ≈ -2 for real stripe), and 227 particles the
# user hand-marked "exclude" also cluster there (median ≈ -32). So a particle the
# model would give a PATTERN, but whose interior is this much darker than the
# image median, is treated as unreliable: dropped from the size stats and not
# classified — rather than mislabelled stripe. Measured on golden: stripe
# precision 0.63 -> 0.76, exclude F 0 -> 0.37, 6-class macro-F1 0.627 -> 0.669.
# `bright` = particle interior mean grey − image median grey (so it is roughly
# contrast-normalised per image). Undercooled predictions are NOT gated (many
# real undercooled are legitimately dim); only pattern predictions are.
# -30 KEPT, 2026-08-01 (the sweep asked for -45 and it was rejected on purpose):
# -45 is worth +0.4 accuracy / +0.25 macro on the held-out golden photos, but it
# costs the exclude class 7 points of recall there (0.396 -> 0.325). Exclude is
# the weakest thing in the whole pipeline and the reason visnet exists, so trading
# a real capability for a fraction of a point is the wrong direction. Note the
# looser eligibility gate ABOVE already improves exclude on its own (0.358 ->
# 0.396): particles that failed the old gate never reached this brightness test
# at all, so the dim exclusion could not fire on exactly the dim, half-buried
# particles it was written for.
PATTERN_MIN_BRIGHT = -30.0

# Detectors whose nameplate settles the question on its own. CBS resolves the
# internal facet contrast the solid/undercooled call is built on; the SE-family
# ones render particles uniformly bright, so internal structure is simply not
# there to read and classifying would be inventing an answer.
CONTRAST_DETECTORS = {"CBS"}
FLAT_DETECTORS = {"ETD", "SE", "SE2", "TLD", "INLENS"}

# When the nameplate is absent, MEASURE the property it was standing in for.
# This is the whole point: the old gate was `detector == "CBS"`, which is a
# lookup against one instrument's info-bar vocabulary. The METU-METE exports
# carry no `det` cell at all, so that test returned "" and silently switched
# classification off for every one of them — not a small error, the feature
# just did not run. Rather than teach the app a second machine's layout (and a
# third, and a fourth), ask the image the physical question directly: is there
# structure inside the particles that stands above this image's own noise?
#
# The statistic is the robust spread of a particle's BLURRED interior divided
# by the image's shot-noise floor. Blur removes noise while coherent facet and
# phase steps survive, and dividing by the noise floor makes it a pure SNR
# ratio — no grey-level constant, so nothing to re-tune per instrument.
#
# Placing the floor: measured over both corpora, the 32 known-CBS photos run
# 0.66 (min) / 1.65 (median) and the 94 METU-METE photos run 0.93 / 2.67, i.e.
# the new instrument carries MORE readable structure than the reference set.
# 0.45 sits below everything known-good with room to spare, so it can only veto
# an image with essentially no internal structure at all.
# HONEST LIMIT: there is not one ETD/SE photo in either corpus, so this floor is
# calibrated only from the positive side. It is deliberately a floor, not a
# discriminator — if a genuine SE image ever shows up and gets classified, the
# fix is to raise this against that measured example, not to guess now.
STRUCTURE_MIN = 0.45

# Sub-classification of SOLID particles by internal pattern (patternnet). Order
# matches the training label ids in tools/train_pattern.py.
PATTERN_CLASSES = ["janus", "stripe", "lamellar", "composite"]

# Hybrid small-particle recovery: Cellpose reliably finds medium/large particles
# but misses many very small, low-contrast ones. A classical top-hat + Otsu pass
# recovers those. Sizes are in nm (converted to px via the scale) so the same
# thresholds hold across magnifications.
SMALL_MAX_NM = 250.0          # upper size of the "small" particles Cellpose misses
SMALL_MIN_NM = 18.0           # below this it's noise, not a resolvable particle


def _small_particle_mask(gray, cp_mask, nm_per_px):
    """Return (label_image, new_id_set) for small bright particles that Cellpose
    missed. Ids continue past cp_mask.max(); anything already covered by cp_mask
    (dilated a little) is skipped so we don't re-detect big-particle edges."""
    import cv2
    from skimage.measure import regionprops, label as sk_label
    npp = nm_per_px if nm_per_px else 8.0
    rmax = 0.5 * SMALL_MAX_NM / npp
    max_area = np.pi * rmax * rmax
    # the nm-based floor goes sub-pixel on coarse photos; MIN_COUNT_AREA_PX is
    # what keeps this from "recovering" 4-pixel texture speckle there
    min_area = max(MIN_COUNT_AREA_PX, np.pi * (0.5 * SMALL_MIN_NM / npp) ** 2)
    ksize = max(5, int(2 * rmax)) | 1                     # odd; > max particle
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    th = cv2.GaussianBlur(th, (0, 0), 0.8)
    _, bw = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    occupied = cv2.dilate((cp_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    lab = sk_label(bw)
    add = np.zeros_like(cp_mask)
    nid = int(cp_mask.max())
    new_ids = set()
    for r in regionprops(lab):
        if not (min_area <= r.area <= max_area):
            continue
        per = r.perimeter or 1.0
        circ = 4 * np.pi * r.area / (per * per)
        if circ < 0.6 or r.solidity < 0.88:          # round & convex only
            continue
        cy, cx = int(r.centroid[0]), int(r.centroid[1])
        if occupied[cy, cx]:
            continue
        nid += 1
        new_ids.add(nid)
        add[lab == r.label] = nid
    return add, new_ids


@dataclass
class Particle:
    id: int
    cx: float
    cy: float
    diam_px: float
    diam_nm: float
    major_px: float
    minor_px: float
    angle: float          # radians, orientation of major axis
    solidity: float
    facet_frac: float     # fraction of interior that is bright facet (patterned)
    is_solid: bool        # True = patterned/crystalline, False = flat/undercooled
    touches_edge: bool
    edge_cut: float = 0.0       # border run / diameter (0 = fully in frame)
    # Crofton perimeter estimate in pixels — the denominator of the sphericity
    # score (see circularity). 0 on analyses saved before it existed; those are
    # filled in from the stored mask by ensure_perimeter().
    perim_px: float = 0.0
    bright: float = 0.0         # interior mean grey − image median (dim = negative)
    pattern_probs: tuple = ()   # patternnet softmax over PATTERN_CLASSES (or ())
    pattern: str = ""           # argmax label for solid particles ("" otherwise)
    excluded: bool = False      # dim/unreliable: dropped from stats & classification
    user_measurable: bool = None   # user override of the occlusion gate (None = auto)
    # "I can't tell what this is, but its diameter is clear" (user rule,
    # 2026-07-30): counted and MEASURED like any particle, but belonging to no
    # class — not undercooled, not solid, not patterned, and left unpainted. Set
    # only by hand (gui), never by the model.
    unclassified: bool = False
    # "the user pressed Solid on this one": solid with no pattern, but a DELIBERATE
    # call rather than the model failing to read one — so the overlay paints it
    # (the model's own ~22% pattern-less solids stay unpainted unless the Solid
    # checkbox is ticked, or every image would light up red).
    user_solid: bool = False


@dataclass
class Analysis:
    image: str
    nm_per_px: float
    scale_label: str
    particles: list = field(default_factory=list)
    label_mask: np.ndarray = None    # integer mask (micrograph coords) for overlays
    micrograph: np.ndarray = None    # cropped grayscale (info bar removed)
    path: str = ""
    detector: str = ""               # "CBS", "ETD", ...
    classifiable: bool = False       # solid/flat only valid on CBS
    evaluated_class: bool = True     # which evaluations this run included —
    evaluated_pattern: bool = True   # lets the GUI re-run when one is missing
    pipeline: str = ""               # pipeline_stamp() when this ran ("" = older
                                     # than the stamp, i.e. stale by definition)

    def reclassify(self, facet_thresh: float):
        for p in self.particles:
            solid = (p.facet_frac >= facet_thresh
                     and p.diam_nm >= SOLID_MIN_DIAM_NM)
            pat = ""
            excluded = False
            if (solid and len(p.pattern_probs)
                    and pattern_eligible(getattr(p, "edge_cut", 0.0), p.solidity,
                                         p.major_px, p.minor_px)):
                pat = size_aware_pattern(p.diam_nm, p.pattern_probs)
                # janus is the net's fallback guess for marginally-solid particles;
                # demand clearer solidity for it than the general bar (see
                # JANUS_MIN_SOLID) or treat the particle as undercooled instead.
                if pat == "janus" and p.facet_frac < JANUS_MIN_SOLID:
                    pat = ""
                if not pat:               # small particle, no credible pattern
                    # …which is evidence of liquid — unless the solid/undercooled
                    # net is sure of the opposite, in which case the particle
                    # stays crystalline with its pattern simply unnamed rather
                    # than being relabelled liquid (see NOPATTERN_SOLID_CONF).
                    solid = p.facet_frac >= NOPATTERN_SOLID_CONF
                elif getattr(p, "bright", 0.0) < PATTERN_MIN_BRIGHT:
                    # too dim to trust the pattern -> drop it entirely
                    pat = ""
                    solid = False
                    excluded = True
            p.is_solid = solid
            p.pattern = pat
            p.excluded = excluded

    @property
    def n(self):
        return len(self.particles)

    @property
    def n_solid(self):
        return sum(p.is_solid for p in self.particles) if self.classifiable else None

    @property
    def n_flat(self):
        # excluded / unclassified particles are neither solid nor undercooled
        return (sum(not p.is_solid and not getattr(p, "excluded", False)
                    and not getattr(p, "unclassified", False)
                    for p in self.particles)
                if self.classifiable else None)

    @property
    def n_excluded(self):
        return sum(getattr(p, "excluded", False) for p in self.particles)

    @property
    def has_patterns(self):
        return self.classifiable and any(len(p.pattern_probs) for p in self.particles)

    def pattern_counts(self):
        """{class: count} over solid particles (only meaningful when classifiable
        and patternnet ran). Undercooled particles are not counted here."""
        c = {k: 0 for k in PATTERN_CLASSES}
        for p in self.particles:
            if p.is_solid and p.pattern:
                c[p.pattern] += 1
        return c

    def diam_array(self, whole_only=True):
        """Diameters that are real sizes. `whole_only` drops both frame-cut and
        neighbour-occluded particles (see analyze.measurable)."""
        return np.array([p.diam_nm for p in self.particles
                         if not whole_only or measurable(p)])

    def title(self):
        return clean_title(os.path.splitext(self.image)[0])

    def stats(self):
        """Size statistics computed on fully visible particles only (not cut by
        the frame, not buried under a neighbour), while the count still reflects
        every detected particle.

        When nothing passes the gate the answer is an EMPTY distribution (zeros /
        "—" upstream), never a fallback to every detected particle: measuring
        buried caps is the exact error the gate exists to prevent, and doing it
        silently — with MEASURED reading the same as the total — is worse than
        reporting nothing.
        """
        d = self.diam_array(whole_only=True)
        return dict(
            count_total=self.n,
            count_measured=int(d.size),
            n_images=1,
            detector=self.detector,
            classifiable=self.classifiable,
            n_solid=self.n_solid, n_flat=self.n_flat,
            has_patterns=self.has_patterns,
            pattern_counts=self.pattern_counts(),
            **_diam_summary(d),
        )


@dataclass
class Aggregate:
    """Pools particles across several analysed images so the user can get one
    combined size distribution / average over a whole experiment (or a chosen
    subset). Diameters are already in nm per each image's own calibration, so
    concatenating them is valid even if the images had different magnifications."""
    analyses: list
    image: str = ""

    def __post_init__(self):
        if not self.image:
            self.image = f"{len(self.analyses)} görsel (birleşik)"

    @property
    def particles(self):
        out = []
        for a in self.analyses:
            out.extend(a.particles)
        return out

    @property
    def classifiable(self):
        return any(a.classifiable for a in self.analyses)

    @property
    def has_patterns(self):
        return any(a.has_patterns for a in self.analyses)

    def pattern_counts(self):
        c = {k: 0 for k in PATTERN_CLASSES}
        for a in self.analyses:
            for k, v in a.pattern_counts().items():
                c[k] += v
        return c

    def title(self):
        titles = [clean_title(os.path.splitext(a.image)[0]) for a in self.analyses]
        if len(set(titles)) == 1:
            return titles[0]
        pref = os.path.commonprefix(titles).rstrip(" -_")
        return pref or f"{len(self.analyses)} images"

    def diam_array(self, whole_only=True):
        parts = [a.diam_array(whole_only) for a in self.analyses]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])

    def stats(self):
        d = self.diam_array(whole_only=True)      # no fallback — see Analysis.stats
        cls = [a for a in self.analyses if a.classifiable]
        n_solid = sum(a.n_solid for a in cls) if cls else None
        n_flat = sum(a.n_flat for a in cls) if cls else None
        return dict(
            count_total=sum(a.n for a in self.analyses),
            count_measured=int(d.size),
            n_images=len(self.analyses),
            detector="+".join(sorted({a.detector or "?" for a in self.analyses})),
            classifiable=bool(cls),
            n_classifiable_images=len(cls),
            n_solid=n_solid, n_flat=n_flat,
            has_patterns=self.has_patterns,
            pattern_counts=self.pattern_counts(),
            **_diam_summary(d),
        )


def _interior_structure(gray: np.ndarray, masks: np.ndarray, regions) -> float:
    """Median over the biggest particles of (blurred interior spread / noise floor).

    Dimensionless by construction — see STRUCTURE_MIN. Uses the largest few
    particles because small ones have too few interior pixels left after the
    erosion to give a stable spread, and the question ("can this image resolve
    what is inside a particle at all?") is answered best where there is most to
    see. Returns 0.0 when nothing is big enough to judge, which reads as "no
    evidence" and leaves the caller to decide.
    """
    import cv2
    g = gray.astype(np.float32)
    nmad = float(np.median(np.abs(g - cv2.GaussianBlur(g, (0, 0), 3.0)))) + 0.3
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    big = sorted(regions, key=lambda rp: -rp.area)[:40]
    ker = np.ones((7, 7), np.uint8)
    vals = []
    for rp in big:
        if rp.area < 400:
            break
        y0, x0, y1, x1 = rp.bbox
        m = cv2.erode((masks[y0:y1, x0:x1] == rp.label).astype(np.uint8), ker).astype(bool)
        if m.sum() < 100:
            continue
        v = blur[y0:y1, x0:x1][m]
        vals.append(float(np.median(np.abs(v - np.median(v)))) / nmad)
    return float(np.median(vals)) if vals else 0.0


# Cellpose segmentation is the slow, CPU/GPU-heavy step (~seconds to a minute per
# image). Its masks depend only on the image pixels + the fixed eval params, so
# we cache them keyed by path+mtime+size. Re-analysing the same image (toggling
# Patterns, or re-running after retraining to see the new model) then skips
# segmentation entirely and only re-runs the fast CNN classification. Bump
# _SEG_VERSION if the eval params below change, to invalidate stale caches.
_SEG_VERSION = 1


def _compact_labels(masks):
    """Label images as the smallest integer type that fits.

    skimage's label() hands back int64: 8 bytes per pixel for ids that never pass
    a few thousand. A 1536x1033 mask is then 12.7 MB instead of 3.2 MB, and every
    analysis keeps one in memory AND in session.pkl (43 analyses = 591 MB of
    session file, 614 MB resident). uint16 covers 65535 particles per image.
    """
    top = int(masks.max()) if masks.size else 0
    dt = np.uint16 if top < np.iinfo(np.uint16).max else np.int32
    return masks.astype(dt, copy=False)


def _mask_cache_path(image_path):
    import hashlib
    from paths import sub_dir
    d = sub_dir("mask_cache")
    try:
        st = os.stat(image_path)
    except OSError:
        return None
    h = hashlib.md5(f"{os.path.abspath(image_path)}|{st.st_mtime_ns}|"
                    f"{st.st_size}|v{_SEG_VERSION}".encode()).hexdigest()
    return os.path.join(d, h + ".npz")


def _segment(gray, image_path):
    """Cellpose masks for `gray`, from cache when available (else compute + store)."""
    from skimage.measure import label as sk_label
    ck = _mask_cache_path(image_path)
    if ck and os.path.exists(ck):
        try:
            cached = np.load(ck)["masks"]
            if cached.shape == gray.shape:
                # .copy(): writable (the small-particle merge edits it in place)
                return _compact_labels(cached).copy()
            # Measuring the info bar (rather than assuming 70 px) moved the
            # micrograph's bottom edge, which would otherwise throw away every
            # cached mask and force a full re-segmentation of the library —
            # ~45 min of sustained Cellpose load for a boundary that shifted by
            # 9 rows. When the cached mask is merely TALLER than the crop we
            # now want, and as wide, the overlap is the same pixels Cellpose
            # already saw, so trim instead of recompute. Rows only ever leave
            # from the bottom, and anything they touch is a frame-edge particle
            # the measurement gate excludes anyway.
            if (cached.ndim == gray.ndim and cached.shape[1] == gray.shape[1]
                    and cached.shape[0] > gray.shape[0]):
                return _compact_labels(cached[:gray.shape[0]]).copy()
        except Exception:
            pass
    masks, _, _ = _get_model().eval(gray, diameter=None, flow_threshold=0.4,
                                    cellprob_threshold=0.0)
    masks = _compact_labels(sk_label(masks))     # contiguous integer labels
    if ck:
        try:
            np.savez_compressed(ck, masks=masks)
        except OSError:
            pass
    return masks


def apply_count_floors(analysis) -> int:
    """Re-apply the MIN_COUNT floors to an analysis that was produced before they
    existed, and return how many particles it dropped.

    The floors are pure post-segmentation filtering — the masks themselves are
    unchanged — so a saved analysis can be brought up to date in milliseconds
    instead of being re-segmented. Used when restoring a session: 36 of the 48
    saved photos carried 1083 speckle "particles" (2026-07-30), which would
    otherwise sit in the counts and the size statistics until each photo happened
    to be analysed again.
    """
    a = analysis
    m = a.label_mask
    if m is None or not a.particles:
        return 0
    ids, cnt = np.unique(m[m > 0], return_counts=True)
    area = dict(zip(ids.tolist(), cnt.tolist()))
    keep = [p for p in a.particles
            if area.get(p.id, 0) >= MIN_COUNT_AREA_PX
            and p.diam_px >= MIN_COUNT_DIAM_PX]
    dropped = len(a.particles) - len(keep)
    if not dropped:
        return 0
    a.particles = keep
    alive = np.zeros(int(m.max()) + 1, bool)
    for p in keep:
        alive[p.id] = True
    m[~alive[m]] = 0            # and out of the mask, so no ghost outlines
    return dropped


def _read_scale_cached(image_path):
    """read_scale() runs EasyOCR on the info bar — ~4 s, and its answer depends
    only on the file's own pixels, exactly like the segmentation cache. Re-running
    an image (a new threshold, a retrained pattern net) shouldn't pay for it
    twice. Cached next to the masks, keyed by path+mtime+size."""
    import json
    ck = _mask_cache_path(image_path)
    ck = ck and ck[:-4] + ".scale.json"
    if ck and os.path.exists(ck):
        try:
            with open(ck) as f:
                return ScaleResult(**json.load(f))
        except Exception:
            pass                                  # stale/corrupt -> just re-read
    sc = read_scale(image_path)
    if ck:
        try:
            d = {k: (v.item() if isinstance(v, np.generic) else v)
                 for k, v in dataclasses.asdict(sc).items()}   # numpy ints -> JSON
            with open(ck, "w") as f:
                json.dump(d, f)
        except OSError:
            pass
    return sc


def analyze_image(image_path: str, facet_thresh: float = DEFAULT_FACET_THRESH,
                  min_diam_nm: float = 0.0,
                  do_class: bool = True, do_pattern: bool = True) -> Analysis:
    """`do_class` / `do_pattern` skip the CNN evaluations the user didn't ask
    for (picked in the Analyze menu), so e.g. a size-only run stays fast.
    Patterns require the solid/undercooled step, so `do_pattern` implies it."""
    from skimage.measure import regionprops
    scale = _read_scale_cached(image_path)
    gray_full = np.array(Image.open(image_path).convert("L"))
    top = _find_info_bar_top(gray_full)
    gray = gray_full[:top, :]  # micrograph only

    masks = _segment(gray, image_path)

    # recover the small particles Cellpose missed and merge them in
    add, small_ids = _small_particle_mask(gray, masks, scale.nm_per_px if scale.ok else 0)
    masks[add > 0] = add[add > 0]

    # Cellpose blobs must clear the area floor; recovered small ones are already
    # shape-filtered, so they always count.
    regions = [rp for rp in regionprops(masks, intensity_image=gray)
               if rp.area >= 30 or rp.label in small_ids]

    # Can this image support a solid/undercooled call at all? The detector
    # nameplate answers it when the instrument writes one; otherwise the image
    # is asked directly (see CONTRAST_DETECTORS / STRUCTURE_MIN).
    det = scale.detector.upper()
    structure = -1.0
    if det in CONTRAST_DETECTORS:
        can_read = True
    elif det in FLAT_DETECTORS:
        can_read = False
    else:
        structure = _interior_structure(gray, masks, regions)
        can_read = structure >= STRUCTURE_MIN
    classifiable = can_read and (do_class or do_pattern)

    # CNN solid/undercooled probability (trained on the user's labeled examples);
    # only run it on the larger Cellpose particles — recovered small ones are
    # < SOLID_MIN_DIAM_NM anyway, so they're undercooled by rule.
    probs = {}
    pat_probs = {}
    if classifiable:
        import model_solid_liquid
        big = [rp for rp in regions if rp.label not in small_ids]
        if model_solid_liquid.available():
            probs = model_solid_liquid.solid_probs(gray, masks, big,
                                         scale.nm_per_px if scale.ok else 8.0)
        if do_pattern:
            import model_pattern
            if model_pattern.available():
                pat_probs = model_pattern.pattern_probs(gray, masks, big,
                                                     scale.nm_per_px if scale.ok else 8.0)

    particles = []
    h, w = gray.shape
    gray_med = float(np.median(gray))     # per-image reference for brightness
    for rp in regions:
        diam_px = float(rp.equivalent_diameter_area)
        diam_nm = diam_px * scale.nm_per_px if scale.ok else 0.0
        if diam_nm < min_diam_nm:
            continue
        # hard floors: sub-15 nm blobs are segmentation noise, and so is anything
        # only a few pixels across whatever the calibration says (see
        # MIN_COUNT_DIAM_PX — that is the one that catches the speckle inside big
        # particles on coarsely-sampled photos). Never counted, never measured.
        if scale.ok and diam_nm <= MIN_COUNT_DIAM_NM:
            continue
        if diam_px < MIN_COUNT_DIAM_PX:
            continue
        cy, cx = rp.centroid
        minr, minc, maxr, maxc = rp.bbox
        touches = (minr <= 1 or minc <= 1 or maxr >= h - 1 or maxc >= w - 1)
        # how much of the particle is cut off by the frame: length of its run
        # along the image border / diameter (0 = fully inside, ~1 = ~half off)
        ys, xs = rp.coords[:, 0], rp.coords[:, 1]
        on_edge = int(((ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)).sum())
        edge_cut = on_edge / diam_px if diam_px else 0.0
        major = float(rp.axis_major_length)
        minor = float(rp.axis_minor_length)
        solidity = float(rp.solidity)
        # interior mean grey relative to the image median: strongly negative for
        # the dim background/occluded particles the pattern CNN over-classifies
        bright = float(rp.intensity_mean) - gray_med
        ff = float(probs.get(rp.label, 0.0))
        pv = pat_probs.get(rp.label)
        pv = tuple(float(x) for x in pv) if pv is not None else ()
        # is_solid / pattern are finalised by reclassify() below (which also
        # applies the size prior); constructed here as a raw first pass.
        is_solid = classifiable and ff >= facet_thresh and diam_nm >= SOLID_MIN_DIAM_NM
        particles.append(Particle(
            id=int(rp.label), cx=cx, cy=cy,
            diam_px=diam_px, diam_nm=diam_nm,
            major_px=major,
            minor_px=minor,
            angle=float(rp.orientation),
            solidity=solidity,
            facet_frac=ff,
            is_solid=is_solid,
            touches_edge=touches,
            edge_cut=edge_cut,
            perim_px=float(rp.perimeter_crofton),
            bright=bright,
            pattern_probs=pv,
            pattern="",
        ))
    # Drop every label that did NOT become a counted particle from the mask too
    # (area-filtered texture specks, sub-15 nm blobs, etc.). They were already
    # excluded from the counts, but the mask still held them, so overlays drew
    # their outlines as ghost black dots inside large particles.
    keep = np.zeros(int(masks.max()) + 1, bool)
    for p in particles:
        keep[p.id] = True
    masks[~keep[masks]] = 0
    analysis = Analysis(os.path.basename(image_path), scale.nm_per_px,
                        scale.label_text, particles,
                        label_mask=masks, micrograph=gray, path=image_path,
                        detector=scale.detector, classifiable=classifiable,
                        evaluated_class=do_class or do_pattern,
                        evaluated_pattern=do_pattern,
                        pipeline=pipeline_stamp())
    if classifiable:
        analysis.reclassify(facet_thresh)
    # Hand the GPU blocks Cellpose just used back to the system. On an 8 GB
    # unified-memory Mac the model, the tiles and the app's own analyses all
    # share the same RAM, and holding the cache across images is what pushes a
    # long batch into swapping (which is also where the heat and the sudden
    # slow-downs come from).
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    return analysis


if __name__ == "__main__":
    import sys
    a = analyze_image(sys.argv[1])
    s = a.stats()
    cls = (f"solid={s['n_solid']} flat={s['n_flat']}" if s["classifiable"]
           else f"det={s['detector'] or '?'} (no readable internal contrast "
                f"-> classification skipped)")
    print(f"{a.image}: {s['count_total']} particles | {cls} "
          f"| mean={s['mean']:.0f} median={s['median']:.0f} "
          f"std={s['std']:.0f} nm (scale {a.nm_per_px:.3f} nm/px)")
