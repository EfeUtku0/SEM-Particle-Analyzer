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

import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from skimage.measure import regionprops, label as sk_label

from scale_reader import read_scale, _find_info_bar_top

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


_model = None


def _get_model():
    global _model
    if _model is None:
        from cellpose import models
        _model = models.CellposeModel(gpu=True)
    return _model


DEFAULT_FACET_THRESH = 0.50   # P(solid) from the CNN classifier; slider adjusts
SOLID_MIN_DIAM_NM = 200       # particles below this are (almost) always undercooled
MIN_COUNT_DIAM_NM = 15.0      # anything this small is a mis-detection, not a real
                              # particle -> dropped entirely (not even counted)

# A particle is pattern-classified only when MOST of it is visible: it may touch
# the image frame a little, but not be cut off by a large chord; and it must be
# reasonably convex (not bitten into by a neighbour) and not a thin sliver.
# `edge_cut` is the length of the particle's run along the image border divided
# by its diameter — ~0 when merely tangent, ~1 when roughly half is off-frame. So
# a particle whose ~60%+ is inside still qualifies (user rule, 2026-07-18: the old
# "touches the edge at all -> reject" was far too aggressive).
PATTERN_MAX_EDGE_CUT = 0.90
PATTERN_MIN_SOLIDITY = 0.90
PATTERN_MIN_AR = 0.60         # minor/major axis ratio


def pattern_eligible(edge_cut, solidity, major_px, minor_px) -> bool:
    if edge_cut > PATTERN_MAX_EDGE_CUT:
        return False
    ar = minor_px / major_px if major_px else 0.0
    return solidity >= PATTERN_MIN_SOLIDITY and ar >= PATTERN_MIN_AR


# --- which particles get a SIZE (user rule, evolved 2026-07-25) ----------------
# A particle lying under its neighbours only shows a cap, so its equivalent
# diameter under-reports its true size. The history of this gate:
#   * "measure everything not frame-cut" -> measured buried caps, badly wrong.
#   * pattern_eligible (solidity>=0.90, ar>=0.60) -> too strict; it dropped fully
#     exposed but elongated / slightly-concave particles the user wanted measured,
#     while (it turns out) the genuinely bad ones — dim, behind neighbours — have
#     high solidity anyway and slipped through regardless.
# The genuinely unreliable particles are now removed up front by the brightness
# `excluded` flag (see reclassify). So the size gate can be generous on geometry:
#   * a particle the model gave a PATTERN is, by definition, exposed -> measure it;
#   * otherwise measure it if it's reasonably whole (looser than pattern_eligible:
#     no aspect-ratio test — an elongated particle still has a valid diameter —
#     and a lower solidity floor), but still drop frame-cut and heavily bitten.
# Still gates only the SIZE STATISTICS: the particle stays detected and counted.
MEASURE_MIN_SOLIDITY = 0.85
# A particle touching / clipped by the frame still has a usable diameter as long
# as most of it is in view. edge_cut = border run / diameter; the missing cap and
# the resulting diameter under-estimate grow with it:
#   0.40 -> ~0.7% ,  0.50 -> ~1.5% ,  0.60 -> ~2.6% ,  0.70 -> ~4.5% ,  0.90 -> ~11%.
# We measure up to 0.60 (~2/3 of the particle in view, <3% error). Beyond that a
# particle is roughly half off-frame — its visible part is a cap and its size
# reads 10-30% too small, so it's dropped (measured on the golden set the dropped
# edge particles have edge_cut median ~0.88, i.e. genuinely half-cut). Replaces
# the old "touches the frame at all -> drop", which lost many measurable ones.
EDGE_MEASURE_MAX = 0.60


def measurable(p) -> bool:
    """True when enough of the particle is exposed for its equivalent diameter to
    be a real size rather than the size of a visible cap."""
    if getattr(p, "user_measurable", None) is not None:
        return bool(p.user_measurable)      # the user's judgement overrides
    if getattr(p, "excluded", False):
        return False                        # dim/unreliable -> not a real size
    if getattr(p, "edge_cut", 0.0) > EDGE_MEASURE_MAX:
        return False                        # a big chord is off-frame -> size wrong
    if p.pattern:
        return True                         # has a pattern -> exposed enough
    return p.solidity >= MEASURE_MIN_SOLIDITY


# Size prior on the pattern decision (user rule, 2026-07-17): a ~250-300 nm
# particle is almost always undercooled (liquid). composite/lamellar are
# physically impossible at that size, stripe is unlikely, janus is possible.
# So below PATTERN_SMALL_NM we forbid composite/lamellar, demand strong evidence
# for stripe, allow janus with moderate evidence, and otherwise fall back to
# undercooled. Above it, the plain argmax over all four classes is used.
PATTERN_SMALL_NM = 350.0
STRIPE_SMALL_MIN_CONF = 0.80   # softmax prob a small particle needs to be stripe
JANUS_SMALL_MIN_CONF = 0.55    # ...to be janus
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
JANUS_MIN_SOLID = 0.55


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
    PATTERN_CLASSES — or "exclude" when there is no credible / reliable pattern
    (small with no confident call, too dim to trust, or edge/occluded). Never
    returns "undercooled": the user's click already says this isn't liquid, so an
    unrecognisable pattern falls through to exclude rather than back to undercooled."""
    if (len(p.pattern_probs)
            and pattern_eligible(getattr(p, "edge_cut", 0.0), p.solidity,
                                 p.major_px, p.minor_px)):
        pat = size_aware_pattern(p.diam_nm, p.pattern_probs)
        if pat and getattr(p, "bright", 0.0) >= PATTERN_MIN_BRIGHT:
            return pat
    return "exclude"


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
PATTERN_MIN_BRIGHT = -30.0

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
    npp = nm_per_px if nm_per_px else 8.0
    rmax = 0.5 * SMALL_MAX_NM / npp
    max_area = np.pi * rmax * rmax
    min_area = max(4.0, np.pi * (0.5 * SMALL_MIN_NM / npp) ** 2)
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
    bright: float = 0.0         # interior mean grey − image median (dim = negative)
    pattern_probs: tuple = ()   # patternnet softmax over PATTERN_CLASSES (or ())
    pattern: str = ""           # argmax label for solid particles ("" otherwise)
    excluded: bool = False      # dim/unreliable: dropped from stats & classification
    user_measurable: bool = None   # user override of the occlusion gate (None = auto)


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
                    solid = False         # -> undercooled (liquid)
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
        # excluded particles are neither solid nor undercooled — drop them
        return (sum(not p.is_solid and not getattr(p, "excluded", False)
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

    def scale_line(self):
        return f"{self.scale_label}   ({self.nm_per_px:.3f} nm/px)   ·   {self.detector or '?'}"

    def stats(self):
        """Size statistics computed on fully visible particles only (not cut by
        the frame, not buried under a neighbour), while the count still reflects
        every detected particle."""
        d = self.diam_array(whole_only=True)
        if d.size == 0:
            d = self.diam_array(whole_only=False)
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

    def scale_line(self):
        dets = sorted({a.detector or "?" for a in self.analyses})
        return f"{len(self.analyses)} görsel birleşik   ·   dedektör: {', '.join(dets)}"

    def diam_array(self, whole_only=True):
        parts = [a.diam_array(whole_only) for a in self.analyses]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])

    def stats(self):
        d = self.diam_array(whole_only=True)
        if d.size == 0:
            d = self.diam_array(whole_only=False)
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


def _facet_fraction(gray_crop: np.ndarray, mask: np.ndarray,
                    margin: float = 18.0) -> float:
    """Fraction of the particle interior covered by bright facets.

    Per the user's rule: an undercooled ('flat') particle is almost entirely one
    greyish tone, whereas a solid (crystalline) particle shows bright facet
    domains over a meaningful part of its area. We take the particle's own median
    grey as its base tone and count the interior pixels that are clearly brighter
    (base + margin). That brightness step is the facet signature; a smooth sphere
    only has a gentle shading gradient and scores near zero.
    """
    import cv2
    ys, xs = np.where(mask)
    if ys.size < 30:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = gray_crop[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    er = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)
    if er.sum() < 30:
        er = m.astype(bool)
    interior = sub[er]
    base = np.median(interior)
    bright = (sub > base + margin) & er
    return float(bright.sum()) / float(er.sum())


# Cellpose segmentation is the slow, CPU/GPU-heavy step (~seconds to a minute per
# image). Its masks depend only on the image pixels + the fixed eval params, so
# we cache them keyed by path+mtime+size. Re-analysing the same image (toggling
# Patterns, or re-running after retraining to see the new model) then skips
# segmentation entirely and only re-runs the fast CNN classification. Bump
# _SEG_VERSION if the eval params below change, to invalidate stale caches.
_SEG_VERSION = 1


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
    ck = _mask_cache_path(image_path)
    if ck and os.path.exists(ck):
        try:
            cached = np.load(ck)["masks"]
            if cached.shape == gray.shape:
                return cached.copy()             # writable (small-particle merge edits it)
        except Exception:
            pass
    masks, _, _ = _get_model().eval(gray, diameter=None, flow_threshold=0.4,
                                    cellprob_threshold=0.0)
    masks = sk_label(masks)                      # contiguous integer labels
    if ck:
        try:
            np.savez_compressed(ck, masks=masks)
        except OSError:
            pass
    return masks


def analyze_image(image_path: str, facet_thresh: float = DEFAULT_FACET_THRESH,
                  min_diam_nm: float = 0.0,
                  do_class: bool = True, do_pattern: bool = True) -> Analysis:
    """`do_class` / `do_pattern` skip the CNN evaluations the user didn't ask
    for (picked in the Analyze menu), so e.g. a size-only run stays fast.
    Patterns require the solid/undercooled step, so `do_pattern` implies it."""
    scale = read_scale(image_path)
    gray_full = np.array(Image.open(image_path).convert("L"))
    top = _find_info_bar_top(gray_full)
    gray = gray_full[:top, :]  # micrograph only

    masks = _segment(gray, image_path)

    # recover the small particles Cellpose missed and merge them in
    add, small_ids = _small_particle_mask(gray, masks, scale.nm_per_px if scale.ok else 0)
    masks[add > 0] = add[add > 0]

    classifiable = scale.detector.upper() == "CBS" and (do_class or do_pattern)
    # Cellpose blobs must clear the area floor; recovered small ones are already
    # shape-filtered, so they always count.
    regions = [rp for rp in regionprops(masks, intensity_image=gray)
               if rp.area >= 30 or rp.label in small_ids]

    # CNN solid/undercooled probability (trained on the user's labeled examples);
    # only run it on the larger Cellpose particles — recovered small ones are
    # < SOLID_MIN_DIAM_NM anyway, so they're undercooled by rule.
    probs = {}
    pat_probs = {}
    if classifiable:
        import solidnet
        big = [rp for rp in regions if rp.label not in small_ids]
        if solidnet.available():
            probs = solidnet.solid_probs(gray, masks, big,
                                         scale.nm_per_px if scale.ok else 8.0)
        if do_pattern:
            import patternnet
            if patternnet.available():
                pat_probs = patternnet.pattern_probs(gray, masks, big,
                                                     scale.nm_per_px if scale.ok else 8.0)

    particles = []
    h, w = gray.shape
    gray_med = float(np.median(gray))     # per-image reference for brightness
    for rp in regions:
        diam_px = float(rp.equivalent_diameter_area)
        diam_nm = diam_px * scale.nm_per_px if scale.ok else 0.0
        if diam_nm < min_diam_nm:
            continue
        # hard floor: sub-15 nm blobs are segmentation noise, never counted
        if scale.ok and diam_nm <= MIN_COUNT_DIAM_NM:
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
                        evaluated_pattern=do_pattern)
    if classifiable:
        analysis.reclassify(facet_thresh)
    return analysis


if __name__ == "__main__":
    import sys
    a = analyze_image(sys.argv[1])
    s = a.stats()
    cls = (f"solid={s['n_solid']} flat={s['n_flat']}" if s["classifiable"]
           else f"det={s['detector']} (no CBS -> classification skipped)")
    print(f"{a.image}: {s['count_total']} particles | {cls} "
          f"| mean={s['mean']:.0f} median={s['median']:.0f} "
          f"std={s['std']:.0f} nm (scale {a.nm_per_px:.3f} nm/px)")
