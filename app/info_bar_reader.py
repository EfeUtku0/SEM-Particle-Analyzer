"""Info-bar reader: calibration (nm per pixel) and instrument metadata.

Every SEM here renders a black info bar across the bottom of its exports, laid
out as a grid of cells with a header row over a value row. Beyond that the
machines agree on very little — BIOMATEN writes date/HV/mag/pressure/temp/WD/
det and a 78 px bar, METU-METE writes WD/mag/HV/HFW and a 59 px one — so
nothing here may assume a fixed height, a fixed cell order, or that a given
field exists at all. The bar's height is measured (_find_info_bar_top) and its
cells are read by NAME (_read_fields), which is what lets one code path serve
both, and the next machine too.

Calibration is taken from the most direct thing the image offers:

  1. HFW ("horizontal field width") IS the micrograph's width in real units,
     so nm-per-pixel falls straight out of it. Preferred wherever present.
  2. Otherwise the scale bar: detect the tick-to-tick pixel span, OCR the
     label ("5 µm" / "1 µm" / "500 nm"), divide. This is what one would measure
     by hand in ImageJ, and it is all BIOMATEN gives us.

The magnification field is never used for calibration. It is not merely
redundant — it is the one field observed to be WRONG: MU 17 is stamped
"15 000 x" while its own HFW and its own scale bar agree on the 30 000 x value.
"""
from __future__ import annotations

import os
import re
import dataclasses
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Make easyocr model downloads work behind macOS' Python.framework cert setup.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

_UNIT_NM = {"nm": 1.0, "um": 1000.0, "µm": 1000.0, "pm": 1000.0, "mm": 1_000_000.0,
            "m": 1000.0}
# Note: OCR often reads "µm" as "pm"/"um"; on this instrument the micron sign is
# the only sub-mm unit that looks like that, so we map both to microns.
# Crucially, easyocr frequently DROPS the "µ" glyph entirely, turning "1 µm" into
# "1 m". This SEM never labels in metres/mm (particles are nm–µm scale), so a bare
# "m" unit is always a micron sign the OCR swallowed -> map "m" to microns too.

_INFO_BAR_H = 70  # legacy fallback only — see _find_info_bar_top

# Row-median bands used to find the info bar without knowing the instrument.
# The bar's background is drawn as pure black (measured medians: 4 on the
# BIOMATEN exports, 0-2 on the METU-METE ones) and stays that dark even on rows
# carrying text, because the glyphs are sparse and the MEDIAN ignores them. No
# micrograph row comes close: the darkest backgrounds in either corpus sit near
# 50. _BAR_RULE_MIN catches the bright full-width rule one instrument draws
# between the micrograph and the bar, which belongs to the chrome, not the image.
_BAR_DARK_MAX = 24.0
_BAR_RULE_MIN = 200.0
_BAR_DARK_FRAC = 0.90  # tolerates the bar's own white rules, rejects micrograph
_BAR_FLAT_MAX = 1.0    # bar background is a synthetic constant; grain is not
# 1.0 is the loosest value that still gets all 126 images in both corpora right
# (0.3-1.0 all score 126/126, 1.5 starts clipping micrograph rows off a photo
# with a black lower half). The bars themselves measure 0.23 (JPEG) and 0.00
# (TIFF), so this leaves ~4x headroom for a noisier bar on a future instrument.
_LINE_MIN = 64         # the bar's 1 px structural lines (see _detect_bar)
_SOFT_RULE_MIN = 50.0   # the rule over the bar after heavy recompression …
_SOFT_RULE_STD = 16.0   # … is still flat across the row (see _find_info_bar_top)
_BAR_MIN_H = 20        # below this a "bar" is a dark image edge, not a bar
_BAR_MAX_FRAC = 0.25   # above this the walk has run into a dark micrograph

# Sanity band for the final calibration (see read_scale). Wide enough to cover
# every magnification this instrument exports (measured: 1.79 – 26.88 nm/px) with
# a large margin either side, tight enough that a unit the OCR mangled cannot
# pass as a plausible scale.
MIN_NM_PER_PX, MAX_NM_PER_PX = 0.05, 500.0

# How far HFW and the measured bar may differ before it is worth saying so.
# Bar geometry carries a genuine ±0.4% tick-convention ambiguity (2026-07-30
# audit) and OCR rounds the printed HFW, so a few percent is normal; 10% is
# not, and means one of the two was misread.
HFW_BAR_TOL = 0.10

# Bump whenever read_scale can give a different answer for the same pixels:
# results are cached next to the masks (analyze._read_scale_cached), and a
# cache from the old reader would keep serving the old, wrong calibration.
# 2: recompressed-copy fixes (soft rule, grey separators) + magnification check.
READER_VERSION = 2

# Magnification × micrograph width is a constant of the instrument (the width
# of its display in mm). Measured 2026-09-26 on BIOMATEN: 412-413 mm across 8
# originals at 5 000-50 000 x, and 413-415 mm on six WhatsApp copies measured
# by hand — so nm/px = K / (width_px × mag). Only instruments listed here are
# checked. NOT METU-METE: its magnification field is the one seen to be wrong
# (see the module docstring), and it writes HFW, which outranks both anyway.
INSTRUMENT_K_MM = {"BIOMATEN": 413.0}
# Bar and magnification must agree this closely. The measured spread is ±1%;
# 5% leaves room for a bar a pixel off on a small copy, and is still far from
# any real misread (a wrong label digit is ≥ 20%, a dropped zero 10x).
MAG_TOL = 0.05
# The labels a scale bar is ever drawn with, in nm. When bar length and
# magnification pin the label to one of these, a misread digit ("5" read as
# "3", 31 AP) is corrected rather than trusted.
STANDARD_LABELS_NM = (100, 200, 250, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
                      10000, 20000, 30000, 40000, 50000, 100000, 200000, 500000)


@dataclass
class ScaleResult:
    nm_per_px: float
    bar_px: int
    label_nm: float
    label_text: str
    left_x: int
    right_x: int
    bar_row: int          # absolute y of the scale-bar line
    ok: bool
    note: str = ""
    detector: str = ""    # "CBS", "ETD", ... ("" when the bar carries no det cell)
    source: str = "bar"   # where the calibration came from: "HFW", "bar", "mag"
                          # (magnification × instrument constant) or "manual"
    hfw_nm: float = 0.0   # micrograph width in nm, when the instrument states it
    # --- cross-check against the magnification field (see _cross_check) ------
    mag: int = 0          # magnification as printed ("16 000 x" -> 16000), 0 = none
    mag_npp: float = 0.0  # nm/px the magnification implies on a known instrument
    check: str = ""       # "agree", "mag_misread", "label_fixed", "conflict",
                          # "mag_only", "manual" or "" (nothing to check against)
    instrument: str = ""  # "BIOMATEN" when recognised by its nameplate
    label_read: str = ""  # what the OCR said, when check == "label_fixed"
    # --- geometry, so the app can SHOW what was read (the scale-check view) --
    info_top: int = 0     # first row of the info bar
    cell_x0: int = 0      # the scale cell's interior, x from … to
    cell_x1: int = 0
    div_row: int = 0      # the divider under the bar's label row
    rot: int = 0          # np.rot90 turns applied to find a side-mounted bar
    version: int = 0      # READER_VERSION that produced this (cache key)


_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def _find_info_bar_top(gray: np.ndarray) -> int:
    """Return the y where the instrument's info bar starts.

    MEASURED, not assumed. The old fixed `h - _INFO_BAR_H` was written for one
    instrument's 1536x1103 exports and is wrong on both machines now in use:
    BIOMATEN actually draws a 78 px bar (so 8 rows of pure black were being fed
    to Cellpose and into the image median that `bright` is measured against),
    and the METU-METE exports draw 59 px (so 11 rows of real micrograph were
    being thrown away). A second instrument is exactly the case a fixed height
    cannot survive, and hard-coding a height per machine is what we are trying
    not to do — so read it off the pixels instead.

    A row is "bar-like" when its MEDIAN is near black: that holds even on rows
    carrying text, because the glyphs are sparse and the median ignores them,
    and it never holds on a micrograph row. Simply walking up while rows stay
    dark is not enough, though — both instruments draw full-width white rules
    that are part of the bar (BIOMATEN closes it with one on the very last row;
    METU-METE rules between the header and value rows). So take the TALLEST
    bottom-anchored band that is overwhelmingly dark and whose own top row is
    dark: embedded rules are absorbed, while reaching up into the micrograph
    dilutes the fraction immediately and is rejected. Fall back to the old
    constant if nothing plausible is found, rather than returning a wild crop.

    Darkness alone is not sufficient: a micrograph can legitimately be black
    across its lower half (UÖ - 15 Janus 10 is, for most of 200 rows). What
    separates them is FLATNESS — the bar's background is a synthetic constant
    (row medians 3-5, std 0.23) while even a black micrograph has grain (std
    10.0 over the same span). So the band's dark rows must also be near-equal.
    """
    h = gray.shape[0]
    med = np.median(gray, axis=1)
    dark = med <= _BAR_DARK_MAX
    limit = h - int(h * _BAR_MAX_FRAC)
    top = None
    for cand in range(h - _BAR_MIN_H, limit, -1):
        band = dark[cand:]
        if dark[cand] and band.mean() >= _BAR_DARK_FRAC \
                and med[cand:][band].std() <= _BAR_FLAT_MAX:
            top = cand
    if top is not None and top > 0 and med[top - 1] >= _BAR_RULE_MIN:
        top -= 1                        # the bright rule is chrome, not image
    # A recompressed copy (a photo forwarded over WhatsApp: ~940 px wide, heavy
    # JPEG) breaks the flatness test above from the inside: ringing lifts some
    # of the bar's own row medians to 6-11, and the band either splits halfway
    # down (25 AP was cut at row 637 of a bar starting at 625) or is not found
    # at all. The noise-tolerant reading in _soft_bar_top takes over then —
    # and ONLY then, so an original that passes the strict test is cropped
    # exactly as before (checked on all 198 library images).
    soft = _soft_bar_top(gray, med, limit)
    if top is None:
        return soft if soft is not None else h - _INFO_BAR_H
    # split = bar rows (dark) lie between the two answers; a thick rule alone
    # is no split, and re-cropping an analysed original would orphan its masks
    if soft is not None and soft < top and (med[soft:top] <= _BAR_DARK_MAX).any():
        return soft
    return top


def _soft_bar_top(gray: np.ndarray, med: np.ndarray, limit: int):
    """The info bar's top by its RULES rather than its flatness.

    Walks up from the bottom through the bar's dark rows and its thin rules,
    and stops at the first row that is neither (the micrograph). The top is
    the highest rule passed. A rule survives recompression as a flat line —
    measured on 40 simulated copies: median 55-182, row std <= 13.4, where a
    micrograph row, dark or bright, has std 37-76 and the bar's background
    rows a median of ~4. A flat run thicker than 3 rows is not a rule but an
    even patch of image (MU 84 has one right above its bar), so it ends the
    walk and is undone. With no rule above the bar at all, the highest dark
    row is the best there is.
    """
    h = gray.shape[0]
    std = gray[limit:].std(axis=1)
    top_rule = top_dark = None
    run, before_run = 0, None
    y = h - 1
    while y > limit:
        if med[y] <= _BAR_DARK_MAX:
            top_dark, run = y, 0
        elif med[y] >= _SOFT_RULE_MIN and std[y - limit] <= _SOFT_RULE_STD:
            if run == 0:
                before_run = top_rule
            run += 1
            if run > 3:
                top_rule = before_run
                break
            if h - y >= _BAR_MIN_H:
                top_rule = y
        else:
            break
        y -= 1
    if top_rule is not None:
        return top_rule
    if top_dark is not None and h - top_dark >= _BAR_MIN_H:
        return top_dark
    return None


def _detect_bar(gray: np.ndarray):
    """Locate the scale bar. Returns (left_x, right_x, bar_row_abs, x0, x1, div_abs).

    The info bar is a grid of cells separated by full-height vertical lines. The
    scale bar lives in the right-most cell as a short horizontal line with two end
    ticks and a centred label ("1 µm"). A full-width horizontal divider inside that
    cell (separating the label row from the "BIOMATEN" row) must NOT be mistaken
    for the bar — the earlier version did exactly that and read ~2x too long.

    `chrome` rows — the ones running white across the WHOLE image — are the
    bar's own outer frame and belong to neither search. They used to be outside
    the band only because the fixed 70 px height happened to start below the top
    border; once the height is measured properly the top border comes into view,
    and the divider search would seize on it as row 0 and then find no bar at
    all. Excluding them by what they are, rather than relying on where a
    hard-coded crop lands, is what makes the two independent of each other.
    """
    h, w = gray.shape
    top = _find_info_bar_top(gray)
    band = gray[top:h, :]
    bh = band.shape[0]
    binz = band > 128
    # The bar's STRUCTURE (cell separators, the divider, its outer frame) is
    # 1 px lines; on a downscaled copy they blur to grey ~100 and fail the
    # glyph threshold, the scale cell then starts one separator too early and
    # the 'det' text is read as part of the bar (31 AP: 287 px for a 181 px
    # bar). The background is ~4, so half of 128 still rejects it by a mile.
    lines = band > _LINE_MIN
    chrome = lines.mean(axis=1) > 0.95

    # 1) vertical separators / border = near-full-height white columns
    col_ext = lines.sum(axis=0)
    seps = [x for x in range(w) if col_ext[x] >= bh * 0.7]
    border = w - 1
    inner = [s for s in seps if s < border - 20]
    if not inner:
        return None
    last_sep = max(inner)
    x0, x1 = last_sep + 6, border - 3         # scale-cell interior
    cellw = x1 - x0

    # 2) horizontal divider row inside the scale cell (spans ~full cell width)
    div = bh - 1
    for r in range(bh):
        if chrome[r]:
            continue
        row = lines[r, x0:x1]
        xs = np.where(row)[0]
        if xs.size and (xs.max() - xs.min()) > 0.9 * cellw and row.mean() > 0.8:
            div = r
            break

    # 3) bar row = the row ABOVE the divider with the widest *bounded* span
    #    (tick-to-tick). The label breaks the line, so the row is sparse; that is
    #    fine — min/max x still mark the two end ticks.
    best = (0, None, 0, 0)
    for r in range(2, div - 1):
        if chrome[r]:
            continue
        row = binz[r, x0:x1]
        xs = np.where(row)[0]
        if xs.size < 4:
            continue
        span = xs.max() - xs.min()
        if span > 0.9 * cellw:            # skip a stray full-width line
            continue
        if span > best[0]:
            best = (span, r, xs.min() + x0, xs.max() + x0)
    span, r, lx, rx = best
    if r is None or span < 60:
        return None
    return lx, rx, top + r, x0, x1, top + div


# common OCR digit confusions on this instrument's font (e.g. "200"->"ZUU",
# "500"->"5uu"). Applied ONLY to the number token, never the unit.
_DIGIT_FIX = str.maketrans({"u": "0", "o": "0", "q": "0", "z": "2", "s": "5",
                            "b": "8", "g": "9", "l": "1", "i": "1", "t": "1"})


def _parse_label(text: str):
    """Extract (value_nm, cleaned_text) from an OCR string like 'det 5 pm'."""
    t = text.replace("μ", "µ").lower()
    # Locate the unit, then read whatever number-ish token precedes it. Longest
    # units first so "nm"/"mm" win over the bare "m" fallback (dropped-µ case).
    #
    # The number group is NON-GREEDY, and that is not cosmetic: 'u' is both a
    # digit confusion for '0' (see _DIGIT_FIX) and the first letter of "um", the
    # spelling easyocr returns most often for the micron sign. With a greedy
    # group, an OCR read with no space — "5um", which happens all the time — let
    # the number swallow the unit's 'u': "5u" -> "50", unit "m" -> 50 µm. That is
    # a silent TENFOLD calibration error on that photo, cached in the .scale.json
    # next to the masks, and every diameter it reports is 10x too large.
    # Non-greedy makes the unit win the 'u' ("5um" -> 5 µm) while a genuine
    # digit-confusion still resolves, because the group grows only as far as the
    # unit match needs ("2uu nm" -> 200 nm).
    m = re.search(r"([0-9uoqzsbgli.,]+?)\s*(nm|µm|um|pm|mm|m)\b", t)
    if not m:
        return None
    raw = m.group(1).translate(_DIGIT_FIX).replace(",", ".").strip(".")
    if not re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return None
    val = float(raw)
    if val <= 0:                      # "5 0m" and friends: a zero-length bar is
        return None                   # not a calibration, it is a failed read
    unit = m.group(2)
    disp = raw if "." not in raw else raw.rstrip("0").rstrip(".")
    return val * _UNIT_NM[unit], f"{disp} {'nm' if unit == 'nm' else 'µm'}"


def read_scale(image_path: str) -> ScaleResult:
    gray = np.array(Image.open(image_path).convert("L"))
    rot = 0
    det = _detect_bar(gray)
    if det is None and gray.shape[0] > gray.shape[1]:
        # Portrait export: the instrument info bar is rotated to a side edge.
        # Rotate back to landscape and retry — nm-per-pixel is orientation-free.
        for k in (3, 1):                       # 90° CW, then CCW
            if _detect_bar(np.rot90(gray, k)) is not None:
                gray, rot = np.rot90(gray, k), k
                det = _detect_bar(gray)
                break
    w = gray.shape[1]
    top = _find_info_bar_top(gray)
    fields = _read_fields(gray, top)
    res = _read_bar(gray, det, fields, top)
    res = _cross_check(res, fields, w)
    res.info_top, res.rot, res.version = top, rot, READER_VERSION
    if det is not None:
        res.cell_x0, res.cell_x1, res.div_row = det[3], det[4], det[5]
    # plain Python numbers out: numpy scalars leak into Qt (setVisible(np.bool_)
    # raises TypeError — the scale check would not open, 2026-09-27)
    for f in dataclasses.fields(res):
        v = getattr(res, f.name)
        if isinstance(v, np.generic):
            setattr(res, f.name, v.item())
    return res


def _read_bar(gray, det, fields, top) -> ScaleResult:
    """Calibration from HFW or the scale bar, before the magnification check."""
    w = gray.shape[1]
    if det is None:
        # No bar geometry — but the metadata table may still carry HFW, which
        # calibrates the image outright. Split the header/value rows at the
        # bar's midline, which is where every layout seen so far puts them.
        return _from_hfw(fields, w, top, "scale bar not found")
    lx, rx, bar_row, x0, x1, div_row = det
    bar_px = rx - lx

    # OCR the label: crop the whole scale cell above the divider. The label
    # ("1 µm") sits between the ticks; the neighbouring 'det' cell is outside x0.
    crop = gray[bar_row - 4:div_row - 1, x0:x1]
    up = np.array(Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3)))
    # constrain OCR to digits + unit characters so "200"/"500" aren't misread
    txt = " ".join(_get_reader().readtext(up, detail=0,
                                          allowlist="0123456789.,nmµup "))
    parsed = _parse_label(txt)
    if parsed is None:  # retry without the allowlist (rare fonts/units)
        txt = " ".join(_get_reader().readtext(up, detail=0))
        parsed = _parse_label(txt)
    if parsed is None:
        return _from_hfw(fields, w, top,
                         f"could not parse label from '{txt}'",
                         bar_px=bar_px, lx=lx, rx=rx, bar_row=bar_row)
    label_nm, clean = parsed
    npp = label_nm / bar_px
    # Last net under the OCR: this instrument works between ~1 and ~30 nm/px, so
    # anything wildly outside that is a misread label rather than a real
    # calibration (a swallowed unit turning "1 µm" into "1 mm" lands here). Better
    # to fail loudly — the caller lists the photo in a warning — than to write
    # plausible-looking diameters that are off by a factor.
    if not (MIN_NM_PER_PX <= npp <= MAX_NM_PER_PX):
        return _from_hfw(fields, w, top,
                         f"implausible calibration: '{clean}' over {bar_px} px "
                         f"= {npp:.3g} nm/px",
                         bar_px=bar_px, lx=lx, rx=rx, bar_row=bar_row)

    # HFW ("horizontal field width") is the width of the micrograph in real
    # units, so it calibrates the image directly: no tick-convention question,
    # no measuring a short line, no OCR of a two-character label. When the
    # instrument writes it, it OUTRANKS the bar. This is not a preference —
    # on the METU-METE exports the bar path reads MU 20 as 21.74 nm/px and
    # reports ok=True, while HFW (and the bar measured by hand) both say 29.1:
    # a silent 34% error on every diameter in that photo. The bar remains the
    # fallback for instruments that write no HFW, which is all of BIOMATEN.
    hfw_npp = _hfw_npp(fields, w)
    if hfw_npp:
        note = ""
        if abs(hfw_npp - npp) / hfw_npp > HFW_BAR_TOL:
            note = (f"HFW and scale bar disagree ({hfw_npp:.3g} vs {npp:.3g} "
                    f"nm/px); using HFW")
        return ScaleResult(hfw_npp, bar_px, label_nm, clean, lx, rx, bar_row,
                           True, note=note, detector=_detector_of(fields),
                           source="HFW", hfw_nm=hfw_npp * w)
    return ScaleResult(npp, bar_px, label_nm, clean, lx, rx, bar_row, True,
                       detector=_detector_of(fields), source="bar")


def _instrument_of(fields: dict) -> str:
    """The instrument, when its nameplate is on the bar (BIOMATEN prints its
    name under the scale bar, so it lands in _read_fields as that cell's value)."""
    for k, v in fields.items():
        for name in INSTRUMENT_K_MM:
            if name in f"{k} {v}".upper():
                return name
    return ""


# OCR's letter-for-digit swaps inside a magnification ("12 O00 x" was read as
# 1 200 and would have called a correct bar wrong by 10x)
_MAG_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1",
                          "I": "1", "|": "1", "S": "5", "B": "8"})


def _mag_of(fields: dict) -> int:
    """The printed magnification ("16 000 x" -> 16000), or 0."""
    vals = [v for k, v in fields.items() if k.startswith("MAG")]
    if not vals:
        # the cells ran together ("10.00 kV 40 000 X 1.00e-2 Pa"): take the one
        # number that is followed by the x sign
        vals = [m.group(1) for v in fields.values()
                for m in re.finditer(r"(\d[\dOo ]*)\s*[xX×]\b", v)]
    for v in vals:
        m = re.match(r"\s*([\dOoDQlIS|B][\dOoDQlIS|B ]*)", v)
        if not m:
            continue
        digits = m.group(1).translate(_MAG_FIX).replace(" ", "")
        if digits.isdigit() and 10 <= int(digits) <= 2_000_000:
            return int(digits)
    return 0


def _cross_check(res: ScaleResult, fields: dict, width_px: int) -> ScaleResult:
    """Hold the calibration against the magnification, on instruments whose
    constant is known. The bar is geometry plus a two-character OCR, and both
    halves have failed silently (31 AP: the 'det' cell read as part of a bar,
    and "5 µm" read as "3 µm" — 2.6x on every diameter, reported ok). The
    magnification is a third, independent witness:

      agree        within MAG_TOL — the bar stands, and now it is confirmed;
      mag_misread  off by a clean power of ten — a zero lost in the mag OCR,
                   the bar stands;
      label_fixed  bar length × mag pins the label to a standard value other
                   than the one read — the digit was misread, the label is
                   replaced (the geometry was right);
      conflict     none of the above — the bar is not trusted, the
                   magnification is used, and the photo is flagged;
      mag_only     the bar could not be read at all — calibrated from mag.

    HFW calibrations are left alone: HFW outranks both, and the instrument
    that writes it is the one with the wrong magnification field.
    """
    res.mag = _mag_of(fields)
    res.instrument = _instrument_of(fields)
    k = INSTRUMENT_K_MM.get(res.instrument)
    if not (k and res.mag) or res.source == "HFW":
        return res
    res.mag_npp = mag_npp = k * 1e6 / (width_px * res.mag)
    if not res.ok:
        why = res.note
        res.nm_per_px, res.ok, res.source, res.check = mag_npp, True, "mag", "mag_only"
        res.note = f"scale bar unreadable ({why}); calibrated from the magnification"
        return res
    ratio = res.nm_per_px / mag_npp
    if abs(ratio - 1) <= MAG_TOL:
        res.check = "agree"
        return res
    decade = round(np.log10(ratio))
    if decade and abs(ratio / 10.0 ** decade - 1) <= MAG_TOL:
        res.check = "mag_misread"
        res.note = (f"magnification read as {res.mag:,} x looks a power of ten off; "
                    f"the scale bar agrees with {res.mag * 10.0 ** decade:,.0f} x")
        return res
    if res.bar_px:
        fits = [L for L in STANDARD_LABELS_NM
                if abs(L / (mag_npp * res.bar_px) - 1) <= MAG_TOL]
        if len(fits) == 1:
            old = res.label_read = res.label_text
            L = fits[0]
            res.label_nm = float(L)
            res.label_text = (f"{L:g} nm" if L < 1000 else f"{L / 1000:g} µm")
            res.nm_per_px = L / res.bar_px
            res.check = "label_fixed"
            res.note = (f"label read as '{old}', but the bar length and the "
                        f"magnification only fit {res.label_text}")
            return res
    res.note = (f"scale bar ({res.nm_per_px:.3g} nm/px) disagrees with the "
                f"magnification ({mag_npp:.3g} nm/px); using the magnification")
    res.nm_per_px, res.source, res.check = mag_npp, "mag", "conflict"
    return res


def _hfw_npp(fields: dict, width_px: int) -> float:
    """nm-per-pixel from the HFW field, or 0.0 if the bar carries no usable one."""
    for k, v in fields.items():
        if not k.startswith("HFW"):
            continue
        parsed = _parse_label(v)
        if parsed is None:
            continue
        npp = parsed[0] / width_px
        if MIN_NM_PER_PX <= npp <= MAX_NM_PER_PX:
            return npp
    return 0.0


def _from_hfw(fields: dict, width_px: int, top: int, why: str,
              bar_px: int = 0, lx: int = 0, rx: int = 0, bar_row: int = 0):
    """Fall back to HFW when the scale bar could not be read."""
    npp = _hfw_npp(fields, width_px)
    det = _detector_of(fields)
    if npp:
        return ScaleResult(npp, bar_px, npp * width_px, f"HFW {npp * width_px / 1000:.3g} µm",
                           lx, rx, bar_row, True,
                           note=f"scale bar unreadable ({why}); calibrated from HFW",
                           detector=det, source="HFW", hfw_nm=npp * width_px)
    return ScaleResult(0, bar_px, 0, "", lx, rx, bar_row, False, why, detector=det)


def _cell_columns(gray: np.ndarray, top: int):
    """x boundaries of the info bar's cells, from its full-height separators."""
    band = gray[top:]
    bh = band.shape[0]
    col = (band > _LINE_MIN).sum(axis=0)
    seps = [x for x in range(gray.shape[1]) if col[x] >= bh * 0.7]
    if not seps:
        return []
    edges, run = [], [seps[0]]
    for x in seps[1:]:
        if x - run[-1] <= 2:
            run.append(x)
        else:
            edges.append(run[-1]); run = [x]
    edges.append(run[-1])
    return edges


def _read_fields(gray: np.ndarray, top: int) -> dict:
    """Read the info bar as a TABLE: {HEADER: value} for every labelled cell.

    The instrument writes its metadata as a header row over a value row, split
    by full-height separators. Reading it generically is what lets the app cope
    with machines that lay the bar out differently — BIOMATEN writes
    date/HV/mag/pressure/temp/WD/det, METU-METE writes WD/mag/HV/HFW, and the
    only thing they agree on is the grid. Asking for a field by NAME therefore
    works on both, where hard-coding "the cell left of the scale bar is the
    detector" silently returned the HFW value on the second machine.

    The header/value split is taken as the bar's own midline rather than the
    divider `_detect_bar` finds: that one is located inside the SCALE cell and
    lands on the wrong rule on the METU-METE layout (the bar line, not the
    caption divider), which garbles every cell. The midline reads both layouts
    cleanly, and it needs no scale bar to have been found at all.
    """
    out = {}
    h = gray.shape[0]
    div_row = top + (h - top) // 2
    edges = _cell_columns(gray, top)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 30:                       # logo / spacer, nothing to read
            continue
        head = _cell_text(gray[top + 2:div_row - 1, a + 4:b - 3])
        if not head:
            continue
        out[head.upper()] = _cell_text(gray[div_row + 2:h - 2, a + 4:b - 3])
    return out


def _cell_text(crop: np.ndarray) -> str:
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return ""
    up = np.array(Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3)))
    return " ".join(_get_reader().readtext(up, detail=0)).strip()


def _detector_of(fields: dict) -> str:
    """The detector nameplate, if this instrument writes one."""
    for k, v in fields.items():
        if k.startswith("DET"):
            m = re.search(r"(CBS|ETD|SE2|INLENS|BSE|TLD|SE)", v.upper())
            if m:
                return m.group(1)
    return ""


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample_images"
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpeg", ".jpg", ".png", ".tif", ".tiff")):
            continue
        r = read_scale(os.path.join(folder, f))
        print(f"{f:26s} {r.label_text:>7s} / {r.bar_px:4d}px = "
              f"{r.nm_per_px:6.3f} nm/px  {'OK' if r.ok else 'FAIL: '+r.note}")
