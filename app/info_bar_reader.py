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
    source: str = "bar"   # which field the calibration came from: "HFW" or "bar"
    hfw_nm: float = 0.0   # micrograph width in nm, when the instrument states it


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
    if top is None:
        return h - _INFO_BAR_H
    if top > 0 and med[top - 1] >= _BAR_RULE_MIN:
        top -= 1                        # the bright rule is chrome, not image
    return top


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
    chrome = binz.mean(axis=1) > 0.95

    # 1) vertical separators / border = near-full-height white columns
    col_ext = binz.sum(axis=0)
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
        row = binz[r, x0:x1]
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
    det = _detect_bar(gray)
    if det is None and gray.shape[0] > gray.shape[1]:
        # Portrait export: the instrument info bar is rotated to a side edge.
        # Rotate back to landscape and retry — nm-per-pixel is orientation-free.
        for k in (3, 1):                       # 90° CW, then CCW
            rot = np.rot90(gray, k)
            if _detect_bar(rot) is not None:
                gray = rot
                det = _detect_bar(gray)
                break
    h, w = gray.shape
    top = _find_info_bar_top(gray)
    if det is None:
        # No bar geometry — but the metadata table may still carry HFW, which
        # calibrates the image outright. Split the header/value rows at the
        # bar's midline, which is where every layout seen so far puts them.
        return _from_hfw(_read_fields(gray, top), w, top, "scale bar not found")
    lx, rx, bar_row, x0, x1, div_row = det
    bar_px = rx - lx

    # OCR the label: crop the whole scale cell above the divider. The label
    # ("1 µm") sits between the ticks; the neighbouring 'det' cell is outside x0.
    crop = gray[bar_row - 4:div_row - 1, x0:x1]
    up = np.array(Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3)))
    # constrain OCR to digits + unit characters so "200"/"500" aren't misread
    txt = " ".join(_get_reader().readtext(up, detail=0,
                                          allowlist="0123456789.,nmµup "))
    fields = _read_fields(gray, top)
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
    # to fail loudly — the caller lists the photo in a warning and shows 0 nm —
    # than to write plausible-looking diameters that are off by a factor.
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
    binz = band > 128
    col = binz.sum(axis=0)
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
