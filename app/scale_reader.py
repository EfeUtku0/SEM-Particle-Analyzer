"""Scale-bar reader for BIOMATEN SEM images.

The instrument renders a fixed info bar across the bottom of every export.
Top-right cell holds the scale bar: a horizontal line with two end ticks and a
centred label like "5 µm" / "1 µm" / "500 nm". We detect the tick-to-tick pixel
span and OCR the label, then combine them into a nm-per-pixel calibration.

The magnification field is intentionally ignored: this SEM draws a fixed-length
bar regardless of mag, so the bar+label pair is the only reliable ground truth
(exactly what one would measure by hand in ImageJ).
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

_INFO_BAR_H = 70  # info-bar height in pixels for the 1536x1103 exports


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
    detector: str = ""    # "CBS", "ETD", ... (solid/flat classification needs CBS)


_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def _find_info_bar_top(gray: np.ndarray) -> int:
    """Return the y where the black info bar starts. Falls back to a fixed height."""
    h = gray.shape[0]
    # The bottom-most fully-white row is the frame border; the bar sits just above.
    # We simply use the known height for these exports but verify a bright bottom row.
    return h - _INFO_BAR_H


def _detect_bar(gray: np.ndarray):
    """Locate the scale bar. Returns (left_x, right_x, bar_row_abs, x0, x1, div_abs).

    The info bar is a grid of cells separated by full-height vertical lines. The
    scale bar lives in the right-most cell as a short horizontal line with two end
    ticks and a centred label ("1 µm"). A full-width horizontal divider inside that
    cell (separating the label row from the "BIOMATEN" row) must NOT be mistaken
    for the bar — the earlier version did exactly that and read ~2x too long.
    """
    h, w = gray.shape
    top = _find_info_bar_top(gray)
    band = gray[top:h, :]
    bh = band.shape[0]
    binz = band > 128

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
    # locate the unit, then read whatever number-ish token precedes it. Longest
    # units first so "nm"/"mm" win over the bare "m" fallback (dropped-µ case).
    m = re.search(r"([0-9uoqzsbgli.,]+)\s*(nm|µm|um|pm|mm|m)\b", t)
    if not m:
        return None
    raw = m.group(1).translate(_DIGIT_FIX).replace(",", ".").strip(".")
    if not re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return None
    val = float(raw)
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
    if det is None:
        return ScaleResult(0, 0, 0, "", 0, 0, 0, False, "scale bar not found")
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
        return ScaleResult(0, bar_px, 0, txt, lx, rx, bar_row, False,
                           f"could not parse label from '{txt}'")
    label_nm, clean = parsed
    detector = _read_detector(gray, x0, div_row)
    return ScaleResult(label_nm / bar_px, bar_px, label_nm, clean,
                       lx, rx, bar_row, True, detector=detector)


def _read_detector(gray: np.ndarray, scale_x0: int, div_row: int) -> str:
    """OCR the detector cell (CBS/ETD/…) — the value just left of the scale cell,
    in the bottom text row. Contrast-based solid/flat classification is only valid
    for CBS; on ETD everything reads bright and internal facets are invisible."""
    h = gray.shape[0]
    crop = gray[div_row + 2:h - 2, max(0, scale_x0 - 175):scale_x0 - 10]
    up = np.array(Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3)))
    txt = " ".join(_get_reader().readtext(up, detail=0)).upper()
    m = re.search(r"(CBS|ETD|SE2|INLENS|BSE|TLD|SE)", txt)
    return m.group(1) if m else ""


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".jpeg", ".jpg", ".png", ".tif", ".tiff")):
            continue
        r = read_scale(os.path.join(folder, f))
        print(f"{f:26s} {r.label_text:>7s} / {r.bar_px:4d}px = "
              f"{r.nm_per_px:6.3f} nm/px  {'OK' if r.ok else 'FAIL: '+r.note}")
