"""Scales the user set by hand, one per photo, in the data folder.

The reader gets most photos right and says so when it cannot (see
info_bar_reader), but a photo it cannot read — a recompressed copy, an
instrument it has never seen — still needs sizes. The scale-check window lets
the user put the bar's ends where they are and type the label; that answer is
kept here and outranks the reader from then on (analyze._read_scale_cached asks
here first), so every later Analyze of the photo uses it.

It lives in the data folder, not next to the mask cache: the cache is
disposable and is left out of the backup, while this is the user's own input.
Keyed by absolute path, with the file's size alongside: a different file saved
under the same name must not inherit a scale that was set for another picture.
"""
from __future__ import annotations

import json
import os

from paths import data_dir

FILE = "scale_overrides.json"


def _path():
    return os.path.join(data_dir(), FILE)


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict):
    tmp = _path() + ".yaziliyor"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _path())


def _size(image_path):
    try:
        return os.path.getsize(image_path)
    except OSError:
        return None


def get(image_path):
    """The hand-set ScaleResult for this photo, or None."""
    rec = _load().get(os.path.abspath(image_path))
    if not rec or rec.get("size") != _size(image_path):
        return None
    from info_bar_reader import ScaleResult, READER_VERSION
    return ScaleResult(
        nm_per_px=float(rec["nm_per_px"]), bar_px=int(rec["bar_px"]),
        label_nm=float(rec["label_nm"]), label_text=rec["label_text"],
        left_x=int(rec["left_x"]), right_x=int(rec["right_x"]),
        bar_row=int(rec["bar_row"]), ok=True,
        note="set by hand in the scale check", source="manual", check="manual",
        info_top=int(rec.get("info_top", 0)), cell_x0=int(rec.get("cell_x0", 0)),
        cell_x1=int(rec.get("cell_x1", 0)), div_row=int(rec.get("div_row", 0)),
        detector=rec.get("detector", ""), instrument=rec.get("instrument", ""),
        mag=int(rec.get("mag", 0)), mag_npp=float(rec.get("mag_npp", 0.0)),
        version=READER_VERSION)


def put(image_path, label_nm, label_text, left_x, right_x, bar_row, auto=None):
    """Keep a hand-set scale. `auto` is the reader's own ScaleResult, whose
    geometry and nameplate fields are carried along for the check window."""
    bar_px = int(right_x) - int(left_x)
    if bar_px <= 0 or label_nm <= 0:
        raise ValueError("the bar must have a length and the label a value")
    rec = {"nm_per_px": float(label_nm) / bar_px, "bar_px": bar_px,
           "label_nm": float(label_nm), "label_text": label_text,
           "left_x": int(left_x), "right_x": int(right_x), "bar_row": int(bar_row),
           "size": _size(image_path)}
    if auto is not None:
        for k in ("info_top", "cell_x0", "cell_x1", "div_row", "detector",
                  "instrument", "mag", "mag_npp"):
            v = getattr(auto, k, None)
            rec[k] = v.item() if hasattr(v, "item") else v
    d = _load()
    d[os.path.abspath(image_path)] = rec
    _save(d)
    return get(image_path)


def clear(image_path):
    d = _load()
    if d.pop(os.path.abspath(image_path), None) is not None:
        _save(d)
