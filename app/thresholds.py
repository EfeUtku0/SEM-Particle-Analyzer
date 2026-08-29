"""The decision thresholds that live in a MODEL's units, kept where they can be
re-measured instead of hand-tuned.

Every constant in this pipeline is one of two kinds, and they age completely
differently:

  * PHYSICAL or GEOMETRIC — 200 nm, solidity 0.90, ellipse fill 0.92, the frame
    cut at 0.85. Written in nanometres or in shape, so they mean the same thing
    whatever network is loaded and whatever microscope took the photo. Those
    stay hard-coded in analyze.py with the measurements that justify them.

  * PROBABILITY-SPACE — the numbers in this file. They are cuts through the
    OUTPUT of a particular trained network, and a retrain rescales that output.
    A hand-tuned 0.55 does not survive the model it was tuned against.

Measured, 2026-08-04, which is why this file exists: with the pre-METU-METE
solid/liquid gate, the best facet threshold was 0.54 on BIOMATEN and 0.88 on
METU-METE — a gap of 0.34, so no single hand-set number could serve both
machines. After retraining that gate on both, the optima moved to 0.40 and 0.43:
one shared value now costs each instrument essentially nothing (0.000 / 0.001
balanced accuracy). The lesson is not "0.40 is the right number" — it is that
the right number is a function of the current model and has to be re-derived
with it. calibrate.py does the deriving; this module stores the answer and hands
it to analyze.py.

The file lives next to the retrained weights in the data folder. Absent (or
unreadable), every value falls back to the DEFAULTS below, which are the
hand-tuned numbers the app shipped with — so a fresh install behaves exactly as
it always did.
"""
from __future__ import annotations

import json
import os
import time

# The shipped values. Each is documented at its use site in analyze.py, with the
# experiment that produced it; do not change one here without changing that.
DEFAULTS = {
    "facet_thresh": 0.50,           # P(solid) above which a particle is crystalline
    "janus_min_solid": 0.55,        # …and how much more a JANUS call needs
    "nopattern_solid_conf": 0.90,   # stays solid though no pattern could be named
    "stripe_small_min_conf": 0.80,  # small-particle bars: stripe…
    "janus_small_min_conf": 0.55,   # …and janus
}

_cache = None


def path():
    from paths import data_dir
    return os.path.join(data_dir(), "thresholds.json")


def stored():
    """The raw record on disk, or {}. Never raises: a broken file must leave the
    app running on its defaults rather than not running at all."""
    try:
        with open(path(), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def all() -> dict:
    """Every threshold in force: the defaults, with any calibrated value laid
    over the top. Values outside (0, 1) are ignored — a nonsensical stored
    number must not be able to switch a gate off entirely."""
    global _cache
    if _cache is None:
        vals = dict(DEFAULTS)
        for k, v in (stored().get("values") or {}).items():
            if k in DEFAULTS:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if 0.0 < v < 1.0:
                    vals[k] = v
        _cache = vals
    return dict(_cache)


def get(name):
    return all()[name]


def reload():
    """Drop the cache so the next read picks up a fresh calibration."""
    global _cache
    _cache = None


def save(values, meta=None):
    """Write a calibration. `meta` records what it was fitted on, because a
    threshold with no provenance is indistinguishable from a guess."""
    rec = dict(values={k: float(v) for k, v in values.items() if k in DEFAULTS},
               meta=dict(meta or {}, saved=time.strftime("%Y-%m-%d %H:%M")))
    tmp = path() + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, path())
    except OSError:
        return False
    reload()
    return True


def describe():
    """One line per threshold: value, and whether it was measured or shipped."""
    cur, base = all(), DEFAULTS
    fitted = set((stored().get("values") or {}).keys())
    return [(k, cur[k], base[k], k in fitted) for k in sorted(DEFAULTS)]
