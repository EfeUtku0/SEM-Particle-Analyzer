"""Which image files the app accepts, and how a .tif is made openable.

Kept apart from the library panel on purpose: the panel decides where a photo
sits in the tree, this decides whether a file is a photo at all and what path
should actually be opened for it. Both the importer and the tree ask here.
"""
from __future__ import annotations

import os
import hashlib
import traceback

import numpy as np
from PIL import Image

IMG_EXT = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")
TIF_EXT = (".tif", ".tiff")


def _converted_dir():
    """Cache for pngs made from imported .tif files (see _importable_path)."""
    from paths import sub_dir
    return sub_dir("converted")


def _importable_path(path):
    """SEM micrographs usually arrive as .tif, which Qt's on-screen preview and
    parts of the pipeline handle poorly (16-bit tiffs in particular). A .tif is
    therefore converted ONCE to an 8-bit png in the app's cache and that png is
    used everywhere in its place — both for a single dropped image and for every
    .tif found inside an imported folder. The original file on disk is never
    touched. Any non-tif path, and any tif we fail to convert, is returned
    unchanged so a problem file still appears in the list rather than vanishing.

    The cache is keyed on the source path + its modification time, so a
    re-import (or a session restore) reuses the same png, and editing the source
    tif produces a fresh conversion. That key names the DIRECTORY, not the file:
    the png keeps the tif's own basename, because this path is what the rest of
    the app treats as the image's identity. Naming the file after the key made
    "MU 1.tif" show up in the library as "20fe53d38bf9…" and — worse, because it
    is silent — handed smart-sort a hex string to parse instead of a sample name,
    so it could not group the photos at all."""
    if not path.lower().endswith(TIF_EXT):
        return path
    try:
        key = hashlib.md5(
            f"{os.path.abspath(path)}|{os.path.getmtime(path)}".encode()).hexdigest()
        d = os.path.join(_converted_dir(), key)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, os.path.splitext(os.path.basename(path))[0] + ".png")
        if not os.path.exists(out):
            with Image.open(path) as im:
                im.load()
                if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                    # 16-bit / float grayscale: scale the real min..max into 0..255
                    arr = np.asarray(im).astype(np.float64)
                    lo, hi = float(arr.min()), float(arr.max())
                    arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
                    conv = Image.fromarray(arr.astype(np.uint8), mode="L")
                elif im.mode in ("RGBA", "P", "LA"):
                    conv = im.convert("RGB")
                else:
                    conv = im
                conv.save(out, "PNG")
        return out
    except Exception:
        traceback.print_exc()
        return path
