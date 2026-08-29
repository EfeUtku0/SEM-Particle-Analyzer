"""The example micrographs a fresh install starts with.

WHY THE APP SHIPS IMAGES AT ALL. Both networks were trained on ONE material
system — BiSn colloidal particles imaged on two specific SEMs — so a stranger
who downloads this and feeds it their own powder gets confident nonsense, not an
error. Starting them on photos the app is actually trained for makes the first
run show what the tool does; the guide and the README say the rest.

WHY THE FILES ARE COPIED OUT rather than referenced where they lie. The library
stores absolute paths, and the folder they ship in is the app bundle (read-only,
and replaced wholesale by the next version) or the source checkout (which the
user is free to move). Either way a referenced path goes stale and the rows turn
grey. A copy in the data folder is the user's own file from then on: it survives
an update, a move, and a re-clone, and deleting it is a normal library delete.

Seeding happens exactly once, on the launch that finds no library.json. After
that this module is never consulted again, so a user who deletes the examples
does not get them back on the next start.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

FOLDER_NAME = "Example images (BiSn)"
_EXT = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")


def bundled_dir():
    """Where the shipped examples live: inside the frozen bundle, else next to
    the source tree (app/ is one level down from the repo root)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "sample_images")
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_images")


def bundled_files():
    d = bundled_dir()
    try:
        names = sorted(n for n in os.listdir(d)
                       if n.lower().endswith(_EXT) and not n.startswith("."))
    except OSError:
        return []
    return [os.path.join(d, n) for n in names]


def install():
    """Copy the examples into the data folder, returning the copies (in order).

    Files that are already there are left alone, so this is safe to call twice.
    """
    from paths import sub_dir
    dst_dir = sub_dir("Examples")
    out = []
    for src in bundled_files():
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            out.append(dst)
        except OSError:
            continue                      # a copy that fails just isn't offered
    return out


def seed_library(path):
    """Write a first library.json holding the examples. True if one was written.

    The format is the one window_library reads; writing the file and letting the
    normal loader parse it keeps a single code path for building the tree.
    """
    files = install()
    if not files:
        return False
    data = {"version": 1, "tree": [{
        "type": "folder", "name": FOLDER_NAME, "expanded": True,
        "children": [{"type": "image", "path": p, "name": os.path.basename(p)}
                     for p in files],
    }]}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except OSError:
        return False
    return True
