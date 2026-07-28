"""Training-mode data layer.

Everything the user teaches the app lives in ONE visible folder on the Desktop
(~/Desktop/SEM Eğitim). Per confirmed photo:
  <stem>.jpeg           copy of the original image
  <stem> (etiketli).png the labelled overlay as the user confirmed it
  <stem>.labels.json    per-particle labels (id, class, user/model source, pos)
  <stem>.crops.npz      training-ready pattern crops (X, y, key)
Plus a one-time copy of the old green-mark dataset ("temel veri.npz") so the
whole training corpus sits in the same place. Re-confirming a photo simply
overwrites its files. At train time everything is merged; if a photo was also
part of the old dataset, the new click-labels win.
"""
from __future__ import annotations

import os
import json
import shutil
import time
import unicodedata

import numpy as np

# pattern label ids — must match analyze.PATTERN_CLASSES / the old dataset
CLASSES = ["janus", "stripe", "lamellar", "composite"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

_APP = os.path.dirname(os.path.abspath(__file__))
_OLD_CROPS = os.path.join(_APP, "..", "training", "crops_pattern.npz")

TRAIN_MIN_PHOTOS = 3          # photos needed before "Train model" activates


def _n(s):
    return unicodedata.normalize("NFC", s)


def _dir_path():
    """The training folder's path WITHOUT creating it or copying anything in —
    for cheap read-only checks (e.g. is_confirmed) called per tree item."""
    from paths import documents_dir
    return (os.environ.get("SEMPA_TRAIN_DIR")
            or os.path.join(documents_dir(), "SEM Eğitim"))


def stem_for(image_name):
    """The on-disk stem save_confirmed uses for an image (name or full path)."""
    return _n(os.path.splitext(os.path.basename(image_name))[0])


def is_confirmed(image_name):
    """True when this image already has saved labels in the training folder."""
    return os.path.exists(os.path.join(_dir_path(), stem_for(image_name) + ".labels.json"))


def train_dir():
    d = _dir_path()
    os.makedirs(d, exist_ok=True)
    readme = os.path.join(d, "OKU BENİ.txt")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write(
                "SEM Particle Analyzer — eğitim verisi klasörü\n\n"
                "Uygulamanın Training modunda 'Add to training set' dediğin her\n"
                "fotoğraf buraya düşer: görselin kopyası, etiketli hâli (png),\n"
                "etiketler (.labels.json) ve eğitime hazır kırpımlar (.crops.npz).\n"
                "'temel veri.npz' eski yeşil-işaret verisidir; eğitimde hepsi\n"
                "birleştirilir (aynı fotoğraf iki yerde varsa tıklamaların kazanır).\n"
                "Bir fotoğrafı silersen bir sonraki eğitimde verisi de çıkar.\n")
    # Copy the green-mark era dataset in as soon as the folder is touched (not
    # only at train time) so it's visible right away, confirming the old data
    # is included even before the user has added enough new photos to train.
    dst = os.path.join(d, "temel veri.npz")
    if not os.path.exists(dst) and os.path.exists(_OLD_CROPS):
        shutil.copy2(_OLD_CROPS, dst)
    return d


def base_crops_path():
    """Path to the green-mark era dataset inside the visible folder, or None
    if there's nothing to copy (e.g. a packaged app with no training/ source)."""
    dst = os.path.join(train_dir(), "temel veri.npz")
    return dst if os.path.exists(dst) else None


def support_dir():
    from paths import data_dir
    return data_dir()


def save_confirmed(analysis, labels, excluded=None):
    """Persist one confirmed photo. `labels` = {pid: (class, from_user)} with
    class in CLASSES or "undercooled". `excluded` = pids the user marked
    0=None/exclude (dropped from pattern data, but kept as training data for the
    visibility model: "this particle can't be reliably classified").
    Returns (n_pattern_crops, n_total)."""
    from skimage.measure import regionprops
    from solidnet import make_crop
    import patterncrop
    import viz

    excluded = set(excluded or ())
    d = train_dir()
    stem = _n(os.path.splitext(analysis.image)[0])
    ext = os.path.splitext(analysis.path)[1] or ".jpeg"
    try:
        shutil.copy2(analysis.path, os.path.join(d, stem + ext))
    except OSError:
        pass                                    # original moved; labels still saved
    viz.render_training(analysis, labels).save(
        os.path.join(d, f"{stem} (etiketli).png"))

    by_id = {p.id: p for p in analysis.particles}

    def meta(pid):
        p = by_id.get(pid)
        return dict(cx=round(float(p.cx), 1) if p else None,
                    cy=round(float(p.cy), 1) if p else None,
                    diam_nm=round(float(p.diam_nm), 1) if p else None)
    rows = [{"id": int(pid), "class": cls, "source": "user" if fu else "model",
             **meta(pid)} for pid, (cls, fu) in sorted(labels.items())]
    rows += [{"id": int(pid), "class": "exclude", "source": "user", **meta(pid)}
             for pid in sorted(excluded)]
    with open(os.path.join(d, stem + ".labels.json"), "w") as f:
        json.dump({"image": analysis.path, "stem": stem,
                   "detector": analysis.detector,
                   "nm_per_px": analysis.nm_per_px,
                   "confirmed": time.strftime("%Y-%m-%d %H:%M"),
                   "labels": rows}, f, ensure_ascii=False, indent=1)

    # training-ready RAW crops for the 4 pattern classes (undercooled labels
    # stay in the json only — patternnet never sees undercooled particles).
    # Stored unprocessed (patterncrop.raw_crop) so normalization/size stay a
    # train-time choice — see patterncrop.py for why.
    gray_f = analysis.micrograph.astype(np.float32)
    masks = analysis.label_mask
    bbox = {rp.label: rp.bbox for rp in regionprops(masks)}
    nmpp = analysis.nm_per_px or 8.0
    imgs, sils, dnm, y = [], [], [], []
    for pid, (cls, _fu) in labels.items():
        if cls not in CLS_ID or pid not in bbox:
            continue
        p = by_id.get(pid)
        im, m = patterncrop.raw_crop(gray_f, masks, pid, bbox[pid])
        if im is not None:
            imgs.append(im.astype(np.float16))
            sils.append(m.astype(np.float16))
            dnm.append(p.diam_nm if p else 0.0)
            y.append(CLS_ID[cls])
    if y:
        np.savez_compressed(os.path.join(d, stem + ".crops.npz"),
                            img=np.array(imgs, np.float16),
                            sil=np.array(sils, np.float16),
                            diam_nm=np.array(dnm, np.float32),
                            y=np.array(y, np.int64),
                            key=np.array([stem] * len(y)))
    else:
        try:
            os.remove(os.path.join(d, stem + ".crops.npz"))
        except OSError:
            pass
    n_pattern = len(y)

    # visibility crops for visnet: 0 = keep (any particle the user could
    # classify — pattern or undercooled), 1 = exclude (marked None). Segmented-
    # but-unlabelled particles are left out (ambiguous — the user may just not
    # have got to them).
    vX, vy = [], []
    for pid in list(labels) + list(excluded):
        if pid not in bbox:
            continue
        c = make_crop(gray_f, masks, pid, bbox[pid], nmpp)
        if c is not None:
            vX.append(c)
            vy.append(1 if pid in excluded else 0)
    vpath = os.path.join(d, stem + ".vis.npz")
    if vX:
        np.savez_compressed(vpath, X=np.array(vX, np.float32),
                            y=np.array(vy, np.int64), key=np.array([stem] * len(vy)))
    else:
        try:
            os.remove(vpath)
        except OSError:
            pass
    return n_pattern, len(labels)


_IMG_EXT = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")


def delete_confirmed(stem):
    """Remove every file one confirmed photo wrote (image copy, labelled png,
    labels json, crops). Returns the basenames actually removed. The stem simply
    drops out of the next training run; any old green-mark rows it shadowed come
    back automatically."""
    d = train_dir()
    stem = _n(stem)
    targets = [stem + ".labels.json", stem + ".crops.npz", stem + ".vis.npz",
               f"{stem} (etiketli).png"] + [stem + e for e in _IMG_EXT]
    removed = []
    for name in targets:
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                os.remove(p)
                removed.append(name)
            except OSError:
                pass
    return removed


def confirmed():
    """[(stem, n_pattern_crops)] for every confirmed photo, sorted by name."""
    d = train_dir()
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".labels.json"):
            continue
        stem = f[:-len(".labels.json")]
        cp = os.path.join(d, stem + ".crops.npz")
        n = 0
        if os.path.exists(cp):
            try:
                n = int(len(np.load(cp)["y"]))
            except Exception:
                n = 0
        out.append((stem, n))
    return out


# The old green-mark dataset ("temel veri.npz") is pre-baked 64px blurred crops
# with known whole-image mislabels (see git history / SMALL_PATTERN_NM, removed
# 2026-07-18) and no raw image to re-extract from in the new pipeline. A sweep
# (tools/exp.py) comparing archs on ONLY the new click-labelled data (no old
# rows mixed in) beat the old small-CNN+old-data baseline outright (acc
# 0.612->0.698, lamellar recall 0.425->0.664 on identical folds), so the
# pattern classifier now trains purely on raw click-labelled crops below; the
# old dataset stays visible in the training folder for transparency but is no
# longer fed to patternnet.
def load_pattern_raw():
    """Concatenated raw crops (imgs, sils, diam_nm, y, keys) over every
    confirmed photo's click-labelled pattern particles."""
    d = train_dir()
    imgs, sils, dnm, y, keys = [], [], [], [], []
    for stem, _ in confirmed():
        cp = os.path.join(d, stem + ".crops.npz")
        if not os.path.exists(cp):
            continue
        c = np.load(cp)
        if "img" not in c:
            continue           # stale pre-migration file; regenerate by re-confirming
        imgs.append(c["img"].astype(np.float32))
        sils.append(c["sil"].astype(np.float32))
        dnm.append(c["diam_nm"].astype(np.float32))
        y.append(c["y"])
        keys.append(c["key"])
    if not y:
        raise RuntimeError("No training data found.")
    return (np.concatenate(imgs), np.concatenate(sils), np.concatenate(dnm),
            np.concatenate(y).astype(np.int64), np.concatenate(keys))
