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
    from paths import named_user_dir
    return named_user_dir("SEMPA_TRAIN_DIR", "SEM Eğitim")


def stem_for(image_name):
    """The on-disk stem save_confirmed uses for an image (name or full path)."""
    return _n(os.path.splitext(os.path.basename(image_name))[0])


def is_confirmed(image_name):
    """True when this image already has saved labels in the training folder."""
    return os.path.exists(os.path.join(_dir_path(), stem_for(image_name) + ".labels.json"))


def signature(labels, excluded=None):
    """A short fingerprint of exactly what a photo would contribute to the
    training set: which particle carries which class, plus the ones excluded.

    This is what makes "added, then edited" visible in the IMAGES panel. It is
    computed over particle id + class ONLY: whether a label came from a click or
    from the model's pre-fill changes nothing in the saved crops, so flipping a
    prediction to the same class by hand must NOT read as a change.
    """
    import hashlib
    parts = [f"{int(pid)}:{cls}" for pid, (cls, _fu) in sorted(labels.items())]
    parts += [f"{int(pid)}:x" for pid in sorted(excluded or ())]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


_SAVED_CACHE = {}          # stem -> (mtime, size, dict)  — the json is re-read
                           # only when it actually changed on disk


def saved(image_name):
    """What is currently in the training set for this image, or None.

    Returns {"labels": {pid: class}, "excluded": {pid}, "signature": str,
    "confirmed": "YYYY-MM-DD HH:MM"}. `signature` is recomputed from the rows
    when the file predates it, so photos added by an older build still compare
    correctly instead of all reading as "edited"."""
    p = os.path.join(_dir_path(), stem_for(image_name) + ".labels.json")
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    hit = _SAVED_CACHE.get(p)
    if hit and hit[0] == key:
        return hit[1]
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return None
    labels, excluded = {}, set()
    for row in data.get("labels", []):
        try:
            pid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("class") == "exclude":
            excluded.add(pid)
        else:
            labels[pid] = row.get("class")
    out = {"labels": labels, "excluded": excluded,
           "confirmed": data.get("confirmed", ""),
           "signature": data.get("signature") or signature(
               {k: (v, True) for k, v in labels.items()}, excluded)}
    _SAVED_CACHE[p] = (key, out)
    return out


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


def support_dir():
    from paths import data_dir
    return data_dir()


def save_confirmed(analysis, labels, excluded=None, meta=None):
    """Persist one confirmed photo. `labels` = {pid: (class, from_user)} with
    class in CLASSES or "undercooled". `excluded` = pids the user marked
    0=None/exclude (dropped from pattern data, but kept as training data for the
    visibility model: "this particle can't be reliably classified").

    `meta` = {pid: {...}} extra fields merged into that particle's json row.
    The bulk review passes what the model originally answered and how sure it
    was (`model_cls`, `conf`, `rank`, `n_items`), and that is the whole point:
    without it a reviewed label is indistinguishable from a hand-placed one, so
    "the model was wrong HERE, while it was confident" — the signal that says a
    model is systematically broken rather than merely imprecise — cannot be
    recovered once the model is retrained and its probabilities have moved.
    Unknown keys are written through untouched, and readers ignore what they
    don't know, so older label files stay valid.

    A photo in the GOLDEN set is refused outright (GoldenPhotoError). It would
    not leave the golden folder — nothing removes it — so it would sit in both
    at once, and the next training run would be graded on a photo it had just
    trained on. `golden_store.conflicts()` does find that, but only afterwards,
    with the model already trained and the ruler already spent. Refusing here
    covers every route in (this button, the bulk review, anything added later),
    because every one of them ends up in this function. Moving a photo the other
    way — golden to training — stays possible and stays deliberate: it is a
    drag in the Finder, which is the whole reason both folders are plain files.

    Returns (n_pattern_crops, n_total)."""
    from skimage.measure import regionprops
    from model_solid_liquid import make_crop
    import model_pattern_crops
    import overlay_draw
    import golden_store

    if golden_store.is_golden(analysis.image):
        raise golden_store.GoldenPhotoError(
            f"“{analysis.image}” is in the golden set, so it is never trained "
            f"on — that is what makes the accuracy figure mean anything. To "
            f"train on it, move its files out of the golden folder first.")

    excluded = set(excluded or ())
    meta_extra = meta or {}
    d = train_dir()
    stem = _n(os.path.splitext(analysis.image)[0])
    ext = os.path.splitext(analysis.path)[1] or ".jpeg"
    try:
        shutil.copy2(analysis.path, os.path.join(d, stem + ext))
    except OSError:
        pass                                    # original moved; labels still saved
    overlay_draw.render_training(analysis, labels).save(
        os.path.join(d, f"{stem} (etiketli).png"))

    by_id = {p.id: p for p in analysis.particles}

    def meta(pid):
        p = by_id.get(pid)
        return dict(cx=round(float(p.cx), 1) if p else None,
                    cy=round(float(p.cy), 1) if p else None,
                    diam_nm=round(float(p.diam_nm), 1) if p else None,
                    **(meta_extra.get(pid) or {}))
    rows = [{"id": int(pid), "class": cls, "source": "user" if fu else "model",
             **meta(pid)} for pid, (cls, fu) in sorted(labels.items())]
    rows += [{"id": int(pid), "class": "exclude", "source": "user", **meta(pid)}
             for pid in sorted(excluded)]
    with open(os.path.join(d, stem + ".labels.json"), "w") as f:
        json.dump({"image": analysis.path, "stem": stem,
                   "detector": analysis.detector,
                   "nm_per_px": analysis.nm_per_px,
                   "confirmed": time.strftime("%Y-%m-%d %H:%M"),
                   # lets the app tell "added" from "added, then edited"
                   "signature": signature(labels, excluded),
                   "labels": rows}, f, ensure_ascii=False, indent=1)

    # training-ready RAW crops for the 4 pattern classes (undercooled labels
    # stay in the json only — patternnet never sees undercooled particles).
    # Stored unprocessed (model_pattern_crops.raw_crop) so normalization/size stay a
    # train-time choice — see model_pattern_crops.py for why.
    gray_f = analysis.micrograph.astype(np.float32)
    masks = analysis.label_mask
    bbox = {rp.label: rp.bbox for rp in regionprops(masks)}
    nmpp = analysis.nm_per_px or 8.0
    imgs, sils, dnm, y = [], [], [], []
    for pid, (cls, _fu) in labels.items():
        if cls not in CLS_ID or pid not in bbox:
            continue
        p = by_id.get(pid)
        im, m = model_pattern_crops.raw_crop(gray_f, masks, pid, bbox[pid])
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
# longer fed to model_pattern.


def _image_for(stem):
    """The image a confirmed photo was labelled on: the original if it is still
    where it was, else the copy save_confirmed put in the training folder (which
    is why that copy exists — originals get renamed and moved)."""
    d = _dir_path()
    lj = os.path.join(d, stem + ".labels.json")
    if os.path.exists(lj):
        try:
            with open(lj) as f:
                ip = json.load(f).get("image")
            if ip and os.path.exists(ip):
                return ip
        except Exception:
            pass
    for e in _IMG_EXT:
        p = os.path.join(d, stem + e)
        if os.path.exists(p):
            return p
    return None


def measurable_now(keys, diam_nm, progress=None):
    """Which training crops belong to particles the app MEASURES today.

    The scoring the app reports has always been over every labelled particle,
    including the ones its own occlusion / frame / shape gate refuses to measure —
    so a janus the user labelled but the app deliberately leaves out of the
    statistics counted as a miss, and the score read worse than the app's actual
    output (user question, 2026-07-31: "training sette janus olarak belirlediğim
    bir parçacık şimdi ölçülmüyorsa sistem bunu yanlış sayıp doğruluğum düşebilir").
    This maps each crop back to its particle so the score can be restricted to the
    set the app actually reports on.

    The crops carry no particle id — they never needed one — so the match is by
    diameter within the crop's own photo: 96.9% of them are unique, and where two
    particles of a photo do share one, the entry is used only if they agree about
    being measurable. Everything unresolved is reported as unknown rather than
    guessed.

    The diameters are compared AS float32, because that is how save_confirmed
    stored them: rounding the analysis' float64 to any number of decimals misses
    the float32 value it was saved as often enough to throw away a quarter of the
    matches (measured: 76.8% matched by rounding, 97% by round-tripping instead).

    Returns (measurable, known) bool arrays aligned with `keys`.
    """
    import analyze
    keys = np.asarray(keys)
    crop_d = np.asarray(diam_nm, np.float32)
    meas = np.zeros(len(keys), bool)
    known = np.zeros(len(keys), bool)
    stems = sorted(set(keys.tolist()))
    for i, stem in enumerate(stems):
        if progress:
            progress(i, len(stems))
        path = _image_for(stem)
        if path is None:
            continue                      # image gone -> those crops stay unknown
        try:
            a = analyze.analyze_image(path)
        except Exception:
            continue
        table = {}
        for p in a.particles:
            k = np.float32(p.diam_nm).tobytes()      # same representation as saved
            m = bool(analyze.measurable(p))
            if k in table:
                if table[k] is not None and table[k] != m:
                    table[k] = None       # two particles, same size, disagree
            else:
                table[k] = m
        for idx in np.where(keys == stem)[0]:
            v = table.get(crop_d[idx].tobytes())
            if v is None:
                continue
            meas[idx] = v
            known[idx] = True
    if progress:
        progress(len(stems), len(stems))
    return meas, known


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
