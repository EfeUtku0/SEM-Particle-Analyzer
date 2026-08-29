"""The golden set: photos the model is scored on and NEVER trained on.

WHY IT IS ITS OWN FOLDER. Until now the only accuracy the app could report was a
cross-validation over the training photos — hold one fold out, train on the
rest, score, repeat. That number has two problems that no amount of arithmetic
fixes. It costs three quarters of every training run, and it is measured on
labels that mostly came out of the bulk review dialog, where the user was shown
the model's own answer and asked to confirm it. A label produced that way agrees
with the model more often than an independent one would, so the score flatters
the model by an amount nobody can quantify.

A golden set has neither problem. The photos sit outside the training folder, so
no model can ever have seen them; scoring is one forward pass, so it is free
next to a training run; and the labels were placed once, deliberately, without a
model's suggestion on screen.

LAYOUT — deliberately identical to the training folder, so a photo can be moved
between the two with the Finder and nothing else has to know:

    <stem>.labels.json    per-particle classes (the same file save_confirmed writes)
    <stem>.<ext>          a copy of the image
    <stem> (etiketli).png the labelled overlay, if it came from training

THE ONE INVARIANT: a photo must not be in both folders. Scoring a model on a
photo it trained on is not a measurement, and the failure is silent — the number
simply comes out high. `conflicts()` finds them and every caller is expected to
show them rather than quietly score anyway.

Photos are grouped by INSTRUMENT when reported (see `instrument_of`), because
"did the new microscope's data spoil the old one" is a question the pooled
number cannot answer — and it is the question that gets asked every time a new
machine's photos enter the training set.
"""
from __future__ import annotations

import json
import os
import unicodedata

IMG_EXT = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")

# the four pattern classes plus the two answers that mean "no pattern"
PATTERN_CLASSES = ["janus", "stripe", "lamellar", "composite"]
NOPAT = "nopat"
REPORT_CLASSES = PATTERN_CLASSES + [NOPAT]


def _n(s):
    return unicodedata.normalize("NFC", s)


def dir_path():
    """The golden folder's path, WITHOUT creating it — for cheap checks."""
    from paths import named_user_dir
    return named_user_dir("SEMPA_GOLDEN_DIR", "SEM Golden")


README = """SEM Particle Analyzer — golden (test) seti

Buradaki fotoğraflar modelin EĞİTİMİNE ASLA girmez; yalnızca eğitim bittiğinde
modelin doğruluğunu ölçmek için kullanılır. Bu yüzden buradaki sayı gerçek
sayıdır: model bu parçacıkları hiç görmemiştir.

Bir fotoğrafı golden yapmak için: önce Training modunda normal şekilde etiketle
ve "Add photo to training set" de, sonra o fotoğrafın dosyalarını (SEM Eğitim
klasöründeki <ad>.labels.json, görsel kopyası ve varsa "(etiketli).png") BURAYA
TAŞI. Taşımak yerine kopyalarsan ölçüm bozulur — aynı fotoğraf hem eğitimde hem
testte olamaz; uygulama bunu fark eder ve rapor ekranında uyarır.

İyi bir golden set: her sınıftan yeterince parçacık içeren, kullandığın her
mikroskoptan en az birkaç fotoğraf. Ne kadar çok fotoğraf, ölçüm o kadar
hassas — ama bu fotoğraflar eğitimden çıktığı için modele de o kadar az veri
kalır; birkaç fotoğraf cihaz başına iyi bir dengedir.
"""


def golden_dir():
    """The golden folder, created, with its README."""
    d = dir_path()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "OKU BENİ.txt")
    if not os.path.exists(p):
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(README)
        except OSError:
            pass
    return d


def image_for(stem, folder=None):
    """The image a golden photo was labelled on: the original where it was
    recorded if it is still there (that path keeps the Cellpose mask cache warm),
    else the copy sitting next to the labels."""
    d = folder or dir_path()
    lj = os.path.join(d, stem + ".labels.json")
    try:
        with open(lj, encoding="utf-8") as fh:
            ip = json.load(fh).get("image")
        if ip and os.path.exists(ip):
            return ip
    except (OSError, ValueError):
        pass
    for e in IMG_EXT:
        p = os.path.join(d, stem + e)
        if os.path.exists(p):
            return p
    return None


def instrument_of(stem, folder=None):
    """Which microscope a photo came from, read from its own labels file.

    The detector nameplate is what the two machines actually differ by and the
    app already stores it when a photo is confirmed: BIOMATEN writes "CBS",
    the METU-METE exports write nothing. Falling back to the file name would
    guess; this reads.
    """
    d = folder or dir_path()
    try:
        with open(os.path.join(d, stem + ".labels.json"), encoding="utf-8") as fh:
            det = (json.load(fh).get("detector") or "").strip().upper()
    except (OSError, ValueError):
        det = ""
    return det or "no nameplate"


def photos():
    """[(stem, image_path, instrument)] for every golden photo, by name.

    Photos whose image can no longer be found are skipped: labels alone cannot
    be scored, and a half-present photo must not turn into a silent zero.
    """
    d = dir_path()
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".labels.json"):
            continue
        stem = _n(f[:-len(".labels.json")])
        img = image_for(stem, d)
        if img:
            out.append((stem, img, instrument_of(stem, d)))
    return out


def is_golden(image_name):
    """True when this image is part of the golden set (name or full path).

    Cheap enough to call per click: one stat, no image work, same shape as
    training_store.is_confirmed."""
    return os.path.exists(os.path.join(
        dir_path(), _n(os.path.splitext(os.path.basename(image_name))[0])
        + ".labels.json"))


class GoldenPhotoError(RuntimeError):
    """Raised when something tries to put a golden photo into the training set.

    A separate class rather than a plain RuntimeError so the bulk-review save
    can tell "this photo is protected" (skip it, say so) apart from "this photo
    failed to save" (a real error), and report them differently.
    """


def truth_of(stem, folder=None):
    """{particle_id: class} as the user labelled it, plus the centroid it was
    labelled at. Returns [(id, class, cx, cy)] — the centroid is what the match
    is actually made on, because particle ids are per-segmentation and a photo
    re-analysed under a newer pipeline renumbers them."""
    d = folder or dir_path()
    try:
        with open(os.path.join(d, stem + ".labels.json"), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    rows = []
    for r in data.get("labels", []):
        try:
            pid = int(r["id"])
        except (KeyError, TypeError, ValueError):
            continue
        cx, cy = r.get("cx"), r.get("cy")
        if cx is None or cy is None:
            continue
        rows.append((pid, r.get("class"), float(cx), float(cy)))
    return rows


def report_class(user_class):
    """The user's label as the five-way answer the report scores.

    "undercooled" and "exclude" both collapse to `nopat`, because that is the
    same thing on screen: the particle is counted and measured, and no pattern
    is painted on it. Scoring them apart would invent a distinction the app does
    not make.
    """
    return user_class if user_class in PATTERN_CLASSES else NOPAT


def conflicts():
    """Stems present in BOTH folders right now — a measurement that would
    silently lie.

    Why nothing OLDER than "right now" is checked: scoring always runs
    immediately after training, against golden as it looks at that instant. A
    photo either sat in training during the run (then it is not in golden yet,
    so `evaluate` never sees it) or it did not (then scoring it is correct) — a
    run this pipeline produces can never end up graded on its own training
    data. The only way around that is a model trained BEFORE golden_store
    existed at all, later graded against a golden set assembled after the fact
    from what used to be training photos: a one-time bootstrapping fact, not a
    recurring risk, and not something a timestamp heuristic can safely catch —
    a photo's `confirmed` field records the last time ANY save_confirmed call
    touched it, which says nothing about whether that particular copy ever sat
    in the training folder THIS model was trained on. (Tried exactly that here
    on 2026-08-05: it flagged four golden photos already proven — by direct
    file-existence comparison — never to have been in the training folder at
    all. Reverted; see [[sem-analyzer-golden-report]].) The only sound check
    for that one-time case is what was used to correct the single affected
    history row by hand: direct on-disk evidence that the stem's crops once
    sat in the training folder, not an inference from a timestamp field.
    """
    import training_store
    try:
        train = {s for s, _ in training_store.confirmed()}
    except Exception:
        return []
    return sorted({s for s, _, _ in photos()} & train)


def summary():
    """(n_photos, {instrument: n_photos}) for the panel, without reading images."""
    ph = photos()
    by = {}
    for _, _, ins in ph:
        by[ins] = by.get(ins, 0) + 1
    return len(ph), by
