"""CNN pattern sub-classifier for SOLID particles (trained in-app — see
model_pattern_training.py). Crops are raw (unblurred) 128px patches, resized and
robust-normalized at train/inference time by model_pattern_crops.featurize_batch —
see that module for why this beat the old blurred-64px pipeline it replaced
(2026-07-18 sweep, tools/exp.py).

Output per particle: softmax probabilities over
  0: janus     1: stripe     2: lamellar     3: composite

Only meaningful on CBS images and on particles already judged solid (patterns are
a property of crystalline particles); the caller applies it accordingly.
"""
from __future__ import annotations

import os
import numpy as np

import model_pattern_crops

SIZE = model_pattern_crops.SIZE
CLASSES = ["janus", "stripe", "lamellar", "composite"]
_BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "patternnet.pt")
# a model retrained in-app (Training mode) lands here and wins over the bundled
# one — the app bundle is read-only when frozen, the data folder isn't


def _retrained_path():
    from paths import data_dir
    return os.path.join(data_dir(), "patternnet.pt")
_net = None


def _model_path():
    p = _retrained_path()
    return p if os.path.exists(p) else _BUNDLED


def available() -> bool:
    return os.path.exists(_model_path())


def reload():
    """Drop the cached net so the next call loads the freshly trained model."""
    global _net
    _net = None


def _get_net():
    global _net
    if _net is None:
        import torch
        _net = torch.jit.load(_model_path(), map_location="cpu")
        _net.eval()
    return _net


def pattern_probs(gray, masks, regions, nm_per_px):
    """Return {label_id: prob_vector[4]} for the given regionprops list."""
    import torch
    imgs, sils, dnm, ids = [], [], [], []
    g = gray.astype(np.float32)
    npp = nm_per_px or 8.0
    for rp in regions:
        im, m = model_pattern_crops.raw_crop(g, masks, rp.label, rp.bbox)
        if im is not None:
            imgs.append(im); sils.append(m)
            dnm.append(2 * np.sqrt(rp.area / np.pi) * npp)   # true mask area, not resized crop
            ids.append(rp.label)
    if not ids:
        return {}
    X = model_pattern_crops.featurize_batch(np.array(imgs), np.array(sils), np.array(dnm))
    Xt = torch.tensor(X)
    net = _get_net()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            out.append(net(Xt[i:i + 256]).softmax(1).numpy())
    p = np.concatenate(out)
    return {lab: p[k] for k, lab in enumerate(ids)}
