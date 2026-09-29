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
    """The net on the fastest device there is (see _infer for why not the CPU)."""
    global _net
    if _net is None:
        import torch
        dev = model_pattern_crops.torch_device()
        try:
            _net = torch.jit.load(_model_path(), map_location=dev)
        except Exception:
            _net = torch.jit.load(_model_path(), map_location="cpu")
        _net.eval()
    return _net


def _net_device(net):
    try:
        return next(net.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def _infer(net, Xt):
    """Softmax over every crop, in batches, on whichever device `net` is on.

    This used to run on the CPU because "the net is tiny". It is not: a resnet18
    at 128 px, run once per particle. On the M2 it was 14.6 s for 1522 crops,
    i.e. 16 of the 19 s a cached re-analysis took; the Apple GPU does the same
    batch in 1.2 s (2026-09-24). Same answers: the largest probability
    difference measured was 7e-6 and every one of the 1522 particles kept its
    class. Batches are smaller on the CPU fallback, where 64 beat 256
    (11.0 vs 14.6 s)."""
    import torch
    dev = _net_device(net)
    bs = 64 if dev.type == "cpu" else 128
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(net(Xt[i:i + bs].to(dev)).softmax(1).cpu().numpy())
    return np.concatenate(out)


def pattern_probs(gray, masks, regions, nm_per_px):
    """Return {label_id: prob_vector[4]} for the given regionprops list."""
    global _net
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
    try:
        p = _infer(net, Xt)
    except RuntimeError:
        if _net_device(net).type == "cpu":
            raise
        # a GPU that fails mid-run (out of memory on a small machine, a driver
        # hiccup) must not fail the analysis: finish on the CPU, and stay there
        _net = net = torch.jit.load(_model_path(), map_location="cpu").eval()
        p = _infer(net, Xt)
    return {lab: p[k] for k, lab in enumerate(ids)}
