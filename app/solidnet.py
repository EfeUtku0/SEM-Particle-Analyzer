"""CNN-based solid/undercooled classifier (trained on the user's hand-labeled
SEM particles — see tools/train_cnn.py; keep make_crop identical to training).

Input per particle: 3-channel 64x64 crop
  ch0: Gaussian-blurred texture normalized by the measured shot-noise floor
       (coherent facet steps survive the blur, noise does not -> kV-invariant)
  ch1: particle silhouette (shape)
  ch2: constant log-size channel (log10 diameter nm - 2.5)
Output: probability the particle is SOLID (crystalline).
"""
from __future__ import annotations

import os

import numpy as np
import cv2

SIZE = 64
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "solidnet.pt")
_net = None


def available() -> bool:
    return os.path.exists(_MODEL_PATH)


def _get_net():
    global _net
    if _net is None:
        import torch
        _net = torch.jit.load(_MODEL_PATH, map_location="cpu")
        _net.eval()
    return _net


def make_crop(gray, masks, label_id, bbox, nm_per_px):
    """Must stay bit-identical to tools/train_cnn.py::make_crop."""
    y0, x0, y1, x1 = bbox
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * 1.25)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    yy0, xx0 = cy - side // 2, cx - side // 2
    crop = np.zeros((side, side), np.float32)
    mcrop = np.zeros((side, side), bool)
    ys0, xs0 = max(0, yy0), max(0, xx0)
    ys1, xs1 = min(gray.shape[0], yy0 + side), min(gray.shape[1], xx0 + side)
    crop[ys0 - yy0:ys1 - yy0, xs0 - xx0:xs1 - xx0] = gray[ys0:ys1, xs0:xs1]
    mcrop[ys0 - yy0:ys1 - yy0, xs0 - xx0:xs1 - xx0] = (masks[ys0:ys1, xs0:xs1] == label_id)
    if mcrop.sum() < 10:
        return None
    inside = crop[mcrop]
    med = float(np.median(inside))
    blur = cv2.GaussianBlur(crop, (0, 0), 2.0)
    noise = crop - cv2.GaussianBlur(crop, (0, 0), 3.0)
    nmad = float(np.median(np.abs(noise[mcrop]))) + 0.3
    tex = np.zeros_like(crop)
    tex[mcrop] = np.clip((blur[mcrop] - med) / (6.0 * nmad), -3, 3)
    tex = cv2.resize(tex, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    sil = cv2.resize(mcrop.astype(np.float32), (SIZE, SIZE),
                     interpolation=cv2.INTER_AREA)
    diam_nm = 2 * np.sqrt(mcrop.sum() / np.pi) * nm_per_px
    sizech = np.full((SIZE, SIZE), np.log10(max(diam_nm, 10.0)) - 2.5, np.float32)
    return np.stack([tex, sil, sizech])


def solid_probs(gray, masks, regions, nm_per_px):
    """Return {label_id: P(solid)} for the given regionprops list."""
    import torch
    crops, ids = [], []
    g = gray.astype(np.float32)
    for rp in regions:
        c = make_crop(g, masks, rp.label, rp.bbox, nm_per_px)
        if c is not None:
            crops.append(c); ids.append(rp.label)
    if not crops:
        return {}
    X = torch.tensor(np.array(crops, np.float32))
    net = _get_net()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.sigmoid(net(X[i:i + 256]).squeeze(1)).numpy())
    p = np.concatenate(out)
    return dict(zip(ids, p.tolist()))
