"""Raw-crop pipeline for the pattern sub-classifier (janus/stripe/lamellar/
composite). Unlike solidnet's crop (blurred, pre-normalized at 64px, baked in
at save time), pattern crops are stored UNPROCESSED at 128px so normalization
stays a train-time choice — this is what the resnet18@96 sweep (2026-07-18)
validated against the old small-CNN/64px/blurred pipeline (acc 0.612->0.698,
lamellar recall 0.425->0.664 on identical image-grouped folds)."""
from __future__ import annotations

import numpy as np
import cv2

RAW = 128
# 128 (was 96): measured on the 32-photo clean set (scratchpad/sweep3_res128.py),
# training at full raw resolution lifts 4-fold OOF macro-recall 0.728->0.739 and
# accuracy 0.737->0.752 — real (not majority-tilt) gains on the fine-detail banded
# classes (stripe +4, lamellar +3). Hand-crafted filter banks (FFT/Gabor/LBP/grad)
# were all tried first and gave only majority-tilt noise; resolution was the lever.
SIZE = 128


def torch_device():
    """Where training runs: Apple GPU, else an NVIDIA GPU, else the CPU.

    Inference is deliberately CPU-only (the two nets are tiny and load with
    map_location="cpu"), but a training run is minutes on a GPU and hours
    without one. This used to read `"mps" if available else "cpu"`, which is
    the right answer on the Mac it was written on and the wrong one on any
    Windows/Linux machine with a CUDA card — the app would quietly train on
    the CPU there.
    """
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def raw_crop(gray, masks, label_id, bbox):
    """Unprocessed intensity + soft silhouette patch, RAWxRAW, 1.30x padded."""
    y0, x0, y1, x1 = bbox
    h, w = y1 - y0, x1 - x0
    side = max(int(max(h, w) * 1.30), 8)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    yy0, xx0 = cy - side // 2, cx - side // 2
    img = np.zeros((side, side), np.float32)
    m = np.zeros((side, side), np.float32)
    ys0, xs0 = max(0, yy0), max(0, xx0)
    ys1, xs1 = min(gray.shape[0], yy0 + side), min(gray.shape[1], xx0 + side)
    img[ys0 - yy0:ys1 - yy0, xs0 - xx0:xs1 - xx0] = gray[ys0:ys1, xs0:xs1]
    m[ys0 - yy0:ys1 - yy0, xs0 - xx0:xs1 - xx0] = (masks[ys0:ys1, xs0:xs1] == label_id)
    if m.sum() < 10:
        return None, None
    interp = cv2.INTER_AREA if side > RAW else cv2.INTER_CUBIC
    img = cv2.resize(img, (RAW, RAW), interpolation=interp)
    m = cv2.resize(m, (RAW, RAW), interpolation=interp)
    return img, np.clip(m, 0, 1)


def featurize_batch(imgs, sils, diam_nm, size=SIZE, masked=True):
    """raw [N,H,W] -> 3ch float32 [N,3,size,size]: robust-normalized texture,
    silhouette, log-size. Must match tools/exp.py::make_x bit-for-bit (that is
    what the reported sweep numbers are trusted against)."""
    N = len(imgs)
    X = np.zeros((N, 3, size, size), np.float32)
    for i in range(N):
        im, m = imgs[i], sils[i]
        if im.shape[0] != size:
            interp = cv2.INTER_AREA if im.shape[0] > size else cv2.INTER_CUBIC
            im = cv2.resize(im, (size, size), interpolation=interp)
            m = cv2.resize(m, (size, size), interpolation=interp)
        inside = m > 0.5
        if inside.sum() < 10:
            inside = m > 0.1
        med = float(np.median(im[inside]))
        mad = float(np.median(np.abs(im[inside] - med))) + 1.0
        tex = np.clip((im - med) / (6.0 * mad), -3, 3)
        if masked:
            tex = tex * (m > 0.5)
        X[i, 0] = tex
        X[i, 1] = m
        X[i, 2] = np.log10(max(diam_nm[i], 10.0)) - 2.5
    return X
