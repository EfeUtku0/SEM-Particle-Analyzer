"""In-app patternnet training.

Runs entirely inside the app (torch is already bundled): merges the visible
training folder via trainmode.load_pattern_raw(), measures an image-grouped
3-fold CV for honest validation, then trains the final model on all data and
saves it as TorchScript into the app's data folder — where patternnet.py picks
it up in preference to the bundled model. The previous model is kept as a
timestamped backup next to it.

Architecture: resnet18 @ 96px on raw (unblurred, median/MAD-normalized) crops.
Chosen 2026-07-18 after a sweep (tools/exp.py) against the app's previous
small-CNN/64px/blurred pipeline on identical image-grouped folds: acc
0.612->0.698, lamellar recall 0.425->0.664 (the worst-performing class by far).
"""
from __future__ import annotations

import os
import time

import numpy as np

import trainmode
import patterncrop
from trainmode import CLASSES

SIZE = patterncrop.SIZE          # 96
VAL_EPOCHS = 30
FINAL_EPOCHS = 35
K = 3


def _get_net():
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(CLASSES)))
    return m


def _augment(xb):
    import torch
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[3])
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        xb = torch.rot90(xb, k, dims=[2, 3])
    tex = xb[:, :1]
    B = tex.shape[0]
    gain = 0.6 + 0.9 * torch.rand(B, 1, 1, 1, device=xb.device)
    tex = tex * gain
    gamma = 0.75 + 0.5 * torch.rand(B, 1, 1, 1, device=xb.device)
    tex = torch.sign(tex) * torch.abs(tex).pow(gamma)
    tex = tex + (0.10 * torch.rand(B, 1, 1, 1, device=xb.device)) * torch.randn_like(tex)
    size = xb[:, 2:3]
    drop = (torch.rand(B, 1, 1, 1, device=xb.device) < 0.25).float()
    return torch.cat([tex, xb[:, 1:2], size * (1 - drop)], dim=1)


def _folds_for(keys):
    stems = sorted(set(keys.tolist()))
    rng = np.random.default_rng(0)
    order = rng.permutation(stems)
    return [list(order[i::K]) for i in range(K)] if len(stems) >= K else []


def _train(X, y, tr_idx, epochs, device, cb=None):
    import torch
    net = _get_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(y[tr_idx], minlength=len(CLASSES)).astype(np.float32)
    w = counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    Xt = torch.tensor(X[tr_idx], device=device)
    yt = torch.tensor(y[tr_idx], dtype=torch.long, device=device)
    n = len(tr_idx)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(net(_augment(Xt[idx])), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if cb:
            cb(ep + 1)
    return net


def _predict(net, X, device):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.tensor(X[i:i + 128], device=device)
            out.append(net(xb).softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def run_training(progress=None):
    """Full training run. `progress(done, total, phase)` is called per epoch.
    Returns a metrics dict; raises RuntimeError with a readable message when
    the data isn't trainable yet."""
    import torch
    imgs, sils, dnm, y, keys = trainmode.load_pattern_raw()
    n = len(y)
    counts = {CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))}
    if n < 60:
        raise RuntimeError(f"Not enough training data yet ({n} particles).")
    if sum(v > 0 for v in counts.values()) < 2:
        raise RuntimeError("Training needs at least two different classes.")

    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    X = patterncrop.featurize_batch(imgs, sils, dnm, size=SIZE)

    # Honest validation: image-grouped K-fold. Every particle's held-out
    # prediction comes from a model that never saw its photo during training
    # (no same-image leakage), so the aggregated accuracy/recalls reflect true
    # generalization to a new photo, not memorized texture from a sibling crop.
    folds = _folds_for(keys)
    total_val = VAL_EPOCHS * len(folds)
    total = total_val + FINAL_EPOCHS

    def cb(base):
        if not progress:
            return None
        return lambda ep: progress(base + ep, total,
                                   "final" if base >= total_val else "validation")

    acc, recalls, n_val_img = None, {}, 0
    if folds:
        yt_all, yp_all = [], []
        for fi, fold in enumerate(folds):
            val = np.isin(keys, fold)
            net = _train(X, y, np.where(~val)[0], VAL_EPOCHS, device, cb(fi * VAL_EPOCHS))
            yp_all.append(_predict(net, X[val], device).argmax(1))
            yt_all.append(y[val])
        yt = np.concatenate(yt_all)
        yp = np.concatenate(yp_all)
        acc = float((yt == yp).mean())
        recalls = {CLASSES[c]: float((yp[yt == c] == c).mean())
                   for c in range(len(CLASSES)) if (yt == c).any()}
        n_val_img = len(set(keys.tolist()))
    elif progress:
        progress(total_val, total, "validation")

    # final model on ALL data -> TorchScript in the app's data folder
    net = _train(X, y, np.arange(n), FINAL_EPOCHS, device, cb(total_val))
    net_cpu = net.to("cpu").eval()
    ts = torch.jit.trace(net_cpu, torch.zeros(1, 3, SIZE, SIZE))
    out = os.path.join(trainmode.support_dir(), "patternnet.pt")
    if os.path.exists(out):
        os.replace(out, out.replace(".pt", time.strftime(" %Y%m%d-%H%M%S.bak.pt")))
    ts.save(out)
    return dict(n=n, counts=counts, acc=acc, recalls=recalls, model=out,
                device=device, n_val_img=n_val_img)
