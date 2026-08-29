"""In-app patternnet training.

Runs entirely inside the app (torch is already bundled): merges the visible
training folder via training_store.load_pattern_raw(), measures an image-grouped
3-fold CV for honest validation, then trains the final model on all data and
saves it as TorchScript into the app's data folder — where model_pattern.py picks
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

import training_store
import model_pattern_crops
import golden_store
from training_store import CLASSES

SIZE = model_pattern_crops.SIZE          # 128
FINAL_EPOCHS = 35


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


class TrainCancelled(Exception):
    """Raised out of `_train` when `should_cancel()` turns true between epochs —
    the "X" next to Train model in the panel, for a click that was a mistake.
    Only ever checked between epochs, so a cancel takes up to one epoch to land;
    nothing is written to disk until every epoch has run, so a cancelled run
    never overwrites the model that was already there."""


def _train(X, y, tr_idx, epochs, device, cb=None, should_cancel=None):
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
        if should_cancel and should_cancel():
            raise TrainCancelled()
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


def run_training(progress=None, saturation=False, should_cancel=None):
    """Full training run. `progress(done, total, phase)` is called per epoch.
    Returns a metrics dict; raises RuntimeError with a readable message when
    the data isn't trainable yet, or TrainCancelled when `should_cancel()` goes
    true during the main training loop — the one part of the run worth
    interrupting, since golden scoring and the saturation curve are quick to
    finish once the model itself is already saved.

    ACCURACY COMES FROM THE GOLDEN SET, NOT FROM CROSS-VALIDATION (2026-08-05).
    This function used to retrain the model K times over image-grouped folds and
    report the aggregate. That number cost three quarters of the run, and it was
    measured on labels that had mostly been confirmed inside the review dialog
    with the model's own answer on screen — so it agreed with the model for
    reasons that have nothing to do with the model being right. Scoring instead
    on photos held permanently outside the training folder costs one forward
    pass and cannot be flattered that way. See model_eval / golden_store.

    With `saturation=True` a learning curve is measured on the way through (see
    model_pattern_curve) and returned under "saturation". It rides along here
    rather than standing alone because the expensive part — loading every crop
    and featurizing it — is already paid for by then, and because "how is the
    model doing" and "is more data still worth labelling" are the same question
    asked twice.
    """
    import torch
    t_start = time.time()
    imgs, sils, dnm, y, keys = training_store.load_pattern_raw()
    n = len(y)
    counts = {CLASSES[c]: int((y == c).sum()) for c in range(len(CLASSES))}
    if n < 60:
        raise RuntimeError(f"Not enough training data yet ({n} particles).")
    if sum(v > 0 for v in counts.values()) < 2:
        raise RuntimeError("Training needs at least two different classes.")

    torch.manual_seed(0)
    np.random.seed(0)
    device = model_pattern_crops.torch_device()

    X = model_pattern_crops.featurize_batch(imgs, sils, dnm, size=SIZE)

    total = FINAL_EPOCHS
    sat_units = 0
    if saturation:
        import model_pattern_curve
        # the curve's cost in the same unit the bar already counts: one "epoch"
        # over a pool-sized training set
        sat_units = int(round(model_pattern_curve._cost_units()
                              * model_pattern_curve.CURVE_EPOCHS))
        total += sat_units

    cb = (lambda ep: progress(ep, total, "final")) if progress else None

    # final model on ALL data -> TorchScript in the app's data folder
    net = _train(X, y, np.arange(n), FINAL_EPOCHS, device, cb,
                should_cancel=should_cancel)
    net_cpu = net.to("cpu").eval()
    ts = torch.jit.trace(net_cpu, torch.zeros(1, 3, SIZE, SIZE))
    out = os.path.join(training_store.support_dir(), "patternnet.pt")
    if os.path.exists(out):
        os.replace(out, out.replace(".pt", time.strftime(" %Y%m%d-%H%M%S.bak.pt")))
    ts.save(out)
    _prune_backups(out)
    train_secs = time.time() - t_start
    res = dict(n=n, counts=counts, model=out, device=device,
               photos=len(set(keys.tolist())), epochs=FINAL_EPOCHS,
               train_secs=round(train_secs, 1))

    # Score the model that was just written, on photos it can never have seen.
    # The weights on disk changed under model_pattern, which caches the loaded
    # net — without the reload the golden score would grade the PREVIOUS model
    # and look, wrongly, like a run that changed nothing.
    try:
        import model_pattern
        model_pattern.reload()
        import model_eval
        if progress:
            progress(total, total, "golden")
        res["golden"] = model_eval.evaluate(
            progress=(lambda d, t, s: progress(total, total, f"golden:{d}/{t}"))
            if progress else None)
        res["golden_conflicts"] = golden_store.conflicts()
    except Exception as exc:                # a broken ruler must not lose weights
        res["golden_error"] = str(exc)

    # LAST, and never fatal: the model the user asked for is already saved by
    # this point, so a curve that cannot be measured (too few photos, a machine
    # that ran out of memory) costs the extra minutes and nothing else.
    if saturation:
        base = FINAL_EPOCHS
        try:
            rec = model_pattern_curve.measure(
                X, y, keys, device,
                progress=(lambda d, t: progress(base + d, total, "saturation"))
                if progress else None)
            rec["device"] = device
            # the run's real held-out score, stored beside the fitted curve for
            # provenance. NOT plotted on the curve's axis: it is a five-class
            # figure over every measured particle, while the curve is pattern
            # classes only (see charts.render_saturation).
            g = (res.get("golden") or {}).get("combined") or {}
            rec["golden"] = dict(acc=g.get("acc"), macro_f1=g.get("macro_f1"),
                                 n=n, photos=res["photos"])
            model_pattern_curve.append_run(rec)
            res["saturation"] = rec
        except Exception as exc:
            res["saturation_error"] = str(exc)

    # the run's own record, kept so past trainings stay comparable
    try:
        import model_eval
        model_eval.append_run(dict(
            ts=time.time(), model=os.path.basename(out), device=device,
            epochs=FINAL_EPOCHS, train_secs=res["train_secs"],
            n_particles=n, n_photos=res["photos"], counts=counts,
            golden=res.get("golden"), golden_error=res.get("golden_error"),
            conflicts=res.get("golden_conflicts") or []))
    except Exception:
        pass
    return res


KEEP_BACKUPS = 2


def _prune_backups(out, keep=KEEP_BACKUPS):
    """Delete all but the newest `keep` timestamped backups of the model.

    Every retrain sets the previous model aside as "patternnet <stamp>.bak.pt",
    and each one is ~43 MB. Nothing used to remove them, so a dozen training
    runs quietly grew to a couple of hundred megabytes in the user's data
    folder (measured: 216 MB over 11 backups). Two are kept — enough to step
    back from a retrain that made the model worse, which is the only reason the
    backups exist. Never raises: losing old copies must not fail a good train.
    """
    d, base = os.path.split(out)
    stem = os.path.splitext(base)[0]
    try:
        old = sorted(f for f in os.listdir(d)
                     if f.startswith(stem + " ") and f.endswith(".bak.pt"))
    except OSError:
        return []
    dropped = []
    for name in old[:-keep] if keep else old:
        try:
            os.remove(os.path.join(d, name))
            dropped.append(name)
        except OSError:
            pass
    return dropped
