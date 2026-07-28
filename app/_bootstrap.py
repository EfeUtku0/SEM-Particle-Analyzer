"""Runtime bootstrap for the packaged app.

When frozen (PyInstaller .app), the ML model weights are bundled read-only inside
the bundle. Cellpose and EasyOCR expect them in writable home folders, so on the
first launch we copy them there (fast no-op afterwards, and on a dev machine that
already has them). Also pins the SSL cert bundle for any first-run download.
"""
import os
import sys
import shutil
from pathlib import Path


def _res(*parts):
    base = getattr(sys, "_MEIPASS", None)
    return os.path.join(base, *parts) if base else None


def _copy_missing_file(src, dst):
    if src and os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def ensure_models():
    if not getattr(sys, "_MEIPASS", None):
        return  # dev mode: models already in ~/.cellpose & ~/.EasyOCR

    home = Path.home()
    # Cellpose cpsam weights
    _copy_missing_file(_res("models", "cellpose", "cpsam_v2"),
                       str(home / ".cellpose" / "models" / "cpsam_v2"))
    # EasyOCR detector + recognizer
    esrc = _res("models", "easyocr")
    if esrc and os.path.isdir(esrc):
        edst = home / ".EasyOCR" / "model"
        edst.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(esrc):
            _copy_missing_file(os.path.join(esrc, f), str(edst / f))


def setup():
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass
    # keep matplotlib/font caches in a writable place
    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path.home() / ".sem_particle_analyzer" / "mpl"))
    try:
        ensure_models()
    except Exception:
        pass  # fall back to Cellpose/EasyOCR's own download on first use


setup()
