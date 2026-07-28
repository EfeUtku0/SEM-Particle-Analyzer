# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the SEM Particle Analyzer — macOS and Windows.

    pyinstaller --noconfirm --clean sem_analyzer.spec

macOS   -> dist/SEM Particle Analyzer.app   (drag to /Applications)
Windows -> dist/SEM Particle Analyzer/      (folder containing the .exe)

The ML weights are bundled and copied into the user's home on first launch by
app/_bootstrap.py, so a fresh machine needs no downloads and runs offline.
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

IS_MAC = sys.platform == "darwin"
IS_WIN = os.name == "nt"

home = Path.home()
APP = os.path.abspath("app")

datas = [("app/assets", "assets")]
binaries = []
hiddenimports = ["_bootstrap", "paths", "solidnet", "scale_reader", "analyze",
                 "viz", "report", "fonts", "patternnet", "patterncrop",
                 "classsize", "trainmode", "pattern_train",
                 "PIL._tkinter_finder", "fontTools.varLib.instancer"]
if IS_MAC:
    # native multi-select folder picker; absent elsewhere, and optional there
    hiddenimports += ["AppKit", "Foundation", "objc"]

# packages whose submodules/data PyInstaller doesn't fully discover on its own
for pkg in ["cellpose", "easyocr", "skimage", "shapely", "fastremap",
            "roifile", "imagecodecs", "tifffile", "natsort", "fill_voids",
            "scipy", "sklearn", "torchvision"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print("collect_all skipped:", pkg, e)

# bundled model weights (copied to ~ on first launch by _bootstrap)
cp = str(home / ".cellpose" / "models" / "cpsam_v2")
if os.path.exists(cp):
    datas.append((cp, "models/cellpose"))
else:
    print("WARNING: Cellpose weights not found at", cp,
          "-> the built app will have to download them on first use")
for f in ["craft_mlt_25k.pth", "english_g2.pth"]:
    p = str(home / ".EasyOCR" / "model" / f)
    if os.path.exists(p):
        datas.append((p, "models/easyocr"))
    else:
        print("WARNING: EasyOCR weight missing:", p)

# fixes an opencv/PyInstaller import clash; harmless where it doesn't apply
runtime_hooks = ["rthook_cv2.py"] if os.path.exists("rthook_cv2.py") else []

a = Analysis(
    ["app/gui.py"],
    pathex=[APP],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=runtime_hooks,
    excludes=["tkinter", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = None
if IS_MAC and os.path.exists("build_icon/icon.icns"):
    icon = "build_icon/icon.icns"
elif IS_WIN and os.path.exists("build_icon/icon.ico"):
    icon = "build_icon/icon.ico"

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="SEM Particle Analyzer",
    debug=False, strip=False, upx=False,
    console=False,                 # no terminal window alongside the GUI
    icon=icon,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="SEM Particle Analyzer",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="SEM Particle Analyzer.app",
        icon=icon,
        bundle_identifier="com.biomaten.semparticleanalyzer",
        info_plist={"NSHighResolutionCapable": True,
                    "LSMinimumSystemVersion": "12.0"},
    )
