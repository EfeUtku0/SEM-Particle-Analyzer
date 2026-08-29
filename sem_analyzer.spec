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
if os.path.isdir("sample_images"):
    # the example micrographs a first launch seeds the library with
    datas.append(("sample_images", "sample_images"))
binaries = []
# Every module under app/. They are reached through the sys.path.insert in
# gui.py rather than as a package, so list them explicitly instead of trusting
# PyInstaller's static analysis to follow that — a module it misses does not
# fail the build, it fails at launch on the user's machine.
hiddenimports = [
    # startup, paths, science pipeline
    "_bootstrap", "paths", "analyze", "info_bar_reader", "image_files",
    "model_solid_liquid", "model_pattern", "model_pattern_crops", "model_pattern_training",
    "model_pattern_curve",
    "training_store", "review_queue", "smartsort", "thresholds", "calibrate",
    "golden_store", "model_eval", "sample_images",
    # rendering and output
    "overlay_draw", "chart_data", "charts", "export_files", "fonts",
    # interface: shared widgets and dialogs
    "ui_theme", "widget_image_view", "widget_tiles", "widget_library_tree",
    "dialog_guide", "dialog_review_sheet", "dialog_saturation",
    "dialog_train_report",
    "background_workers", "window_pattern_size",
    # interface: the window, one module per area (mixed into MainWindow)
    "window_library", "window_results", "window_particle_edit",
    "window_review", "window_training", "window_analysis",
    "window_layout", "session_store",
    # third-party bits PyInstaller cannot see
    "PIL._tkinter_finder", "fontTools.varLib.instancer",
]
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
