"""PyInstaller runtime hook: load the OpenCV native extension directly.

On Python 3.13 OpenCV ships an abi3 wheel (`cv2.abi3.so`) whose `__init__.py`
loader manipulates sys.path and re-imports "cv2"; under PyInstaller's
FrozenImporter that re-import resolves back to the package, raising
"recursion is detected during loading of cv2 binary extensions".

We pre-load the extension as the top-level `cv2` module before any code imports
it, so the recursing bootstrap in `cv2/__init__.py` never runs. All cv2 functions
and constants used by the app and by easyocr live in the extension itself.
"""
import os
import sys
import glob
import importlib.util


def _preload_cv2():
    if "cv2" in sys.modules:
        return
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    matches = glob.glob(os.path.join(base, "cv2", "cv2*.so"))
    if not matches:
        return
    try:
        spec = importlib.util.spec_from_file_location("cv2", matches[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules["cv2"] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("cv2", None)  # let the normal loader try instead


_preload_cv2()
