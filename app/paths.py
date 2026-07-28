"""Where the app keeps its data, on every platform.

One place decides this, because several modules need the same folder (the image
library, the analysis session, the mask cache, retrained weights, baked fonts).
They used to hard-code "~/Library/Application Support/…" each, which is macOS
only and drifts apart the moment one of them changes.

macOS keeps EXACTLY the folder it always used, so an existing install — and the
image library inside it — is untouched by this change.
"""
import os
import sys

APP_NAME = "SEM Particle Analyzer"

# the macOS location this app has always used; also the migration source
_MAC_DIR = os.path.join(os.path.expanduser("~/Library/Application Support"), APP_NAME)


def data_dir():
    """The app's writable data folder, created if missing.

    macOS   ~/Library/Application Support/SEM Particle Analyzer
    Windows %APPDATA%\\SEM Particle Analyzer
    Linux   ~/.local/share/SEM Particle Analyzer
    """
    if sys.platform == "darwin":
        d = _MAC_DIR
    elif os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, APP_NAME)
    else:
        base = (os.environ.get("XDG_DATA_HOME")
                or os.path.expanduser("~/.local/share"))
        d = os.path.join(base, APP_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.expanduser("~")          # last resort: never fail to start
    return d


def sub_dir(*parts):
    """A subfolder of the data folder (created)."""
    d = os.path.join(data_dir(), *parts)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def documents_dir():
    """Where user-facing files (the training set) live. Kept in Documents rather
    than Desktop on Windows/Linux, where a Desktop folder may not exist or may be
    redirected into OneDrive."""
    if sys.platform == "darwin":
        return os.path.expanduser("~/Desktop")
    for cand in ("~/Documents", "~"):
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return os.path.expanduser("~")
