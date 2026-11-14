"""Tell the user when a newer download is on GitHub. Never installs anything.

Once per launch, a few seconds after the window is up, one request goes to the
repository's release list. If a release for THIS platform carries a higher
version than APP_VERSION, a button appears in the toolbar that opens its page
in the browser. No answer (offline, rate-limited, GitHub down) = no button and
no message: the app works exactly as before.

Only the packaged app checks. Running from source, `git pull` is the update;
SEMPA_UPDATE_CHECK=1 forces the check anyway (for testing), =0 turns it off.
SEMPA_UPDATE_FEED points the check at another URL (a file:// JSON in tests).

Releases are per platform (v1.1.0 is the Mac zip, v1.2.0-windows the Windows
one), so a release counts only if it has an asset for this OS — a Windows-only
release must not tell a Mac user to update.
"""
import json
import os
import re
import sys
import threading
import urllib.request

APP_VERSION = "1.3.0"

REPO = "EfeUtku0/SEM-Particle-Analyzer"
FEED = f"https://api.github.com/repos/{REPO}/releases?per_page=20"
TIMEOUT_S = 5


def parse_version(tag):
    """'v1.2.0-windows' -> (1, 2, 0); anything without a number -> None."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", tag or "")
    if not m:
        return None
    parts = [int(x) for x in m.group(1).split(".")]
    return tuple((parts + [0, 0, 0])[:3])


def platform_word(platform=None):
    p = platform or sys.platform
    if p == "darwin":
        return "macos"
    if p.startswith("win"):
        return "windows"
    return None


def newest_for_platform(releases, current=APP_VERSION, platform=None):
    """The newest published release for this OS that beats `current`, as
    (version string, page url); None when there is nothing newer."""
    word = platform_word(platform)
    if word is None:
        return None
    cur = parse_version(current)
    best = None
    for r in releases:
        if r.get("draft") or r.get("prerelease"):
            continue
        assets = [a.get("name", "").lower() for a in r.get("assets", [])]
        if not any(word in a for a in assets):
            continue
        v = parse_version(r.get("tag_name"))
        if v is None or v <= cur:
            continue
        if best is None or v > best[0]:
            best = (v, r.get("html_url") or f"https://github.com/{REPO}/releases")
    if best is None:
        return None
    return ".".join(str(x) for x in best[0]), best[1]


def _ssl_context():
    """certifi's bundle, explicitly. python.org's Python on macOS ships with no
    CA store of its own (measured 2026-11-12: CERTIFICATE_VERIFY_FAILED against
    api.github.com from the dev venv), and SSL_CERT_FILE from _bootstrap only
    helps if it was set before this context is made."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_releases(url=None):
    url = url or os.environ.get("SEMPA_UPDATE_FEED") or FEED
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"SEM-Particle-Analyzer/{APP_VERSION}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_ssl_context()) as resp:
        data = json.load(resp)
    return data if isinstance(data, list) else []


def enabled():
    flag = os.environ.get("SEMPA_UPDATE_CHECK")
    if flag is not None:
        return flag.strip() not in ("", "0", "false", "no")
    return bool(getattr(sys, "frozen", False))


def check_in_background(on_found):
    """Run the check off the main thread. `on_found(version, url)` is called
    from that thread only when a newer release exists; the caller hops back
    to the GUI thread (a Qt signal does that)."""
    if not enabled():
        return None

    def run():
        try:
            hit = newest_for_platform(fetch_releases())
        except Exception:
            return              # offline / rate-limited / bad JSON: say nothing
        if hit:
            on_found(*hit)

    t = threading.Thread(target=run, name="update-check", daemon=True)
    t.start()
    return t
