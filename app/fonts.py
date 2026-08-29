"""Shared fonts for PIL rendering, Qt UI and matplotlib charts.

- Source Serif 4  -> UI text, data panel, chart labels
- Roboto Slab     -> measurement labels drawn onto the micrograph (legible on
                     busy imagery)
"""
import os
from PIL import ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
ROBOTO = os.path.join(FONT_DIR, "RobotoSlab.ttf")
SOURCE_SERIF = os.path.join(FONT_DIR, "SourceSerif4.ttf")

SERIF_FAMILY = "Source Serif 4"


def pil_font(size, weight="Bold", path=ROBOTO):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


_registered = {}


def matplotlib_family(path=SOURCE_SERIF, family=SERIF_FAMILY):
    """Register a font with matplotlib and return its family name."""
    import matplotlib.font_manager as fm
    if path not in _registered:
        try:
            fm.fontManager.addfont(path)
            _registered[path] = family
        except Exception:
            return "DejaVu Sans"
    return _registered[path]


# ---- the app's own UI face, for chart pieces that reproduce a bit of UI ----
# Qt draws the panel in macOS's system font (.AppleSystemUIFont = SFNS.ttf), and
# an export has to match it exactly. That file is a VARIABLE font, though, and
# matplotlib can't pull a weight out of one — asking for bold silently gave
# regular, so every exported tile came out thin.
#
# So we bake two STATIC instances (Regular 400 / Bold 700) out of it once, cache
# them next to the app's data, and hand matplotlib a normal two-weight family.
# Anything that goes wrong (not macOS, no fontTools, read-only system) falls back
# to the nearest face that ships a real bold.
_SF_VAR = "/System/Library/Fonts/SFNS.ttf"
_UI_FAMILY = "SEMPA UI"
_UI_FALLBACKS = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
_ui_family = None


def _ui_cache_dir():
    from paths import sub_dir
    return sub_dir("fonts")


_SF_STYLES = (("Regular", 400), ("Semibold", 600), ("Bold", 700))


def _bake_sf_instances(styles=_SF_STYLES):
    """Static Regular/Semibold/Bold cut from the variable system font. Cached on
    disk — baking costs a few seconds, and only ever happens once."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    d = _ui_cache_dir()
    made = []
    for style, wght in styles:
        p = os.path.join(d, f"sempa_ui_v4_{style}.ttf")
        if not os.path.exists(p):
            f = TTFont(_SF_VAR)
            # opsz=17 is the axis minimum, which is what macOS itself uses for
            # UI text at the app's sizes: at 13 pt the baked face measures 182 px
            # for the same string as .AppleSystemUIFont, i.e. metric-identical
            # (opsz=20, used before, came out 3% narrow and would have reflowed
            # panels tuned pixel by pixel).
            instancer.instantiateVariableFont(
                f, {"wght": wght, "opsz": 17, "wdth": 100, "GRAD": 400},
                inplace=True)
            # SFNS's name table is missing entries that updateFontNames needs,
            # so name the instance ourselves: one family, two styles.
            nt = f["name"]
            # 16/17 (typographic family/style) matter as much as 1/2: Qt reads 16
            # first, and SFNS ships ".SF NS" there — leaving it made Qt file all
            # three faces under the hidden system family and ignore ours.
            for nid, val in ((1, _UI_FAMILY), (2, style),
                             (4, f"{_UI_FAMILY} {style}"),
                             (6, f"{_UI_FAMILY}-{style}"),
                             (16, _UI_FAMILY), (17, style)):
                nt.setName(val, nid, 3, 1, 0x409)
                nt.setName(val, nid, 1, 0, 0)
            # instancer sets OS/2.usWeightClass but leaves the style BITS saying
            # "Regular", and CoreText believes the bits: without this Qt read all
            # three faces as regular, ignored Bold/Semibold and synthesised a
            # smeared fake bold instead (same advance width, darker pixels).
            os2, head = f["OS/2"], f["head"]
            os2.fsSelection &= ~(1 << 5 | 1 << 6)        # clear BOLD | REGULAR
            head.macStyle &= ~1
            if wght >= 700:
                os2.fsSelection |= 1 << 5                # BOLD
                head.macStyle |= 1
            elif wght < 600:
                os2.fsSelection |= 1 << 6                # REGULAR
            f.save(p)
        made.append(p)
    return made


def qt_ui_family():
    """Register the baked static faces with Qt and return the family name, or
    None to leave Qt on the system font.

    WHY (crash fix, 2026-07-30): macOS's UI font is the VARIABLE SFNS.ttf. Every
    time Qt builds a font engine for it, CoreText walks the variation axes — and
    five of the app's ten recorded crashes are a SIGSEGV inside
    CTFontCopyVariationAxes / CopyLocalizedFontNameInternal, reached from
    QCoreTextFontEngine::init() during an ordinary QLabel size hint. It is a
    CoreText fault we cannot guard against from Python; the only fix is to stop
    handing Qt a variable font. The faces baked for the exports are instances of
    that same system font, so the UI keeps its look (Regular / Semibold / Bold —
    Semibold exists for the file panel's DemiBold rows) without any variable
    font in play.
    """
    try:
        from PySide6 import QtGui
        if not os.path.exists(_SF_VAR):
            return None
        ok = False
        for p in _bake_sf_instances():
            ok = QtGui.QFontDatabase.addApplicationFont(p) >= 0 or ok
        if not ok:
            return None
        # Self-check before handing the family to the whole UI: the faces are
        # only usable if Qt actually picks Bold/Semibold when a widget asks for
        # those weights. (Under the offscreen platform it does not — it keeps
        # Regular and fakes the bold.) An all-regular UI would be a worse
        # regression than the crash we are avoiding, so verify, and stay on the
        # system font if the check fails.
        s = "Undercooled 123 Karışık"
        adv = {}
        for w in (QtGui.QFont.Normal, QtGui.QFont.DemiBold, QtGui.QFont.Bold):
            f = QtGui.QFont(_UI_FAMILY, 13)
            f.setWeight(w)
            adv[w] = QtGui.QFontMetrics(f).horizontalAdvance(s)
        if not (adv[QtGui.QFont.Normal] < adv[QtGui.QFont.DemiBold]
                < adv[QtGui.QFont.Bold]):
            return None
        return _UI_FAMILY
    except Exception:
        return None                    # not macOS / no fontTools -> system font


def matplotlib_ui_family():
    """The app's UI face for matplotlib, with a genuine bold."""
    global _ui_family
    if _ui_family is not None:
        return _ui_family
    import matplotlib.font_manager as fm
    if os.path.exists(_SF_VAR):
        try:
            for p in _bake_sf_instances():
                fm.fontManager.addfont(p)
            fm.findfont(fm.FontProperties(family=_UI_FAMILY, weight="bold"),
                        fallback_to_default=False)
            _ui_family = _UI_FAMILY
            return _ui_family
        except Exception:
            pass
    for cand in _UI_FALLBACKS:
        try:
            fm.findfont(fm.FontProperties(family=cand, weight="bold"),
                        fallback_to_default=False)
            _ui_family = cand
            break
        except Exception:
            continue
    _ui_family = _ui_family or "DejaVu Sans"
    return _ui_family
