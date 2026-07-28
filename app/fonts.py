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


def _bake_sf_instances():
    """Static Regular/Bold cut from the variable system font. Cached on disk —
    baking costs a few seconds, and only ever happens once."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    d = _ui_cache_dir()
    made = []
    for style, wght in (("Regular", 400), ("Bold", 700)):
        p = os.path.join(d, f"sempa_ui_{style}.ttf")
        if not os.path.exists(p):
            f = TTFont(_SF_VAR)
            instancer.instantiateVariableFont(
                f, {"wght": wght, "opsz": 20, "wdth": 100, "GRAD": 400},
                inplace=True)
            # SFNS's name table is missing entries that updateFontNames needs,
            # so name the instance ourselves: one family, two styles.
            nt = f["name"]
            for nid, val in ((1, _UI_FAMILY), (2, style),
                             (4, f"{_UI_FAMILY} {style}"),
                             (6, f"{_UI_FAMILY}-{style}")):
                nt.setName(val, nid, 3, 1, 0x409)
                nt.setName(val, nid, 1, 0, 0)
            f.save(p)
        made.append(p)
    return made


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
