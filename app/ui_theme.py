"""Every colour, shape and font rule the app's widgets are styled with.

One place, because the look is cross-cutting: the QSS sheet below is applied
once to the whole QApplication (see main), and the helpers next to it build the
few styles that have to be computed per widget — a chip that takes its class's
colour, a wash of that colour, the tick drawn into the checkbox indicator.

Keeping them together is also what stops the sheet and the helpers drifting
apart: the greys here (#e2e7ec borders, #f4f6f8 cards) are the same greys
_chip_qss writes inline, and a chip restyled on its own would stop matching the
cards around it.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


STYLE = """
QWidget { background:#ffffff; color:#1a2129; font-size:13px; font-weight:600; }
QLabel#h { color:#8a95a1; font-size:11px; font-weight:800; letter-spacing:1px; }
QLabel#ctl { color:#4a5560; font-size:12px; font-weight:700; background:transparent; }
QLabel#ctl:disabled { color:#b3bcc5; }
/* a caption that NAMES a group of controls rather than being one — quieter, so
   it doesn't read as another item in the row */
QLabel#ctlmute { color:#9aa4ae; font-size:12px; font-weight:600; background:transparent; }
QLabel#ctlmute:disabled { color:#c3cad1; }
/* one grey for every card in the app (matches the RESULTS figure's own
   background, so the plot melts into its card instead of sitting on a slab) */
QFrame#card { background:#f4f6f8; border:1px solid #e2e7ec; border-radius:8px; }
QTreeWidget#imgtree { background:#f4f6f8; border:1px solid #e2e7ec; border-radius:10px;
                      padding:4px 1px 4px 4px; outline:0; show-decoration-selected:0; }
/* Rows are painted by LibraryDelegate (indent, highlight, chevron, text), so
   there are no ::item background/colour rules here — they would fight it.
   NOTE: never add a ::item:drop-enabled rule: that pseudo-state is true for
   every folder at all times, so it paints the whole tree rather than the drag
   target (it once made every row look permanently selected). */
QTreeWidget#imgtree::branch { background:transparent; }
/* the rename box: the panel's own grey, never a white slab, and a soft blue
   text selection instead of the macOS accent colour (which can be red) */
QTreeWidget#imgtree QLineEdit { background:#e9eef4; border:1px solid #bcd2ff;
    border-radius:5px; padding:0px 3px; color:#1a2129;
    selection-background-color:#cfe0fb; selection-color:#12386e; }
/* both scrollbars take zero size: the list still scrolls (wheel/trackpad) but
   no bar is drawn, and the strip it used to occupy goes to the names */
QTreeWidget#imgtree QScrollBar:horizontal { height:0px; }
QTreeWidget#imgtree QScrollBar:vertical { width:0px; }
/* IMAGES search: the same soft grey as the tree it sits above, so the panel
   still reads as one surface; it only lights up (white + blue rim) when typing */
QLineEdit#searchbox { background:#eef1f5; border:1px solid #e2e7ec; border-radius:9px;
    padding:6px 9px; font-size:12px; color:#1a2129;
    selection-background-color:#cfe0fb; selection-color:#12386e; }
QLineEdit#searchbox:focus { background:#ffffff; border:1px solid #2b6fff; }
/* the results list mirrors the tree's shell exactly (rows are drawn by
   SearchDelegate, so no ::item colour rules here either) */
QListWidget#searchlist { background:#f4f6f8; border:1px solid #e2e7ec;
    border-radius:10px; padding:4px 1px 4px 4px; outline:0; }
QListWidget#searchlist QScrollBar:vertical { width:0px; }
QListWidget#searchlist QScrollBar:horizontal { height:0px; }
QPushButton { background:#f4f6f8; border:1px solid #dce1e7; border-radius:8px;
              padding:8px 12px; font-weight:600; color:#1a2129; }
QPushButton:hover { background:#eaeef2; }
/* the three IMAGES-panel actions share one narrow row — trim the padding so
   their labels are never clipped */
QPushButton#sidebtn { padding:7px 4px; font-size:12px; font-weight:700; color:#4a5560; }
QPushButton:checked { background:#eaf1fb; border:1px solid #2b6fff; color:#12386e; }
QListWidget { background:#f4f6f8; border:1px solid #e2e7ec; border-radius:8px; padding:4px; }
QListWidget::item { padding:5px 6px; border-radius:5px; color:#1a2129; }
QListWidget::item:selected { background:#d9e6fb; color:#12386e; }
QPushButton:disabled { color:#aab2bc; background:#f4f6f8; }
QPushButton#primary { background:#2b6fff; border:none; color:white; }
QPushButton#primary:hover { background:#1c5df0; }
QPushButton#primary:disabled { background:#b9ccf6; color:#eef2ff; }
/* the TRAINING card's secondary actions: quiet and low, so the card's height
   goes to what is actually being read (the class table) rather than to a stack
   of full-size buttons */
QPushButton#trainmini { padding:5px 8px; font-size:11.5px; font-weight:700;
                        color:#5a6472; border-radius:7px; }
QPushButton#trainmini::menu-indicator { width:0px; image:none; }
QCheckBox { padding:3px; spacing:9px; background:transparent; }
QCheckBox:disabled { color:#aab2bc; }
QCheckBox::indicator { width:18px; height:18px; border-radius:5px;
                       border:2px solid #b6c0cc; background:transparent; }
QCheckBox::indicator:hover { border:2px solid #2b6fff; }
QCheckBox::indicator:checked { background:#2b6fff; border:2px solid #2b6fff;
    image:url(CHECKMARK); }
QCheckBox::indicator:checked:disabled { background:#b9ccf6; border:2px solid #b9ccf6; }
QCheckBox::indicator:disabled { border:2px solid #cfd6de; background:transparent; }
QProgressBar { border:none; border-radius:5px; background:#e6ebf0; height:8px;
               text-align:center; color:transparent; }
QProgressBar::chunk { background:#2b6fff; border-radius:5px; }
/* the analysis "working" bar in the status area: taller, with readable text so
   the first (slow) run never looks frozen */
QProgressBar#busybar { height:20px; border-radius:6px; background:#e6ebf0;
                       color:#12386e; font-weight:700; font-size:11.5px; }
QProgressBar#busybar::chunk { background:#bcd2ff; border-radius:6px; width:40px;
                              margin:1px; }
QScrollArea { border:none; background:transparent; }
QMenu { background:#ffffff; border:1px solid #e2e7ec; border-radius:12px; padding:6px; }
QMenu::item { padding:7px 18px 7px 8px; border-radius:7px; color:#1a2129;
              font-weight:600; font-size:12.5px; }
QMenu::item:selected { background:#eef3fb; color:#12386e; }
/* the small grey section headings inside the Save menu (disabled = heading) */
QMenu::item:disabled { color:#a2acb8; font-size:10.5px; font-weight:800;
                       padding:8px 8px 3px 8px; background:transparent; }
QMenu::separator { height:1px; background:#eff2f5; margin:5px 8px; }
/* checkable items reuse the app's own rounded checkbox instead of the native tick */
QMenu::indicator { width:15px; height:15px; margin-left:8px; margin-right:9px;
                   border-radius:4px; border:1.6px solid #c2ccd8;
                   background:transparent; }
QMenu::indicator:checked { background:#2b6fff; border:1.6px solid #2b6fff;
                           image:url(CHECKMARK); }
QSlider { min-height:22px; background:transparent; }
QSlider::groove:horizontal { height:6px; background:#e6ebf0; border-radius:3px; }
QSlider::sub-page:horizontal { background:#2b6fff; border-radius:3px; }
QSlider::add-page:horizontal { background:#e6ebf0; border-radius:3px; }
QSlider::handle:horizontal { width:18px; height:18px; margin:-7px 0; border-radius:10px;
                             background:#ffffff; border:2px solid #2b6fff; }
QSlider::handle:horizontal:hover { border:2px solid #1c5df0; background:#eef4ff; }
QSlider::groove:horizontal:disabled { background:#f4f6f8; }
QSlider::sub-page:horizontal:disabled { background:#cdd5de; }
QSlider::handle:horizontal:disabled { background:#f4f6f8; border:2px solid #cdd5de; }
QStatusBar { color:#5b6672; }
QLineEdit#rangebox { background:#f4f6f8; border:1px solid #dce1e7; border-radius:6px;
                    padding:5px 4px; font-weight:700; color:#1a2129; }
/* small stat cards under the result figure */
QFrame#tile { background:#ffffff; border:1px solid #e6ebf0; border-radius:8px; }
QLabel#tilecap { color:#8a95a1; font-size:10px; font-weight:800; letter-spacing:0.7px;
                 background:transparent; }
QLabel#tileval { color:#1a2129; font-size:15px; font-weight:800; background:transparent; }
QLabel#tilesub { color:#6b7580; font-size:11.5px; font-weight:700; background:transparent; }
QLineEdit#rangebox:focus { border:1px solid #2b6fff; background:#ffffff; }
/* the chart-title box inside the Save menu */
QLineEdit#titlebox { background:#f4f6f8; border:1px solid #dce1e7; border-radius:7px;
                     padding:6px 8px; font-weight:600; color:#1a2129; min-width:210px; }
QLineEdit#titlebox:focus { border:1px solid #2b6fff; background:#ffffff; }
QWidget#vsep { background:#c7cfd8; }
QSplitter::handle:horizontal { background:transparent; width:9px; }
QSplitter::handle:horizontal:hover { background:#bcd2ff; border-radius:4px; }
QSplitter::handle:horizontal:pressed { background:#2b6fff; border-radius:4px; }
QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:#c7d0da; border-radius:5px; min-height:32px; }
QScrollBar::handle:vertical:hover { background:#aab6c3; }
QScrollBar:horizontal { background:transparent; height:10px; margin:2px; }
QScrollBar::handle:horizontal { background:#c7d0da; border-radius:5px; min-width:32px; }
QScrollBar::handle:horizontal:hover { background:#aab6c3; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }
"""


def _sep():
    w = QtWidgets.QWidget(); w.setObjectName("vsep"); w.setFixedWidth(1)
    # a bare QWidget ignores a stylesheet background unless it is told to draw
    # one — without this the hairline is there in the layout but invisible
    w.setAttribute(QtCore.Qt.WA_StyledBackground, True)
    return w


def _tint(hexcol, f=0.18):
    """A very light wash of a class colour (mixed towards white) — the fill of a
    selected filter chip, so it reads as that class without shouting."""
    c = QtGui.QColor(hexcol)
    mix = lambda v: int(255 + (v - 255) * f)
    return QtGui.QColor(mix(c.red()), mix(c.green()), mix(c.blue())).name()


def _chip_qss(col=None, align="center", radius=12, pad="1px 8px", size=11.5):
    """Pill-shaped chip. Unselected chips are quiet grey; the selected one takes
    its class colour (blue for 'All'). Shared by the RESULTS filter chips and
    the particle-tool buttons, so both read as one family."""
    on_bg, on_bd, on_fg = ("#e7edf8", "#2b6fff", "#12386e") if col is None else \
        (_tint(col), col, "#1a2129")
    return (
        f"QPushButton{{border-radius:{radius}px;padding:{pad};font-size:{size}px;"
        f"font-weight:700;background:#ffffff;border:1px solid #e2e7ec;"
        f"color:#5b6672;text-align:{align};}}"
        "QPushButton:hover{background:#eef2f6;}"
        "QPushButton:disabled{background:#f0f3f6;border:1px solid #e8ecf1;color:#c3cad2;}"
        f"QPushButton:checked{{background:{on_bg};border:1px solid {on_bd};color:{on_fg};}}")


def _checkmark_png():
    """White checkmark for the checkbox indicator; returns a file path.
    Written to a writable temp dir (the app bundle is read-only when frozen).
    Drawn oversized with round caps for a clean, classic tick."""
    import tempfile
    from PIL import Image as _I, ImageDraw as _D
    path = os.path.join(tempfile.gettempdir(), "sempa_check2.png")
    s = 64
    im = _I.new("RGBA", (s, s), (0, 0, 0, 0))
    d = _D.Draw(im)
    w = 8
    pts = [(15, 33), (27, 45), (49, 20)]
    d.line(pts, fill=(255, 255, 255, 255), width=w, joint="curve")
    for ex, ey in (pts[0], pts[-1]):          # rounded line caps
        r = w / 2 - 0.5
        d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(255, 255, 255, 255))
    im.save(path)
    return path
