"""SEM Particle Analyzer — desktop GUI (light theme).

Load SEM micrographs, auto-read the scale bar, segment particles with Cellpose,
measure diameters and (on CBS images) classify solid vs flat particles. Select
one or several images to get per-image or pooled (combined) results, shown as a
histogram + data panel directly in the app.
"""
from __future__ import annotations

import os
import sys
import random
import dataclasses
import traceback

import numpy as np

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401  (sets model paths / SSL before heavy imports)
import analyze
from analyze import analyze_image, Aggregate, DEFAULT_FACET_THRESH
from viz import render, render_training, pick_spread, PATTERN_COLORS, GREEN, RED
import trainmode
from report import (export, render_report, class_diams,
                    CLASS_ORDER, CLASS_LABELS, CLASS_COLORS)
from classsize import ClassSizeWindow, _Plot as PatternSizePlot
from fonts import SOURCE_SERIF, SERIF_FAMILY

IMG_EXT = (".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp")

ROLE_PATH = QtCore.Qt.UserRole        # image item -> file path (never mutated)
ROLE_KIND = QtCore.Qt.UserRole + 1    # node kind -> "folder" or "image"

STYLE = """
QWidget { background:#ffffff; color:#1a2129; font-size:13px; font-weight:600; }
QLabel#h { color:#8a95a1; font-size:11px; font-weight:800; letter-spacing:1px; }
QLabel#ctl { color:#4a5560; font-size:12px; font-weight:700; background:transparent; }
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
QWidget#vsep { background:#dce1e7; }
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


def pil_to_qpix(im):
    im = im.convert("RGB")
    qim = QtGui.QImage(im.tobytes("raw", "RGB"), im.width, im.height,
                       im.width * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qim.copy())


class ImageView(QtWidgets.QGraphicsView):
    """Pannable/zoomable image (trackpad pinch + ⌘-wheel; drag to pan; double-click fits)."""

    clicked = QtCore.Signal(float, float)   # image coords of a plain (no-drag) click

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._fit = True
        self._press = None
        self.setRenderHints(QtGui.QPainter.SmoothPixmapTransform | QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        # zooming in still pans (drag / trackpad), but without the scrollbars
        # appearing along the edges of the micrograph
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QtGui.QColor(0, 0, 0, 0))
        self.setStyleSheet("QGraphicsView{background:transparent;border:none;}")
        self.viewport().setAutoFillBackground(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def set_image(self, pil):
        pix = pil_to_qpix(pil)
        if self._item is None:
            self._item = self._scene.addPixmap(pix)
        else:
            self._item.setPixmap(pix)
        self._scene.setSceneRect(QtCore.QRectF(pix.rect()))
        if self._fit:
            self.fit()

    def clear_image(self):
        if self._item is not None:
            self._scene.removeItem(self._item)
            self._item = None

    def fit(self):
        if self._item is not None:
            self.resetTransform()
            self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)
            self._fit = True

    def _zoom(self, f):
        self.scale(f, f); self._fit = False

    def wheelEvent(self, e):
        if e.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
            self._zoom(1.0015 ** e.angleDelta().y()); e.accept()
        else:
            super().wheelEvent(e)

    def event(self, e):
        if e.type() == QtCore.QEvent.NativeGesture and \
                e.gestureType() == QtCore.Qt.ZoomNativeGesture:
            self._zoom(1.0 + e.value()); return True
        return super().event(e)

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        press, self._press = self._press, None
        super().mouseReleaseEvent(e)
        # a click is a press+release without a drag in between (pan untouched)
        if press is not None and \
                (e.position().toPoint() - press).manhattanLength() < 5:
            p = self.mapToScene(e.position().toPoint())
            self.clicked.emit(p.x(), p.y())

    def resizeEvent(self, e):
        if self._fit:
            self.fit()
        super().resizeEvent(e)


class ScaledImage(QtWidgets.QLabel):
    """A QLabel that keeps a pixmap and scales it to fit (aspect-preserving)."""

    def __init__(self, placeholder="", on_resize=None, radius=0):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("color:#9aa4ae;")
        self._placeholder = placeholder
        self.setText(placeholder)
        self._pix = None
        self._on_resize = on_resize
        self._radius = radius     # display-px corner radius (matches the card)

    def set_image(self, pil):
        self._pix = pil_to_qpix(pil)
        self._rescale()

    def clear_img(self):
        self._pix = None
        self.setText(self._placeholder)

    def resizeEvent(self, e):
        self._rescale()
        if self._on_resize is not None:
            self._on_resize()
        super().resizeEvent(e)

    def _rescale(self):
        if self._pix is None:
            return
        # Render at physical (Retina) resolution so downscaled figures stay crisp.
        dpr = self.devicePixelRatioF()
        target = QtCore.QSize(max(1, int(self.width() * dpr)),
                              max(1, int(self.height() * dpr)))
        scaled = self._pix.scaled(target, QtCore.Qt.KeepAspectRatio,
                                  QtCore.Qt.SmoothTransformation)
        if self._radius > 0:
            rounded = QtGui.QPixmap(scaled.size())
            rounded.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(rounded)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            clip = QtGui.QPainterPath()
            r = self._radius * dpr
            clip.addRoundedRect(QtCore.QRectF(rounded.rect()), r, r)
            p.setClipPath(clip)
            p.drawPixmap(0, 0, scaled)
            p.end()
            scaled = rounded
        scaled.setDevicePixelRatio(dpr)
        self.setPixmap(scaled)


class Worker(QtCore.QThread):
    done = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)

    def __init__(self, path, do_class=True, do_pattern=True):
        super().__init__()
        self.path = path
        self.do_class = do_class
        self.do_pattern = do_pattern

    def run(self):
        try:
            self.done.emit(self.path, analyze_image(
                self.path, do_class=self.do_class, do_pattern=self.do_pattern))
        except Exception:
            self.failed.emit(self.path, traceback.format_exc())


class TrainWorker(QtCore.QThread):
    """Runs the in-app patternnet retraining off the GUI thread."""
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def run(self):
        try:
            import pattern_train
            res = pattern_train.run_training(
                progress=lambda d, t, ph: self.progress.emit(d, t, ph))
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(str(e) or traceback.format_exc())


def _listdir(d):
    """Sorted directory listing that never raises (permission/IO errors -> [])."""
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


SESSION_VERSION = 2   # v1 sessions still load (the occlusion gate only reads
                      # fields v1 already has) — never drop a user's analyses.


def _support_dir():
    from paths import data_dir
    return data_dir()


def _session_path():
    """Where the last session (analyses, thresholds…) is persisted."""
    return os.path.join(_support_dir(), "session.pkl")


def _library_path():
    """Where the IMAGES panel's folder tree is persisted — its own file, wholly
    independent of the analysis session, so images and folders survive even when
    they've never been analysed. It is only ever rewritten to reflect an edit the
    user made; nothing here is pruned automatically."""
    return os.path.join(_support_dir(), "library.json")


def _native_pick():
    """One native macOS picker for BOTH kinds of import: pick folders to bring
    them in as folders, or pick image files inside a folder to bring in just
    those. Returns a list of paths (empty if cancelled), or None if the native
    panel can't be used so the caller can fall back to a Qt dialog."""
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSOpenPanel, NSModalResponseOK
    except Exception:
        return None
    panel = NSOpenPanel.openPanel()
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(True)
    panel.setAllowsMultipleSelection_(True)
    panel.setResolvesAliases_(True)
    panel.setTitle_("Import images or folders")
    panel.setMessage_("Pick folders to import whole, or pick individual images.")
    panel.setPrompt_("Import")
    if panel.runModal() != NSModalResponseOK:
        return []
    return [str(u.path()) for u in panel.URLs()]


def _sep():
    w = QtWidgets.QWidget(); w.setObjectName("vsep"); w.setFixedWidth(1)
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


class CheckMenu(QtWidgets.QMenu):
    """A drop-down menu whose checkable items toggle in place without closing the
    menu, so several can be ticked in one open (used for the Evaluate button)."""

    def mouseReleaseEvent(self, e):
        act = self.activeAction()
        if act is not None and act.isCheckable() and act.isEnabled():
            act.trigger()          # flips checked state + fires toggled/triggered
            e.accept()
            return
        super().mouseReleaseEvent(e)


class Tile(QtWidgets.QFrame):
    """A small white stat card under the result figure: caption, big value and
    an optional trailing note (e.g. the share that goes with a count)."""

    def __init__(self, cap="", val="—", sub=""):
        super().__init__()
        self.setObjectName("tile")
        self._accent = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 7); lay.setSpacing(2)
        self.cap = QtWidgets.QLabel(cap); self.cap.setObjectName("tilecap")
        row = QtWidgets.QHBoxLayout(); row.setSpacing(7); row.setContentsMargins(0, 0, 0, 0)
        self.val = QtWidgets.QLabel(val); self.val.setObjectName("tileval")
        self.sub = QtWidgets.QLabel(sub); self.sub.setObjectName("tilesub")
        # the small trailing note sits at the vertical middle of the big value,
        # so the two read as one centred pair rather than the note hanging low
        row.addWidget(self.val, 0, QtCore.Qt.AlignVCenter)
        row.addWidget(self.sub, 1, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        lay.addWidget(self.cap); lay.addLayout(row)

    def set(self, cap=None, val=None, sub=None, accent=None):
        if cap is not None and cap != self.cap.text():
            self.cap.setText(cap)
        if val is not None and val != self.val.text():
            self.val.setText(val)
        if sub is not None and sub != self.sub.text():
            self.sub.setText(sub)
        # re-applying a stylesheet re-polishes the widget — far too heavy to do
        # on every step of a slider drag, so only when the colour really changes
        if accent is not None and accent != self._accent:
            self._accent = accent
            self.val.setStyleSheet(f"color:{accent};background:transparent;")


ROLE_MISSING = QtCore.Qt.UserRole + 2     # image whose file is gone from disk


class LibraryDelegate(QtWidgets.QStyledItemDelegate):
    """Draws every row of the IMAGES tree by hand.

    Doing it here (instead of leaving it to Qt + QSS) is what makes the narrow
    panel work: the nesting indent is ours, so the rounded highlight can hug the
    row content and never bleed into the indent gutter; folder rows get an
    animated chevron; image rows start at the very left of their level (no icon
    column) and their file extension is hidden — both purely to win width for
    the names, which is the scarcest thing in this panel.
    """

    INDENT = 13        # px per nesting level
    CHEV = 15          # chevron column, folder rows only
    GAP = 3
    # An image hangs back from its folder's own text by this much, which is what
    # makes "inside a folder" visible at a glance: a photo in a folder lands just
    # right of that folder's name, while a loose photo at the root sits back at
    # the chevron column, clearly outside any of them.
    IMG_BASE = 9
    PADX = 9           # breathing room to the left of the text inside the pill

    FOLDER_FG = "#3c4753"
    IMAGE_FG = "#4a5560"
    MISSING_FG = "#aeb6c0"
    SEL_FG = "#12386e"
    SEL_BG = "#dce8fa"
    HOVER_BG = "#e9eef4"

    def __init__(self, tree):
        super().__init__(tree)
        self.tree = tree

    @staticmethod
    def _depth(item):
        d, p = 0, item.parent()
        while p is not None:
            d, p = d + 1, p.parent()
        return d

    @staticmethod
    def display_name(item):
        """What the row shows: an image drops a trailing image extension, so
        '… Janus 1.jpeg' reads as '… Janus 1'. The item's real text (and the
        file path it points to) is untouched — this is display only."""
        text = item.text(0)
        if item.data(0, ROLE_KIND) == "image":
            stem, ext = os.path.splitext(text)
            if ext.lower() in IMG_EXT and stem:
                return stem
        return text

    def _text_left(self, item):
        """x of the row text, relative to the row rect."""
        if item.data(0, ROLE_KIND) == "folder":
            return self._depth(item) * self.INDENT + self.CHEV + self.GAP
        return self._depth(item) * self.INDENT + self.IMG_BASE

    def _row_font(self, base, item):
        f = QtGui.QFont(base)
        if item.data(0, ROLE_KIND) == "folder":
            f.setBold(True)
        else:
            f.setWeight(QtGui.QFont.DemiBold)   # the panel's long-standing weight
        return f

    def paint(self, painter, option, index):
        item = self.tree.itemFromIndex(index)
        if item is None:
            return super().paint(painter, option, index)
        is_folder = item.data(0, ROLE_KIND) == "folder"
        r = option.rect
        # The pill starts a fixed gap left of the text — except on a folder, where
        # it starts at the chevron, so the expand arrow sits INSIDE the highlight
        # instead of being stranded next to it.
        lead = (self._depth(item) * self.INDENT if is_folder
                else self._text_left(item) - self.PADX)
        x0 = r.left() + max(0, lead - (4 if is_folder else 0))

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.State_MouseOver)
        if selected or hovered:
            # Starts at this row's own indent (never in the gutter) and runs out
            # to the panel edge rather than stopping at the end of the text —
            # a short name would otherwise get a stubby little pill.
            right = max(r.right(), self.tree.viewport().width())
            box = QtCore.QRectF(x0, r.top() + 1.0, right - x0, r.height() - 2.0)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(self.SEL_BG if selected else self.HOVER_BG))
            painter.drawRoundedRect(box, 6, 6)

        if is_folder:                       # animated chevron: ▸ closed, ▾ open
            phase = self.tree.chevron_phase(item)
            cx = r.left() + self._depth(item) * self.INDENT + self.CHEV / 2.0
            cy = r.center().y() + 0.5
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(90.0 * phase)
            pen = QtGui.QPen(QtGui.QColor("#94a0ad"), 1.7)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPolyline(QtGui.QPolygonF([
                QtCore.QPointF(-2.0, -3.6), QtCore.QPointF(1.6, 0.0),
                QtCore.QPointF(-2.0, 3.6)]))
            painter.restore()

        painter.setFont(self._row_font(option.font, item))
        if selected:
            fg = self.SEL_FG
        elif is_folder:
            fg = self.FOLDER_FG
        elif item.data(0, ROLE_MISSING):
            fg = self.MISSING_FG
        else:
            fg = self.IMAGE_FG
        painter.setPen(QtGui.QColor(fg))
        tx = r.left() + self._text_left(item)
        painter.drawText(QtCore.QRect(tx, r.top(), max(0, r.right() - tx), r.height()),
                         QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                         self.display_name(item))
        painter.restore()

    def sizeHint(self, option, index):
        item = self.tree.itemFromIndex(index)
        if item is None:
            return super().sizeHint(option, index)
        fm = QtGui.QFontMetrics(self._row_font(option.font, item))
        return QtCore.QSize(self._text_left(item) + fm.horizontalAdvance(
            self.display_name(item)) + 8, max(31, fm.height() + 15))

    # renaming edits the REAL name (extension included), positioned over the text
    def updateEditorGeometry(self, editor, option, index):
        item = self.tree.itemFromIndex(index)
        r = QtCore.QRect(option.rect)
        if item is not None:
            r.setLeft(r.left() + self._text_left(item) - 4)
        # inset so the editor's own frame and text highlight can never reach the
        # row edge (it used to look like it was about to spill out)
        r.adjust(0, 3, -5, -3)
        editor.setGeometry(r)

    def createEditor(self, parent, option, index):
        ed = super().createEditor(parent, option, index)
        item = self.tree.itemFromIndex(index)
        if item is not None and isinstance(ed, QtWidgets.QLineEdit):
            ed.setFont(self._row_font(option.font, item))
        return ed


class LibraryTree(QtWidgets.QTreeWidget):
    """The IMAGES panel: a persistent, virtual folder tree.

    Folders are pure organisation — they never correspond to anything on disk,
    so renaming / moving / nesting them is free and touches no real file. Images
    are references to files on disk; an image's display name can be renamed
    without changing the file it points to. Nothing here is ever removed except
    by the user (see MainWindow._remove_items / _save_library).

    The tree supports dragging to reorder rows, to move an image into another
    folder, and to nest a folder inside another. Image rows are deliberately
    *not* drop-enabled, so an image can never become the child of another image —
    dropping onto one lands the row beside it instead. External file/folder
    drops (from Finder) are forwarded to the window as an "add images" request.
    """

    changed = QtCore.Signal()                 # structure moved -> persist
    urls_dropped = QtCore.Signal(object, object)   # (paths, target_folder_item)

    def __init__(self):
        super().__init__()
        self.setObjectName("imgtree")
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        # Qt's own indent/branch gutter is switched OFF and LibraryDelegate draws
        # the indent itself. Qt painted that gutter in the selection colour (the
        # stray blue block beside a selected row); drawing it ourselves keeps the
        # nesting visible while the highlight stays on the row content alone.
        self.setIndentation(0)
        self.setRootIsDecorated(False)
        self.setAnimated(True)                 # smooth expand / collapse
        # per-item chevron angle (0 = closed, 1 = open), driven by an animation
        # so folders don't snap open — see _animate_chevron
        self._chev = {}
        self._press = None                 # left-press origin, for the drag guard
        self._chev_click = False
        self.itemExpanded.connect(lambda it: self._animate_chevron(it, 1.0))
        self.itemCollapsed.connect(self._on_collapsed)
        self.setExpandsOnDoubleClick(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  # rename via menu / F2
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.header().setStretchLastSection(False)
        self.setTextElideMode(QtCore.Qt.ElideNone)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setItemDelegate(LibraryDelegate(self))

    def _on_collapsed(self, item):
        """Collapsing a folder drops the selection of everything it just hid.

        Qt keeps hidden rows selected, so a folder holding ten photos left ten
        invisible selections behind — the results panel then showed an aggregate
        of ten images with nothing visibly selected. If any of them WAS selected,
        the folder itself takes over the selection, which means the same set of
        images but stated visibly."""
        hid = []

        def walk(f):
            for j in range(f.childCount()):
                c = f.child(j)
                if c.isSelected():
                    hid.append(c)
                walk(c)

        walk(item)
        if hid:
            was_current = self.currentItem() in hid
            self.blockSignals(True)
            for c in hid:
                c.setSelected(False)
            self.blockSignals(False)
            item.setSelected(True)
            if was_current:
                self.setCurrentItem(item)
            self.itemSelectionChanged.emit()
        self._animate_chevron(item, 0.0)

    def chevron_phase(self, item):
        a = self._chev.get(item)
        return a if a is not None else (1.0 if item.isExpanded() else 0.0)

    def _animate_chevron(self, item, target):
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(float(self.chevron_phase(item)))
        anim.setEndValue(float(target))
        anim.setDuration(170)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        def step(v):
            self._chev[item] = float(v)
            self.viewport().update()

        anim.valueChanged.connect(step)
        anim.finished.connect(lambda: self._chev.pop(item, None))
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def _drop_pos(self, e):
        return self.itemAt(e.position().toPoint())

    # Qt starts a drag after ~10px, which on a trackpad fires on almost every
    # click; a row also drag-selects its neighbours over that same tiny distance.
    # Both are swallowed until the pointer has really travelled.
    DRAG_START_PX = 30

    def mousePressEvent(self, e):
        """Qt draws no branch arrow (indentation is 0 and the delegate paints the
        chevron), so clicking that chevron is what toggles a folder — a
        double-click anywhere on the row still works too."""
        self._chev_click = False
        self._press = None
        it = self.itemAt(e.position().toPoint())
        if (it is not None and it.data(0, ROLE_KIND) == "folder"
                and e.button() == QtCore.Qt.LeftButton):
            d = self.itemDelegate()
            x = e.position().toPoint().x() - self.visualItemRect(it).left()
            lo = d._depth(it) * d.INDENT
            if lo <= x <= lo + d.CHEV + d.GAP:
                # toggling only — no selection change, and (via the guards in
                # mouseMove/Release) no drag and no drag-select either
                self._chev_click = True
                it.setExpanded(not it.isExpanded())
                e.accept()
                return
        if e.button() == QtCore.Qt.LeftButton:
            self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._chev_click:
            e.accept()                     # a chevron click can never drag
            return
        if (self._press is not None and (e.buttons() & QtCore.Qt.LeftButton)
                and (e.position().toPoint() - self._press).manhattanLength()
                < self.DRAG_START_PX):
            e.accept()                     # too small a move to mean anything
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        if self._chev_click:
            self._chev_click = False
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # external drops (Finder) add images; internal drags move rows -----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            target = self._drop_pos(e)
            if target is not None and target.data(0, ROLE_KIND) == "image":
                target = target.parent()      # land in the image's folder
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
            self.urls_dropped.emit(paths, target)
            e.acceptProposedAction()
            return
        # internal move: block only the impossible case — dropping a folder INTO
        # itself or one of its own descendants (reordering above/below is fine)
        if self.dropIndicatorPosition() == QtWidgets.QAbstractItemView.OnItem:
            dragged = set(self.selectedItems())
            t = self._drop_pos(e)
            while t is not None:
                if t in dragged:
                    e.ignore(); return
                t = t.parent()
        super().dropEvent(e)
        self.changed.emit()


class AspectCard(QtWidgets.QFrame):
    """A rounded 'card' whose height is locked to width / aspect, hugging its
    content with an equal border. The inner widget is positioned directly (a
    QGraphicsView won't expand vertically through a layout), so it always fills."""

    def __init__(self, inner, aspect, margin=4):
        super().__init__()
        self.setObjectName("card")
        self._aspect = aspect
        self._inner = inner
        self._m = margin
        inner.setParent(self)

    def resizeEvent(self, e):
        h = int(self.width() / self._aspect)
        if self.minimumHeight() != h or self.maximumHeight() != h:
            self.setFixedHeight(h)
        m = self._m
        self._inner.setGeometry(m, m, max(1, self.width() - 2 * m),
                                max(1, self.height() - 2 * m))
        super().resizeEvent(e)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Particle Analyzer")
        self.setMinimumSize(1380, 640)   # keeps the View bar (incl. 🎓 Training) from clipping
        self.setAcceptDrops(True)

        self.results = {}
        self.chosen = {}       # path -> [particle] whose size line is drawn (click-picked)
        self.size_range = None  # (lo, hi) nm for the size-fraction tool; None = off
        # Click tool in normal mode: "measure" (toggle a size line), a class name
        # (paint that class as a view-only correction), or None (clicks do nothing).
        self.click_tool = "measure"
        self.certainty = {}         # path -> {pid: (text, rgb)}  Certainty tool badges
        self.class_overrides = {}   # path -> {pid: class}  (view-only, not training data)
        self.view_excluded = {}     # path -> set(pid)  (view-only hide from count/report)
        # Particles the user re-admitted into the size statistics after the
        # occlusion gate (analyze.measurable) had greyed them out: measuring one
        # with the Measure tool IS the statement "this one's outline is whole".
        self.measure_include = {}   # path -> set(pid)
        self.current = None
        self.worker = None
        self.queue = []
        self._result_target = None
        self._scale_warnings = []
        self._empty_warnings = []
        # ---- training-mode state ----
        self.train_mode = False
        self.train_labels = {}        # path -> {particle_id: class} (user clicks)
        self.train_show_overlay = True
        self._overlay_hidden = False   # normal-mode Space: hide pattern/class fills
        self.train_blank = set()      # paths where the model's pre-fill is hidden,
        #                               so the photo can be labelled neutrally
        self._train_undo = []         # [(path, pid, previous or None)]
        self._train_worker = None

        def hdr(t):
            l = QtWidgets.QLabel(t); l.setObjectName("h"); return l

        def ctl(t):
            l = QtWidgets.QLabel(t); l.setObjectName("ctl"); return l

        # ---- top toolbar (live preview controls) ----
        # "Analyze" runs the whole measurement in one go — size, solid /
        # undercooled and the patterns. (It used to be a menu where each of the
        # three was ticked separately; every run wants all of them, and the
        # RESULTS panel now filters afterwards, so the choice was just friction.)
        self.analyze_btn = QtWidgets.QPushButton("🔬  Analyze")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.setMinimumHeight(38)
        self.analyze_btn.setToolTip("Measure the selected image(s): size, "
                                    "solid / undercooled and patterns")
        self.analyze_btn.clicked.connect(self.analyze_selected)

        # ---- View toggles. Every overlay the preview can show, spelled out —
        # including each pattern class — and PERMANENTLY in the bar (they don't
        # come and go with the evaluation choice; they're merely disabled until
        # an image actually has the matching results). Ticking e.g. only Janus +
        # Lamellar paints just those classes. Solid and the pattern classes are
        # mutually exclusive (a solid particle would get two colours at once).
        self.cb_under = QtWidgets.QCheckBox("Undercooled")
        self.cb_solid = QtWidgets.QCheckBox("Solid")
        self.cb_outline = QtWidgets.QCheckBox("Borders")
        self._pat_cbs = {}
        for key, lab in (("janus", "Janus"), ("stripe", "Stripe"),
                         ("composite", "Composite"), ("lamellar", "Lamellar")):
            self._pat_cbs[key] = QtWidgets.QCheckBox(lab)
        self._pat_ok = False
        self.cb_solid.toggled.connect(self._sync_pattern_enable)
        for cb in (self.cb_under, self.cb_solid,
                   *self._pat_cbs.values(), self.cb_outline):
            cb.stateChanged.connect(self._view_toggle)

        # size-fraction tool: what % of particles fall in [min, max] nm (either
        # box empty => open-ended, e.g. only max => "< max"). Live readout.
        def _numbox(ph):
            e = QtWidgets.QLineEdit(); e.setObjectName("rangebox")
            e.setFixedWidth(58); e.setPlaceholderText(ph)
            e.setAlignment(QtCore.Qt.AlignCenter)
            e.setValidator(QtGui.QDoubleValidator(0.0, 1e7, 2))
            e.textChanged.connect(self._range_changed)
            return e
        self.range_lo = _numbox("min")
        self.range_hi = _numbox("max")
        # coalesce fast keystrokes ("2"→"25"→"250") into a single re-render
        self._range_timer = QtCore.QTimer(self); self._range_timer.setSingleShot(True)
        self._range_timer.timeout.connect(self._render_result)

        # ---- Save drop-down (top right): tick what to write, then Export.
        # Two sections: WHAT to export (images / charts) and, for the charts,
        # WHICH distribution — the whole set or one class on its own.
        self.save_btn = QtWidgets.QPushButton("💾  Save")
        self.save_btn.setMinimumHeight(40)
        self._save_menu = CheckMenu(self.save_btn)

        def _sect(lab):
            a = self._save_menu.addAction(lab)
            a.setEnabled(False)                     # a heading, not an item
            f = a.font(); f.setBold(True); f.setPointSize(f.pointSize() - 2)
            a.setFont(f)
            return a

        def _exp(lab, on=False):
            a = self._save_menu.addAction(lab)
            a.setCheckable(True); a.setChecked(on)
            return a

        _sect("SEM IMAGE")
        self.ex_line = _exp("Particle Size Measurement", True)
        self.ex_under = _exp("Solid / Undercooled")
        self.ex_pattern = _exp("Patterns")
        self._save_menu.addSeparator()
        # Ticking a distribution IS what asks for its chart — with several images
        # selected each one is a single pooled chart over all of them.
        _sect("CHARTS")
        self.ex_cls = {}
        self.ex_cls["all"] = _exp("All", True)
        for _k in CLASS_ORDER:
            self.ex_cls[_k] = _exp(CLASS_LABELS[_k])
        self.ex_cls["patternsize"] = _exp("Pattern × Size")
        self._save_menu.addSeparator()
        _sect("CHART TITLE")
        self.ex_title = QtWidgets.QLineEdit()
        self.ex_title.setObjectName("titlebox")
        self.ex_title.setClearButtonEnabled(True)
        _tw = QtWidgets.QWidgetAction(self._save_menu)
        _box = QtWidgets.QWidget()
        _bl = QtWidgets.QHBoxLayout(_box); _bl.setContentsMargins(12, 2, 12, 6)
        _bl.addWidget(self.ex_title)
        _tw.setDefaultWidget(_box)
        self._save_menu.addAction(_tw)
        self._save_menu.addSeparator()
        act_export = self._save_menu.addAction("Export…")
        f = act_export.font(); f.setBold(True); act_export.setFont(f)
        act_export.triggered.connect(self.export_selected)
        self.save_btn.clicked.connect(self._open_save_menu)

        # ---- top bar: Analyze | view toggles | Save. One evenly-spaced run of
        # toggles — no "View:" caption and no group separators, so every gap is
        # the same and the row reads as a single set.
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(20); view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self.cb_outline)
        view_row.addWidget(self.cb_under)
        view_row.addWidget(self.cb_solid)
        for cb in self._pat_cbs.values():
            view_row.addWidget(cb)
        view_w = QtWidgets.QWidget(); view_w.setLayout(view_row)

        # ---- Training-mode toggle: flips the right panel to the labelling
        # tools and turns image clicks into per-particle class corrections.
        self.train_btn = QtWidgets.QPushButton("🎓  Training")
        self.train_btn.setCheckable(True)
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setToolTip("Correct particle classes by clicking; feed them back into the model")
        self.train_btn.toggled.connect(self._toggle_train_mode)
        # (the old toolbar "Pattern × Size" button is gone — it's now a chip/tab
        # inside the RESULTS panel)

        tb = QtWidgets.QHBoxLayout(); tb.setSpacing(16); tb.setContentsMargins(2, 4, 2, 4)
        tb.addWidget(self.analyze_btn)
        tb.addWidget(_sep())
        tb.addWidget(view_w)
        tb.addStretch(1)
        tb.addWidget(self.train_btn)
        tb.addWidget(self.save_btn)
        tbw = QtWidgets.QWidget(); tbw.setLayout(tb)

        def card(widget, margin=3):
            f = QtWidgets.QFrame(); f.setObjectName("card")
            lay = QtWidgets.QVBoxLayout(f)
            lay.setContentsMargins(margin, margin, margin, margin)
            lay.addWidget(widget)
            return f

        # ---- left column: a persistent, virtual folder tree of images ----
        # Folders are pure organisation (never touch disk); images are references
        # to files. Drag to reorder, to move an image into a folder, or to nest
        # folders. Rename via F2 / right-click; remove via right-click or Delete.
        # The whole tree persists to library.json and is NEVER pruned on its own.
        self.tree = LibraryTree()
        # Long filenames get clipped to the panel width otherwise. Let the
        # column size to its actual content (instead of stretching to fit the
        # viewport) and never elide text, so a horizontal scroll — trackpad /
        # shift+wheel — reveals the rest. The scrollbar itself stays invisible
        # (QSS height:0 above) since it's a narrow side panel.
        self.tree.itemExpanded.connect(self._on_expand_toggle)
        self.tree.itemCollapsed.connect(self._on_expand_toggle)
        # Neutralise the native row highlight (its colour is the OS accent —
        # possibly red). Setting it to the tree's own background makes the native
        # highlight invisible; the visible selection then comes solely from the
        # QSS ::item:selected rule, which is rounded and hugs just the row.
        _pal = self.tree.palette()
        _pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#f4f6f8"))
        _pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#12386e"))
        self.tree.setPalette(_pal)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemSelectionChanged.connect(self._refresh_results)
        self.tree.itemChanged.connect(self._item_renamed)     # inline rename commit
        self.tree.changed.connect(self._after_move)           # after a drag-move
        self.tree.urls_dropped.connect(self._on_external_drop)
        # Delete / Backspace removes the highlighted folder(s) or image(s)
        for seq in (QtGui.QKeySequence.Delete, QtGui.QKeySequence("Backspace")):
            sc = QtGui.QShortcut(seq, self.tree)
            sc.setContext(QtCore.Qt.WidgetShortcut)
            sc.activated.connect(self._delete_current)
        sc = QtGui.QShortcut(QtGui.QKeySequence("F2"), self.tree)   # rename in place
        sc.setContext(QtCore.Qt.WidgetShortcut)
        sc.activated.connect(self._rename_current)

        # two compact actions on one row — the panel is narrow and every pixel
        # of width belongs to the file names, not to button padding
        imp_btn = QtWidgets.QPushButton("＋  Import"); imp_btn.clicked.connect(self.import_items)
        imp_btn.setToolTip("Pick folders to import them whole, or pick individual "
                           "images inside a folder — the files on disk are never "
                           "moved or changed.")
        newf_btn = QtWidgets.QPushButton("📁  Folder"); newf_btn.clicked.connect(self._new_folder)
        newf_btn.setToolTip("New folder — with rows selected, it groups them inside.")
        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(6)
        for b in (imp_btn, newf_btn):
            b.setObjectName("sidebtn")
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Fixed)
            btn_row.addWidget(b)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(7); left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(hdr("IMAGES"))
        left.addWidget(self.tree, 1)
        left.addLayout(btn_row)
        leftw = QtWidgets.QWidget(); leftw.setLayout(left)
        leftw.setMinimumWidth(160); leftw.setMaximumWidth(380)

        # ---- middle column: image (hugged, top) + save settings (bottom) ----
        self.view = ImageView()
        image_card = AspectCard(self.view, aspect=1536 / 1103, margin=4)

        # ---- adjustments card: the live analysis knobs (used to be squeezed
        # into the toolbar; the old export checkboxes moved to the Save menu).
        # ---- click tool: what a click on a particle does. Exactly one is active
        # (mutually exclusive) so measure and class-paint never collide. Measure
        # toggles a size line; a class paints a view-only correction; Exclude
        # hides the particle from the count/report (view-only).
        self._tool_btns = {}
        # not exclusive: an exclusive group refuses to uncheck the active button,
        # but the user wants "click the active tool again -> nothing selected".
        # Mutual exclusivity is enforced by hand in _set_click_tool.
        self._tool_grp = QtWidgets.QButtonGroup(self); self._tool_grp.setExclusive(False)
        tool_grid = QtWidgets.QGridLayout(); tool_grid.setSpacing(6)
        # same muted family as the RESULTS chips (CLASS_COLORS), not the bright
        # saturated swatches the overlay itself paints with — a tool button is a
        # quiet control, the overlay is where the strong colour belongs
        tools = [("measure", "📏  Measure", None, "M"),
                 ("certainty", "🎯  Certainty", None, "C"),
                 ("janus", "Janus", CLASS_COLORS["janus"], "1"),
                 ("stripe", "Stripe", CLASS_COLORS["stripe"], "2"),
                 ("composite", "Composite", CLASS_COLORS["composite"], "3"),
                 ("lamellar", "Lamellar", CLASS_COLORS["lamellar"], "4"),
                 ("undercooled", "Undercooled", CLASS_COLORS["undercooled"], "5"),
                 ("solid", "Solid", "#cf8481", "6"),
                 ("exclude", "Exclude", "#9aa4ae", "0")]
        # 3 x 3 grid: the two inspection tools (Measure, Certainty) share the top
        # row; the card stays compact and the class labels get more width
        for i, (key, lab, col, sc) in enumerate(tools):
            b = QtWidgets.QPushButton(f"{lab}  ({sc})")
            b.setCheckable(True); b.setMinimumHeight(30)
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
            b.setStyleSheet(_chip_qss(col, align="left", radius=10,
                                      pad="4px 10px", size=12.5))
            if col is not None:
                pm = QtGui.QPixmap(11, 11); pm.fill(QtGui.QColor(col))
                b.setIcon(QtGui.QIcon(pm))
            b.clicked.connect(lambda _=False, k=key: self._toggle_click_tool(k))
            self._tool_grp.addButton(b); self._tool_btns[key] = b
            r, c = divmod(i, 3)
            tool_grid.addWidget(b, r, c)
        for c in range(3):
            tool_grid.setColumnStretch(c, 1)
        for r in range(3):
            tool_grid.setRowStretch(r, 1)
        self._tool_btns["measure"].setChecked(True)

        # (the size-range boxes and solid-threshold slider moved to the RESULTS
        # panel / were retired; the tool grid alone now fills this card's grey,
        # growing with it via the stretch factors above and below)
        # no caption above the grid — the tools say what they do, and the card
        # keeps its height, so the nine buttons take over the freed space
        adj_body = QtWidgets.QVBoxLayout()
        adj_body.setContentsMargins(12, 10, 12, 10); adj_body.setSpacing(8)
        adj_body.addLayout(tool_grid, 1)
        adj_card = QtWidgets.QFrame(); adj_card.setObjectName("card")
        adj_card.setLayout(adj_body)

        mid = QtWidgets.QVBoxLayout(); mid.setSpacing(8); mid.setContentsMargins(0, 0, 0, 0)
        mid.addWidget(hdr("PARTICLES"))
        mid.addWidget(image_card)            # fixed height (width / aspect)
        # the adjustments card takes the leftover height. It must be the
        # stretchable item: AspectCard is fixed-height, so with nothing able to
        # stretch QVBoxLayout centres the whole column and the PARTICLES header
        # drifts out of line with IMAGES / RESULTS. Its content is only two rows
        # tall now, so growing here just extends the grey, it never squeezes the
        # image (whose height follows the column width, not this layout).
        mid.addWidget(adj_card, 1)
        midw = QtWidgets.QWidget(); midw.setLayout(mid); midw.setMinimumWidth(420)

        # ---- right column: result card fills to the bottom; figure fits its aspect ----
        self._result_timer = QtCore.QTimer(self); self._result_timer.setSingleShot(True)
        self._result_timer.timeout.connect(self._render_result)
        self.result = ScaledImage("Results appear here after analysis",
                                  on_resize=lambda: self._result_timer.start(120),
                                  radius=8)
        # pinned under the chips (top); it fills all the space down to the tile
        # block, which is a fixed height, so the chart lands in the same rectangle
        # in every tab and the tiles never collide with the x-axis label
        self.result.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.result.setMinimumHeight(150)
        self.result.setStyleSheet("background:transparent;color:#9aa4ae;")

        # ---- class filter chips, on top of the result card. "All" is the whole
        # distribution (the familiar green histogram); picking a class narrows
        # the histogram and the statistics to that class alone — e.g. the size
        # distribution of the janus particles — while the size axis stays the
        # one of the full distribution, so the views stay comparable. The
        # composition band inside the figure covers the other direction
        # (what each size bin is made of).
        self.result_filter = None
        self._ps_active = False     # Pattern × Size tab showing instead of a chart
        self._res_groups = {}       # class -> diameters of the current selection
        self._res_all = np.array([])   # every measured diameter in the selection
        self._res_stats = None      # cached target.stats() (recomputing it per
        #                             keystroke made the readout stutter)
        self._chip_btns = {}
        chip_grid = QtWidgets.QGridLayout()
        chip_grid.setSpacing(5); chip_grid.setContentsMargins(2, 2, 2, 0)
        # 7 chips over two rows, four columns: All + the five classes are the
        # per-class filters; "Pattern × Size" is an overview tab that swaps the
        # whole panel for the size-composition view. It spans two cells so its
        # longer label sits comfortably.
        #   row 0: All · Undercooled · Janus · Stripe
        #   row 1: Lamellar · Composite · [ Pattern × Size ── ]
        chips = ([("all", "All", None)]
                 + [(k, CLASS_LABELS[k], CLASS_COLORS[k]) for k in CLASS_ORDER]
                 + [("patternsize", "Pattern × Size", None)])
        cells = {"all": (0, 0, 1), "undercooled": (0, 1, 1), "janus": (0, 2, 1),
                 "stripe": (0, 3, 1), "lamellar": (1, 0, 1), "composite": (1, 1, 1),
                 "patternsize": (1, 2, 2)}
        for key, lab, col in chips:
            b = QtWidgets.QPushButton(lab)
            b.setCheckable(True)
            b.setFixedHeight(25)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setStyleSheet(_chip_qss(col))
            if col is not None:
                pm = QtGui.QPixmap(9, 9); pm.fill(QtGui.QColor(col))
                b.setIcon(QtGui.QIcon(pm))
            b.clicked.connect(lambda _=False, k=key: self._set_result_filter(k))
            self._chip_btns[key] = b
            r, c, span = cells[key]
            chip_grid.addWidget(b, r, c, 1, span)
        for c in range(4):
            chip_grid.setColumnStretch(c, 1)
        self._chip_btns["all"].setChecked(True)
        self.chip_bar = QtWidgets.QWidget()
        self.chip_bar.setLayout(chip_grid)
        self.chip_bar.setStyleSheet("background:transparent;")
        self.chip_bar.setVisible(False)      # only once something is classified

        # ---- size range: type two numbers and read, right beside them, what
        # share of the shown particles falls inside — and how many that is. The
        # matching bars are highlighted in the plot too.
        rng_row = QtWidgets.QHBoxLayout(); rng_row.setSpacing(7)
        rng_row.setContentsMargins(2, 0, 2, 0)
        rng_row.addWidget(ctl("Size range:"))
        rng_row.addWidget(self.range_lo)
        rng_row.addWidget(ctl("\u2013"))
        rng_row.addWidget(self.range_hi)
        rng_row.addWidget(ctl("nm"))
        self.range_pct = QtWidgets.QLabel("\u2014")
        self.range_pct.setStyleSheet("color:#1a2129;font-size:15px;font-weight:800;"
                                     "background:transparent;")
        self.range_cnt = QtWidgets.QLabel("")
        self.range_cnt.setStyleSheet("color:#6b7580;font-size:12.5px;font-weight:700;"
                                     "background:transparent;")
        rng_row.addStretch(1)
        rng_row.addWidget(self.range_pct)
        rng_row.addWidget(self.range_cnt)

        # tile pool \u2014 the grid below shows a different subset per view:
        #   \u2022 a class filter -> total \u00b7 that class (count+%) \u00b7 mean \u00b7 range
        #   \u2022 All, classified -> total \u00b7 measured \u00b7 patterned, then one tile per
        #     class showing its share (coloured) and count
        #   \u2022 All, unclassified (ETD) -> total \u00b7 measured \u00b7 mean \u00b7 range
        self.tile_total = Tile("TOTAL PARTICLES", "\u2014")
        self.tile_measured = Tile("MEASURED", "\u2014")
        self.tile_patterned = Tile("PATTERNED", "\u2014")
        self.tile_n = Tile("PARTICLES", "\u2014")
        self.tile_mean = Tile("MEAN SIZE", "\u2014")
        self.tile_range = Tile("RANGE", "\u2014")
        # class breakdown tiles (Solid = crystalline but no pattern assigned)
        self._stat_classes = ["undercooled", "solid", "janus",
                              "stripe", "composite", "lamellar"]
        # Solid = every non-undercooled particle (the classic crystalline split),
        # so it gets the crystalline red, not a neutral grey.
        self._stat_colors = dict(CLASS_COLORS); self._stat_colors["solid"] = "#e0685f"
        self._stat_labels = {**{k: CLASS_LABELS[k] for k in CLASS_LABELS},
                             "solid": "Solid"}
        self.tile_cls = {k: Tile(self._stat_labels[k].upper(), "\u2014")
                         for k in self._stat_classes}

        self.stat_grid = QtWidgets.QGridLayout()
        self.stat_grid.setHorizontalSpacing(6); self.stat_grid.setVerticalSpacing(6)
        self.stat_grid.setContentsMargins(0, 0, 0, 0)
        self._stat_mode = None                # which layout is currently built

        stat_lay = QtWidgets.QVBoxLayout(); stat_lay.setSpacing(6)
        stat_lay.setContentsMargins(0, 0, 0, 0)
        stat_lay.addLayout(rng_row)
        # the tiles grow to fill the whole block below Size range: in class views
        # (fewer rows) they get taller instead of pinning to the bottom with a gap
        stat_lay.addLayout(self.stat_grid, 1)
        self.stat_bar = QtWidgets.QWidget(); self.stat_bar.setLayout(stat_lay)
        self.stat_bar.setStyleSheet("background:transparent;")
        # the tile block reserves the SAME height in every view (sized to the
        # 3-row All layout) so the chart above it lands identically in all tabs
        self._stat_h_locked = False
        self.stat_bar.setVisible(False)

        # ---- Pattern × Size page: swaps in for the histogram + tiles when that
        # chip is picked. The size-composition plot (reused from the standalone
        # window) with a threshold slider under it — pure QPainter, so dragging
        # is instant. Built lazily the first time it's shown.
        self._ps_plot = None
        self.ps_head = QtWidgets.QLabel("")
        self.ps_head.setStyleSheet("color:#1a2129;font-size:14px;font-weight:800;"
                                   "background:transparent;")
        self.ps_thlab = QtWidgets.QLabel("400 nm")
        self.ps_thlab.setStyleSheet("color:#1a2129;font-size:13px;font-weight:800;"
                                    "background:transparent;min-width:70px;")
        self.ps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ps_slider.setRange(0, 2000); self.ps_slider.setValue(400)
        self.ps_slider.setSingleStep(25); self.ps_slider.setPageStep(50)
        self.ps_slider.valueChanged.connect(self._ps_thresh_changed)
        ps_body = QtWidgets.QVBoxLayout()
        ps_body.setContentsMargins(0, 0, 0, 0); ps_body.setSpacing(8)
        ps_body.addWidget(self.ps_head)
        self._ps_holder = QtWidgets.QVBoxLayout()   # the plot drops in here
        self._ps_holder.setContentsMargins(0, 0, 0, 0)
        ps_body.addLayout(self._ps_holder, 1)
        ps_srow = QtWidgets.QHBoxLayout(); ps_srow.setSpacing(10)
        ps_srow.setContentsMargins(2, 0, 2, 2)
        ps_srow.addWidget(ctl("Size threshold"))
        ps_srow.addWidget(self.ps_thlab)
        ps_srow.addWidget(self.ps_slider, 1)
        ps_body.addLayout(ps_srow)
        self.ps_page = QtWidgets.QWidget(); self.ps_page.setLayout(ps_body)
        self.ps_page.setStyleSheet("background:transparent;")
        self.ps_page.setVisible(False)

        res_card = QtWidgets.QFrame(); res_card.setObjectName("card")
        # normal page: the histogram is pinned right under the chips (fixed
        # height, same in every tab) and the tiles are pinned to the bottom, so
        # switching All <-> a class never moves the chart or the tile block — the
        # flexible gap between them absorbs the differing tile-row count.
        normal_lay = QtWidgets.QVBoxLayout()
        normal_lay.setContentsMargins(0, 0, 0, 0); normal_lay.setSpacing(8)
        normal_lay.addWidget(self.result, 1)   # fills down to just above the tiles
        normal_lay.addWidget(self.stat_bar)    # … which stay pinned to the bottom
        self.normal_page = QtWidgets.QWidget(); self.normal_page.setLayout(normal_lay)
        self.normal_page.setStyleSheet("background:transparent;")

        res_lay = QtWidgets.QVBoxLayout(res_card)
        res_lay.setContentsMargins(8, 8, 8, 8); res_lay.setSpacing(8)
        res_lay.addWidget(self.chip_bar)
        res_lay.addWidget(self.normal_page, 1)
        res_lay.addWidget(self.ps_page, 1)

        right = QtWidgets.QVBoxLayout(); right.setSpacing(8); right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(hdr("RESULTS"))
        right.addWidget(res_card, 1)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right)

        # ---- training page (swaps in for RESULTS while Training mode is on) ----
        # Labelling uses the SAME "Click a particle to" tools in the Adjustments
        # card (Janus/Stripe/…/Exclude, keys 1–5/0) — no separate class picker
        # here. In training mode a click writes a training label; the View
        # checkboxes above filter which labelled classes are shown.
        tr_help = ctl("Space toggles the overlay — hide / show.")
        tr_help.setWordWrap(True)
        tr_help.setStyleSheet("color:#6a7484;font-size:11.5px;font-weight:600;"
                              "background:transparent;")

        self.tr_info = QtWidgets.QLabel()
        self.tr_info.setWordWrap(True)
        self.tr_info.setTextFormat(QtCore.Qt.RichText)
        self.tr_info.setStyleSheet("font-size:12px;color:#2c3442;background:transparent;")
        self.tr_info.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Expanding)
        self.tr_info.setAlignment(QtCore.Qt.AlignTop)

        # Clear: a menu — clear all my labels (blank slate) or one class at a time
        self.tr_clear_btn = QtWidgets.QPushButton("🧹   Clear")
        self.tr_clear_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._clear_menu = QtWidgets.QMenu(self)
        self._clear_menu.aboutToShow.connect(self._build_clear_menu)
        self.tr_clear_btn.setMenu(self._clear_menu)

        self.tr_confirm_btn = QtWidgets.QPushButton("✓   Add photo to training set")
        self.tr_confirm_btn.setObjectName("primary")
        self.tr_confirm_btn.setMinimumHeight(36)
        self.tr_confirm_btn.clicked.connect(self._train_confirm)
        self.tr_train_btn = QtWidgets.QPushButton("🧠   Train model")
        self.tr_train_btn.clicked.connect(self._train_go)
        self.tr_prog = QtWidgets.QProgressBar(); self.tr_prog.setVisible(False)
        self.tr_metrics = ctl(""); self.tr_metrics.setWordWrap(True)
        folder_btn = QtWidgets.QPushButton("📁   Open training folder")
        folder_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(trainmode.train_dir())))

        tr_body = QtWidgets.QVBoxLayout()
        tr_body.setContentsMargins(14, 14, 14, 14); tr_body.setSpacing(10)
        tr_body.addWidget(tr_help)
        tr_body.addWidget(self.tr_info, 1)
        tr_body.addWidget(self.tr_clear_btn)
        tr_body.addWidget(self.tr_confirm_btn)
        tr_body.addWidget(self.tr_train_btn)
        tr_body.addWidget(self.tr_prog)
        tr_body.addWidget(self.tr_metrics)
        tr_body.addWidget(folder_btn)
        tr_card = QtWidgets.QFrame(); tr_card.setObjectName("card")
        tr_card.setLayout(tr_body)
        tpage = QtWidgets.QVBoxLayout(); tpage.setSpacing(8); tpage.setContentsMargins(0, 0, 0, 0)
        tpage.addWidget(hdr("TRAINING"))
        tpage.addWidget(tr_card, 1)
        tpagew = QtWidgets.QWidget(); tpagew.setLayout(tpage)

        self.right_stack = QtWidgets.QStackedWidget()
        self.right_stack.addWidget(rightw)
        self.right_stack.addWidget(tpagew)
        self.right_stack.setFixedWidth(430)

        # image clicks label particles (only acted on while Training mode is on)
        self.view.clicked.connect(self._on_view_click)
        # Class keys work in BOTH modes: training paints the label, normal mode
        # picks the click tool. (M = Measure, normal mode only.)
        for key, cls in (("1", "janus"), ("2", "stripe"), ("3", "composite"),
                         ("4", "lamellar"), ("5", "undercooled"), ("6", "solid"),
                         ("0", "exclude")):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda c=cls: self._key_class(c))
        sc = QtGui.QShortcut(QtGui.QKeySequence("M"), self)
        sc.activated.connect(lambda: self._key_class("measure"))
        sc = QtGui.QShortcut(QtGui.QKeySequence("C"), self)
        sc.activated.connect(lambda: self._key_class("certainty"))
        # Space flips the overlay off/on in BOTH modes (training: show the plain
        # micrograph; normal: hide the pattern/class fills, then restore them).
        sc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        sc.activated.connect(self._flip_overlay)
        # ⌘Z (undo a label) is training-only
        self._train_shortcuts = []
        sc = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        sc.activated.connect(self._train_undo_last)
        self._train_shortcuts.append(sc)
        for sc in self._train_shortcuts:
            sc.setEnabled(False)

        leftw.setFixedWidth(230)

        # 3-column layout; middle (image) flexes so the window fits any screen
        body = QtWidgets.QHBoxLayout(); body.setSpacing(18)
        body.addWidget(leftw)
        body.addWidget(midw, 1)
        body.addWidget(self.right_stack)

        root = QtWidgets.QVBoxLayout(); root.setContentsMargins(16, 12, 16, 10); root.setSpacing(12)
        root.addWidget(tbw); root.addLayout(body, 1)
        central = QtWidgets.QWidget(); central.setLayout(root)
        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.status.showMessage("Ready — select image(s) and press Analyze.  "
                                "In the image: pinch / ⌘+scroll to zoom, double-click to fit.")
        self._update_class_controls(None)
        # a click anywhere outside the size-range boxes lets go of them — without
        # this a QLineEdit keeps focus until you tab away, so you can never leave
        QtWidgets.QApplication.instance().installEventFilter(self)
        # reserve the tile-block height once styling is applied
        QtCore.QTimer.singleShot(0, self._lock_stat_height)
        # bring back the previous session's analyses once the window is up
        QtCore.QTimer.singleShot(0, self._boot)

    def _lock_stat_height(self):
        """Freeze the tile block to its tallest (3-row 'all') layout so the chart
        above it sits in the same place whether All or a single class is shown."""
        prev = self._stat_mode
        self._relayout_stats("all")
        self.stat_bar.setFixedHeight(self.stat_bar.sizeHint().height())
        self._stat_h_locked = True
        self._stat_mode = None            # force the real view to rebuild
        if prev is not None:
            self._update_stats()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.MouseButtonPress:
            fw = self.focusWidget()
            if isinstance(fw, QtWidgets.QLineEdit):
                w = QtWidgets.QApplication.widgetAt(ev.globalPosition().toPoint())
                # clicked off the focused box (or on a non-input) -> release it
                if w is not fw and not isinstance(w, QtWidgets.QLineEdit):
                    fw.clearFocus()
        return super().eventFilter(obj, ev)

    # ---- session persistence ----
    def _save_session(self):
        """Persist the analyses so closing the app doesn't lose them."""
        import pickle
        import shutil
        try:
            p = _session_path()
            # never let an empty run wipe a session that still holds analyses,
            # and always keep the previous file as .bak (a bad save used to be
            # unrecoverable — the user lost a session that way)
            if os.path.exists(p):
                try:
                    if not self.results and os.path.getsize(p) > 200:
                        return
                    shutil.copyfile(p, p + ".bak")
                except OSError:
                    pass
            data = {"version": SESSION_VERSION,
                    "results": self.results,
                    "chosen": self.chosen,
                    # view-only per-particle corrections, so labelled photos keep
                    # their applied labels across restarts (sets aren't picklable
                    # as-is via defaults, so store lists)
                    "class_overrides": self.class_overrides,
                    "view_excluded": {k: list(v) for k, v in self.view_excluded.items()},
                    "measure_include": {k: list(v) for k, v in self.measure_include.items()},
                    "train_labels": self.train_labels,
                    "train_blank": list(self.train_blank)}
            with open(_session_path(), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            traceback.print_exc()      # never let saving break the app

    def _load_session(self):
        """Restore the previous session. Anything incompatible (old app
        version, moved files, unpicklable data) is silently discarded."""
        import pickle
        try:
            p = _session_path()
            if not os.path.exists(p):
                return
            with open(p, "rb") as f:
                data = pickle.load(f)
            ver = data.get("version")
            if ver not in (1, SESSION_VERSION):
                return
            results = {k: v for k, v in data.get("results", {}).items()
                       if os.path.exists(k)}
            if not results:
                return
            self.results.update(results)
            self.chosen.update(data.get("chosen", {}))
            # restore view-only corrections only for analyses that came back
            self.class_overrides.update({k: v for k, v in
                                         data.get("class_overrides", {}).items()
                                         if k in results})
            self.view_excluded.update({k: set(v) for k, v in
                                       data.get("view_excluded", {}).items()
                                       if k in results})
            self.measure_include.update({k: set(v) for k, v in
                                         data.get("measure_include", {}).items()
                                         if k in results})
            self.train_labels.update({k: v for k, v in
                                      data.get("train_labels", {}).items()
                                      if os.path.exists(k)})
            self.train_blank.update(p for p in data.get("train_blank", [])
                                    if os.path.exists(p))
            self._add_paths(list(results))
            self.status.showMessage(
                f"Restored {len(results)} analysis(es) from the last session.")
        except Exception:
            traceback.print_exc()

    def closeEvent(self, e):
        self._save_session()
        self._save_library()
        super().closeEvent(e)

    # ---- files ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        # a drop onto the window background (not the tree) goes to the root level
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        self._on_external_drop(paths, None)

    def import_items(self):
        """One picker for both kinds of import (the real macOS Finder panel, so
        several things can be picked at once): whatever folders you select come
        in as folders, whatever image files you select come in as images. Falls
        back to Qt's dialog if PyObjC isn't available. Nothing on disk is moved
        or renamed — the library only ever references these files."""
        paths = _native_pick()
        if paths is None:                       # PyObjC unavailable
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Import images", "",
                "Images (*.jpeg *.jpg *.png *.tif *.tiff *.bmp)")
        if paths:
            self._on_external_drop(paths, self._drop_target())

    def _on_external_drop(self, paths, target):
        """A Finder drop: folders come in as new virtual folders, loose files as
        images (into the folder they were dropped on, else the root)."""
        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if not os.path.isdir(p)]
        if dirs:
            self._add_folders_as_groups(dirs)
        if files:
            self._add_paths(files, target)

    # ---- tree helpers ----
    def _collect_images(self, paths):
        """Flatten a mix of files/folders into a de-duplicated list of images.

        Only files that actually exist are taken in — a dead path (a stale alias,
        a file deleted between the pick and the drop) would otherwise be added as
        a permanently-missing row. This does NOT affect rows already in the
        library: those are kept even when their file disappears later."""
        out = []
        for p in paths:
            if os.path.isdir(p):
                out += self._collect_images([os.path.join(p, f) for f in _listdir(p)])
            elif p.lower().endswith(IMG_EXT) and os.path.isfile(p):
                out.append(p)
        return out

    def _iter_items(self, kind=None):
        """Depth-first walk over every node, optionally filtered to one kind."""
        root = self.tree.invisibleRootItem()
        stack = [root.child(i) for i in range(root.childCount() - 1, -1, -1)]
        while stack:
            it = stack.pop()
            for j in range(it.childCount() - 1, -1, -1):
                stack.append(it.child(j))
            if kind is None or it.data(0, ROLE_KIND) == kind:
                yield it

    def _iter_image_items(self):
        yield from self._iter_items("image")

    def _all_paths(self):
        return {it.data(0, ROLE_PATH) for it in self._iter_image_items()}

    def _drop_target(self):
        """The folder a new item should land in: the selected folder, or the
        folder holding the selected image, else None (root)."""
        it = self.tree.currentItem()
        if it is None:
            return None
        return it if it.data(0, ROLE_KIND) == "folder" else it.parent()

    def _style_row(self, it):
        """Row colours/weights live in LibraryDelegate; all that's recorded here
        is whether the file is still on disk (checked once, not on every repaint)
        and the tooltip that goes with it. A vanished file is only greyed out —
        it is KEPT in the list until the user removes it."""
        if it.data(0, ROLE_KIND) == "folder":
            return
        path = it.data(0, ROLE_PATH)
        gone = bool(path) and not os.path.exists(path)
        it.setData(0, ROLE_MISSING, gone)
        it.setToolTip(0, "File not found on disk — kept until you remove it"
                      if gone else (path or ""))

    def _restyle_rows(self):
        """Re-check every row's on-disk state (after a load, a move or a rename)."""
        self.tree.blockSignals(True)
        for it in self._iter_items():
            self._style_row(it)
        self.tree.blockSignals(False)
        self.tree.viewport().update()

    _FOLDER_FLAGS = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                     | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
                     | QtCore.Qt.ItemIsEditable)
    # images are NOT drop-enabled, so nothing can ever be nested *under* an image
    _IMAGE_FLAGS = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsEditable)

    def _make_folder_item(self, name):
        it = QtWidgets.QTreeWidgetItem([name])
        it.setData(0, ROLE_KIND, "folder")
        it.setFlags(self._FOLDER_FLAGS)
        self._style_row(it)
        return it

    def _make_image_item(self, path, name=None):
        it = QtWidgets.QTreeWidgetItem([name or os.path.basename(path)])
        it.setData(0, ROLE_KIND, "image")
        it.setData(0, ROLE_PATH, path)
        it.setFlags(self._IMAGE_FLAGS)
        self._style_row(it)
        return it

    def _add_paths(self, paths, target=None):
        """Add image files as references, into `target` folder (else the root).
        Already-present paths are skipped, so re-adds and session restores never
        duplicate a row."""
        files = self._collect_images(paths)
        existing = self._all_paths()
        parent = target if (target is not None
                            and target.data(0, ROLE_KIND) == "folder") else None
        holder = parent or self.tree.invisibleRootItem()
        added = []
        self.tree.blockSignals(True)
        for path in files:
            if path in existing:
                continue
            it = self._make_image_item(path)
            holder.addChild(it)
            existing.add(path); added.append(it)
        self.tree.blockSignals(False)
        if added:
            self._restyle_rows()
            if parent is not None:
                parent.setExpanded(True)
            self.tree.resizeColumnToContents(0)
            if self.current is None:
                self.tree.setCurrentItem(added[0])
            self._save_library()
            self._refresh_results()
        return added

    def _add_folders_as_groups(self, dir_paths):
        """Import each disk folder as a new top-level virtual folder holding its
        images (recursively found, flattened). The files on disk are untouched."""
        existing = self._all_paths()
        created = []
        self.tree.blockSignals(True)
        for d in dir_paths:
            imgs = [p for p in self._collect_images([d]) if p not in existing]
            if not imgs:
                continue
            folder = self._make_folder_item(os.path.basename(d.rstrip("/\\")) or "Folder")
            self.tree.addTopLevelItem(folder)
            for p in imgs:
                folder.addChild(self._make_image_item(p)); existing.add(p)
            folder.setExpanded(True); created.append(folder)
        self.tree.blockSignals(False)
        if created:
            self._restyle_rows()
            self.tree.resizeColumnToContents(0)
            if self.current is None:
                first = next(self._iter_image_items(), None)
                if first is not None:
                    self.tree.setCurrentItem(first)
            self._save_library()
            self._refresh_results()

    def _get_or_create_top_folder(self, name):
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            it = root.child(i)
            if it.data(0, ROLE_KIND) == "folder" and it.text(0) == name:
                return it
        it = self._make_folder_item(name)
        self.tree.addTopLevelItem(it)
        return it

    # ---- new folder / rename ----
    def _new_folder(self):
        """New Folder with rows selected groups THOSE rows into it (the natural
        'put these together' gesture); with nothing selected it just makes an
        empty folder beside the current one."""
        sel = [it for it in self.tree.selectedItems()]
        # keep only the top-most of a nested selection, so moving a folder
        # doesn't also try to move a child that travelled with it
        selset = set(sel)

        def nested(it):
            p = it.parent()
            while p is not None:
                if p in selset:
                    return True
                p = p.parent()
            return False

        tops = [it for it in sel if not nested(it)]
        if not tops:
            self._new_folder_in(self._drop_target())
            return
        # the new folder takes the place of the first selected row
        anchor = tops[0]
        parent = anchor.parent()
        idx = ((parent or self.tree.invisibleRootItem()).indexOfChild(anchor))
        folder = self._make_folder_item("New Folder")
        self.tree.blockSignals(True)
        for it in tops:                       # detach, then re-home under folder
            (it.parent() or self.tree.invisibleRootItem()).removeChild(it)
        (parent or self.tree.invisibleRootItem()).insertChild(idx, folder)
        for it in tops:
            folder.addChild(it)
        self.tree.blockSignals(False)
        folder.setExpanded(True)
        self._restyle_rows()
        self.tree.resizeColumnToContents(0)
        self._save_library()
        self.tree.setCurrentItem(folder)
        self.tree.editItem(folder, 0)         # name it straight away

    def _new_folder_in(self, parent):
        it = self._make_folder_item("New Folder")
        if parent is not None and parent.data(0, ROLE_KIND) == "folder":
            parent.addChild(it); parent.setExpanded(True)
        else:
            self.tree.addTopLevelItem(it)
        self._save_library()
        self.tree.setCurrentItem(it)
        self.tree.editItem(it, 0)             # drop straight into rename

    def _rename_current(self):
        it = self.tree.currentItem()
        if it is not None:
            self.tree.editItem(it, 0)

    def _item_renamed(self, item, _col):
        """An in-place rename committed. For an image this changes only its
        display name — the file path it points to is left exactly as it was."""
        if not item.text(0).strip():          # refuse an empty name
            path = item.data(0, ROLE_PATH)
            fallback = os.path.basename(path) if path else "Folder"
            self.tree.blockSignals(True)
            item.setText(0, fallback)
            self.tree.blockSignals(False)
        self.tree.resizeColumnToContents(0)
        self._save_library()

    def _on_expand_toggle(self, _item=None):
        # the chevron itself animates in LibraryTree; nothing to redraw here
        self.tree.resizeColumnToContents(0)
        self._save_library()                  # remember which folders are open

    def _after_move(self):
        """A drag-move finished. Qt's internal move recreates the dragged rows
        with DEFAULT flags/icons, so re-assert them — most importantly, images
        must stay non-drop-enabled so nothing can be nested under them."""
        self.tree.blockSignals(True)
        for it in self._iter_items():
            it.setFlags(self._FOLDER_FLAGS if it.data(0, ROLE_KIND) == "folder"
                        else self._IMAGE_FLAGS)
        self.tree.blockSignals(False)
        self._restyle_rows()                  # icons/colours are reset by the move
        self.tree.resizeColumnToContents(0)
        self._save_library()
        self._refresh_results()

    # ---- persistence: the folder tree, independent of the analysis session ----
    def _serialize_node(self, it):
        if it.data(0, ROLE_KIND) == "image":
            return {"type": "image", "path": it.data(0, ROLE_PATH), "name": it.text(0)}
        return {"type": "folder", "name": it.text(0), "expanded": it.isExpanded(),
                "children": [self._serialize_node(it.child(j))
                             for j in range(it.childCount())]}

    def _count_images(self, data):
        """How many image rows a serialised tree holds."""
        def walk(nodes):
            n = 0
            for x in nodes:
                if x.get("type") == "image":
                    n += 1
                else:
                    n += walk(x.get("children", []))
            return n
        return walk(data.get("tree", []))

    def _save_library(self):
        """Write the whole tree to library.json — atomically, and behind several
        layers of backup, because losing this file loses the user's organisation.

        Layers, in order of age:
          library.json.bak       previous save (rolls every time)
          library.startup.json   how the library looked when the app opened
          library.shrink.json    the last state before the row count dropped

        Together these mean an accidental delete is recoverable even after the
        user has kept working — a single rolling .bak would already be gone.
        """
        import json
        import shutil
        # If the file could not be READ at startup (corrupt/unreadable), never
        # write over it: the tree in memory is empty only because loading failed,
        # and saving it would destroy a library the user can still recover from.
        # (This is the lesson from the lost session.pkl.)
        if not getattr(self, "_lib_ok", True):
            return
        try:
            root = self.tree.invisibleRootItem()
            data = {"version": 1,
                    "tree": [self._serialize_node(root.child(j))
                             for j in range(root.childCount())]}
            n_now = self._count_images(data)
            p = _library_path()
            if os.path.exists(p):
                try:
                    shutil.copyfile(p, p + ".bak")
                    # first save of this run: keep the state we opened with
                    if not getattr(self, "_startup_backup_done", False):
                        shutil.copyfile(p, os.path.join(
                            os.path.dirname(p), "library.startup.json"))
                        self._startup_backup_done = True
                    # the row count is going DOWN — keep the fuller version too
                    if n_now < getattr(self, "_lib_count", n_now):
                        shutil.copyfile(p, os.path.join(
                            os.path.dirname(p), "library.shrink.json"))
                except OSError:
                    pass
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, p)
            self._lib_count = n_now
            self._save_failed = False
        except Exception:
            traceback.print_exc()             # never let saving break the app
            # …but do tell the user once, or they'd lose work silently
            if not getattr(self, "_save_failed", False):
                self._save_failed = True
                self.status.showMessage(
                    "Could not save the image library — check disk space and "
                    "permissions for the app's data folder.", 15000)

    def _load_library(self):
        """Rebuild the tree from library.json. Missing files are kept (greyed),
        never dropped — the user's organisation always survives a restart.

        On any failure `_lib_ok` goes False, which stops _save_library from
        overwriting a file we could not read (see the note there)."""
        import json
        self._lib_ok = True
        try:
            p = _library_path()
            if not os.path.exists(p):
                return
            data = json.load(open(p, encoding="utf-8"))
            if data.get("version") != 1:
                self._lib_ok = False          # unknown format: leave it alone
                return
            # remember how many images we started with, so the very first delete
            # of the session still triggers the "shrink" backup
            self._lib_count = self._count_images(data)

            def build(node, parent):
                if node.get("type") == "image":
                    if not node.get("path"):
                        return
                    parent.addChild(self._make_image_item(node["path"], node.get("name")))
                else:
                    it = self._make_folder_item(node.get("name") or "Folder")
                    parent.addChild(it)
                    for ch in node.get("children", []):
                        build(ch, it)
                    it.setExpanded(bool(node.get("expanded", True)))

            self.tree.blockSignals(True)
            root = self.tree.invisibleRootItem()
            for node in data.get("tree", []):
                build(node, root)
            self.tree.blockSignals(False)
            self._restyle_rows()
            self.tree.resizeColumnToContents(0)
            if self.current is None:
                first = next(self._iter_image_items(), None)
                if first is not None:
                    self.tree.setCurrentItem(first)
        except Exception:
            self._lib_ok = False              # never write over what we can't read
            traceback.print_exc()
            self.status.showMessage(
                "Could not read the image library — it has been left untouched "
                "(a copy is in library.json.bak).")

    def _boot(self):
        """Startup order: rebuild the saved folder tree first, then restore the
        analysis session (which only re-attaches analyses to images already here,
        adding any stray analysed image that isn't in the tree)."""
        self._load_library()
        self._load_session()

    def _selected_paths(self):
        """Selected image paths (the analysis / results set). Selecting a folder
        counts as selecting every image anywhere beneath it."""
        paths = []

        def collect(it):
            if it.data(0, ROLE_KIND) == "image":
                paths.append(it.data(0, ROLE_PATH))
            else:
                for j in range(it.childCount()):
                    collect(it.child(j))

        for it in self.tree.selectedItems():
            collect(it)
        return list(dict.fromkeys(paths))     # de-dup, keep order

    def _current_changed(self, cur, _prev):
        if cur is None or cur.data(0, ROLE_KIND) != "image":
            return
        self.current = cur.data(0, ROLE_PATH)
        if self.current in self.results:
            self._rerender()
        else:
            try:
                self.view.set_image(Image.open(self.current))
            except Exception:
                self.view.clear_image()
                self.status.showMessage(f"Could not open: {os.path.basename(self.current)}")
        self._update_class_controls(self.results.get(self.current))
        if self.train_mode:
            self._train_update_panel()

    # ---- analysis ----
    def analyze_selected(self):
        sel = self._selected_paths()
        if not sel and self.current:   # nothing ticked -> just the previewed image
            sel = [self.current]
        # pressing Analyze always re-runs the selected image(s) from scratch, even
        # if already analysed — that's how the user sees the effect of changed
        # rules/thresholds without having to delete and re-add the image. (The
        # Cellpose masks are cached, so a re-run only redoes the fast CNN steps.)
        if sel:
            self._run(sel)
        elif self.current in self.results:
            self._rerender()

    def _run(self, paths, do_class=None, do_pattern=None):
        if not paths or self.worker is not None:
            return
        # Analyze evaluates everything; callers may still switch a step off —
        # loading labelled photos skips the pattern CNN because the saved labels
        # override it anyway.
        self._do_class = True if do_class is None else do_class
        self._do_pattern = True if do_pattern is None else do_pattern
        self.queue = list(paths)
        self._prog_total = len(paths)
        self._prog_done = 0
        self._scale_warnings = []
        self._empty_warnings = []
        self._next()

    def _next(self):
        if not self.queue:
            self.analyze_btn.setEnabled(True)
            self.status.showMessage("Analysis complete.")
            self._refresh_results()
            self._save_session()          # keep results across restarts
            warns = []
            if self._scale_warnings:
                warns.append("Scale bar could not be read (sizes shown as 0 nm):\n  "
                             + ", ".join(self._scale_warnings))
            if self._empty_warnings:
                warns.append("No particles detected:\n  " + ", ".join(self._empty_warnings))
            if warns:
                QtWidgets.QMessageBox.warning(self, "Warning", "\n\n".join(warns))
            return
        path = self.queue.pop(0)
        self.analyze_btn.setEnabled(False)
        import analyze
        # the models load lazily on the very first analysis of the session; say so
        # only then (it's the one slow, misleading-looking run), not every time
        note = "loading model, first run…" if analyze._model is None else "…"
        self.status.showMessage(
            f"Analyzing: {os.path.basename(path)}  "
            f"({self._prog_done + 1}/{self._prog_total})  {note}")
        self.worker = Worker(path, self._do_class, self._do_pattern)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, path, analysis):
        analysis.reclassify(DEFAULT_FACET_THRESH)
        self.results[path] = analysis
        self.chosen.setdefault(path, [])   # measurements are click-picked now
        self.worker = None
        self._prog_done += 1
        if not analysis.nm_per_px:            # scale bar could not be read
            self._scale_warnings.append(os.path.basename(path))
        if not analysis.particles:            # nothing segmented
            self._empty_warnings.append(os.path.basename(path))
        if path == self.current:
            self._rerender(); self._update_class_controls(analysis)
        self._refresh_results()
        self._next()

    def _on_failed(self, path, tb):
        self.worker = None
        self._prog_done += 1
        QtWidgets.QMessageBox.critical(self, "Error", f"{os.path.basename(path)}:\n\n{tb}")
        self._next()

    # ---- tree edit: remove folders/images (Delete key or right-click) ----
    #  This is the ONLY path that ever removes anything from the panel, and it
    #  only runs on a deliberate user action. Removing an image also forgets its
    #  analysis; the files on disk are never touched.
    def _delete_current(self):
        items = self.tree.selectedItems()
        if not items and self.tree.currentItem() is not None:
            items = [self.tree.currentItem()]
        if items:
            self._remove_items(items)

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        if item is None:                       # empty space -> just offer a folder
            menu.addAction("📁   New Folder", self._new_folder)
            menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        kind = item.data(0, ROLE_KIND)
        sel = [it for it in self.tree.selectedItems()] or [item]
        menu.addAction("✏️   Rename", lambda: self.tree.editItem(item, 0))
        if kind == "folder":
            menu.addAction("📁   New subfolder", lambda: self._new_folder_in(item))
        menu.addSeparator()
        if len(sel) > 1:
            menu.addAction(f"🗑   Remove {len(sel)} selected",
                           lambda: self._remove_items(sel))
        else:
            label = "🗑   Remove folder" if kind == "folder" else "🗑   Remove image"
            menu.addAction(label, lambda: self._remove_items([item]))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _remove_items(self, items):
        """Remove the given rows (folders take their whole subtree). Confirms
        first when a folder or several rows are involved, so nothing substantial
        goes on a stray keypress."""
        itemset = set(items)

        def has_ancestor_in_set(it):
            p = it.parent()
            while p is not None:
                if p in itemset:
                    return True
                p = p.parent()
            return False

        # keep only the top-most of any nested selection, so we never remove a
        # parent and then try to remove a child that went with it
        tops = [it for it in items if not has_ancestor_in_set(it)]
        has_folder = any(it.data(0, ROLE_KIND) == "folder" for it in tops)
        if has_folder or len(tops) > 1:
            if QtWidgets.QMessageBox.question(
                    self, "Remove",
                    "Remove the selected item(s) from the library?\n\n"
                    "The files on disk are not deleted.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return

        # gather the image paths going away (to forget their analyses and to pick
        # a nearby replacement preview)
        removed = set()
        for it in tops:
            if it.data(0, ROLE_KIND) == "image":
                removed.add(it.data(0, ROLE_PATH))
            else:
                for c in self._descendant_images(it):
                    removed.add(c.data(0, ROLE_PATH))
        for p in removed:
            self.results.pop(p, None)
            self.chosen.pop(p, None)

        # line up a NEARBY replacement for the previewed image before mutating,
        # so the selection lands next to where the user was, not at the top
        next_path = None
        if self.current in removed:
            flat = [it.data(0, ROLE_PATH) for it in self._iter_image_items()]
            if self.current in flat:
                i = flat.index(self.current)
                after = next((p for p in flat[i + 1:] if p not in removed), None)
                before = next((p for p in reversed(flat[:i]) if p not in removed), None)
                next_path = after or before

        self.tree.blockSignals(True)
        for it in tops:
            parent = it.parent() or self.tree.invisibleRootItem()
            parent.removeChild(it)
        self.tree.blockSignals(False)

        if self.current in removed:
            self.current = None
            cur = None
            if next_path is not None:
                cur = next((it for it in self._iter_image_items()
                           if it.data(0, ROLE_PATH) == next_path), None)
            if cur is None:
                cur = next(self._iter_image_items(), None)
            if cur is not None:
                self.tree.setCurrentItem(cur)
                self._current_changed(cur, None)   # signal may not fire after edits
            else:
                self.view.clear_image()
                self.result.clear_img()
        self._save_library()
        self._refresh_results()

    def _descendant_images(self, folder):
        for j in range(folder.childCount()):
            c = folder.child(j)
            if c.data(0, ROLE_KIND) == "image":
                yield c
            else:
                yield from self._descendant_images(c)

    # ---- view / results ----
    def _open_save_menu(self):
        """Drop the save menu under the Save button, right-aligned to it."""
        m = self._save_menu
        x = self.save_btn.width() - m.sizeHint().width()
        pos = self.save_btn.mapToGlobal(QtCore.QPoint(x, self.save_btn.height() + 4))
        m.popup(pos)

    def _update_class_controls(self, a):
        ok = bool(a and a.classifiable)
        for wdg in (self.cb_under, self.cb_solid):
            wdg.setEnabled(ok)
        # Class-paint tools need a classifiable (CBS) image; Measure always works.
        for key in ("certainty", "janus", "stripe", "composite", "lamellar",
                    "undercooled", "solid", "exclude"):
            self._tool_btns[key].setEnabled(ok)
        if not ok and self.click_tool not in (None, "measure"):
            self._tool_btns["measure"].setChecked(True)
            self._set_click_tool("measure")
        if a and not ok:
            self.cb_under.setChecked(False)
            self.cb_solid.setChecked(False)
        # The pattern-class toggles are a permanent part of the View bar (the
        # layout never reshuffles); they only *enable* once the current image
        # actually has pattern results — and never together with Solid.
        self._pat_ok = bool(a and a.has_patterns)
        self._sync_pattern_enable()

    def _sync_pattern_enable(self):
        """Solid and the pattern classes are mutually exclusive: a solid
        particle can't be tinted red and janus-orange at once. In training mode
        the pattern checkboxes are always available (they filter labelled classes)."""
        on = self.train_mode or (self._pat_ok and not self.cb_solid.isChecked())
        for cb in self._pat_cbs.values():
            if not on and cb.isChecked():
                cb.setChecked(False)
            cb.setEnabled(on)

    def _view_toggle(self):
        self._overlay_hidden = False   # ticking a class un-hides the overlay
        self._rerender()

    def _rerender(self):
        a = self.results.get(self.current)
        if not a:
            return
        if self.train_mode:
            # the Measure tool draws its size lines here too; chosen is per-photo
            chosen = self.chosen.get(self.current) if self.click_tool == "measure" else None
            # the top View checkboxes filter which labelled classes are shown;
            # none ticked = show all (so entering training isn't a blank image)
            sel = {k for k, cb in self._pat_cbs.items() if cb.isChecked()}
            if self.cb_under.isChecked():
                sel.add("undercooled")
            self.view.set_image(render_training(
                a, self._train_effective(a), show_overlay=self.train_show_overlay,
                chosen=chosen, show_classes=(sel or None),
                certainty=self._certainty_overlay()))
            self._train_update_panel()
            return
        # measurement lines are hidden while a fill/border overlay is shown, so
        # the cyan lines don't clutter those views. Any ticked pattern class
        # switches to the per-class overlay (painting just the ticked classes,
        # plus green undercooled if that's on); otherwise Solid / Undercooled
        # tick the plain red / green fills individually.
        va = self._view_analysis(a, self.current)   # apply view-only class paints
        # The overlay is driven ONLY by the top View checkboxes — a class-paint
        # tool being active no longer forces the full pattern overlay on (the user
        # wants whatever is ticked above to stay; corrections still apply to the
        # ticked classes live).
        if self._overlay_hidden:
            # Space toggled the overlay off: show the plain micrograph (measure
            # lines stay). A second press restores the ticked classes.
            sel, show_pat, cf, show_cls = set(), False, set(), False
        else:
            sel = {k for k, cb in self._pat_cbs.items() if cb.isChecked()}
            show_pat = bool(sel)
            if show_pat and self.cb_under.isChecked():
                sel.add("undercooled")
            cf = set()
            if self.cb_solid.isChecked():
                cf.add("solid")
            if self.cb_under.isChecked():
                cf.add("undercooled")
            show_cls = bool(cf) and not show_pat
        im = render(va, show_measurements=(self.click_tool == "measure"),
                    show_classification=show_cls,
                    class_filter=cf,
                    show_pattern=show_pat,
                    pattern_classes=sel,
                    show_outlines=self.cb_outline.isChecked(),
                    # with the Measure tool active, grey out everything that is
                    # not contributing a diameter to the statistics
                    mark_unmeasured=(self.click_tool == "measure"),
                    chosen=self.chosen.get(self.current),
                    certainty=self._certainty_overlay())
        self.view.set_image(im)

    def _refresh_results(self):
        sel = self._selected_paths()
        if not sel and self.current:      # nothing ticked -> show the previewed one
            sel = [self.current]
        analyzed = [self._view_analysis(self.results[p], p)
                    for p in sel if p in self.results]
        if not analyzed:
            self._result_target = None
            self._update_chips(None)
            self.result.clear_img(); return
        self._result_target = analyzed[0] if len(analyzed) == 1 else Aggregate(analyzed)
        self._update_chips(self._result_target)
        # Debounced: the matplotlib render takes a noticeable moment, and this
        # fires on EVERY selection change (arrow keys, drag-select, analysis
        # batches). Rendering through the timer coalesces bursts into one draw
        # and keeps the file tree responsive instead of freezing per click.
        self._result_timer.start(120)

    def _update_chips(self, target):
        """One chip per class present in the current selection; the bar hides
        when nothing is classified (ETD, or not analysed). Everything the panel
        needs is cached here so the readouts touch no heavy code."""
        self._res_groups = class_diams(target) if target is not None else {}
        self._res_all = (target.diam_array(whole_only=True) if target is not None
                         else np.array([]))
        self._res_stats = target.stats() if target is not None else None
        groups = self._res_groups
        self.chip_bar.setVisible(bool(groups))
        for key, b in self._chip_btns.items():
            if key in ("all", "patternsize"):
                continue
            d = groups.get(key)
            n = int(d.size) if d is not None else 0
            b.setEnabled(n > 0)
            b.setToolTip(f"{n} {CLASS_LABELS[key].lower()} particles" if n else
                         f"no {CLASS_LABELS[key].lower()} particles here")
        self._chip_btns["patternsize"].setEnabled(bool(groups))
        # the active view vanished with the selection -> fall back to All
        rf = self.result_filter
        gone = (rf == "patternsize" and not groups) or \
               (rf not in (None, "patternsize") and rf not in groups)
        if gone:
            self.result_filter = None
            for k, b in self._chip_btns.items():
                b.setChecked(k == "all")
        self._apply_result_view(target)

    def _set_result_filter(self, key):
        """Switch the RESULTS view: All, one class, or the Pattern × Size tab."""
        self.result_filter = None if key == "all" else key
        for k, b in self._chip_btns.items():
            b.setChecked(k == key)
        self._apply_result_view(getattr(self, "_result_target", None))
        if not self._ps_active:
            self._result_timer.start(10)

    def _apply_result_view(self, target):
        """Show the histogram+tiles, or the Pattern × Size page, per the chip."""
        ps = (self.result_filter == "patternsize")
        self._ps_active = ps
        self.normal_page.setVisible(not ps)
        self.ps_page.setVisible(ps)
        if ps:
            self._refresh_ps()
        else:
            self.stat_bar.setVisible(target is not None)
            self._update_stats()

    # ---- Pattern × Size tab (embedded size-composition view) ----
    def _ps_pairs(self):
        return [(cls, float(d)) for cls, arr in self._res_groups.items()
                for d in arr]

    def _refresh_ps(self):
        pairs = self._ps_pairs()
        if self._ps_plot is None:
            self._ps_plot = PatternSizePlot(pairs)
            self._ps_holder.addWidget(self._ps_plot)
        else:
            self._ps_plot.set_data(pairs)
        top = int(max(500, self._ps_plot.dmax))
        self.ps_slider.blockSignals(True)
        self.ps_slider.setMaximum(top)
        v = min(self.ps_slider.value(), top)
        self.ps_slider.setValue(v)
        self.ps_slider.blockSignals(False)
        self.ps_thlab.setText(f"{v} nm")
        self._ps_plot.set_thresh(v)
        self.ps_head.setText(f"{self._ps_plot.n_shown} particles")

    def _ps_thresh_changed(self, v):
        self.ps_thlab.setText(f"{v} nm")
        if self._ps_plot is not None:
            self._ps_plot.set_thresh(v)

    def _filtered_diams(self):
        """The diameters the RESULTS panel is currently about: one class, or
        every measured particle when no chip is active (both cached)."""
        if self.result_filter and self.result_filter != "patternsize":
            return self._res_groups.get(self.result_filter, np.array([]))
        return getattr(self, "_res_all", np.array([]))

    def _relayout_stats(self, mode):
        """Arrange the tile grid for the current view (built once per mode
        change, not every value update)."""
        if self._stat_mode == mode:
            return
        self._stat_mode = mode
        g = self.stat_grid
        while g.count():                       # detach the previous set (reused)
            w = g.takeAt(0).widget()
            if w is not None:
                w.hide()
        if mode == "filtered":
            rows = [(self.tile_total, self.tile_n),
                    (self.tile_mean, self.tile_range)]
        elif mode == "all":
            # MEAN SIZE replaces TOTAL PARTICLES here: the mean was the one
            # number the All view never showed, and the total is already implied
            # by MEASURED + the class counts below it.
            rows = [(self.tile_mean, self.tile_measured, self.tile_patterned),
                    (self.tile_cls["undercooled"], self.tile_cls["solid"],
                     self.tile_cls["janus"]),
                    (self.tile_cls["stripe"], self.tile_cls["composite"],
                     self.tile_cls["lamellar"])]
        else:                                  # unclassified (ETD): no breakdown
            rows = [(self.tile_total, self.tile_measured),
                    (self.tile_mean, self.tile_range)]
        ncol = max(len(r) for r in rows)
        for c in range(3):
            g.setColumnStretch(c, 1 if c < ncol else 0)
        # the rows share the block height evenly, so on a class view the two tile
        # rows grow to fill the space Size range leaves — no gap under it
        for r in range(3):
            g.setRowStretch(r, 1 if r < len(rows) else 0)
        for r, row in enumerate(rows):
            for c, t in enumerate(row):
                g.addWidget(t, r, c); t.show()

    def _update_stats(self):
        """The tiles and the size-range readout, from the cached diameters."""
        if getattr(self, "_res_stats", None) is None or self._ps_active:
            return
        groups = self._res_groups
        cls = self.result_filter
        total = int(self._res_stats["count_total"])
        measured = int(getattr(self, "_res_all", np.array([])).size)
        d = np.asarray(self._filtered_diams(), float)
        n = int(d.size)

        if cls:                                # one class in focus
            self._relayout_stats("filtered")
            tot = sum(int(x.size) for x in groups.values())
            pct = f"{100.0 * n / tot:.0f}%" if tot else ""
            self.tile_total.set(val=f"{total}")
            self.tile_n.set(cap=CLASS_LABELS[cls].upper(), val=f"{n}", sub=pct,
                            accent=CLASS_COLORS[cls])
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            self.tile_range.set(val=(f"{d.min():.0f} – {d.max():.0f}" if n else "—"),
                                sub="nm" if n else "")
        elif groups:                           # All, with a class breakdown
            self._relayout_stats("all")
            gsz = {k: int(v.size) for k, v in groups.items()}
            und = gsz.get("undercooled", 0)
            pat4 = sum(gsz.get(k, 0)
                       for k in ("janus", "stripe", "composite", "lamellar"))
            # undercooled counts as a pattern too, so "patterned" is everything
            # that got a class (only crystalline particles the net couldn't
            # pattern are left out). "solid" is simply every non-undercooled one.
            counts = dict(gsz)
            counts["solid"] = max(0, measured - und)
            patterned = und + pat4
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            self.tile_measured.set(val=f"{measured}")
            self.tile_patterned.set(val=f"{patterned}")
            for k in self._stat_classes:
                c = counts.get(k, 0)
                pct = f"{100.0 * c / measured:.0f}%" if measured else "—"
                self.tile_cls[k].set(val=pct, sub=f"{c}",
                                     accent=self._stat_colors[k])
        else:                                  # All, nothing classified (ETD)
            self._relayout_stats("plain")
            self.tile_total.set(val=f"{total}")
            self.tile_measured.set(val=f"{measured}")
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            self.tile_range.set(val=(f"{d.min():.0f} – {d.max():.0f}" if n else "—"),
                                sub="nm" if n else "")
        self._update_range_readout(d)

    def _update_range_readout(self, d=None):
        """How much of what's on the plot sits inside the typed size range."""
        if d is None:
            d = np.asarray(self._filtered_diams(), float)
        lo, hi = self.size_range or (None, None)
        if d.size == 0 or self.size_range is None:
            self.range_pct.setText("")       # nothing typed -> no readout at all
            self.range_cnt.setText("")
            return
        m = np.ones(d.shape, bool)
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
        k = int(m.sum())
        self.range_pct.setText(f"{100.0 * k / d.size:.0f}%")
        self.range_cnt.setText(f"({k})")

    # The chart fills the whole space above the tile block (which is a fixed
    # height, the same in every tab), so the histogram lands in exactly the same
    # rectangle and never shifts when you switch chips — it just reaches down to
    # just above the Size-range row.
    def _render_result(self):
        target = getattr(self, "_result_target", None)
        if target is None or self._ps_active:
            return                              # Pattern × Size page is showing
        w = max(60, self.result.width())
        # no height cap: the figure fills the whole space above the tiles, down to
        # just above the Size-range row, and renders at that box's real aspect so
        # it always fills the width (no side letterboxing)
        h = max(120, self.result.height())
        try:
            self.result.set_image(render_report(target, aspect=w / h,
                                                 size_range=self.size_range,
                                                 cls_filter=self.result_filter))
        except Exception:
            traceback.print_exc()

    # ---- training mode ----
    def _toggle_train_mode(self, on):
        self.train_mode = on
        self.right_stack.setCurrentIndex(1 if on else 0)
        for sc in self._train_shortcuts:
            sc.setEnabled(on)
        self._sync_pattern_enable()          # View pattern filters work in training
        if on:
            self._train_refresh_confirmed()
            self._train_update_panel()
            self.status.showMessage(
                "Training mode — pick a class in Adjustments (1–6, 0 = exclude), "
                "click particles; the View checkboxes filter classes; "
                "Space hides the overlay; ⌘Z undoes.")
        else:
            self.status.showMessage("Training mode off.")
            self._refresh_results()
        self._rerender()

    def _train_effective(self, a):
        """{pid: (class, from_user)}: the model's current view of every particle,
        overridden by the user's clicks. 'exclude' drops the particle entirely."""
        out = {}
        if a.classifiable and self.current not in self.train_blank:
            for p in a.particles:
                if getattr(p, "excluded", False):
                    continue                       # model dropped it -> unlabelled
                if p.is_solid and p.pattern:
                    out[p.id] = (p.pattern, False)
                elif not p.is_solid:
                    out[p.id] = ("undercooled", False)
        for pid, cls in self.train_labels.get(self.current, {}).items():
            if cls == "exclude":
                out.pop(pid, None)
            else:
                out[pid] = (cls, True)
        return out

    # ---- Class × Size distribution ----
    @staticmethod
    def _classsize_pairs_model(target):
        """(class, diam_nm) for every classified particle in the model's output.
        Only CBS images are classified; ETD ones are skipped."""
        analyses = getattr(target, "analyses", None) or [target]
        pairs = []
        for a in analyses:
            if not getattr(a, "classifiable", False):
                continue
            for p in a.particles:
                if not getattr(p, "diam_nm", 0):
                    continue
                if getattr(p, "excluded", False):
                    continue               # dim/unreliable -> not in the distribution
                if not p.is_solid:
                    pairs.append(("undercooled", p.diam_nm))
                elif p.pattern:
                    pairs.append((p.pattern, p.diam_nm))
                else:                      # solid but no pattern (edge/occluded/small)
                    pairs.append(("solid", p.diam_nm))
        return pairs

    def _open_classsize(self):
        # In training mode the top Class × Size button reflects the TRAINING
        # result (the user's corrected ground-truth labels for the current
        # photo); otherwise it's the model's output over the selected images.
        if self.train_mode:
            a = self.results.get(self.current)
            if a is None:
                self.status.showMessage("Analyze this image first.")
                return
            eff = self._train_effective(a)
            diam = {p.id: p.diam_nm for p in a.particles}
            pairs = [(cls, diam[pid]) for pid, (cls, _) in eff.items()
                     if pid in diam and diam[pid]]
            if not pairs:
                self.status.showMessage("No labelled particles in this photo yet.")
                return
            title = os.path.basename(getattr(a, "image", "") or self.current or "photo")
            self._cs_win = ClassSizeWindow(pairs, title, ground_truth=True, parent=self)
            self._cs_win.show()
            return
        target = getattr(self, "_result_target", None) or self.results.get(self.current)
        if target is None:
            self.status.showMessage("Analyze an image first.")
            return
        pairs = self._classsize_pairs_model(target)
        if not pairs:
            self.status.showMessage("No classified (CBS) particles in the selection.")
            return
        title = os.path.basename(getattr(target, "image", "") or "selection")
        self._cs_win = ClassSizeWindow(pairs, title, ground_truth=False, parent=self)
        self._cs_win.show()

    # ---- normal-mode click tools (measure / class-paint), view-only ----
    def _key_class(self, cls):
        """Class/measure keyboard shortcut, routed by mode."""
        if isinstance(QtWidgets.QApplication.focusWidget(), QtWidgets.QLineEdit):
            return                                 # typing digits in a box
        # both modes use the Adjustments tool selector; only the click ACTION
        # differs (train label vs view-only correction), handled in _on_view_click
        self._toggle_click_tool(cls)

    def _toggle_click_tool(self, key):
        """Clicking the active tool again turns it off (no tool selected)."""
        self._set_click_tool(None if key == self.click_tool else key)

    def _set_click_tool(self, key):
        self.click_tool = key
        # reflect the single active tool (or none) in the button row
        for k, b in self._tool_btns.items():
            b.setChecked(k == key)
        if key is None:
            self.status.showMessage("No tool selected — clicks on the image do nothing.")
        elif key == "measure":
            self.status.showMessage(
                "Measure: click a particle to show its size, click again to hide. "
                "Grey particles are left out of the size statistics (buried or "
                "frame-cut) — measuring one puts it back in.")
        elif key == "solid":
            self.status.showMessage(
                "Solid: click a particle to force it solid — the model then picks "
                "its pattern automatically, or excludes it if none is clear.")
        elif key == "certainty":
            self.status.showMessage(
                "Certainty: click a particle to show the model's confidence in its "
                "current class; click again to hide it.")
        elif self.train_mode:
            self.status.showMessage(
                f"Click a particle to label it “{key}” (training).")
        else:
            self.status.showMessage(
                f"Click a particle to mark it “{key}” (view-only correction).")
        self._rerender()

    def _particle_at(self, x, y):
        a = self.results.get(self.current)
        if a is None:
            return None, None
        masks = a.label_mask
        iy, ix = int(y), int(x)
        if masks is None or not (0 <= iy < masks.shape[0] and 0 <= ix < masks.shape[1]):
            return a, None
        pid = int(masks[iy, ix])
        return a, (pid if pid else None)

    def _on_view_click(self, x, y):
        # Certainty is an inspection tool: it reads the model's confidence and
        # works the same in both modes (never writes a label / correction).
        if self.click_tool == "certainty":
            a, pid = self._particle_at(x, y)
            if a is not None and pid is not None:
                self._toggle_certainty(pid)
            return
        if self.train_mode:
            # Measure works in training mode too: pick the center Measure tool and
            # clicks measure size; otherwise clicks label with the training class.
            if self.click_tool == "measure":
                a, pid = self._particle_at(x, y)
                if a is not None and pid is not None:
                    self._toggle_measure(a, pid)
            else:
                self._train_click(x, y)
            return
        tool = self.click_tool
        if tool is None:
            return
        a, pid = self._particle_at(x, y)
        if a is None or pid is None:
            return
        if tool == "measure":
            self._toggle_measure(a, pid)
        else:
            self._set_particle_class(pid, tool)

    def _toggle_measure(self, a, pid):
        lst = self.chosen.setdefault(self.current, [])
        inc = self.measure_include.setdefault(self.current, set())
        if any(p.id == pid for p in lst):
            self.chosen[self.current] = [p for p in lst if p.id != pid]
            inc.discard(pid)
        else:
            p = next((q for q in a.particles if q.id == pid), None)
            if p is not None:
                lst.append(p)
                # measuring a greyed-out (occluded / frame-cut) particle is the
                # user overruling the gate -> it counts towards the statistics
                if not analyze.measurable(p):
                    inc.add(pid)
        self._rerender()
        self._refresh_results()

    # ---- Certainty tool (inspect the model's confidence, both modes) ----
    def _overlay_modes(self):
        """(show_pattern, show_class): which normal-mode overlay is active — mirrors
        _rerender, so Certainty reports the class the particle is coloured as."""
        if self._overlay_hidden:
            return False, False
        show_pat = any(cb.isChecked() for cb in self._pat_cbs.values())
        show_cls = ((self.cb_solid.isChecked() or self.cb_under.isChecked())
                    and not show_pat)
        return show_pat, show_cls

    def _certainty_label(self, pid):
        """The class particle `pid` is currently shown as (so its certainty matches
        the colour on screen), or None when it carries no usable label."""
        a = self.results.get(self.current)
        if a is None:
            return None
        if self.train_mode:
            eff = self._train_effective(a)
            if pid in eff:
                return eff[pid][0]
            p = next((q for q in a.particles if q.id == pid), None)
            if p is None or getattr(p, "excluded", False):
                return None
            if p.is_solid and p.pattern:
                return p.pattern
            return "undercooled" if not p.is_solid else "solid"
        va = self._view_analysis(a, self.current)
        vp = next((q for q in va.particles if q.id == pid), None)
        if vp is None or getattr(vp, "excluded", False):
            return None                          # excluded (model or view) -> no label
        show_pat, show_cls = self._overlay_modes()
        if show_cls:                             # Solid / Undercooled overlay
            return "solid" if vp.is_solid else "undercooled"
        if vp.is_solid and vp.pattern:           # pattern overlay / plain view
            return vp.pattern
        return "solid" if vp.is_solid else "undercooled"

    def _certainty_value(self, pid, label):
        """Model confidence in [0,1] for `label`: solidnet P(solid) for solid /
        undercooled, patternnet softmax for a pattern class."""
        a = self.results.get(self.current)
        p = next((q for q in a.particles if q.id == pid), None) if a else None
        if p is None or label is None:
            return None
        if label == "undercooled":
            return 1.0 - float(getattr(p, "facet_frac", 0.0))
        if label == "solid":
            return float(getattr(p, "facet_frac", 0.0))
        if label in analyze.PATTERN_CLASSES:
            pp = getattr(p, "pattern_probs", ())
            if pp:
                return float(pp[analyze.PATTERN_CLASSES.index(label)])
        return None

    @staticmethod
    def _certainty_color(label):
        if label == "undercooled":
            return GREEN
        if label == "solid":
            return RED
        return PATTERN_COLORS.get(label, (200, 200, 200))

    def _toggle_certainty(self, pid):
        store = self.certainty.setdefault(self.current, {})
        if pid in store:                         # click again -> hide this badge
            del store[pid]
            self._rerender()
            return
        label = self._certainty_label(pid)
        val = self._certainty_value(pid, label)
        if label is None or val is None:
            self.status.showMessage(
                "This particle is excluded / not classified — no certainty to show.")
            return
        store[pid] = (f"{val * 100:.0f}%", self._certainty_color(label))
        self.status.showMessage(
            f"{label.capitalize()} certainty: {val * 100:.0f}%")
        self._rerender()

    def _certainty_overlay(self):
        """Badges (cx, cy, text, rgb) for the Certainty tool, or None. Only drawn
        while the tool is active, so the numbers don't clutter other views."""
        if self.click_tool != "certainty":
            return None
        store = self.certainty.get(self.current)
        a = self.results.get(self.current)
        if not store or a is None:
            return None
        pos = {p.id: (p.cx, p.cy) for p in a.particles}
        out = [(pos[pid][0], pos[pid][1], text, rgb)
               for pid, (text, rgb) in store.items() if pid in pos]
        return out or None

    def _resolve_solid(self, pid):
        """The manual 'Solid' tool: treat particle `pid` as solid and let the model
        pick its pattern (or 'exclude' if it can't recognise one). Returns the
        resolved class name, or None when patterns weren't evaluated for this run."""
        a = self.results.get(self.current)
        if a is None:
            return None
        p = next((q for q in a.particles if q.id == pid), None)
        if p is None:
            return None
        if not len(getattr(p, "pattern_probs", ())):
            self.status.showMessage(
                "Enable “Patterns” in Analyze to sub-classify a particle as Solid.")
            return None
        return analyze.solidify_pattern(p)

    def _set_particle_class(self, pid, cls):
        ov = self.class_overrides.setdefault(self.current, {})
        exc = self.view_excluded.setdefault(self.current, set())
        inc = self.measure_include.setdefault(self.current, set())
        if cls == "solid":
            cls = self._resolve_solid(pid)
            if cls is None:
                return
            msg = ("excluded (no clear pattern)" if cls == "exclude"
                   else f"solid → {cls}")
            self.status.showMessage(f"Particle re-analysed as {msg}.")
        if cls == "exclude":
            exc.discard(pid) if pid in exc else exc.add(pid)  # click again = undo
            ov.pop(pid, None)
            inc.discard(pid)
        else:
            exc.discard(pid)
            if ov.get(pid) == cls:
                ov.pop(pid, None)                             # same class = undo
                inc.discard(pid)
            else:
                ov[pid] = cls
                # classifying a particle BY HAND says it's exposed enough to see,
                # so it joins the size statistics (bulk-loaded labels do NOT — the
                # occlusion gate still decides those)
                inc.add(pid)
        self._rerender()
        self._refresh_results()

    def _view_analysis(self, a, path):
        """Return `a` with the user's view-only corrections applied (class paints
        + excludes). The original analysis is never mutated, so corrections are
        transient and never leak into training data or the saved model."""
        ov = self.class_overrides.get(path, {})
        exc = self.view_excluded.get(path, set())
        inc = self.measure_include.get(path, set())
        if not ov and not exc and not inc:
            return a
        parts = []
        for p in a.particles:
            if p.id in exc:
                continue
            if p.id in inc:
                p = dataclasses.replace(p, user_measurable=True)
            if p.id in ov:
                # class display only; whether it counts towards the size stats is
                # governed by `inc` (measure_include) above, not by having a class
                c = ov[p.id]
                if c == "undercooled":
                    p = dataclasses.replace(p, is_solid=False, pattern="")
                else:
                    p = dataclasses.replace(p, is_solid=True, pattern=c)
            parts.append(p)
        return dataclasses.replace(a, particles=parts)

    def _flip_overlay(self):
        """Space: hide the overlay, then restore whatever was showing."""
        if isinstance(QtWidgets.QApplication.focusWidget(), QtWidgets.QLineEdit):
            return                                  # typing in a size box
        if self.train_mode:
            self.train_show_overlay = not self.train_show_overlay
        else:
            self._overlay_hidden = not self._overlay_hidden
        self._rerender()

    _CLEAR_CLASSES = ["janus", "stripe", "composite", "lamellar",
                      "undercooled", "exclude"]

    def _build_clear_menu(self):
        """Populate the Temizle menu on demand: clear one class's labels, or all
        (all also blanks the model's pre-fill so the photo can be labelled from
        scratch — for a golden test set)."""
        m = self._clear_menu
        m.clear()
        ov = self.train_labels.get(self.current, {})
        per = {}
        for c in ov.values():
            per[c] = per.get(c, 0) + 1
        for cls in self._CLEAR_CLASSES:
            n = per.get(cls, 0)
            lab = "Exclude" if cls == "exclude" else cls.capitalize()
            act = m.addAction(f"Clear {lab} labels  ({n})")
            act.setEnabled(n > 0)
            act.triggered.connect(lambda _=False, c=cls: self._clear_class(c))
        m.addSeparator()
        blank = self.current in self.train_blank
        if blank:
            act = m.addAction("↩  Restore model predictions")
            act.triggered.connect(self._restore_model_view)
        else:
            act = m.addAction("🧹  Clear all · label from scratch")
            act.triggered.connect(self._clear_all)

    def _clear_class(self, cls):
        ov = self.train_labels.get(self.current)
        if not ov:
            return
        gone = [pid for pid, c in ov.items() if c == cls]
        for pid in gone:
            ov.pop(pid, None)
        self._train_undo = [u for u in self._train_undo
                            if not (u[0] == self.current and u[1] in gone)]
        self._rerender()
        self.status.showMessage(f"Cleared {len(gone)} “{cls}” label(s).")

    def _restore_model_view(self):
        self.train_blank.discard(self.current)
        self._rerender()
        self.status.showMessage("Model predictions restored for this photo.")

    def _clear_all(self):
        if not self.train_mode:
            return
        a = self.results.get(self.current)
        if a is None or not a.classifiable:
            self.status.showMessage("Analyze a CBS image first.")
            return
        ov = self.train_labels.get(self.current, {})
        if ov and QtWidgets.QMessageBox.question(
                self, "Clear all",
                f"Delete all {len(ov)} labels on this photo and hide the model's "
                "predictions (to label from scratch)?") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.train_labels.pop(self.current, None)
        self._train_undo = [u for u in self._train_undo if u[0] != self.current]
        self.train_blank.add(self.current)
        self._rerender()
        self.status.showMessage("Cleared — label from scratch; only your clicks are saved.")

    def _train_click(self, x, y):
        if not self.train_mode:
            return
        cls = self.click_tool
        if cls in (None, "measure"):
            self.status.showMessage(
                "Pick a class in Adjustments (1–6, 0 = exclude), then click a particle.")
            return
        a = self.results.get(self.current)
        if a is None:
            self.status.showMessage("Analyze this image first, then label it.")
            return
        if not a.classifiable:
            self.status.showMessage("Pattern labels need a CBS image.")
            return
        masks = a.label_mask
        iy, ix = int(y), int(x)
        if masks is None or not (0 <= iy < masks.shape[0] and 0 <= ix < masks.shape[1]):
            return
        pid = int(masks[iy, ix])
        if pid == 0:
            return                                    # background click
        if cls == "solid":
            # 'Solid' isn't a training class: resolve it to the pattern the model
            # would assign this now-solid particle (or 'exclude' if unrecognisable)
            cls = self._resolve_solid(pid)
            if cls is None:
                return
        # Any SEGMENTED particle can be labelled — the user's judgement overrides
        # the model. (The model itself won't auto-assign a pattern to edge/occluded
        # particles — see analyze.pattern_eligible — but if the user sees a clear
        # pattern the model missed, they set it here. 0 = exclude drops it.)
        ov = self.train_labels.setdefault(self.current, {})
        self._train_undo.append((self.current, pid, ov.get(pid)))
        ov[pid] = cls
        self._rerender()

    def _train_undo_last(self):
        if not self._train_undo:
            return
        path, pid, prev = self._train_undo.pop()
        ov = self.train_labels.setdefault(path, {})
        if prev is None:
            ov.pop(pid, None)
        else:
            ov[pid] = prev
        if path == self.current:
            self._rerender()

    _DISP = ["undercooled", "janus", "stripe", "lamellar", "composite"]

    @staticmethod
    def _model_class(p):
        if getattr(p, "excluded", False):
            return "excluded"
        if not p.is_solid:
            return "undercooled"
        if p.pattern:
            return p.pattern
        return "solid"

    def _train_update_panel(self):
        a = self.results.get(self.current)
        if a is None:
            self.tr_info.setText("Analyze an image, then click particles to label.")
            return
        if not a.classifiable:
            self.tr_info.setText("This image isn't CBS — patterns can't be labelled here.")
            return
        blank = self.current in self.train_blank
        ov = self.train_labels.get(self.current, {})
        eff = self._train_effective(a)

        # model's own prediction per particle, and the final (effective) labels
        model = {k: 0 for k in self._DISP + ["solid", "excluded"]}
        for p in a.particles:
            model[self._model_class(p)] += 1
        final = {k: 0 for k in self._DISP}
        for cls, _fu in eff.values():
            if cls in final:
                final[cls] += 1

        user_excl = sum(1 for c in ov.values() if c == "exclude")
        # dim particles the model dropped that the user re-labelled as a class
        rescued = {}
        for p in a.particles:
            if getattr(p, "excluded", False):
                c = ov.get(p.id)
                if c and c != "exclude":
                    rescued[c] = rescued.get(c, 0) + 1
        n_rescued = sum(rescued.values())

        COL = {"undercooled": GREEN, **PATTERN_COLORS}

        def chip(c):
            r, g, b = COL[c]
            return f"<span style='color:rgb({r},{g},{b});'>■</span>"

        rows = ""
        for c in self._DISP:
            m, f = model[c], final[c]
            if m == f:
                you = f"<span style='color:#1a2129;font-weight:700;'>{f}</span>"
            else:
                you = (f"<span style='color:#1a2129;font-weight:700;'>{f}</span>"
                       f"&nbsp;<span style='color:#2b6fff;font-weight:700;'>"
                       f"{f-m:+d}</span>")
            rows += (
                f"<tr>"
                f"<td style='padding:4px 0;'>{chip(c)}&nbsp;&nbsp;"
                f"<span style='color:#3a4351;font-weight:600;'>{c.capitalize()}</span></td>"
                f"<td align='right' style='padding:4px 10px;color:#aab2bc;'>{m}</td>"
                f"<td align='right' style='padding:4px 0;'>{you}</td>"
                f"</tr>")

        blank_badge = ""
        if blank:
            blank_badge = ("&nbsp;&nbsp;<span style='color:#c08a30;"
                           "font-weight:700;'>· from scratch</span>")

        def stat(label, value, color="#1a2129"):
            return (f"<tr>"
                    f"<td style='padding:3px 0;color:#7a8492;'>{label}</td>"
                    f"<td align='right' style='padding:3px 0;color:{color};"
                    f"font-weight:700;'>{value}</td></tr>")

        secondary = (
            f"<table width='100%' cellspacing='0' cellpadding='0' "
            f"style='font-size:11.5px;'>"
            + stat("Dropped by model", model['excluded'])
            + stat("Excluded by you", user_excl)
            + stat("Pattern unclear", model['solid'])
            + "</table>")

        rescue_line = ""
        if n_rescued:
            parts = ", ".join(f"{k.capitalize()} {v}" for k, v in rescued.items())
            rescue_line = (
                f"<div style='margin-top:12px;padding:8px 10px;"
                f"background:#e7f2ec;border-radius:8px;color:#2f6b48;"
                f"font-size:11.5px;font-weight:600;'>"
                f"Rescued from model &nbsp;<span style='font-weight:800;'>"
                f"{n_rescued}</span>"
                f"<span style='color:#5f9c78;font-weight:600;'> &nbsp;·&nbsp; "
                f"{parts}</span></div>")

        html = (
            f"<div style='font-size:20px;font-weight:800;color:#1a2129;'>{a.n}"
            f"<span style='font-size:12px;font-weight:600;color:#8a95a1;'>"
            f"&nbsp;particles</span></div>"
            f"<div style='font-size:11.5px;color:#8a95a1;font-weight:600;"
            f"margin-top:1px;'>{len(eff)} labelled &nbsp;·&nbsp; "
            f"{len(ov)} edits{blank_badge}</div>"
            f"<div style='margin:14px 0 4px;'>"
            f"<table width='100%' cellspacing='0' cellpadding='0' "
            f"style='font-size:12.5px;'>"
            f"<tr style='color:#a0a9b4;font-size:10.5px;font-weight:700;'>"
            f"<td style='padding-bottom:4px;'>CLASS</td>"
            f"<td align='right' style='padding-bottom:4px;'>MODEL</td>"
            f"<td align='right' style='padding-bottom:4px;'>YOU</td></tr>"
            f"{rows}</table></div>"
            f"<div style='border-top:1px solid #dce1e7;margin:12px 0 10px;'></div>"
            f"{secondary}"
            f"{rescue_line}")
        self.tr_info.setText(html)

    def _train_confirm(self):
        a = self.results.get(self.current)
        if a is None or not a.classifiable:
            self.status.showMessage("Analyze a CBS image first."); return
        eff = self._train_effective(a)
        excluded = [pid for pid, cls in self.train_labels.get(self.current, {}).items()
                    if cls == "exclude"]
        if not eff and not excluded:
            self.status.showMessage("Nothing labelled on this image yet."); return
        try:
            ncrops, ntot = trainmode.save_confirmed(a, eff, excluded=excluded)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Save failed", traceback.format_exc())
            return
        exmsg = f", {len(excluded)} excluded" if excluded else ""
        self.status.showMessage(
            f"Added {a.image}: {ntot} labels{exmsg}, {ncrops} pattern crops → "
            f"{trainmode.train_dir()}")
        # visible confirmation: flash the button, so the user isn't left
        # guessing whether the click registered
        self.tr_confirm_btn.setText("✓   Added to training set")
        QtCore.QTimer.singleShot(1900, lambda: self.tr_confirm_btn.setText(
            "✓   Add photo to training set"))
        self._train_refresh_confirmed()

    def _train_refresh_confirmed(self):
        """Update just the Train button's ready-state / photo count (the confirmed
        photos list was removed — the training folder button opens them)."""
        n = len(trainmode.confirmed())
        need = trainmode.TRAIN_MIN_PHOTOS
        self.tr_train_btn.setEnabled(n >= need and self._train_worker is None)
        self.tr_train_btn.setText(
            "🧠   Train model" if n >= need
            else f"🧠   Train model  ({n}/{need} photos)")

    def _train_go(self):
        if self._train_worker is not None:
            return
        self.tr_train_btn.setEnabled(False)
        self.tr_prog.setVisible(True); self.tr_prog.setValue(0)
        self.tr_metrics.setText("Preparing dataset…")
        w = TrainWorker(); self._train_worker = w
        w.progress.connect(self._train_progress)
        w.done.connect(self._train_done)
        w.failed.connect(self._train_failed)
        w.start()

    def _train_progress(self, done, total, phase):
        self.tr_prog.setMaximum(total); self.tr_prog.setValue(done)
        self.tr_metrics.setText(f"Training ({phase})…  {done}/{total} epochs")

    def _train_done(self, res):
        self._train_worker = None
        self.tr_prog.setVisible(False)
        import patternnet
        patternnet.reload()
        # existing analyses hold the OLD model's predictions; dropping the flag
        # makes the next "Start analysis" genuinely re-run them with the new one
        for a in self.results.values():
            a.evaluated_pattern = False
        rec = "   ".join(f"{k.capitalize()} {v:.0%}" for k, v in res["recalls"].items())
        if res.get("acc") is not None:
            head = (f"Done — accuracy {res['acc']:.0%} cross-validated across "
                    f"{res['n_val_img']} photos, your clean labels  ({rec})")
        else:
            head = f"Done — trained on {res['n']} particles (need ≥4 photos to score)"
        self.tr_metrics.setText(
            f"{head}  ·  {res['n']} particles total.\n"
            "New analyses use the retrained model; re-run Analyze to see it.")
        self._train_refresh_confirmed()

    def _train_failed(self, msg):
        self._train_worker = None
        self.tr_prog.setVisible(False)
        self.tr_metrics.setText(f"Training failed: {msg}")
        self._train_refresh_confirmed()

    # ---- size-fraction tool ----
    @staticmethod
    def _parse_num(text):
        text = text.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _range_changed(self):
        lo, hi = self._parse_num(self.range_lo.text()), self._parse_num(self.range_hi.text())
        self.size_range = None if lo is None and hi is None else (lo, hi)
        self._update_range_readout()   # the numbers answer instantly …
        self._range_timer.start(300)   # … the bar highlight re-renders after a beat

    def export_selected(self):
        sel = self._selected_paths()
        # Export what is ON SCREEN: _view_analysis applies the user's own class
        # paints and excludes. Exporting the raw analysis instead was why saved
        # images didn't match the colours in the app.
        analyzed = [self._view_analysis(self.results[p], p)
                    for p in sel if p in self.results]
        if not analyzed and self.current in self.results:
            analyzed = [self._view_analysis(self.results[self.current], self.current)]
        if not analyzed:
            QtWidgets.QMessageBox.information(self, "Info", "Analyze image(s) first.")
            return
        types = [t for cb, t in [(self.ex_line, "line"),
                                 (self.ex_under, "undercooled"),
                                 (self.ex_pattern, "pattern")] if cb.isChecked()]
        cls_filters = [k for k, a in self.ex_cls.items() if a.isChecked()]
        if not types and not cls_filters:
            QtWidgets.QMessageBox.information(self, "Info", "Select at least one export type.")
            return
        title = self.ex_title.text().strip() or None
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if not d:
            return
        try:
            out = export(analyzed, d, types, chosen_map=self.chosen, k=0,
                         size_range=self.size_range,
                         cls_filters=cls_filters, title=title)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Export failed", traceback.format_exc())
            return
        folder = os.path.dirname(next(iter(out.values()))) if out else d
        QtWidgets.QMessageBox.information(
            self, "Exported",
            f"{len(out)} file(s) saved to folder:\n{os.path.basename(folder)}/")


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


def main():
    # headless self-test for verifying the packaged bundle end-to-end
    if os.environ.get("SEMPA_SELFTEST"):
        from analyze import analyze_image
        from report import render_report
        a = analyze_image(os.environ["SEMPA_SELFTEST"])
        s = a.stats()
        print(f"SELFTEST: {s['count_total']} particles, mean {s['mean']:.0f} nm, "
              f"det {a.detector}, solid {s['n_solid']}")
        render_report(a)
        print("SELFTEST OK")
        return

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SEM Particle Analyzer")
    QtGui.QFontDatabase.addApplicationFont(SOURCE_SERIF)  # for the report figure
    # clean native UI font (SF Pro on macOS) for the controls
    f = app.font(); f.setPointSize(13); app.setFont(f)
    check = _checkmark_png().replace("\\", "/")
    app.setStyleSheet(STYLE.replace("CHECKMARK", check))
    w = MainWindow()
    # open at a size that fits the user's screen, centred
    avail = app.primaryScreen().availableGeometry()
    W = min(1420, avail.width() - 60)
    H = min(820, avail.height() - 60)
    w.resize(W, H)
    w.move(avail.center() - w.rect().center())
    w.show()
    # launching a plain script from Terminal often leaves the window behind
    # other apps on macOS (it never becomes "key"). raise_()/activateWindow()
    # called immediately can be ignored by the window server before the window
    # is actually mapped, so retry once on the next event-loop tick too.
    w.raise_()
    w.activateWindow()
    QtCore.QTimer.singleShot(0, lambda: (w.raise_(), w.activateWindow()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
