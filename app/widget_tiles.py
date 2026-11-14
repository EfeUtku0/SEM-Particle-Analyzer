"""Small reusable pieces of the interface: the stat card under a chart, the
aspect-locked frame a figure sits in, one particle crop in the review sheet,
and the drop-down that stays open while several items are ticked.

They share a file because they share nothing else — each is a few dozen lines
with no logic of its own, and giving each a module would only add imports.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from chart_data import CLASS_COLORS, CLASS_LABELS


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


class FitButton(QtWidgets.QPushButton):
    """A button whose label breaks onto a second line when one line won't fit.

    The particle tools sit in a card that gets NARROW and TALL together (the
    photo above it keeps its shape, so a narrower column leaves more height
    under it). A one-line label was simply cut off there — "Change to
    Undercool" — with half the button's height standing empty. Given the label
    in two parts, this shows "Change to Janus  (1)" for as long as that fits,
    exactly as it always looked, and only when it no longer does

        Change to
        Janus  (1)

    Each button decides for itself, at the exact pixel where its one line stops
    fitting. The column can never get narrower than the two-line form needs
    (minimumSizeHint), so nothing is ever cut.
    """

    SWATCH = 11        # the class-colour square, px
    GAP_ONE = 4        # Qt's own icon-to-text gap, all there is on one line
    GAP_TWO = 9        # on two lines the square stands clear of both of them
    FRAME = 22         # the stylesheet's 10 px padding each side + 1 px border

    def __init__(self, lead, tail):
        super().__init__()
        self._colour = None
        self._one = f"{lead} {tail}" if lead else tail
        self._two = f"{lead}\n{tail}" if lead else tail
        self._wrapped = False
        self.setText(self._one)

    def set_swatch(self, colour):
        self._colour = colour
        self._show(self._wrapped, force=True)

    def set_glyph(self, icon, size):
        """A line icon (icons.icon) in place of the colour square, for the
        tools that are not a class. `size` includes the icon's own gap."""
        self._glyph = size
        self.setIconSize(size)
        self.setIcon(icon)

    def _show(self, wrapped, force=False):
        if wrapped == self._wrapped and not force:
            return
        self._wrapped = wrapped
        self.setText(self._two if wrapped else self._one)
        if self._colour is not None:
            # the extra gap is transparent pixmap to the right of the square —
            # QPushButton has no setting for the space between icon and text
            pad = (self.GAP_TWO - self.GAP_ONE) if wrapped else 0
            pm = QtGui.QPixmap(self.SWATCH + pad, self.SWATCH)
            pm.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pm)
            p.fillRect(0, 0, self.SWATCH, self.SWATCH, QtGui.QColor(self._colour))
            p.end()
            self.setIconSize(pm.size())
            self.setIcon(QtGui.QIcon(pm))

    def _need(self, wrapped):
        """Width the button must have to show that form of its label whole."""
        self.ensurePolished()
        fm = self.fontMetrics()
        text = max(fm.horizontalAdvance(line)
                   for line in (self._two if wrapped else self._one).split("\n"))
        icon = 0
        if self._colour is not None:
            icon = self.SWATCH + (self.GAP_TWO if wrapped else self.GAP_ONE)
        elif getattr(self, "_glyph", None) is not None:
            icon = self._glyph.width() + self.GAP_ONE
        return text + icon + self.FRAME + 2          # 2 px of slack

    def one_line_width(self):
        return self._need(False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show(self._two != self._one and self._need(False) > self.width())

    def sizeHint(self):
        self.ensurePolished()
        return QtCore.QSize(self._need(False), self.fontMetrics().height() + 14)

    def minimumSizeHint(self):
        self.ensurePolished()
        lines = self._two.count("\n") + 1
        return QtCore.QSize(self._need(True),
                            lines * self.fontMetrics().height() + 12)


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


class CropTile(QtWidgets.QFrame):
    """One particle in the review grid: its crop, and the class it will be saved
    as. Clicking selects it; a class key then re-labels it."""

    # the modifiers travel with the click: the sheet needs to tell a plain click
    # (toggle this one) from a shift-click (take everything since the last one),
    # and reading the keyboard state later in the slot is a guess about when it
    # was read, not a fact about the click
    clicked = QtCore.Signal(object, int)

    def __init__(self, item, pix, parent=None):
        super().__init__(parent)
        self.item = item
        self.selected = False
        self.setFixedSize(pix.width() + 8, pix.height() + 26)
        self._img = QtWidgets.QLabel(self)
        self._img.setPixmap(pix)
        self._img.setGeometry(4, 4, pix.width(), pix.height())
        self._cap = QtWidgets.QLabel(self)
        self._cap.setGeometry(4, pix.height() + 5, pix.width(), 17)
        self._cap.setAlignment(QtCore.Qt.AlignCenter)
        self.refresh()

    def refresh(self):
        cls = self.item["cls"]
        col = CLASS_COLORS.get(cls, "#9aa4b0")
        changed = cls != self.item["model_cls"]
        self.setStyleSheet(
            f"CropTile {{ border:{3 if self.selected else 2}px solid "
            f"{'#2b6fff' if self.selected else col}; border-radius:7px; "
            f"background:{'#eaf1ff' if self.selected else '#ffffff'}; }}")
        self._cap.setStyleSheet(
            f"color:{col}; font-size:11px; font-weight:800; background:transparent;")
        self._cap.setText(("● " if changed else "") + CLASS_LABELS.get(cls, cls.title()))

    def mousePressEvent(self, e):
        self.clicked.emit(self, int(e.modifiers().value))
