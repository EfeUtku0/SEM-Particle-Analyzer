"""Interactive Class × Size distribution window (live size-range selection).

Opened from the main toolbar (model-predicted classes of the selected images)
or from Training mode (the user's corrected ground-truth labels). Drag the two
handles and read, live, what the particles BETWEEN them are made of — the data
behind size rules like "between 400 and 700 nm it is mostly janus". Pure Qt
painting so dragging stays instant.

It used to be a single threshold with below / above columns; a range answers the
same question (put a handle at 0 and you have "below x") and answers the ones a
split cannot — a band in the middle of the axis.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# Classes shown, bottom→top in the stack. "solid" (crystalline but no pattern
# assigned) is intentionally NOT shown — it's noise for size-rule design — and is
# filtered out of the data entirely, so the percentages are shares of the
# actually-classified particles.
CLASSES = ["undercooled", "janus", "stripe", "lamellar", "composite"]
LABELS = {"undercooled": "Undercooled", "janus": "Janus", "stripe": "Stripe",
          "lamellar": "Lamellar", "composite": "Composite"}
# Muted, harmonious palette (softer than the bright viz overlay colours, which
# the user found too loud here) — same hue identity so classes stay recognisable.
COLORS = {
    "undercooled": (109, 191, 139),
    "janus":       (227, 168, 87),
    "stripe":      (110, 155, 214),
    "lamellar":    (208, 129, 174),
    "composite":   (167, 139, 208),
}
_INK = QtGui.QColor(44, 52, 66)
_MUTED = QtGui.QColor(140, 150, 164)
_FAINT = QtGui.QColor(180, 188, 198)
_TRACK = QtGui.QColor(232, 236, 241)
_LINE = QtGui.QColor(220, 226, 234)
_MARK = QtGui.QColor(228, 96, 92)         # the range handles / their lines
_VEIL = QtGui.QColor(244, 246, 248, 176)  # laid over the out-of-range bins

# size axis cap and bin count — shared with the exported figure (chart_data),
# so the panel and its export cannot draw different axes
from chart_data import PANEL_AXIS_MAX as _AXIS_MAX, PANEL_BINS as _NB


def _qc(name, a=255):
    r, g, b = COLORS[name]
    return QtGui.QColor(r, g, b, a)


class _Plot(QtWidgets.QWidget):
    """Top: size-axis stacked composition (0–1500 + a 1500+ bucket), with the
    selected range marked by two lines and everything outside it veiled.
    Bottom: the in-range composition bar + the class legend.

    The two lines are draggable in place — the slider under the plot and the
    lines drive the same pair of numbers, whichever the user reaches for."""

    NB = _NB                              # 50 nm bins over 0–1500 for the top strip
    GRAB = 14                             # px around a line that grabs it
    rangeChanged = QtCore.Signal(float, float)   # emitted only on a line drag

    def __init__(self, pairs, parent=None):
        super().__init__(parent)
        self.lo, self.hi = 400.0, _AXIS_MAX
        self._drag = None                 # "lo" / "hi" while a line is held
        self.setMinimumHeight(430)
        self.setMouseTracking(True)       # for the resize cursor over a line
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.set_data(pairs)

    def set_data(self, pairs):
        """(Re)load the particles. Everything the paint needs is precomputed
        here — per-class sorted diameters (for the range counts via
        searchsorted) and a fixed 30-bin histogram (for the top strip) — so
        dragging only repaints, never re-scans the data."""
        self.pairs = [(c, float(d)) for c, d in pairs if c in COLORS and d]
        edges = list(_AXIS_MAX * np.arange(self.NB) / self.NB) + [np.inf]
        self._sorted = {}
        self._binned = {}
        for cls in CLASSES:
            arr = np.sort(np.array([d for c, d in self.pairs if c == cls], float))
            self._sorted[cls] = arr
            self._binned[cls] = (np.histogram(arr, bins=edges)[0]
                                 if arr.size else np.zeros(self.NB, int))
        self.dmax = max((d for _, d in self.pairs), default=1000.0)
        self.n_shown = len(self.pairs)
        self.update()

    def set_range(self, lo, hi):
        lo, hi = float(lo), float(hi)
        self.lo, self.hi = min(lo, hi), max(lo, hi)
        self.update()

    def _bin(self, i):
        c = {cls: int(self._binned[cls][i]) for cls in CLASSES}
        return c, sum(c.values())

    def _in_range(self):
        """Per-class counts of lo ≤ d < hi — except that a hi sitting at the top
        of the slider means "and everything above", so the largest particles
        are never silently left out of their own range."""
        top = self.hi >= self._top_value()
        cnt = {}
        for cls in CLASSES:
            arr = self._sorted[cls]
            a = int(np.searchsorted(arr, self.lo, side="left"))
            b = arr.size if top else int(np.searchsorted(arr, self.hi, side="right"))
            cnt[cls] = max(0, b - a)
        return cnt, sum(cnt.values())

    def _top_value(self):
        """The largest value the range can take: the axis cap. A handle parked
        there means "and everything above", exactly like the strip's 1500+ bin —
        the alternative (running the handles out to the biggest particle) spends
        most of the groove on a tail of a few dozen particles and puts the
        handles somewhere other than the lines they control."""
        return _AXIS_MAX

    # ---- painting helpers ------------------------------------------------
    def _hbar(self, p, x, y, w, h, counts, n):
        """One horizontal stacked composition bar for a group of particles."""
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(x, y, w, h), h / 2, h / 2)
        p.fillPath(path, _TRACK)
        if not n:
            return
        p.save()
        p.setClipPath(path)
        cx = x
        for cls in CLASSES:
            seg = w * counts[cls] / n
            if seg > 0:
                p.fillRect(QtCore.QRectF(cx, y, seg + 0.5, h), _qc(cls))
                cx += seg
        p.restore()

    # ---- geometry (one definition; paint and the mouse must agree) --------
    def _geom(self):
        ml, mr = 20, 20
        plot_w = self.width() - ml - mr
        top_y, top_h = 6, int(self.height() * 0.60)
        return ml, plot_w, top_y, top_h

    def _x_of(self, v):
        """Pixel x of a diameter on the (capped) size axis."""
        ml, plot_w, _, _ = self._geom()
        return ml + plot_w * min(float(v), _AXIS_MAX) / _AXIS_MAX

    def _v_of(self, x):
        """The inverse — a diameter from a pixel x, snapped to 25 nm. Dragging
        to the very end means the cap's "and above" bucket, not exactly 1500."""
        ml, plot_w, _, _ = self._geom()
        if plot_w <= 0:
            return 0.0
        f = min(1.0, max(0.0, (x - ml) / plot_w))
        if f >= 0.999:
            return self._top_value()
        return round(f * _AXIS_MAX / 25.0) * 25.0

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        if not self.pairs:
            p.setPen(_MUTED)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "No classified particles")
            return

        mr = 20
        # ---------- TOP: size-axis stacked composition ----------
        # One continuous histogram 0–1500; the final bin absorbs everything ≥1500
        # (labelled "1500+"), so the tail is part of the same chart.
        ml, plot_w, top_y, top_h = self._geom()
        nb = self.NB
        bw = plot_w / nb

        def col(x, w, counts, n):
            if not n:
                return
            yy = top_y + top_h
            for cls in CLASSES:
                h = top_h * counts[cls] / n
                if h > 0.4:
                    p.fillRect(QtCore.QRectF(x, yy - h, w + 0.6, h), _qc(cls))
                yy -= h

        for i in range(nb):
            c, n = self._bin(i)
            col(ml + i * bw, bw, c, n)

        # axis labels only — no gridlines (250 / 500 / 750 / 1000, then 1500+)
        f = p.font(); f.setPointSizeF(9.5); p.setFont(f); p.setPen(_MUTED)
        for t in (250, 500, 750, 1000):
            x = ml + plot_w * t / _AXIS_MAX
            p.drawText(QtCore.QRectF(x - 28, top_y + top_h + 4, 56, 15),
                       QtCore.Qt.AlignCenter, f"{t}")
        p.drawText(QtCore.QRectF(ml + plot_w - 62, top_y + top_h + 4, 62, 15),
                   QtCore.Qt.AlignRight, "1500+")

        # ---------- the selected range: veil outside, a line each side --------
        # veiling (rather than only drawing the lines) is what makes the strip
        # answer the question the numbers below it answer: what is IN the range
        xlo, xhi = self._x_of(self.lo), self._x_of(self.hi)
        for a, b in ((ml, xlo), (xhi, ml + plot_w)):
            if b > a:
                p.fillRect(QtCore.QRectF(a, top_y, b - a, top_h), _VEIL)
        p.setPen(QtGui.QPen(_MARK, 2, QtCore.Qt.DashLine))
        for x in (xlo, xhi):
            p.drawLine(QtCore.QPointF(x, top_y - 2),
                       QtCore.QPointF(x, top_y + top_h + 2))
        # a solid grip on each line, so it reads as something to take hold of.
        # It sits at the very top of the strip, out of the bars' way — in the
        # middle it read as a stray red dot sitting on the data.
        p.setPen(QtCore.Qt.NoPen); p.setBrush(_MARK)
        for x in (xlo, xhi):
            p.drawRoundedRect(QtCore.QRectF(x - 4, top_y - 4, 8, 14), 4, 4)
        p.setBrush(QtCore.Qt.NoBrush)

        # ---------- BOTTOM: the in-range composition + legend ----------
        cnt, n_in = self._in_range()

        name_w = 108
        bar_x = ml + name_w
        bar_w = W - mr - bar_x

        y = top_y + top_h + 30
        # header: the range on the left, how many particles fall in it on the
        # right (bare number — the panel is narrow, so "particles" is dropped)
        f.setPointSizeF(11); f.setBold(True); p.setFont(f); p.setPen(_INK)
        p.drawText(QtCore.QRectF(bar_x, y, bar_w, 18), QtCore.Qt.AlignLeft,
                   self.range_text())
        f.setBold(False); f.setPointSizeF(10); p.setFont(f); p.setPen(_MUTED)
        p.drawText(QtCore.QRectF(bar_x, y, bar_w, 18), QtCore.Qt.AlignRight,
                   f"n = {n_in}")

        # composition bar
        by = y + 24
        self._hbar(p, bar_x, by, bar_w, 14, cnt, n_in)

        # legend: chip + name on the left, % and (count) under the bar
        yy = by + 30
        row_h = 26
        for cls in CLASSES:
            # chip + name
            chip = QtCore.QRectF(ml + 2, yy + row_h / 2 - 5, 10, 10)
            pth = QtGui.QPainterPath(); pth.addRoundedRect(chip, 2.5, 2.5)
            p.fillPath(pth, _qc(cls))
            f.setPointSizeF(11); p.setFont(f); p.setPen(_INK)
            p.drawText(QtCore.QRectF(ml + 20, yy, name_w - 20, row_h),
                       QtCore.Qt.AlignVCenter, LABELS[cls])
            # values
            k = cnt[cls]
            pct = 100.0 * k / n_in if n_in else 0.0
            z = (k == 0)
            f.setPointSizeF(11.5); f.setBold(not z); p.setFont(f)
            p.setPen(_FAINT if z else _INK)
            p.drawText(QtCore.QRectF(bar_x, yy, bar_w - 62, row_h),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{pct:.1f}%")
            f.setBold(False); f.setPointSizeF(10.5); p.setFont(f); p.setPen(_FAINT)
            p.drawText(QtCore.QRectF(bar_x + bar_w - 58, yy, 58, row_h),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"({k})")
            yy += row_h
        p.end()

    def range_text(self):
        """'400 – 700 nm', or '400+ nm' when the upper handle is at the top."""
        if self.hi >= self._top_value():
            return f"{int(self.lo)}+ nm"
        return f"{int(self.lo)} – {int(self.hi)} nm"

    # ---- dragging the lines themselves ------------------------------------
    def _hit(self, x):
        """Which handle, if any, the pointer is on — the nearer one when the
        two are sitting on top of each other."""
        dlo, dhi = abs(x - self._x_of(self.lo)), abs(x - self._x_of(self.hi))
        if min(dlo, dhi) > self.GRAB:
            return None
        return "lo" if dlo <= dhi else "hi"

    def _in_plot(self, pos):
        _, _, top_y, top_h = self._geom()
        return top_y - 8 <= pos.y() <= top_y + top_h + 8

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self.pairs:
            return
        if not self._in_plot(ev.position()):
            return
        self._drag = self._hit(ev.position().x())
        if self._drag:
            self.setCursor(QtCore.Qt.SizeHorCursor)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            on = self.pairs and self._in_plot(ev.position()) and \
                self._hit(ev.position().x())
            self.setCursor(QtCore.Qt.SizeHorCursor if on else QtCore.Qt.ArrowCursor)
            return
        v = self._v_of(ev.position().x())
        # the handles may meet but not cross: a dragged handle that would pass
        # the other one stops there, rather than swapping roles mid-drag
        if self._drag == "lo":
            self.lo = min(v, self.hi)
        else:
            self.hi = max(v, self.lo)
        self.update()
        self.rangeChanged.emit(self.lo, self.hi)

    def mouseReleaseEvent(self, ev):
        self._drag = None
        self.setCursor(QtCore.Qt.ArrowCursor)


class RangeSlider(QtWidgets.QWidget):
    """Two handles on one groove — the low and high end of a size range.

    Qt has no range slider, and two stacked QSliders read as two unrelated
    controls; the span between the handles is the whole point, so it is drawn.
    Values are nm, snapped to `step`."""

    valuesChanged = QtCore.Signal(float, float)

    H = 26                     # widget height
    R = 8                      # handle radius

    def __init__(self, lo=0.0, hi=1500.0, step=25.0, parent=None):
        super().__init__(parent)
        self.minv, self.maxv, self.step = 0.0, float(hi), float(step)
        self.lo, self.hi = float(lo), float(hi)
        self._drag = None
        self.setFixedHeight(self.H)
        self.setMinimumWidth(140)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)

    def sizeHint(self):
        # a plain QWidget has no size hint, so a layout would hand this one the
        # few pixels its minimum asks for instead of the stretch it is given
        return QtCore.QSize(240, self.H)

    # ---- values ----------------------------------------------------------
    def set_bounds(self, maxv):
        """Re-scale to a new dataset, keeping the current selection where it
        still fits (a narrower dataset must not leave a handle off the end)."""
        self.maxv = float(max(self.minv + self.step, maxv))
        self.lo = min(self.lo, self.maxv)
        self.hi = min(self.hi, self.maxv)
        self.update()

    def values(self):
        return self.lo, self.hi

    def set_values(self, lo, hi, notify=False):
        lo, hi = float(lo), float(hi)
        self.lo = min(max(self.minv, min(lo, hi)), self.maxv)
        self.hi = min(max(self.minv, max(lo, hi)), self.maxv)
        self.update()
        if notify:
            self.valuesChanged.emit(self.lo, self.hi)

    # ---- geometry --------------------------------------------------------
    def _track(self):
        return self.R + 1, self.width() - self.R - 1

    def _x_of(self, v):
        x0, x1 = self._track()
        span = self.maxv - self.minv or 1.0
        return x0 + (x1 - x0) * (float(v) - self.minv) / span

    def _v_of(self, x):
        x0, x1 = self._track()
        f = min(1.0, max(0.0, (x - x0) / max(1.0, x1 - x0)))
        v = self.minv + f * (self.maxv - self.minv)
        return min(self.maxv, max(self.minv, round(v / self.step) * self.step))

    # ---- paint -----------------------------------------------------------
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        cy = self.height() / 2
        x0, x1 = self._track()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(218, 224, 232))   # darker than the panel's grey,
        p.drawRoundedRect(QtCore.QRectF(x0, cy - 3, x1 - x0, 6), 3, 3)  # or the
        # groove disappears into the background and the handles look unmoored
        xa, xb = self._x_of(self.lo), self._x_of(self.hi)
        p.setBrush(QtGui.QColor(47, 125, 225))
        p.drawRoundedRect(QtCore.QRectF(xa, cy - 3, max(1.0, xb - xa), 6), 3, 3)
        for x in (xa, xb):
            p.setBrush(QtGui.QColor(255, 255, 255))
            p.setPen(QtGui.QPen(QtGui.QColor(47, 125, 225), 2))
            p.drawEllipse(QtCore.QPointF(x, cy), self.R, self.R)
        p.end()

    # ---- interaction -----------------------------------------------------
    def _hit(self, x):
        da, db = abs(x - self._x_of(self.lo)), abs(x - self._x_of(self.hi))
        if min(da, db) <= self.R + 5:
            return "lo" if da <= db else "hi"
        return None

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        x = ev.position().x()
        self._drag = self._hit(x)
        if self._drag is None:          # a click on the groove moves the nearer
            v = self._v_of(x)           # handle to it, as a plain slider would
            self._drag = ("lo" if abs(v - self.lo) <= abs(v - self.hi) else "hi")
        self._move_to(ev.position().x())

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            self.setCursor(QtCore.Qt.PointingHandCursor
                           if self._hit(ev.position().x())
                           else QtCore.Qt.ArrowCursor)
            return
        self._move_to(ev.position().x())

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def _move_to(self, x):
        v = self._v_of(x)
        if self._drag == "lo":
            self.lo = min(v, self.hi)   # the handles meet but never cross
        else:
            self.hi = max(v, self.lo)
        self.update()
        self.valuesChanged.emit(self.lo, self.hi)


class ClassSizeWindow(QtWidgets.QDialog):
    def __init__(self, pairs, title, ground_truth=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pattern × Size — " + title)
        self.resize(700, 560)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(12)

        self.plot = _Plot(pairs)
        shown = self.plot.n_shown             # excludes filtered "solid"
        head = QtWidgets.QLabel("<b>%d particles</b>" % shown)
        head.setStyleSheet("color:#2c3442;font-size:14px;")
        lay.addWidget(head)
        lay.addWidget(self.plot, 1)

        row = QtWidgets.QHBoxLayout(); row.setSpacing(12)
        cap = QtWidgets.QLabel("Size range")
        cap.setStyleSheet("color:#6a7484;font-size:13px;")
        self.thlab = QtWidgets.QLabel("")
        self.thlab.setStyleSheet(
            "font-size:14px;font-weight:600;color:#2c3442;min-width:120px;")
        top = self.plot._top_value()
        self.sld = RangeSlider(400.0, top)
        self.sld.set_bounds(top)
        self.sld.set_values(400.0, top)
        self.sld.valuesChanged.connect(self._changed)
        # the two lines on the plot and the two handles are the same range
        self.plot.rangeChanged.connect(self._from_plot)
        row.addWidget(cap)
        row.addWidget(self.thlab)
        row.addWidget(self.sld, 1)
        lay.addLayout(row)
        self._changed(*self.sld.values())

    def _changed(self, lo, hi):
        self.plot.set_range(lo, hi)
        self.thlab.setText(self.plot.range_text())

    def _from_plot(self, lo, hi):
        self.sld.set_values(lo, hi)
        self.thlab.setText(self.plot.range_text())
