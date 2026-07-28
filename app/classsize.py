"""Interactive Class × Size distribution window (live threshold slider).

Opened from the main toolbar (model-predicted classes of the selected images)
or from Training mode (the user's corrected ground-truth labels). Drag the size
threshold and read, live, each class's share below / above it — the data behind
size rules like "below 400 nm, don't call stripe". Pure Qt painting so the
slider stays instant.
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

_AXIS_MAX = 1500.0          # size axis is capped here; everything above is "1500+"


def _qc(name, a=255):
    r, g, b = COLORS[name]
    return QtGui.QColor(r, g, b, a)


class _Plot(QtWidgets.QWidget):
    """Top: size-axis stacked composition (0–1500 + a 1500+ bucket).
    Bottom: below / above threshold composition bars + a shared legend."""

    NB = 30                               # 50 nm bins over 0–1500 for the top strip

    def __init__(self, pairs, parent=None):
        super().__init__(parent)
        self.thresh = 400.0
        self.setMinimumHeight(430)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.set_data(pairs)

    def set_data(self, pairs):
        """(Re)load the particles. Everything the paint needs is precomputed
        here — per-class sorted diameters (for the below/above split via
        searchsorted) and a fixed 30-bin histogram (for the top strip) — so
        dragging the threshold only repaints, never re-scans the data."""
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

    def set_thresh(self, v):
        self.thresh = float(v)
        self.update()

    def _bin(self, i):
        c = {cls: int(self._binned[cls][i]) for cls in CLASSES}
        return c, sum(c.values())

    def _split(self, t):
        below = {cls: int(np.searchsorted(self._sorted[cls], t, side="left"))
                 for cls in CLASSES}
        above = {cls: int(self._sorted[cls].size - below[cls]) for cls in CLASSES}
        return below, sum(below.values()), above, sum(above.values())

    # ---- painting helpers ------------------------------------------------
    def _hbar(self, p, x, y, w, h, counts, n):
        """One horizontal stacked composition bar for a below/above group."""
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

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        if not self.pairs:
            p.setPen(_MUTED)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "No classified particles")
            return

        ml, mr = 20, 20
        # ---------- TOP: size-axis stacked composition ----------
        # One continuous histogram 0–1500; the final bin absorbs everything ≥1500
        # (labelled "1500+"), so the tail is part of the same chart.
        plot_w = W - ml - mr
        top_y, top_h = 6, int(H * 0.60)   # the colourful strip is the centrepiece
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

        # threshold line
        tx = ml + plot_w * min(self.thresh, _AXIS_MAX) / _AXIS_MAX
        p.setPen(QtGui.QPen(QtGui.QColor(228, 96, 92), 2, QtCore.Qt.DashLine))
        p.drawLine(QtCore.QPointF(tx, top_y - 2), QtCore.QPointF(tx, top_y + top_h + 2))

        # ---------- BOTTOM: below / above composition + shared legend ----------
        clo, nlo, chi, nhi = self._split(self.thresh)

        name_w = 108
        col_gap = 18
        cols_x = ml + name_w
        col_w = (W - mr - cols_x - col_gap) / 2
        bx_lo = cols_x
        bx_hi = cols_x + col_w + col_gap

        y = top_y + top_h + 30
        # group headers: threshold on the left, the group's particle count on the
        # right (bare number — the panel is narrow, so "particles" is dropped)
        f.setPointSizeF(11); f.setBold(True); p.setFont(f); p.setPen(_INK)
        p.drawText(QtCore.QRectF(bx_lo, y, col_w, 18), QtCore.Qt.AlignLeft,
                   f"< {int(self.thresh)} nm")
        p.drawText(QtCore.QRectF(bx_hi, y, col_w, 18), QtCore.Qt.AlignLeft,
                   f"≥ {int(self.thresh)} nm")
        f.setBold(False); f.setPointSizeF(10); p.setFont(f); p.setPen(_MUTED)
        p.drawText(QtCore.QRectF(bx_lo, y, col_w, 18), QtCore.Qt.AlignRight,
                   f"n = {nlo}")
        p.drawText(QtCore.QRectF(bx_hi, y, col_w, 18), QtCore.Qt.AlignRight,
                   f"n = {nhi}")

        # composition bars
        by = y + 24
        self._hbar(p, bx_lo, by, col_w, 14, clo, nlo)
        self._hbar(p, bx_hi, by, col_w, 14, chi, nhi)

        # shared legend: chip + name on the left, %(count) under each bar
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
            for cx, counts, n in ((bx_lo, clo, nlo), (bx_hi, chi, nhi)):
                k = counts[cls]
                pct = 100.0 * k / n if n else 0.0
                z = (k == 0)
                f.setPointSizeF(11.5); f.setBold(not z); p.setFont(f)
                p.setPen(_FAINT if z else _INK)
                p.drawText(QtCore.QRectF(cx, yy, col_w - 52, row_h),
                           QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{pct:.1f}%")
                f.setBold(False); f.setPointSizeF(10.5); p.setFont(f); p.setPen(_FAINT)
                p.drawText(QtCore.QRectF(cx + col_w - 48, yy, 48, row_h),
                           QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"({k})")
            yy += row_h
        p.end()


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
        cap = QtWidgets.QLabel("Size threshold")
        cap.setStyleSheet("color:#6a7484;font-size:13px;")
        self.thlab = QtWidgets.QLabel("400 nm")
        self.thlab.setStyleSheet(
            "font-size:14px;font-weight:600;color:#2c3442;min-width:78px;")
        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld.setRange(0, int(max(500, self.plot.dmax)))
        self.sld.setValue(400)
        self.sld.setSingleStep(25)
        self.sld.valueChanged.connect(self._changed)
        row.addWidget(cap)
        row.addWidget(self.thlab)
        row.addWidget(self.sld, 1)
        lay.addLayout(row)

    def _changed(self, v):
        self.thlab.setText(f"{v} nm")
        self.plot.set_thresh(v)
