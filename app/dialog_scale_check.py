"""The scale check: how this photo's nm-per-pixel was arrived at, shown so it
can be believed — or fixed.

Every size the app reports is pixels × this one number, and it is read off the
info bar by OCR and line-finding, both of which have failed silently before
(31 AP, 2026-09-26: the 'det' cell taken for part of the bar and "5 µm" read
as "3 µm" — every diameter 2.6x too small, reported as fine). So the window
shows the evidence rather than a verdict alone: the scale cell magnified, with
the two ends the reader found drawn on it; the label it read; the arithmetic;
and what the magnification says independently.

If any of that is wrong, the markers can be dragged onto the bar's real ends
and the label retyped. "Use this scale" keeps that per photo
(scale_overrides) and re-analyses; "Back to automatic" forgets it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from info_bar_reader import _parse_label

MARK = QtGui.QColor("#2b6fff")          # the app's accent, as on the checkboxes

# verdict -> (banner colour, background, headline)
GOOD, WARN, BAD, INFO = "#1f7a4d", "#9a5b00", "#b3261e", "#12386e"
BG = {GOOD: "#e8f5ee", WARN: "#fff4e0", BAD: "#fdecea", INFO: "#eaf1fb"}


def verdict(sr, used_npp=None):
    """(colour, short text for the chip, headline, explanation) for a ScaleResult
    (or its dict). Shared by the window and the chip above the photo so the two
    can never tell different stories."""
    g = (lambda k, d=None: sr.get(k, d)) if isinstance(sr, dict) else \
        (lambda k, d=None: getattr(sr, k, d))
    if sr is None:
        if used_npp:
            return (INFO, "not checked", "Not checked yet",
                    "This analysis predates the scale check. Open it to read "
                    "the bar again.")
        return (BAD, "scale not read", "Scale not read",
                "No calibration: sizes can't be measured until a scale is set.")
    if not g("ok") or not g("nm_per_px"):
        return (BAD, "scale not read", "Scale not read",
                "The scale bar could not be read" +
                (f" ({g('note')})" if g("note") else "") +
                ". Put the markers on the bar's ends and type its label below.")
    chk, src = g("check", ""), g("source", "bar")
    if src == "manual":
        return (INFO, "set by hand", "Scale set by hand",
                "You set this scale; the reader's own answer is shown for "
                "comparison.")
    if src == "HFW":
        return (GOOD, "from HFW", "Scale from the field width (HFW)",
                "The instrument printed the image's real width; that calibrates "
                "the photo directly and outranks the bar.")
    if chk == "agree":
        return (GOOD, "confirmed", "Scale confirmed",
                "The scale bar and the magnification agree.")
    if chk == "mag_misread":
        return (GOOD, "confirmed", "Scale confirmed by the bar",
                "The magnification text looks misread by a power of ten; the "
                "bar itself is consistent.")
    if chk == "label_fixed":
        return (WARN, "label corrected", "Label corrected",
                "The label was misread; the bar's length and the magnification "
                "only fit the standard value used here.")
    if chk == "mag_only":
        return (WARN, "from magnification", "Scale from the magnification",
                "The bar couldn't be read, so the scale was worked out from the "
                "magnification. Check it against the bar below.")
    if chk == "conflict":
        return (WARN, "check this", "Bar and magnification disagree",
                "They point to different scales; the magnification was used. "
                "Check the markers below.")
    return (INFO, "not cross-checked", "Read from the scale bar",
            "Nothing on this instrument's info bar to cross-check it with — "
            "the markers below are the check.")


class BarZoom(QtWidgets.QWidget):
    """The scale cell, magnified, with the bar's two ends as draggable markers."""

    changed = QtCore.Signal(int, int)

    def __init__(self, gray, y0, y1, x0, x1, lx, rx, row):
        super().__init__()
        self._crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
        self._x0, self._y0 = x0, y0
        self.lx, self.rx, self.row = int(lx), int(rx), int(row)
        h, w = self._crop.shape
        qi = QtGui.QImage(self._crop.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        self._pix = QtGui.QPixmap.fromImage(qi.copy())
        self._drag = None
        self.setMouseTracking(True)
        self.setMinimumHeight(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Preferred)

    def sizeHint(self):
        return QtCore.QSize(860, int(860 * self._pix.height() / max(1, self._pix.width())))

    def heightForWidth(self, w):
        return int(w * self._pix.height() / max(1, self._pix.width()))

    def hasHeightForWidth(self):
        return True

    # image x <-> widget x
    def _k(self):
        return self.width() / max(1, self._pix.width())

    def _wx(self, x):
        return (x - self._x0 + 0.5) * self._k()

    def _ix(self, wx):
        return int(round(wx / self._k() - 0.5)) + self._x0

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        k = self._k()
        # nearest-neighbour: at 3-5x the individual pixels ARE the evidence
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        p.drawPixmap(QtCore.QRectF(0, 0, self.width(), self._pix.height() * k),
                     self._pix, QtCore.QRectF(self._pix.rect()))
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        yb = (self.row - self._y0 + 0.5) * k
        # the measured span, just under the bar so the bar itself stays visible
        a, b = self._wx(self.lx), self._wx(self.rx)
        # the marker lines break around the bar row: the instrument's own end
        # ticks are the evidence, and a line drawn over them would hide exactly
        # what the user is checking
        gap = max(6.0, 5 * k)
        for x in (a, b):
            p.setPen(QtGui.QPen(MARK, 1.5))
            p.drawLine(QtCore.QPointF(x, 9), QtCore.QPointF(x, yb - gap))
            p.drawLine(QtCore.QPointF(x, yb + gap), QtCore.QPointF(x, self._pix.height() * k - 2))
            p.setPen(QtCore.Qt.NoPen); p.setBrush(MARK)
            tri = QtGui.QPolygonF([QtCore.QPointF(x - 7, 0), QtCore.QPointF(x + 7, 0),
                                   QtCore.QPointF(x, 9)])
            p.drawPolygon(tri)
        p.end()

    def resizeEvent(self, e):
        # the height follows the width (a QLabel-like fixed aspect), so the
        # layout neither squashes the cell nor leaves a gap under it
        self.setFixedHeight(self.heightForWidth(self.width()))
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        x = e.position().x()
        da, db = abs(x - self._wx(self.lx)), abs(x - self._wx(self.rx))
        if min(da, db) <= 14:
            self._drag = "l" if da <= db else "r"

    def mouseMoveEvent(self, e):
        x = e.position().x()
        near = min(abs(x - self._wx(self.lx)), abs(x - self._wx(self.rx))) <= 14
        self.setCursor(QtCore.Qt.SizeHorCursor if (near or self._drag)
                       else QtCore.Qt.ArrowCursor)
        if not self._drag:
            return
        ix = min(max(self._ix(x), self._x0), self._x0 + self._pix.width() - 1)
        if self._drag == "l":
            self.lx = min(ix, self.rx - 1)
        else:
            self.rx = max(ix, self.lx + 1)
        self.update()
        self.changed.emit(self.lx, self.rx)

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def set_ends(self, lx, rx):
        self.lx, self.rx = int(lx), int(rx)
        self.update()
        self.changed.emit(self.lx, self.rx)


def _fmt_npp(v):
    return f"{v:.3g}" if v < 10 else f"{v:.2f}".rstrip("0").rstrip(".")


class ScaleCheckDialog(QtWidgets.QDialog):
    """`sr` is the scale in force (maybe hand-set), `auto` the reader's own
    answer (the same object when nothing was set by hand), `used_npp` what the
    analysis on screen was made with."""

    def __init__(self, parent, path, sr, auto, used_npp):
        super().__init__(parent)
        import os
        self.setWindowTitle(f"Scale check — {os.path.basename(path)}")
        self.action, self.new = None, None
        # the analysis hands numpy scalars (np.float64 nm/px); comparisons on
        # them give np.bool_, which Qt's setVisible/setEnabled reject outright
        used_npp = float(used_npp or 0.0)
        self._sr, self._auto, self._used = sr, auto, used_npp
        gray = np.array(Image.open(path).convert("L"))
        if getattr(sr, "rot", 0):
            gray = np.rot90(gray, sr.rot)
        h, w = gray.shape
        top = sr.info_top or int(h * 0.93)
        # the scale cell (with a sliver either side), or the right half of the
        # bar when no cell was found
        x0 = max(0, sr.cell_x0 - 6) if sr.cell_x1 else w // 2
        x1 = min(w, sr.cell_x1 + 6) if sr.cell_x1 else w
        lx, rx = (sr.left_x, sr.right_x) if sr.right_x > sr.left_x else \
            (x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4)
        row = sr.bar_row or (top + (h - top) // 3)
        self.zoom = BarZoom(gray, top, h, x0, x1, lx, rx, row)
        self.zoom.changed.connect(self._update_text)

        col, _chip, head, why = verdict(sr)
        self.banner = QtWidgets.QLabel(
            f"<span style='font-size:15px;font-weight:800'>{head}</span><br>"
            f"<span style='font-weight:600'>{why}</span>")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(f"background:{BG[col]};color:{col};"
                                  f"border-radius:8px;padding:10px 12px;")

        self.facts = QtWidgets.QLabel()
        self.facts.setTextFormat(QtCore.Qt.RichText)
        self.facts.setWordWrap(True)
        self.facts.setStyleSheet("color:#1a2129;font-size:13px;background:transparent;")

        cap = QtWidgets.QLabel("If something is off: drag the blue markers onto the "
                               "ends of the bar, correct the label, and use it.")
        cap.setWordWrap(True)
        cap.setObjectName("ctlmute")
        self.label_edit = QtWidgets.QLineEdit(sr.label_text if sr.label_nm else "")
        self.label_edit.setPlaceholderText("e.g. 5 µm")
        self.label_edit.setFixedWidth(120)
        self.label_edit.textChanged.connect(self._update_text)
        self.result_lab = QtWidgets.QLabel()
        self.result_lab.setObjectName("ctl")

        self.btn_use = QtWidgets.QPushButton("Use this scale and re-analyze")
        self.btn_use.setObjectName("primary")
        self.btn_use.clicked.connect(self._use)
        self.btn_auto = QtWidgets.QPushButton("Back to automatic")
        self.btn_auto.clicked.connect(self._reset)
        self.btn_auto.setVisible(bool(sr.source == "manual"))
        self.btn_reread = QtWidgets.QPushButton(
            f"Re-analyze with {_fmt_npp(sr.nm_per_px)} nm/px")
        self.btn_reread.clicked.connect(self._reanalyze)
        stale = bool(used_npp) and sr.ok and abs(sr.nm_per_px / used_npp - 1) > 0.005
        self.btn_reread.setVisible(bool(stale))
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)

        edit_row = QtWidgets.QHBoxLayout(); edit_row.setSpacing(8)
        lab = QtWidgets.QLabel("Label:"); lab.setObjectName("ctl")
        edit_row.addWidget(lab)
        edit_row.addWidget(self.label_edit)
        edit_row.addSpacing(10)
        edit_row.addWidget(self.result_lab)
        edit_row.addStretch(1)

        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_auto)
        btn_row.addWidget(self.btn_reread)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        btn_row.addWidget(self.btn_use)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 14); lay.setSpacing(12)
        lay.addWidget(self.banner)
        lay.addWidget(self.zoom)
        lay.addWidget(self.facts)
        lay.addSpacing(4)
        lay.addWidget(cap)
        lay.addLayout(edit_row)
        lay.addLayout(btn_row)
        self.setMinimumWidth(860)
        self._update_text()

    # ---- the readout -----------------------------------------------------
    def _parsed(self):
        return _parse_label(self.label_edit.text().strip() or "")

    def _edited(self):
        sr = self._sr
        p = self._parsed()
        return (self.zoom.lx != sr.left_x or self.zoom.rx != sr.right_x
                or p is None or abs(p[0] - sr.label_nm) > 1e-6 or not sr.ok)

    def _update_text(self, *_):
        sr, auto = self._sr, self._auto
        rows = []
        px = self.zoom.rx - self.zoom.lx
        p = self._parsed()
        if sr.source == "HFW":
            rows.append(("Field width (HFW)",
                         f"{sr.hfw_nm / 1000:.3g} µm ÷ {sr.hfw_nm / sr.nm_per_px:.0f} px "
                         f"= <b>{_fmt_npp(sr.nm_per_px)} nm/px</b>"))
        label = sr.label_text or "—"
        if sr.check == "label_fixed" and sr.label_read:
            label += f" <span style='color:{WARN}'>(read as “{sr.label_read}”)</span>"
        rows.append(("Label on the bar", label))
        rows.append(("Bar length",
                     f"x {self.zoom.lx} → {self.zoom.rx} = <b>{px} px</b>"))
        if p and px > 0:
            rows.append(("Scale", f"{p[1]} ÷ {px} px = <b>{_fmt_npp(p[0] / px)} nm/px</b>"))
        # compared with what the markers and label say NOW, so dragging a marker
        # onto the right tick turns this line green as it lands
        live = (p[0] / px) if (p and px > 0) else sr.nm_per_px
        if auto is not None and auto.mag:
            if auto.mag_npp:
                d = (live / auto.mag_npp - 1) * 100 if live else None
                agree = d is not None and abs(d) <= 5
                mark = (f"<span style='color:{GOOD}'>✓ agrees ({abs(d):.1f}%)</span>"
                        if agree else
                        f"<span style='color:{WARN}'>⚠ differs"
                        + (f" by {d:+.0f}%" if d is not None else "") + "</span>")
                rows.append(("Magnification",
                             f"{auto.mag:,} x on {auto.instrument} → "
                             f"{_fmt_npp(auto.mag_npp)} nm/px &nbsp; {mark}"
                             .replace(",", " ")))
            else:
                rows.append(("Magnification",
                             f"{auto.mag:,} x — not used to check: this instrument's "
                             f"constant isn't known".replace(",", " ")))
        if sr.source == "manual" and auto is not None:
            rows.append(("Reader's own answer",
                         (f"{auto.label_text} over {auto.bar_px} px = "
                          f"{_fmt_npp(auto.nm_per_px)} nm/px") if auto.ok else
                         "could not read the bar"))
        if self._used and sr.ok and abs(sr.nm_per_px / self._used - 1) > 0.005:
            rows.append(("This analysis",
                         f"<span style='color:{WARN}'>was made with "
                         f"{_fmt_npp(self._used)} nm/px — re-analyze to use "
                         f"{_fmt_npp(sr.nm_per_px)}</span>"))
        self.facts.setText(
            "<table cellspacing='0' cellpadding='3'>" + "".join(
                f"<tr><td style='color:#6b7580;padding-right:14px'>{k}</td>"
                f"<td>{v}</td></tr>" for k, v in rows) + "</table>")
        if p and px > 0:
            self.result_lab.setText(f"= {_fmt_npp(p[0] / px)} nm/px")
        else:
            self.result_lab.setText("type a label like “5 µm” or “500 nm”")
        self.btn_use.setEnabled(bool(p and px > 0 and self._edited()))

    # ---- actions ---------------------------------------------------------
    def _use(self):
        p = self._parsed()
        if not p:
            return
        self.action = "apply"
        self.new = (p[0], p[1], self.zoom.lx, self.zoom.rx, self.zoom.row)
        self.accept()

    def _reset(self):
        self.action = "reset"
        self.accept()

    def _reanalyze(self):
        self.action = "reanalyze"
        self.accept()
