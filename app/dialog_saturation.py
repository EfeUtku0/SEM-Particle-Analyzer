"""The Data saturation window: is more labelling still buying accuracy?

Opened from the TRAINING panel. Shows the most recent learning curve (see
model_pattern_curve for how it is measured), the sentence that reads it, and the
two ways of taking it with you — the chart as a PNG for a report, and the raw
points as a CSV for anyone who wants to plot them their own way.

The window does no science: everything on it comes from the stored record, so
what is on screen and what was saved to disk cannot drift apart.
"""
from __future__ import annotations

import csv
import os

from PySide6 import QtCore, QtGui, QtWidgets

import model_pattern_curve as curve
from widget_image_view import pil_to_qpix

EMPTY = (
    "No saturation measurement yet.\n\n"
    "Tick “Measure data saturation” before a training run — or press "
    "“Measure now” below, which runs the measurement on its own without "
    "retraining. It trains several small models on 20 – 100 % of your photos "
    "and scores each on the same held-out ones, which is what gives the curve "
    "its shape.\n\n"
    "It takes roughly as long as a third of a training run, and it needs at "
    "least 6 confirmed photos.")


class SaturationWindow(QtWidgets.QDialog):
    """Non-modal: the measurement takes minutes and the user should be able to
    keep working (and watch the training panel) while it runs."""

    def __init__(self, parent=None, on_measure=None):
        super().__init__(parent)
        self.setWindowTitle("Data saturation")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self._on_measure = on_measure
        self._record = None
        self._image = None

        self.canvas = QtWidgets.QLabel()
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setWordWrap(True)
        self.canvas.setMargin(18)
        self.canvas.setStyleSheet("color:#4a5560; font-size:13px;")
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                  QtWidgets.QSizePolicy.Ignored)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#6b7580; font-size:12px;")
        self.prog = QtWidgets.QProgressBar()
        self.prog.setVisible(False)

        self.btn_measure = QtWidgets.QPushButton("↻   Measure now")
        self.btn_measure.clicked.connect(self._measure)
        self.btn_png = QtWidgets.QPushButton("⤓   Save chart…")
        self.btn_png.clicked.connect(self._save_png)
        self.btn_csv = QtWidgets.QPushButton("⤓   Save data…")
        self.btn_csv.clicked.connect(self._save_csv)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)

        row = QtWidgets.QHBoxLayout(); row.setSpacing(8)
        row.addWidget(self.btn_measure)
        row.addStretch(1)
        row.addWidget(self.btn_png)
        row.addWidget(self.btn_csv)
        row.addWidget(btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(10)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.prog)
        lay.addWidget(self.status)
        lay.addLayout(row)
        self.resize(900, 760)
        self.refresh()

    # ---------------------------------------------------------------- data

    def refresh(self, record=None):
        """Redraw from `record`, or from the newest one on disk."""
        self._record = record or curve.latest()
        self._image = None
        if not self._record:
            self.canvas.setText(EMPTY)
            self._enable_saves(False)
            self.status.setText("")
            return
        try:
            import charts
            f = curve.fit(self._record["points"])
            v = curve.verdict(self._record, f)
            self._image = charts.render_saturation(
                self._record, f, v, history=curve.load_history())
        except Exception as exc:                     # a broken chart must not
            self.canvas.setText(f"Could not draw the chart: {exc}")   # eat the
            self._enable_saves(False)                                 # window
            return
        self._enable_saves(True)
        self._paint()
        ts = self._record.get("ts")
        when = QtCore.QDateTime.fromSecsSinceEpoch(int(ts)).toString(
            "d MMM yyyy, HH:mm") if ts else "unknown time"
        self.status.setText(
            f"Measured {when} · {self._record['n_photos']} photos · "
            f"{self._record['n_total']} labelled particles")

    def _enable_saves(self, on):
        self.btn_png.setEnabled(on)
        self.btn_csv.setEnabled(on)

    def _paint(self):
        if self._image is None:
            return
        pix = pil_to_qpix(self._image)
        self.canvas.setPixmap(pix.scaled(
            self.canvas.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._paint()

    # ------------------------------------------------------------ measuring

    def measuring(self, on, done=0, total=1):
        """Driven by the window that owns the worker thread (see
        window_training) — this window never starts a QThread itself, because
        the app's one rule about them is that MainWindow retires every one."""
        self.prog.setVisible(on)
        self.btn_measure.setEnabled(not on)
        if on:
            self.prog.setMaximum(max(1, int(total)))
            self.prog.setValue(int(done))
            pct = 100.0 * done / max(1e-9, total)
            self.status.setText(
                f"Measuring… {pct:.0f}% — training the subset models. "
                "You can leave this open and keep working.")

    def _measure(self):
        if self._on_measure:
            self._on_measure()

    # -------------------------------------------------------------- saving

    def _default_name(self, ext):
        n = self._record.get("n_total", 0) if self._record else 0
        from paths import documents_dir
        return os.path.join(documents_dir(),
                            f"data saturation {n} particles.{ext}")

    def _save_png(self):
        if self._image is None:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save chart", self._default_name("png"), "PNG image (*.png)")
        if not p:
            return
        try:
            self._image.save(p)
            self.status.setText(f"Saved {os.path.basename(p)}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def _save_csv(self):
        """One row per measured model — including each repeat separately, so the
        spread the chart draws as a bar is in the file too, not just its mean."""
        if not self._record:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save data", self._default_name("csv"), "CSV (*.csv)")
        if not p:
            return
        pts = self._record["points"]
        insts = sorted({g for q in pts for g in (q.get("by_instrument") or {})})
        try:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["fraction_of_pool", "draw", "photos_train",
                            "particles_train", "held_out_accuracy",
                            "macro_recall"]
                           + [f"acc_{g}" for g in insts]
                           + [c for c in curve.CLASSES])
                for q in pts:
                    bi = q.get("by_instrument") or {}
                    w.writerow(
                        [q["frac"], q["rep"], q["photos_train"], q["n_train"],
                         round(q["acc"], 5),
                         round(q["macro_recall"], 5) if q.get("macro_recall") else ""]
                        + [round(bi[g]["acc"], 5) if g in bi else "" for g in insts]
                        + [round(q["recalls"].get(c, float("nan")), 5)
                           for c in curve.CLASSES])
            self.status.setText(f"Saved {os.path.basename(p)}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
