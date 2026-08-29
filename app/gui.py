"""SEM Particle Analyzer — desktop GUI (light theme).

Load SEM micrographs, auto-read the scale bar, segment particles with Cellpose,
measure diameters and (on CBS images) classify solid vs flat particles. Select
one or several images to get per-image or pooled (combined) results, shown as a
histogram + data panel directly in the app.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from PySide6 import QtCore, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401  (sets model paths / SSL before heavy imports)
from ui_theme import STYLE, _checkmark_png
from window_library import LibraryPanel
from window_results import ResultsPanel
from window_particle_edit import ParticleEditor
from window_review import ReviewFlow
from window_training import TrainingMode
from window_analysis import AnalysisFlow
from session_store import SessionStore
from window_layout import WindowLayout


# The window is assembled from one mixin per area of the app. They are not
# independent objects — every one of them reaches across the whole window (the
# status bar, the tree, the results panel), so handing each a slice of state
# would only move the coupling somewhere less obvious. Splitting them by
# SUBJECT keeps each file readable on its own while `self` stays one window.
# None of them defines __init__ or overrides another's methods, so the order
# below is for reading, not for resolution.


class MainWindow(LibraryPanel,      # the IMAGES panel: import, file, sort, search
                 ResultsPanel,      # the RESULTS panel and the preview redraw
                 ParticleEditor,    # click tools: view-only per-particle fixes
                 ReviewFlow,        # the contact-sheet review round
                 TrainingMode,      # labelling, the training panel, retraining
                 AnalysisFlow,      # the Analyze queue and its worker handover
                 SessionStore,      # session.pkl, and an orderly shutdown
                 WindowLayout,      # builds the widgets (no behaviour of its own)
                 QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Particle Analyzer")
        self.setMinimumSize(1380, 640)   # keeps the View bar (incl. 🎓 Training) from clipping
        self.setAcceptDrops(True)

        self._init_state()
        tbw = self._build_toolbar()
        leftw = self._build_left_panel()
        midw = self._build_middle_panel()
        self._build_results_panel()
        self._wire_shortcuts()


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

        # The 1380 floor above was tuned before the four pattern checkboxes
        # existed; the toolbar now needs more than that to lay out without
        # squeezing the centre group (which is what clips checkbox labels like
        # "Undercooled" / "Composite"). Read the toolbar's own natural width
        # instead of re-guessing a number each time it grows.
        needed_w = tbw.sizeHint().width() + 32  # root's left+right margins
        if needed_w > self.minimumWidth():
            self.setMinimumWidth(needed_w)

        self.status = self.statusBar()
        # a moving (indeterminate) bar shown while an analysis runs, so the app
        # never looks frozen during the slow first run — see _next
        self.busy = QtWidgets.QProgressBar()
        self.busy.setObjectName("busybar")
        self.busy.setRange(0, 0)               # indeterminate: animates on its own
        self.busy.setTextVisible(True)
        self.busy.setFixedWidth(280)
        self.busy.setVisible(False)
        self.status.addPermanentWidget(self.busy)
        self.status.showMessage("Ready — select image(s) and press Analyze.  "
                                "In the image: pinch / ⌘+scroll to zoom, ⌘0 to fit.")
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


def main():
    # headless self-test for verifying the packaged bundle end-to-end
    if os.environ.get("SEMPA_SELFTEST"):
        from analyze import analyze_image
        from charts import render_report
        a = analyze_image(os.environ["SEMPA_SELFTEST"])
        s = a.stats()
        print(f"SELFTEST: {s['count_total']} particles, mean {s['mean']:.0f} nm, "
              f"det {a.detector}, solid {s['n_solid']}")
        render_report(a)
        print("SELFTEST OK")
        return

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SEM Particle Analyzer")
    # NOTE (crash fix, 2026-07-30): Source Serif 4 used to be registered here
    # "for the report figure" — but the report goes through matplotlib, which
    # gets the file via fonts.matplotlib_family() and its own font manager, so Qt
    # never needed it: no widget and no QSS rule names the family. What the
    # registration DID do was put a VARIABLE font (SourceSerif4.ttf has wght and
    # opsz axes) into Qt's font database, and seven of the app's ten recorded
    # crashes are a SIGSEGV at the same bogus address (0x100000000) inside
    # CoreText's CTFontCopyVariationAxes / CopyLocalizedFontNameInternal, reached
    # from QCoreTextFontEngine::init() while Qt was merely laying out a QLabel or
    # shaping text — i.e. CoreText reading variation data that is no longer
    # mapped. Registering nothing variable removes that whole path. If a Qt
    # widget ever needs the serif face, register a STATIC instance of it (see
    # fonts._bake_sf_instances for how), never the variable file.
    # The controls keep the native look (SF Pro), but as STATIC faces baked from
    # the system font — handing Qt the variable original crashes inside CoreText
    # (see fonts.qt_ui_family). Falls back to whatever Qt picked if baking fails.
    # (fonts.qt_ui_family() would swap in static faces baked from the system font
    # — see its docstring. NOT called here: Qt does not match Bold/Semibold to
    # application fonts, so its own self-check refuses the family anyway, and
    # baking three faces would cost ~5 s on a first launch for nothing.)
    f = app.font()
    f.setPointSize(13); app.setFont(f)
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
