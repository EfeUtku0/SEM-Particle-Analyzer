"""The model report: what the last training run produced, and how it scored.

Opened automatically when "Train model" finishes, and from the TRAINING panel
at any time. One tab per past run, newest first, so two trainings can be laid
next to each other — which is the only way to tell whether a change helped.

BUILT FROM WIDGETS, NOT FROM A RENDERED IMAGE. Everything on the page is real
Qt, styled by the app's own sheet, so it reads as part of the app and stays
sharp at any window size. "Save" then renders that same widget tree at 3x into a
PNG: what is exported is what is on screen, and there is no second layout that
can drift away from the first.

WHAT THE NUMBERS ARE. Every figure comes from the golden set — photos that live
outside the training folder and that no model has ever trained on (see
golden_store). They are computed over the particles the app MEASURES, because
that is the population every other number in the app is computed over; the
all-labelled figure is shown next to it so the gap is visible rather than
chosen. Scores are broken out per microscope as well as pooled: "did the new
instrument's photos spoil the old one" is a question the pooled number cannot
answer.
"""
from __future__ import annotations

import os
import time

from PySide6 import QtCore, QtGui, QtWidgets

import model_eval
import golden_store
from golden_store import REPORT_CLASSES, NOPAT

# the class colours the rest of the app paints with, plus one for "no pattern"
_CHIP = {"janus": "#e3a857", "stripe": "#6e9bd6", "lamellar": "#d081ae",
         "composite": "#a78bd0", NOPAT: "#6dbf8b"}
_NAME = {"janus": "Janus", "stripe": "Stripe", "lamellar": "Lamellar",
         "composite": "Composite", NOPAT: "Desensiz (undercooled + exclude)"}
_SHORT = {"janus": "Janus", "stripe": "Stripe", "lamellar": "Lamellar",
          "composite": "Composite", NOPAT: "Desensiz"}
_INSTRUMENT = {"CBS": "BIOMATEN (CBS)", "no nameplate": "METU-METE (etiketsiz)"}

EMPTY = (
    "Henüz kayıtlı bir eğitim raporu yok.\n\n"
    "“Train model” ile bir eğitim çalıştır: bittiğinde model, hiç eğitilmediği "
    "golden set fotoğrafları üzerinde puanlanır ve sonuç buraya düşer.")

NO_GOLDEN = (
    "Golden set boş — doğruluk ölçülemedi.\n\n"
    "Golden set, modelin ASLA eğitilmediği fotoğraflardan oluşur; doğruluğun "
    "tek dürüst ölçüsü budur. Kurmak için: bir fotoğrafı Training modunda "
    "etiketleyip eğitim setine ekle, sonra o fotoğrafın dosyalarını "
    "“SEM Eğitim” klasöründen “SEM Golden” klasörüne TAŞI (kopyalama — aynı "
    "fotoğraf iki yerde olamaz).")


def _pct(v):
    return "—" if v is None else f"%{100 * v:.1f}"


def _lab(text, obj=None, color=None, size=None, bold=None):
    w = QtWidgets.QLabel(text)
    if obj:
        w.setObjectName(obj)
    css = []
    if color:
        css.append(f"color:{color}")
    if size:
        css.append(f"font-size:{size}px")
    if bold is not None:
        css.append(f"font-weight:{bold}")
    if css:
        w.setStyleSheet(";".join(css) + ";background:transparent;")
    return w


def _note(text):
    """An explanatory paragraph that WRAPS and has no say in how wide the page is.

    Both halves matter. A QLabel without setWordWrap reports its whole sentence
    as its minimum width, so one long line silently became the layout's minimum
    and pushed the tables off the side of the window (measured: the page wanted
    1480 px). Ignored horizontal policy is what stops it happening again the next
    time someone writes a longer sentence.
    """
    w = _lab(text, color="#6b7580", size=12)
    w.setWordWrap(True)
    w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
    return w


def _card(title=None):
    f = QtWidgets.QFrame()
    f.setObjectName("card")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(10)
    if title:
        lay.addWidget(_lab(title, "h"))
    return f, lay


def _chip(cls, text=None):
    """A class name preceded by its own colour dot — the same colours the
    overlay and the charts use, so a row is recognisable without reading it."""
    w = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(7)
    dot = QtWidgets.QLabel()
    dot.setFixedSize(10, 10)
    dot.setStyleSheet(f"background:{_CHIP.get(cls, '#9aa4ae')};border-radius:5px;")
    row.addWidget(dot)
    row.addWidget(_lab(text or _SHORT.get(cls, cls), color="#1a2129", size=12, bold=700))
    row.addStretch(1)
    return w


class _Stat(QtWidgets.QFrame):
    """One headline number, in the app's existing tile shape."""

    def __init__(self, caption, value, sub=""):
        super().__init__()
        self.setObjectName("tile")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(2)
        lay.addWidget(_lab(caption.upper(), "tilecap"))
        v = _lab(value, "tileval")
        v.setStyleSheet("color:#1a2129;font-size:22px;font-weight:800;background:transparent;")
        lay.addWidget(v)
        if sub:
            lay.addWidget(_lab(sub, "tilesub"))


def _metrics_table(block):
    """recall / precision / F1 per class, with the overall row underneath."""
    w = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(w)
    g.setContentsMargins(0, 2, 0, 0)
    g.setHorizontalSpacing(10)
    g.setVerticalSpacing(6)
    heads = ["Sınıf", "Parçacık", "Recall", "Precision", "F1"]
    for c, h in enumerate(heads):
        lb = _lab(h.upper(), color="#8a95a1", size=10.5, bold=800)
        lb.setStyleSheet(lb.styleSheet() + "letter-spacing:0.6px;")
        g.addWidget(lb, 0, c,
                    QtCore.Qt.AlignLeft if c == 0 else QtCore.Qt.AlignRight)
    line = QtWidgets.QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet("background:#e2e7ec;")
    g.addWidget(line, 1, 0, 1, len(heads))
    r = 2
    for cls in REPORT_CLASSES:
        m = (block.get("per_class") or {}).get(cls) or {}
        if not m.get("n"):
            continue
        g.addWidget(_chip(cls), r, 0)
        for c, v in enumerate([str(m["n"]), _pct(m.get("recall")),
                               _pct(m.get("prec")), _pct(m.get("f1"))], start=1):
            lb = _lab(v, color="#1a2129" if c > 1 else "#5a6472", size=12.5, bold=700)
            g.addWidget(lb, r, c, QtCore.Qt.AlignRight)
        r += 1
    line2 = QtWidgets.QFrame()
    line2.setFixedHeight(1)
    line2.setStyleSheet("background:#e2e7ec;")
    g.addWidget(line2, r, 0, 1, len(heads)); r += 1
    g.addWidget(_lab("GENEL", color="#1a2129", size=12, bold=800), r, 0)
    g.addWidget(_lab(str(block.get("n") or 0), color="#5a6472", size=12.5, bold=700),
                r, 1, QtCore.Qt.AlignRight)
    g.addWidget(_lab(f"doğruluk {_pct(block.get('acc'))}", color="#1a2129",
                     size=12.5, bold=800), r, 2, 1, 2, QtCore.Qt.AlignRight)
    g.addWidget(_lab(_pct(block.get("macro_f1")), color="#2b6fff", size=12.5, bold=800),
                r, 4, QtCore.Qt.AlignRight)
    g.setColumnStretch(0, 1)
    return w


def _confusion(block):
    """Truth down the side, the app's answer across the top.

    Cells are tinted by their share of the TRUE row, not by raw count, so a
    small class's mistakes are as visible as a large one's — the whole point of
    looking at the matrix rather than the accuracy."""
    conf = block.get("confusion") or []
    present = [i for i, c in enumerate(REPORT_CLASSES)
               if (block.get("per_class") or {}).get(c, {}).get("n")]
    w = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(w)
    g.setContentsMargins(0, 2, 0, 0)
    g.setHorizontalSpacing(4)
    g.setVerticalSpacing(4)
    g.addWidget(_lab("gerçek ↓ / model →", color="#8a95a1", size=10.5, bold=800), 0, 0)
    for c, j in enumerate(present, start=1):
        g.addWidget(_chip(REPORT_CLASSES[j]), 0, c)
    for r, i in enumerate(present, start=1):
        g.addWidget(_chip(REPORT_CLASSES[i]), r, 0)
        row = conf[i] if i < len(conf) else []
        tot = sum(row) or 1
        for c, j in enumerate(present, start=1):
            v = row[j] if j < len(row) else 0
            share = v / tot
            if i == j:
                bg = f"rgba(43,111,255,{0.06 + 0.42 * share:.3f})"
                fg = "#12386e"
            elif v:
                bg = f"rgba(214,110,74,{0.07 + 0.55 * share:.3f})"
                fg = "#7a3517"
            else:
                bg, fg = "#ffffff", "#c3cad1"
            cell = QtWidgets.QLabel(f"{v}\n%{100 * share:.0f}" if v else "0")
            cell.setAlignment(QtCore.Qt.AlignCenter)
            cell.setMinimumWidth(62)
            cell.setStyleSheet(
                f"background:{bg};color:{fg};border:1px solid #e6ebf0;"
                f"border-radius:6px;padding:6px 4px;font-size:12px;font-weight:700;")
            g.addWidget(cell, r, c)
    return w


class ReportPage(QtWidgets.QWidget):
    """One training run, rendered. Also the unit that Save exports."""

    def __init__(self, run, parent=None):
        super().__init__(parent)
        self.run = run
        self.setStyleSheet("background:#ffffff;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        self._header(lay)
        g = run.get("golden") or {}
        if run.get("conflicts"):
            lay.addWidget(self._warning(
                "⚠ Modelin eğitiminde de yer alan golden fotoğraf: "
                + ", ".join(run["conflicts"]) + ".\n"
                + (run.get("conflict_note")
                   or "Bu fotoğrafların skoru sızıntılı — birinden silmelisin.")))
        if not g:
            box, bl = _card("GOLDEN SET")
            bl.addWidget(_note(run.get("golden_error") or NO_GOLDEN))
            lay.addWidget(box)
            lay.addStretch(1)
            return
        self._golden(lay, g)
        lay.addStretch(1)

    # ------------------------------------------------------------- sections

    def _warning(self, text):
        f = QtWidgets.QFrame()
        f.setStyleSheet("background:#fff4ec;border:1px solid #f0c8ae;border-radius:8px;")
        l = QtWidgets.QVBoxLayout(f)
        l.setContentsMargins(14, 11, 14, 11)
        w = _lab(text, color="#7a3517", size=12.5, bold=700)
        w.setWordWrap(True)
        w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        l.addWidget(w)
        return f

    def _header(self, lay):
        run = self.run
        card, cl = _card()
        when = time.strftime("%d.%m.%Y  %H:%M", time.localtime(run.get("ts") or 0))
        title = ("Mevcut model puanlandı" if run.get("scored_only")
                 else "Model eğitildi")
        cl.addWidget(_lab(f"{title} — {when}", color="#1a2129", size=17, bold=800))
        secs = run.get("train_secs") or 0
        bits = [f"{run.get('n_photos', 0)} fotoğraf",
                f"{run.get('n_particles', 0)} etiketli parçacık",
                f"{run.get('epochs')} epoch" if run.get("epochs") else "",
                str(run.get("device", "")).upper(),
                (f"{secs / 60:.0f} dk" if secs >= 90 else f"{secs:.0f} sn") if secs else ""]
        cl.addWidget(_note("  ·  ".join(b for b in bits if b)))
        if run.get("scored_only"):
            # a history row that was never a training run must say so, or two
            # rows that mean different things get compared as though they didn't
            cl.addWidget(_note(
                "Bu satır yeni bir eğitim değil: o sırada kurulu olan model "
                "golden sete karşı ölçüldü. Eğitim tarihi ve parçacık sayısı "
                "modelin kendi eğitimine aittir."))
        counts = run.get("counts") or {}
        if counts:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(16)
            for cls in ("janus", "stripe", "lamellar", "composite"):
                if counts.get(cls):
                    row.addWidget(_chip(cls, f"{_SHORT[cls]}  {counts[cls]}"))
            row.addStretch(1)
            cl.addLayout(row)
        lay.addWidget(card)

    def _golden(self, lay, g):
        card, cl = _card("GOLDEN SET — MODELİN HİÇ GÖRMEDİĞİ FOTOĞRAFLAR")
        comb = g.get("combined") or {}
        sub = (f"{g.get('photos', 0)} fotoğraf · {g.get('particles', 0)} etiketli "
               f"parçacık · ölçülen {comb.get('n', 0)}")
        if g.get("unmatched"):
            sub += f" · {g['unmatched']} etiket bu segmentasyonda eşleşmedi"
        cl.addWidget(_note(sub))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(_Stat("macro-F1", _pct(comb.get("macro_f1")), "sınıf ortalaması"))
        row.addWidget(_Stat("doğruluk", _pct(comb.get("acc")), "5 sınıf birden"))
        row.addWidget(_Stat("katı / sıvı", _pct(comb.get("solid_acc")),
                            "desen var mı yok mu"))
        all_lab = (comb.get("all_labelled") or {}).get("macro_f1")
        row.addWidget(_Stat("tüm etiketler", _pct(all_lab),
                            "ölçüm kapısı olmadan"))
        row.addStretch(1)
        cl.addLayout(row)
        cl.addWidget(_metrics_table(comb))
        cl.addWidget(_note(
            "Ölçülen parçacıklar üzerinde — uygulamanın istatistiklerinde "
            "gösterdiği parçacık kümesi. Desensiz satırı undercooled ile "
            "exclude'u birlikte sayar; ekranda ikisi de aynı görünür."))
        cl.addWidget(_note(
            "KARIŞIKLIK MATRİSİ — satır senin etiketin, sütun modelin cevabı. "
            "Renk o satırın yüzdesine göre koyulaşır, böylece küçük bir sınıfın "
            "hatası da büyüğünki kadar görünür."))
        cl.addWidget(_confusion(comb))
        lay.addWidget(card)

        # Each microscope on its own, stacked rather than side by side: two full
        # tables next to each other squeeze to about eighty pixels a column and
        # stop being readable, and this page is allowed to scroll.
        by = g.get("by_instrument") or {}
        if len(by) > 1:
            for ins in sorted(by):
                b = by[ins]
                icard, icl = _card(_INSTRUMENT.get(ins, ins).upper())
                icl.addWidget(_lab(
                    f"macro-F1 {_pct(b.get('macro_f1'))}   ·   doğruluk "
                    f"{_pct(b.get('acc'))}   ·   katı/sıvı "
                    f"{_pct(b.get('solid_acc'))}   ·   {b.get('n', 0)} ölçülen "
                    f"parçacık", color="#2b6fff", size=13, bold=800))
                icl.addWidget(_metrics_table(b))
                icl.addWidget(_confusion(b))
                lay.addWidget(icard)

        per = [p for p in (g.get("per_photo") or []) if p.get("n_measured")]
        if per:
            pcard, pcl = _card("FOTOĞRAF BAŞINA")
            pcl.addWidget(_note(
                "En zordan başlayarak. Bir fotoğrafta bir sınıf çoğu zaman iki "
                "üç parçacıktır ve macro-F1 onu sıfıra çevirip fotoğrafın "
                "sayısını yirmi puan oynatır — sıralama bu yüzden doğruluğa "
                "göre; macro-F1 ise bir sınıfın hiç bulunamadığını gösterir."))
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(5)
            for c, h in enumerate(["FOTOĞRAF", "CİHAZ", "ÖLÇÜLEN", "DOĞRULUK",
                                   "MACRO-F1"]):
                grid.addWidget(_lab(h, color="#8a95a1", size=10.5, bold=800), 0, c,
                               QtCore.Qt.AlignLeft if c < 2 else QtCore.Qt.AlignRight)
            for r, p in enumerate(sorted(per, key=lambda x: x.get("acc") or 0),
                                  start=1):
                grid.addWidget(_lab(p["stem"], color="#1a2129", size=12.5, bold=700), r, 0)
                grid.addWidget(_lab(_INSTRUMENT.get(p.get("instrument"),
                                                    p.get("instrument", "")),
                                    color="#6b7580", size=12), r, 1)
                grid.addWidget(_lab(str(p.get("n_measured", 0)), color="#6b7580",
                                    size=12.5), r, 2, QtCore.Qt.AlignRight)
                grid.addWidget(_lab(_pct(p.get("acc")), color="#1a2129",
                                    size=12.5, bold=700), r, 3, QtCore.Qt.AlignRight)
                grid.addWidget(_lab(_pct(p.get("macro_f1")), color="#5a6472",
                                    size=12.5, bold=700), r, 4, QtCore.Qt.AlignRight)
            grid.setColumnStretch(0, 1)
            pcl.addLayout(grid)
            lay.addWidget(pcard)


class TrainReportWindow(QtWidgets.QDialog):
    """Every stored run, one tab each, newest first."""

    SCALE = 3          # export resolution multiplier

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model raporu")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.empty = _lab(EMPTY, color="#4a5560", size=13)
        self.empty.setWordWrap(True)
        self.empty.setAlignment(QtCore.Qt.AlignCenter)
        self.empty.setMargin(30)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.tabs)
        self.stack.addWidget(self.empty)

        self.status = _lab("", color="#6b7580", size=12)
        self.btn_save = QtWidgets.QPushButton("⤓   Save report…")
        self.btn_save.clicked.connect(self._save)
        btn_folder = QtWidgets.QPushButton("📁  Golden klasörü")
        btn_folder.setToolTip("Golden set klasörünü aç")
        btn_folder.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(golden_store.golden_dir())))
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(btn_folder)
        row.addWidget(self.status, 1)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        lay.addWidget(self.stack, 1)
        lay.addLayout(row)
        self.resize(1120, 860)
        self.refresh()

    def refresh(self, select_latest=True):
        """Rebuild the tabs from the stored history."""
        runs = model_eval.load_history()
        self.tabs.clear()
        for run in reversed(runs):                    # newest first
            page = ReportPage(run)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            # as-needed, not off: the tables have a real minimum width, and a
            # narrow window must let the user reach the right-hand columns
            # rather than silently clipping them
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            g = ((run.get("golden") or {}).get("combined") or {}).get("macro_f1")
            when = time.strftime("%d.%m %H:%M", time.localtime(run.get("ts") or 0))
            self.tabs.addTab(scroll, f"{when}   {_pct(g)}" if g else when)
        has = bool(runs)
        self.stack.setCurrentIndex(0 if has else 1)
        self.btn_save.setEnabled(has)
        if has and select_latest:
            self.tabs.setCurrentIndex(0)
        self.status.setText(f"{len(runs)} kayıtlı eğitim" if has else "")

    # ------------------------------------------------------------- exporting

    def _page(self):
        w = self.tabs.currentWidget()
        return w.widget() if isinstance(w, QtWidgets.QScrollArea) else w

    def _save(self):
        page = self._page()
        if page is None:
            return
        run = getattr(page, "run", {})
        name = "model-raporu-" + time.strftime(
            "%Y%m%d-%H%M", time.localtime(run.get("ts") or time.time())) + ".png"
        start = os.path.join(
            os.path.expanduser("~/Desktop") if os.path.isdir(
                os.path.expanduser("~/Desktop")) else os.path.expanduser("~"), name)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save report", start, "PNG image (*.png)")
        if not p:
            return
        try:
            self._render(page).save(p, "PNG")
            self.status.setText(f"Kaydedildi: {os.path.basename(p)}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def _render(self, page):
        """The page at SCALE x, whole — not just the part on screen.

        The widget is laid out at its own preferred size first, because inside a
        scroll area its current height is the viewport's, and rendering that
        would export exactly the crop the user can already see."""
        w = max(page.width(), 900)
        page.setFixedWidth(w)
        h = page.heightForWidth(w) if page.hasHeightForWidth() else page.sizeHint().height()
        h = max(h, page.sizeHint().height())
        page.resize(w, h)
        page.layout().activate()
        h = max(h, page.sizeHint().height())
        page.resize(w, h)
        pix = QtGui.QPixmap(w * self.SCALE, h * self.SCALE)
        pix.setDevicePixelRatio(self.SCALE)
        pix.fill(QtGui.QColor("#ffffff"))
        page.render(pix)
        page.setMinimumWidth(0)
        page.setMaximumWidth(16777215)
        return pix
