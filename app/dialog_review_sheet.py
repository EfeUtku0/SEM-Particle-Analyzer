"""The contact sheet: a page of particle crops the user confirms at a glance.

This is the cheap half of teaching the model. Labelling on the micrograph costs
one click per particle and a dense photo holds a thousand; confirming forty
crops laid out in a grid costs one look plus a click on the few that are wrong.
The ordering (least-confident first) is what makes that work — see review_queue.py.
"""
from __future__ import annotations

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

import review_queue
from analyze import DEFAULT_FACET_THRESH
from chart_data import CLASS_LABELS
from widget_tiles import CropTile


class ReviewDialog(QtWidgets.QDialog):
    """Confirm a whole page of the model's calls at a glance; correct the few
    that are wrong.

    The point is throughput without giving up ground truth — see review_queue.py for
    why a reviewed label is worth the same as a hand-placed one and a
    model-prefilled label is not. Everything shown on a page the user accepts is
    written as an explicit USER label, and the photos touched are switched to
    "labels only" (train_blank) so nothing the user never looked at can ride
    along into the training set.
    """

    # 4 rows of 8 fits a page on screen without scrolling, which is the whole
    # premise: a page you have to scroll is not taken in "at a glance", and the
    # bottom row ends up confirmed unseen.
    PER_PAGE = 32
    COLS = 8

    GRID_GAP = 8            # spacing between crops, both axes
    CAP_H = 26              # the class caption strip under each crop
    TILE_PAD = 8            # CropTile's own border/padding around the pixmap
    # Head + sub paragraph + button row + margins. Measured on the built
    # dialog rather than guessed would be nicer, but the tile size has to be
    # known BEFORE the crops are scaled, so this is a deliberate over-estimate:
    # too large only costs a few pixels of tile, too small brings the scrollbar
    # back, and the scrollbar is the bug.
    CHROME_H = 250
    CHROME_W = 32           # the dialog's own left + right contents margins
    SCROLLBAR_W = 18        # the scroll area's vertical bar, if it ever shows
    TILE_MIN, TILE_MAX = 64, 132

    def __init__(self, parent, analyses, cls, tile=None, include_done=False):
        super().__init__(parent)
        self.setWindowTitle("Review")
        self.analyses = analyses
        self.cls = cls
        self.main = parent
        self.include_done = include_done
        labelled = {p: dict(v) for p, v in (parent.train_labels or {}).items()}
        # Normally a particle you have already answered for is out of the queue
        # — the point is to get through the ones nobody has looked at. Asked for
        # again (include_done), the whole class comes back so an earlier answer
        # can be changed.
        skip = None if include_done else {p: set(v) for p, v in labelled.items()}
        self.items = review_queue.collect(analyses, cls, DEFAULT_FACET_THRESH, skip)
        if include_done:
            # each tile opens on YOUR last answer, not the model's — otherwise a
            # re-review silently proposes reverting every correction you made,
            # and one absent-minded "Accept page" would do exactly that.
            for it in self.items:
                mine = labelled.get(it["path"], {}).get(it["pid"])
                it["model_cls"] = it["cls"]          # what the model still says
                if mine:
                    it["cls"] = mine
                    it["was_mine"] = True
        # Position in the least-confident-first ordering, stamped once. This is
        # the only place it exists: `conf` itself is recomputed from the model's
        # probabilities, so it changes the moment the model is retrained and the
        # photo re-analysed. Corrections still arriving near the END of the list
        # — where the model was most sure — is the difference between "the model
        # is imprecise" and "the model is broken on this instrument", and it is
        # unrecoverable if not written down now.
        for i, it in enumerate(self.items):
            it["rank"] = i
        self.page = 0
        self.accepted = {}          # path -> {pid: verdict dict}
        self._tiles = []

        # Tile size fit to the SCREEN, not a fixed 132px (user report,
        # 2026-08-04: the dialog still didn't fit a MacBook screen). 4 rows at
        # 132px need ~250px of chrome plus ~640px of grid — comfortably over a
        # 1366x768-class laptop's usable height. Shrinking the tile is the only
        # lever that keeps "a whole page at a glance" true on a small screen;
        # scrolling would defeat the dialog's entire premise (see class docstring).
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        rows = -(-self.PER_PAGE // self.COLS)      # ceil
        grid_w = avail.width() - 60 - self.CHROME_W - self.SCROLLBAR_W
        grid_h = avail.height() - 60 - self.CHROME_H
        row_h = max(1, (grid_h - (rows - 1) * self.GRID_GAP) / rows)
        col_w = max(1, (grid_w - (self.COLS - 1) * self.GRID_GAP) / self.COLS)
        tile = tile or int(min(row_h - self.CAP_H, col_w) - self.TILE_PAD)
        tile = max(self.TILE_MIN, min(self.TILE_MAX, tile))
        self.tile = tile

        self.head = QtWidgets.QLabel(); self.head.setObjectName("h")
        self.sub = QtWidgets.QLabel()
        # Wrap, and don't let this paragraph have a say in how wide the dialog
        # is. Without both, its natural single-line width (~1600px) becomes the
        # layout's minimum and the window opens wider than a laptop screen —
        # which is what "the review page still doesn't fit" actually was, not
        # the grid (user report, 2026-08-04).
        self.sub.setWordWrap(True)
        self.sub.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                               QtWidgets.QSizePolicy.Minimum)
        self.sub.setStyleSheet("color:#6b7683;font-size:12px;background:transparent;")
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(8)
        holder = QtWidgets.QWidget(); holder.setLayout(self.grid)
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(holder)
        scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # The grid is sized to fit the width by construction (see the tile
        # calculation above), so a horizontal bar can only mean the arithmetic
        # slipped — and it would hide the last column, which is worse than the
        # crops being a pixel narrower.
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.btn_prev = QtWidgets.QPushButton("◀  Back")
        self.btn_all = QtWidgets.QPushButton("Select page")
        self.btn_all.setToolTip("⌘A — select the crops on THIS page only, then "
                                "one class key re-labels them")
        self.lbl_sel = QtWidgets.QLabel()
        self.lbl_sel.setStyleSheet("color:#2b6fff;font-size:12px;font-weight:700;"
                                   "background:transparent;")
        self.btn_accept = QtWidgets.QPushButton("✓  Accept page")
        self.btn_accept.setObjectName("primary")
        self.btn_accept.setMinimumHeight(38)
        self.btn_done = QtWidgets.QPushButton("Finish")
        self.btn_prev.clicked.connect(lambda: self._go(-1))
        self.btn_all.clicked.connect(self._select_all)
        self.btn_accept.clicked.connect(self._accept)
        self.btn_done.clicked.connect(self.accept)
        QtGui.QShortcut(QtGui.QKeySequence.SelectAll, self, self._select_all)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_prev); row.addWidget(self.btn_all)
        row.addWidget(self.lbl_sel); row.addStretch(1)
        row.addWidget(self.btn_accept); row.addStretch(1); row.addWidget(self.btn_done)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12); lay.setSpacing(8)
        lay.addWidget(self.head); lay.addWidget(self.sub)
        lay.addWidget(scroll, 1); lay.addLayout(row)
        # size to fit the actual screen, centred — a fixed 1500x900 can run off
        # the edges on a smaller display or when the dock/menu bar eats into it.
        # Height follows the tile size computed above, not a flat 900: on a
        # screen too short even for the shrunk minimum tile, this is what stops
        # the grid itself from needing a scrollbar (the CHROME_H budget above
        # already assumed no vertical squeeze — matching that here keeps the
        # two in agreement instead of guessing twice).
        cell_w = tile + self.TILE_PAD
        cell_h = tile + self.CAP_H
        w = min(self.COLS * cell_w + (self.COLS - 1) * self.GRID_GAP
                + self.CHROME_W + self.SCROLLBAR_W,
                avail.width() - 60)
        h = min(rows * cell_h + (rows - 1) * self.GRID_GAP + self.CHROME_H,
                avail.height() - 60)
        self.resize(w, h)
        self.move(avail.center() - self.rect().center())
        self._build()

    # ---- pages ----
    def _slice(self):
        i = self.page * self.PER_PAGE
        return self.items[i:i + self.PER_PAGE]

    def _build(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)
        self._tiles = []
        # a page is a fresh sheet: the shift anchor cannot point into the last one
        self._anchor = None
        self._range = []
        cur = self._slice()
        for n, it in enumerate(cur):
            a = self.analyses[it["path"]]
            arr = review_queue.crop_rgb(a, it["pid"])
            if arr is None:
                continue
            img = Image.fromarray(arr).resize((self.tile, self.tile), Image.LANCZOS)
            qim = QtGui.QImage(img.tobytes(), img.width, img.height,
                               img.width * 3, QtGui.QImage.Format_RGB888)
            it.setdefault("model_cls", it["cls"])
            t = CropTile(it, QtGui.QPixmap.fromImage(qim))
            t.clicked.connect(self._select)
            self.grid.addWidget(t, n // self.COLS, n % self.COLS)
            self._tiles.append(t)
        npages = max(1, (len(self.items) + self.PER_PAGE - 1) // self.PER_PAGE)
        done = sum(len(v) for v in self.accepted.values())
        self.head.setText(
            f"REVIEW · {CLASS_LABELS.get(self.cls, self.cls.title()).upper()}  ·  "
            f"PAGE {self.page + 1} / {npages}")
        mine = sum(1 for it in self.items if it.get("was_mine"))
        again = (f"  <b>{mine} of them already carry your answer</b> and open on "
                 f"it, not on the model's." if mine else "")
        self.sub.setText(
            f"{len(self.items)} particles the model calls "
            f"“{CLASS_LABELS.get(self.cls, self.cls)}”, least confident first — "
            f"so the mistakes come first and you can stop when the corrections "
            f"dry up.{again}  Click a wrong one — or click one and shift-click "
            f"another to take every particle between them — then press its class key "
            f"(1 Janus · 2 Stripe · 3 Composite · 4 Lamellar · 5 Undercooled · "
            f"0 Exclude).  Confirmed so far: {done}.")
        self.btn_prev.setEnabled(self.page > 0)
        self.btn_accept.setEnabled(bool(self._tiles))
        self._sel_line()

    def _go(self, d):
        self.page = max(0, self.page + d)
        self._build()

    def _select(self, tile, mods=0):
        """Toggle, leaving the rest alone — selection is MULTI on purpose.

        When a model is wrong on a new instrument it is usually wrong the same
        way for most of a page (on METU-METE it calls whole pages of plain
        spheres "janus"). Single-select would make the common case one click per
        particle, which is the cost this dialog exists to remove; toggling lets
        the user sweep the wrong ones and fix them all with one key.

        SHIFT extends: click one, shift-click another, and everything between
        them in reading order comes along. The queue is sorted least-confident
        first, so a run of wrong particles IS a contiguous stretch — that is the
        shape the sheet is read in, and clicking thirty of them one at a time
        was the remaining cost.
        """
        try:
            idx = self._tiles.index(tile)
        except ValueError:
            return
        shift = bool(mods & int(QtCore.Qt.ShiftModifier.value))
        if shift and self._anchor is not None and self._anchor < len(self._tiles):
            # a second shift-click re-draws the range from the SAME anchor
            # instead of piling ranges up, so it can shrink as well as grow —
            # and it only ever gives back tiles this range itself took
            for i in self._range:
                if i != self._anchor:
                    self._tiles[i].selected = False
                    self._tiles[i].refresh()
            lo, hi = sorted((self._anchor, idx))
            self._range = list(range(lo, hi + 1))
            for i in self._range:
                self._tiles[i].selected = True
                self._tiles[i].refresh()
        else:
            tile.selected = not tile.selected
            tile.refresh()
            # the anchor is where the NEXT shift-click measures from; a click
            # that clears a tile is still a place to start a range from
            self._anchor = idx
            self._range = []
        self._sel_line()

    def _select_all(self):
        """Select (or clear) THIS PAGE only — never the whole queue.

        `self._tiles` is rebuilt per page, so the scope is structural rather
        than a filter that could drift. It is spelled out because ⌘A reads as
        "everything" and a user who believed it took the whole queue would
        think one class key had just relabelled thousands of particles.
        """
        want = not all(t.selected for t in self._tiles)
        for t in self._tiles:
            t.selected = want
            t.refresh()
        # a page-wide sweep is not a range: leave nothing for a later shift-click
        # to "give back"
        self._anchor = None
        self._range = []
        self._sel_line()

    def _sel_line(self):
        n = sum(1 for t in self._tiles if t.selected)
        self.btn_all.setText("Clear page" if n == len(self._tiles) and n
                             else f"Select page ({len(self._tiles)})")
        self.lbl_sel.setText(f"{n} of {len(self._tiles)} on this page"
                             if n else "")

    def keyPressEvent(self, e):
        k = e.text()
        cls = {"1": "janus", "2": "stripe", "3": "composite", "4": "lamellar",
               "5": "undercooled", "0": "exclude"}.get(k)
        if cls:
            for t in self._tiles:
                if t.selected:
                    t.item["cls"] = cls
                    t.selected = False
                    t.refresh()
            self._sel_line()
            return
        if e.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Space,
                      QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            # Return/Enter would otherwise reach QDialog's own handling, which
            # fires whatever button is marked "default" — closing the sheet
            # instead of accepting the page (user report, 2026-08-04).
            self._accept(); return
        if e.key() == QtCore.Qt.Key_Left:
            self._go(-1); return
        super().keyPressEvent(e)

    def _accept(self):
        """Bank this page as user labels and move on."""
        for t in self._tiles:
            it = t.item
            self.accepted.setdefault(it["path"], {})[it["pid"]] = dict(
                cls=it["cls"], model_cls=it["model_cls"],
                conf=round(float(it["conf"]), 4), rank=int(it["rank"]),
                n_items=len(self.items))
        self.main._apply_review(self.accepted)
        if (self.page + 1) * self.PER_PAGE >= len(self.items):
            self.accept(); return
        self.page += 1
        self._build()

    def done(self, r):
        """Persist once, on the way out, and say plainly what was written —
        the counts live in the training store, which nothing on the main
        RESULTS panel displays, so without this the work leaves no trace the
        user can see."""
        n, changed = self.main._apply_review(self.accepted, save=True)
        self._written = (n, changed, len(self.accepted))
        super().done(r)
