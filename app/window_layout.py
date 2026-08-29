"""Building the window: what goes where, once, at startup.

Separated from behaviour so the two do not have to be read together. Nothing
here decides anything — it creates widgets, sets their look, wires their
signals to handlers that live in the other window_* modules, and hands back
the finished column. If you are looking for what a button DOES, it is not in
this file.

Each _build_* returns the column it made (or, for the results side, leaves it
on self.right_stack) and MainWindow.__init__ assembles them.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import training_store

from chart_data import CLASS_ORDER, CLASS_LABELS, CLASS_COLORS
from ui_theme import _sep, _chip_qss
from widget_image_view import ImageView, ScaledImage
from widget_tiles import CheckMenu, Tile, AspectCard
from widget_library_tree import LibraryTree, SearchBox, SearchDelegate
from window_pattern_size import RangeSlider


def hdr(t):
    """A section caption (IMAGES, RESULTS, TRAINING)."""
    l = QtWidgets.QLabel(t); l.setObjectName("h"); return l


def ctl(t):
    """A label that names a control, in the quieter control colour."""
    l = QtWidgets.QLabel(t); l.setObjectName("ctl"); return l


class WindowLayout:
    """Mixin: builds the window's widgets. `self` is the window."""

    # What every sphericity tile explains about itself. The panel appends the
    # plain circularity behind the number to it (see _set_sph_tile).
    SPH_TIP = (
        "How round the particles are, averaged over them.\n"
        "1.00 is a perfect circle and 0 is an outline no rounder than a square "
        "or a ripple — the scale is stretched onto the band real particles "
        "occupy, so photos spread out instead of all reading 0.9-something. "
        "It re-scales the measurement; it does not change which photo is "
        "rounder than which.\n"
        "The small number beside it is the share of the measured particles the "
        "score rests on: a particle touching the edge of the photo is cut by "
        "the frame rather than by its own shape, so it is left out.")

    def _init_state(self):
        """Every piece of per-session state the window carries.

        Declared in one place so the stores that must stay in step are visible
        together — the four per-photo correction dictionaries especially, since
        a single class click can touch three of them and the undo snapshot has
        to copy all of them."""
        self.results = {}
        self.chosen = {}       # path -> [particle] whose size line is drawn (click-picked)
        self.size_range = None  # (lo, hi) nm for the size-fraction tool; None = off
        # Click tool in normal mode: "measure" (toggle a size line), a class name
        # (paint that class as a view-only correction), or None (clicks do nothing).
        self.click_tool = "measure"
        self.certainty = {}         # path -> {pid: (text, rgb)}  Certainty tool badges
        self.class_overrides = {}   # path -> {pid: class}  (view-only, not training data)
        self.view_excluded = {}     # path -> set(pid)  (view-only: dropped from the
        #                             classes/size stats, kept as an "excluded"
        #                             particle so Measure still shows it blue)
        # Particles the user re-admitted into the size statistics after the
        # occlusion gate (analyze.measurable) had greyed them out: measuring one
        # with the Measure tool IS the statement "this one's outline is whole".
        self.measure_include = {}   # path -> set(pid)
        # The mirror of measure_include: particles the app DID measure but the
        # user right-clicked to drop from the size statistics (a partly-hidden
        # particle the gate wrongly let through). They render light-blue like any
        # other unmeasured particle.
        self.measure_exclude = {}   # path -> set(pid)
        # ⌘Z history for the normal-mode corrections. Snapshots of the per-photo
        # edit state pushed BEFORE each change: a snapshot is a handful of small
        # sets/dicts, and restoring one is exact — a per-action inverse would not
        # be, since one class click can touch three stores at once.
        self._edit_undo = []        # [(path, snapshot)]
        # ⌘Z also reaches the library: one slot holding the last removal, with
        # the time it happened, so the newest action is the one that gets undone
        # whichever kind it was.
        self._lib_undo = None       # (when, [(folder_trail, index, node), …])
        self._undo_mark = 0.0       # when the newest per-photo step was pushed
        self.current = None
        self.worker = None
        self.queue = []
        self._result_target = None
        self._scale_warnings = []
        self._empty_warnings = []
        # ---- training-mode state ----
        self.train_mode = False
        self.review_mode = False      # training mode, but showing the model's
                                      # disagreements with the saved labels
        self.accept_mode = False      # ...and a click takes the model's answer
        self.train_labels = {}        # path -> {particle_id: class} (user clicks)
        self.train_show_overlay = True
        self._overlay_hidden = False   # normal-mode Space: hide pattern/class fills
        self.train_blank = set()      # paths where the model's pre-fill is hidden,
        #                               so the photo can be labelled neutrally
        # {path: {"seen": {pid}, "fixed": {pid}}} — how much of a photo has been
        # through the bulk review, and how much of that the user had to correct.
        # "fixed" is the number worth showing: it is the model's error rate on
        # the part of this photo a human has actually checked.
        self.review_stats = {}
        # {path: {pid: {"model_cls", "conf", "rank", "n_items"}}} — what the
        # model answered before the user overruled it, and where that particle
        # sat in the confidence ordering. Rides along into the label file so the
        # record survives a retrain; see ReviewDialog.__init__ for why it cannot
        # be reconstructed later.
        self.review_meta = {}
        self._train_undo = []         # [(path, pid, previous or None)]
        self._train_worker = None

    def _build_toolbar(self):
        """The top bar: Analyze, the view toggles, Save, Training."""
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
        # Two pairs of boxes — one under the histogram, one under the Solid vs
        # Undercooled chart — but ONE range: they mirror each other, so a range
        # typed on either page still means the same thing after switching tabs
        # (two independent ranges with the same label would just be a trap).
        self._range_sync = False

        def _numbox(ph, pair):
            e = QtWidgets.QLineEdit(); e.setObjectName("rangebox")
            e.setFixedWidth(58); e.setPlaceholderText(ph)
            e.setAlignment(QtCore.Qt.AlignCenter)
            e.setValidator(QtGui.QDoubleValidator(0.0, 1e7, 2))
            e.textChanged.connect(lambda _t, p=pair: self._range_changed(p))
            return e
        self.range_lo = _numbox("min", "main")
        self.range_hi = _numbox("max", "main")
        self.srange_lo = _numbox("min", "split")
        self.srange_hi = _numbox("max", "split")
        # coalesce fast keystrokes ("2"→"25"→"250") into a single re-render
        self._range_timer = QtCore.QTimer(self); self._range_timer.setSingleShot(True)
        self._range_timer.timeout.connect(self._rerender_chart)

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
        self.ex_cls["cumulative"] = _exp("Cumulative")
        self.ex_cls["solidsplit"] = _exp("Solid / Liquid")
        self._save_menu.addSeparator()
        _sect("CHART TITLE  ·  FILE NAME")
        self.ex_title = QtWidgets.QLineEdit()
        self.ex_title.setObjectName("titlebox")
        self.ex_title.setClearButtonEnabled(True)
        # the caption is also what the file is called, so the name the user
        # typed is the name they find on disk
        self.ex_title.setPlaceholderText("also names the saved file")
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

        # ---- top bar: Analyze | view toggles | Save. The toggles read as THREE
        # groups — Borders (the outline), then the two states a particle can be
        # in, then the four patterns a SOLID one can have — separated by
        # hairlines and by gaps wider than the ones inside a group, so the
        # grouping is carried by the spacing and the hairline only confirms it.
        # (A "Solid phases:" caption named the third group for a day and was
        # dropped, 2026-08-02: another word in a row of words is noise, and the
        # separator already says where the group starts.)
        def _vsep():
            w = _sep(); w.setFixedHeight(20)
            return w

        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(20); view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self.cb_outline)
        view_row.addSpacing(14); view_row.addWidget(_vsep()); view_row.addSpacing(14)
        view_row.addWidget(self.cb_under)
        view_row.addWidget(self.cb_solid)
        view_row.addSpacing(14); view_row.addWidget(_vsep()); view_row.addSpacing(14)
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

        # A short in-app manual of the controls a first-time user can't guess
        # (Space, right-click-to-drop, view-only vs. training…), opened beside
        # Training so it's the first thing a newcomer reaches for.
        self.guide_btn = QtWidgets.QPushButton("📖  Guide")
        self.guide_btn.setMinimumHeight(40)
        self.guide_btn.setToolTip("The controls that aren't obvious — worth a read")
        self.guide_btn.clicked.connect(self._open_guide)

        # Analyze keeps the width of its widest caption ("⏳  Analyzing…") so the
        # bar doesn't shift under the user the moment an analysis starts — and so
        # the centred block below stays put.
        _fm = self.analyze_btn.fontMetrics()
        self.analyze_btn.setFixedWidth(max(
            _fm.horizontalAdvance(t) for t in ("🔬  Analyze", "⏳  Analyzing…")) + 34)

        right_row = QtWidgets.QHBoxLayout()
        right_row.setSpacing(16); right_row.setContentsMargins(0, 0, 0, 0)
        right_row.addWidget(self.guide_btn)
        right_row.addWidget(self.train_btn)
        right_row.addWidget(self.save_btn)
        right_w = QtWidgets.QWidget(); right_w.setLayout(right_row)

        # The toggles are centred in the GAP between Analyze and the right-hand
        # buttons — not on the window (user rule, 2026-08-04: "tam ortada
        # olmasın, analyze ile guide'ı ortalasın"). Window-centring was tried:
        # it needs column 0 padded out to match the wider right flank (Guide +
        # Training + Save), which opens a dead hole after Analyze that grows
        # with the window. Splitting the leftover space evenly instead keeps
        # both margins tied to the window's actual spare width, so the row sits
        # visually between its two neighbours and neither side balloons.
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(16); tb.setContentsMargins(2, 4, 2, 4)
        tb.addWidget(self.analyze_btn)
        tb.addStretch(1)
        tb.addWidget(view_w)
        tb.addStretch(1)
        tb.addWidget(right_w)
        tbw = QtWidgets.QWidget(); tbw.setLayout(tb)

        def card(widget, margin=3):
            f = QtWidgets.QFrame(); f.setObjectName("card")
            lay = QtWidgets.QVBoxLayout(f)
            lay.setContentsMargins(margin, margin, margin, margin)
            lay.addWidget(widget)
            return f
        return tbw

    def _build_left_panel(self):
        """The IMAGES column: the tree, its search box and the buttons under it."""
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
        sort_btn = QtWidgets.QPushButton("✨  Sort"); sort_btn.clicked.connect(self._smart_sort)
        sort_btn.setToolTip("Smart sort — read the file names and file every image "
                            "under its sample number, its region (Alt / Karışık / "
                            "Üst) or its pattern. Shows the plan first.")
        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(6)
        for b in (imp_btn, newf_btn, sort_btn):
            b.setObjectName("sidebtn")
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Fixed)
            btn_row.addWidget(b)
        # ---- search: a box over the list, and a results view that REPLACES the
        # tree while you type (the panel is 230 px wide — a second list stacked
        # under the tree would leave neither of them usable)
        self.search = SearchBox()
        self.search.setObjectName("searchbox")
        self.search.setPlaceholderText("🔍   Search images")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        self.search.returnPressed.connect(self._search_reveal_current)
        self.search.escaped.connect(self._search_cancel)
        self.search.down.connect(lambda: self.results_list.setFocus())
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setObjectName("searchlist")
        self.results_list.setItemDelegate(SearchDelegate())
        self.results_list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.results_list.itemClicked.connect(self._search_preview)
        # double-click = "show me where this lives": jumps back to the tree with
        # the row revealed and selected (user rule, 2026-07-30)
        self.results_list.itemDoubleClicked.connect(self._search_reveal)
        self.list_stack = QtWidgets.QStackedWidget()
        self.list_stack.addWidget(self.tree)
        self.list_stack.addWidget(self.results_list)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(7); left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(hdr("IMAGES"))
        left.addWidget(self.search)
        left.addWidget(self.list_stack, 1)
        left.addLayout(btn_row)
        leftw = QtWidgets.QWidget(); leftw.setLayout(left)
        leftw.setMinimumWidth(160); leftw.setMaximumWidth(380)
        return leftw

    def _build_middle_panel(self):
        """The middle column: the micrograph, and the adjustment tools under it."""
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
        # The class buttons say "Change to …", not just the class name: a bare
        # "Janus" reads as a filter ("show me the janus ones"), which is what
        # the RESULTS chips above actually do. Spelling out the verb is the one
        # thing that tells a newcomer these REWRITE the particle they click.
        tools = [("measure", "📏  Measure", None, "M"),
                 ("certainty", "🎯  Certainty", None, "C"),
                 ("janus", "Change to Janus", CLASS_COLORS["janus"], "1"),
                 ("stripe", "Change to Stripe", CLASS_COLORS["stripe"], "2"),
                 ("composite", "Change to Composite", CLASS_COLORS["composite"], "3"),
                 ("lamellar", "Change to Lamellar", CLASS_COLORS["lamellar"], "4"),
                 ("undercooled", "Change to Undercooled", CLASS_COLORS["undercooled"], "5"),
                 ("solid", "Change to Solid", "#cf8481", "6"),
                 # not a class change — it takes the particle out of the counts
                 ("exclude", "Exclude particle", "#9aa4ae", "0")]
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
        return midw

    def _build_results_panel(self):
        """The right column: the RESULTS pages, and the TRAINING panel that
        swaps in for them. Sets self.right_stack."""
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
        self._res_split = {}        # solid / undercooled diameters (both or none)
        self._res_stats = None      # cached target.stats() (recomputing it per
        #                             keystroke made the readout stutter)
        self._chip_btns = {}
        # All + the five classes narrow the histogram to one class; "Pattern ×
        # Size" and "Cumulative" are overview tabs that swap the whole panel.
        # Solid / Liquid sits directly beside All: it is the other whole-selection
        # view of the same distribution, not a per-class filter like the six that
        # follow it.
        chips = ([("all", "All", None), ("solidsplit", "Solid / Liquid", None)]
                 + [(k, CLASS_LABELS[k], CLASS_COLORS[k]) for k in CLASS_ORDER]
                 + [("patternsize", "Pattern × Size", None),
                    ("cumulative", "Cumulative", None)])
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
        self._chip_btns["all"].setChecked(True)
        self._chip_rows = QtWidgets.QVBoxLayout()
        self._chip_rows.setSpacing(5); self._chip_rows.setContentsMargins(2, 2, 2, 0)
        self.chip_bar = QtWidgets.QWidget()
        self.chip_bar.setLayout(self._chip_rows)
        self.chip_bar.setStyleSheet("background:transparent;")
        self.chip_bar.setVisible(False)      # only once something is classified
        self._chip_order = [k for k, _l, _c in chips]
        self._chip_rowcount = 0
        # laid out for real once the panel has its width (see _layout_chips)
        QtCore.QTimer.singleShot(0, self._layout_chips)

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
        #   \u2022 All, classified -> mean \u00b7 measured \u00b7 classified, then one tile per
        #     class showing its share (coloured) and count
        #   \u2022 All, unclassified (ETD) -> total \u00b7 measured \u00b7 mean \u00b7 range
        self.tile_total = Tile("TOTAL PARTICLES", "\u2014")
        self.tile_measured = Tile("MEASURED", "\u2014")
        # How round the particles are, 1.00 = perfect circles (user request,
        # 2026-08-08). Shown in the All view (where CLASSIFIED used to be) and in
        # a class view (where RANGE used to be), so the same tile object serves
        # both layouts. The trailing note is the share of the measured particles
        # the score could be taken over \u2014 frame-touching ones are left out.
        self.tile_sph = Tile("SPHERICITY", "\u2014")
        self.tile_sph.setToolTip(self.SPH_TIP)
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
        # window) with a two-handle size range under it — pure QPainter, so
        # dragging is instant. Built lazily the first time it's shown.
        self._ps_plot = None
        self.ps_head = QtWidgets.QLabel("")
        self.ps_head.setStyleSheet("color:#1a2129;font-size:14px;font-weight:800;"
                                   "background:transparent;")
        self.ps_thlab = QtWidgets.QLabel("")
        self.ps_thlab.setStyleSheet("color:#1a2129;font-size:13px;font-weight:800;"
                                    "background:transparent;min-width:112px;")
        self.ps_slider = RangeSlider(400.0, 1500.0)
        self.ps_slider.valuesChanged.connect(self._ps_range_changed)
        ps_body = QtWidgets.QVBoxLayout()
        ps_body.setContentsMargins(0, 0, 0, 0); ps_body.setSpacing(8)
        ps_body.addWidget(self.ps_head)
        self._ps_holder = QtWidgets.QVBoxLayout()   # the plot drops in here
        self._ps_holder.setContentsMargins(0, 0, 0, 0)
        ps_body.addLayout(self._ps_holder, 1)
        ps_srow = QtWidgets.QHBoxLayout(); ps_srow.setSpacing(10)
        ps_srow.setContentsMargins(2, 0, 2, 2)
        ps_srow.addWidget(ctl("Size range"))
        ps_srow.addWidget(self.ps_thlab)
        ps_srow.addWidget(self.ps_slider, 1)
        ps_body.addLayout(ps_srow)
        self.ps_page = QtWidgets.QWidget(); self.ps_page.setLayout(ps_body)
        self.ps_page.setStyleSheet("background:transparent;")
        self.ps_page.setVisible(False)

        # ---- Cumulative page: the size distribution read the other way round —
        # "what share of the particles is at or below this diameter" — with the
        # D-values that go with it. Same shape as the normal page (chart on top,
        # tiles pinned underneath), so switching tabs doesn't move anything.
        self.cdf_view = ScaledImage("Analyze an image to see the distribution",
                                    on_resize=lambda: self._cdf_timer.start(120),
                                    radius=8)
        self.cdf_view.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.cdf_view.setMinimumHeight(150)
        self.cdf_view.setStyleSheet("background:transparent;color:#9aa4ae;")
        self._cdf_timer = QtCore.QTimer(self); self._cdf_timer.setSingleShot(True)
        self._cdf_timer.timeout.connect(self._render_cdf)
        self.cdf_tiles = {}
        cdf_grid = QtWidgets.QGridLayout()
        cdf_grid.setHorizontalSpacing(6); cdf_grid.setVerticalSpacing(6)
        cdf_grid.setContentsMargins(0, 0, 0, 0)
        for i, key in enumerate(("d10", "d50", "d90")):
            t = Tile(key.upper(), "—"); self.cdf_tiles[key] = t
            cdf_grid.addWidget(t, 0, i)
        for i, key in enumerate(("span", "measured")):
            t = Tile(key.upper(), "—"); self.cdf_tiles[key] = t
            cdf_grid.addWidget(t, 1, i, 1, 2 if i else 1)
        cdf_body = QtWidgets.QVBoxLayout()
        cdf_body.setContentsMargins(0, 0, 0, 0); cdf_body.setSpacing(8)
        cdf_body.addWidget(self.cdf_view, 1)
        cdf_body.addLayout(cdf_grid)
        self.cdf_page = QtWidgets.QWidget(); self.cdf_page.setLayout(cdf_body)
        self.cdf_page.setStyleSheet("background:transparent;")
        self.cdf_page.setVisible(False)

        # ---- Solid / Liquid page: the same size axis as the All chart,
        # but every bar split into the two states — crystalline at the bottom,
        # undercooled on top — with the solid share per bin drawn over it. The
        # size-range boxes here answer the one question the stack can't be read
        # off precisely: "of the particles between X and Y nm, how many are
        # solid?". Same shape as the other pages (chart on top, numbers pinned
        # underneath) so switching tabs doesn't move anything.
        self.split_view = ScaledImage("Analyze an image to compare the two states",
                                      on_resize=lambda: self._split_timer.start(120),
                                      radius=8)
        self.split_view.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.split_view.setMinimumHeight(150)
        self.split_view.setStyleSheet("background:transparent;color:#9aa4ae;")
        self._split_timer = QtCore.QTimer(self); self._split_timer.setSingleShot(True)
        self._split_timer.timeout.connect(self._render_split)

        srng_row = QtWidgets.QHBoxLayout(); srng_row.setSpacing(7)
        srng_row.setContentsMargins(2, 0, 2, 0)
        srng_row.addWidget(ctl("Size range:"))
        srng_row.addWidget(self.srange_lo)
        srng_row.addWidget(ctl("–"))
        srng_row.addWidget(self.srange_hi)
        srng_row.addWidget(ctl("nm"))
        self.split_pct = QtWidgets.QLabel("")
        self.split_pct.setStyleSheet("color:#1a2129;font-size:15px;font-weight:800;"
                                     "background:transparent;")
        self.split_cnt = QtWidgets.QLabel("")
        self.split_cnt.setStyleSheet("color:#6b7580;font-size:12.5px;font-weight:700;"
                                     "background:transparent;")
        srng_row.addStretch(1)
        srng_row.addWidget(self.split_pct)
        srng_row.addWidget(self.split_cnt)

        self.split_tiles = {}
        split_grid = QtWidgets.QGridLayout()
        split_grid.setHorizontalSpacing(6); split_grid.setVerticalSpacing(6)
        split_grid.setContentsMargins(0, 0, 0, 0)
        # Two columns, not three: "MEAN UNDERCOOLED" and "MOSTLY SOLID ABOVE"
        # are long captions and a third of the panel elides them. The pairs also
        # line up the way they are read — solid on the left, undercooled on the
        # right. Captions mirror chart_data.split_tiles(), which draws the same block
        # into an exported figure.
        # The last pair was MOSTLY SOLID ABOVE + CLASSIFIED until 2026-08-08, when
        # the user asked for the sphericity of each state instead — which also
        # makes every row of this block a solid/undercooled pair.
        _split_caps = [("solid", "SOLID"), ("undercooled", "UNDERCOOLED"),
                       ("meansolid", "MEAN SOLID"), ("meanunder", "MEAN UNDERCOOLED"),
                       ("sphsolid", "SPHERICITY SOLID"),
                       ("sphunder", "SPHERICITY UNDERCOOLED")]
        for i, (key, cap) in enumerate(_split_caps):
            t = Tile(cap, "—"); self.split_tiles[key] = t
            if key.startswith("sph"):
                t.setToolTip(self.tile_sph.toolTip())
            split_grid.addWidget(t, i // 2, i % 2)
        for c in range(2):
            split_grid.setColumnStretch(c, 1)
        split_body = QtWidgets.QVBoxLayout()
        split_body.setContentsMargins(0, 0, 0, 0); split_body.setSpacing(8)
        split_body.addWidget(self.split_view, 1)
        split_body.addLayout(srng_row)
        split_body.addLayout(split_grid)
        self.split_page = QtWidgets.QWidget(); self.split_page.setLayout(split_body)
        self.split_page.setStyleSheet("background:transparent;")
        self.split_page.setVisible(False)

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
        res_lay.addWidget(self.cdf_page, 1)
        res_lay.addWidget(self.split_page, 1)

        right = QtWidgets.QVBoxLayout(); right.setSpacing(8); right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(hdr("RESULTS"))
        right.addWidget(res_card, 1)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right)

        # ---- training page (swaps in for RESULTS while Training mode is on) ----
        # Labelling uses the SAME "Click a particle to" tools in the Adjustments
        # card (Janus/Stripe/…/Exclude, keys 1–5/0) — no separate class picker
        # here. In training mode a click writes a training label; the View
        # checkboxes above filter which labelled classes are shown.
        # (no standing hint line here: what the card shows is worth the height,
        #  and Space / the click tools are covered in the Guide)

        # Where THIS photo stands in the training set, spelled out — the dot in
        # the IMAGES panel says it at a glance for every photo, this says it in
        # words (and carries the date) for the one being looked at.
        self.tr_state = QtWidgets.QLabel()
        self.tr_state.setWordWrap(True)
        self.tr_state.setTextFormat(QtCore.Qt.RichText)
        self.tr_state.setStyleSheet("background:transparent;")

        # Only ever offered on a photo that is in the set AND has drifted from
        # it: puts back exactly what was saved, for the mis-click case.
        self.tr_restore_btn = QtWidgets.QPushButton("↩  Restore saved")
        self.tr_restore_btn.setObjectName("trainmini")
        self.tr_restore_btn.setToolTip(
            "Go back to the labels this photo was added to the training set with")
        self.tr_restore_btn.clicked.connect(self._train_restore)
        self.tr_restore_btn.setVisible(False)

        # The review switch: instead of the labels, paint the particles where the
        # model's own answer differs from the ground truth this photo was added
        # with. Answers "what does the model still get wrong here?".
        self.tr_review_cb = QtWidgets.QCheckBox("🔍   Show where the model disagrees")
        self.tr_review_cb.setToolTip(
            "Compare the model's current answer against the labels this photo "
            "was added to the training set with, and paint only the differences")
        self.tr_review_cb.toggled.connect(self._toggle_review)

        # In review, most of the model's "mistakes" turn out to be right. Fixing
        # those by hand means picking the matching class tool for each one; this
        # turns a click into "you're right, take yours" instead.
        self.tr_accept_cb = QtWidgets.QCheckBox("✓   Click a particle to accept the model's answer")
        self.tr_accept_cb.setToolTip(
            "While this is on, clicking a particle in the review adopts the "
            "model's class as your label — whatever class tool is selected. "
            "Add the photo to the training set again when you're done.")
        self.tr_accept_cb.setVisible(False)
        self.tr_accept_cb.toggled.connect(self._toggle_accept)

        self.tr_info = QtWidgets.QLabel()
        self.tr_info.setWordWrap(True)
        self.tr_info.setTextFormat(QtCore.Qt.RichText)
        self.tr_info.setStyleSheet("font-size:12px;color:#2c3442;background:transparent;")
        self.tr_info.setAlignment(QtCore.Qt.AlignTop)
        # A long confusion table has to be reachable. The scrollbar itself stays
        # hidden (the panel hides them everywhere), but the area still takes the
        # wheel / two-finger scroll, which is how it is meant to be read.
        self.tr_scroll = QtWidgets.QScrollArea()
        self.tr_scroll.setWidget(self.tr_info)
        self.tr_scroll.setWidgetResizable(True)
        self.tr_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tr_scroll.setStyleSheet("background:transparent;")
        self.tr_scroll.viewport().setStyleSheet("background:transparent;")
        self.tr_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tr_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tr_scroll.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                     QtWidgets.QSizePolicy.Expanding)

        # Clear: a menu — clear all my labels (blank slate) or one class at a time
        self.tr_clear_btn = QtWidgets.QPushButton("🧹  Clear")
        self.tr_clear_btn.setObjectName("trainmini")
        self.tr_clear_btn.setToolTip(
            "Clear labels — one class at a time, or all of them")
        self.tr_clear_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._clear_menu = QtWidgets.QMenu(self)
        self._clear_menu.aboutToShow.connect(self._build_clear_menu)
        self.tr_clear_btn.setMenu(self._clear_menu)

        self.tr_confirm_btn = QtWidgets.QPushButton("✓   Add photo to training set")
        self.tr_confirm_btn.setObjectName("primary")
        self.tr_confirm_btn.setMinimumHeight(32)
        self.tr_confirm_btn.clicked.connect(self._train_confirm)
        # The bulk route in: check a whole class at a glance instead of clicking
        # a thousand particles one at a time on the micrograph.
        self.tr_review_btn = QtWidgets.QPushButton("▦   Review particles")
        self.tr_review_btn.setObjectName("trainmini")
        self.tr_review_btn.setToolTip(
            "Show every particle the model gave one class as a grid of crops, "
            "least confident first. Confirm a page at a glance; click only the "
            "wrong ones. Uses the photos selected in IMAGES.")
        self.tr_review_btn.clicked.connect(self._open_review)
        self.tr_train_btn = QtWidgets.QPushButton("🧠  Train model")
        self.tr_train_btn.setObjectName("trainmini")
        self.tr_train_btn.clicked.connect(self._train_go)
        # undoes an accidental click on Train model: hidden until a run is
        # actually in progress, so it never sits next to the button as a
        # second thing to misclick
        self.tr_cancel_btn = QtWidgets.QPushButton("✕")
        self.tr_cancel_btn.setObjectName("trainmini")
        self.tr_cancel_btn.setToolTip("Cancel training")
        self.tr_cancel_btn.setFixedWidth(30)
        self.tr_cancel_btn.setVisible(False)
        self.tr_cancel_btn.clicked.connect(self._train_cancel)
        self.tr_prog = QtWidgets.QProgressBar(); self.tr_prog.setVisible(False)
        self.tr_metrics = ctl(""); self.tr_metrics.setWordWrap(True)
        folder_btn = QtWidgets.QPushButton("📁")
        folder_btn.setObjectName("trainmini")
        folder_btn.setToolTip("Open the training folder")
        folder_btn.setFixedWidth(38)
        folder_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(training_store.train_dir())))

        # Pattern × Size while labelling: the RESULTS panel (with its chart tabs)
        # is swapped out for this card in training mode, so there was no way to
        # read a composition off the labels you are making without leaving the
        # mode. It opens in its own window and reads the USER's labels here —
        # ground truth, not the model's answer (see _open_classsize).
        # Training labels are ground truth, and the user analyses from normal
        # mode — so their labels belong there too. One way only: normal-mode
        # edits are transient view corrections and must never become training
        # data. The checkbox does it on confirm; the button is the catch-up for
        # a photo labelled before this existed, or not confirmed yet.
        self.tr_reflect_btn = QtWidgets.QPushButton("⇩   Apply labels to normal mode")
        self.tr_reflect_btn.setObjectName("trainmini")
        self.tr_reflect_btn.setToolTip(
            "Copy THIS photo's training labels onto its normal-mode view, so an "
            "analysis run from the normal panel uses your corrections.\n"
            "Never the other way round: normal-mode edits stay out of training.")
        self.tr_reflect_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tr_reflect_btn.clicked.connect(self._reflect_training_now)
        self.tr_reflect_cb = QtWidgets.QCheckBox(
            "Apply labels to normal mode when I add a photo")
        self.tr_reflect_cb.setChecked(True)
        self.tr_reflect_cb.setToolTip(
            "On by default: a photo you have taught should read the same way in "
            "the panel you analyse from.")

        self.tr_ps_btn = QtWidgets.QPushButton("📊  Pattern × Size")
        self.tr_ps_btn.setObjectName("trainmini")
        self.tr_ps_btn.setToolTip(
            "Open the Pattern × Size chart for YOUR labels on this photo")
        self.tr_ps_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tr_ps_btn.clicked.connect(self._open_classsize)

        # "Is more labelling still worth it?" — the one question the training
        # panel could never answer. Opens the last measured learning curve; the
        # checkbox next to Train decides whether the next run measures a fresh
        # one (it costs real minutes, so it is a choice, not a default tax).
        self.tr_sat_btn = QtWidgets.QPushButton("📈  Data saturation")
        self.tr_sat_btn.setObjectName("trainmini")
        self.tr_sat_btn.setToolTip(
            "Accuracy against how much data it was trained on — whether more "
            "labelled photos are still buying accuracy, or the curve has "
            "flattened and the model itself is now the limit")
        self.tr_sat_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tr_sat_btn.clicked.connect(self._open_saturation)
        # Every training run's score, kept and comparable. There is no checkbox
        # beside it: scoring on the golden set is one forward pass over a handful
        # of photos, so there is nothing to opt out of. The cross-validation this
        # replaced cost three quarters of the run, which is why it HAD a switch.
        self.tr_report_btn = QtWidgets.QPushButton("📋  Model raporu")
        self.tr_report_btn.setObjectName("trainmini")
        self.tr_report_btn.setToolTip(
            "Her eğitimin golden set skoru: sınıf sınıf recall / precision / F1, "
            "karışıklık matrisi, mikroskop başına ayrım — ve geçmiş eğitimler "
            "sekme sekme, kıyaslayabilmen için")
        self.tr_report_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tr_report_btn.clicked.connect(self._open_train_report)
        self.tr_sat_cb = QtWidgets.QCheckBox("Measure data saturation while training")
        # on by default: the question it answers ("keep labelling, or is the
        # model the limit now?") is worth a third of a training run every time,
        # and a run that skips it leaves no record of where the data stood
        self.tr_sat_cb.setChecked(True)
        self.tr_sat_cb.setToolTip(
            "Trains several extra models on 20 – 100 % of your photos to plot "
            "the learning curve. Adds roughly a third to the training time.")

        # two quiet rows for the secondary actions, instead of full-width buttons
        # stacked down the card
        # Clear and the training-folder button are BUILT but not placed (user,
        # 2026-08-04: "hiç kullanmıyorum zaten"). They stay constructed because
        # the Clear menu is wired to _build_clear_menu and the restore/state
        # code reads these widgets; dropping them from the layout is the whole
        # of the change, and re-adding one is a single addWidget.
        tr_row = QtWidgets.QHBoxLayout(); tr_row.setSpacing(6)
        tr_row.addWidget(self.tr_restore_btn, 2)
        tr_row.addWidget(self.tr_train_btn, 1)
        tr_row.addWidget(self.tr_cancel_btn)
        tr_row2 = QtWidgets.QHBoxLayout(); tr_row2.setSpacing(6)
        tr_row2.addWidget(self.tr_ps_btn, 1)
        tr_row2.addWidget(self.tr_review_btn, 1)
        tr_row3 = QtWidgets.QHBoxLayout(); tr_row3.setSpacing(6)
        tr_row3.addWidget(self.tr_report_btn, 1)
        tr_row3.addWidget(self.tr_sat_btn, 1)
        tr_row4 = QtWidgets.QHBoxLayout(); tr_row4.setSpacing(6)
        tr_row4.addWidget(self.tr_reflect_btn, 1)

        tr_body = QtWidgets.QVBoxLayout()
        tr_body.setContentsMargins(14, 14, 14, 14); tr_body.setSpacing(9)
        tr_body.addWidget(self.tr_state)
        tr_body.addWidget(self.tr_scroll, 1)
        tr_body.addWidget(self.tr_review_cb)
        tr_body.addWidget(self.tr_accept_cb)
        tr_body.addLayout(tr_row)
        tr_body.addLayout(tr_row2)
        tr_body.addLayout(tr_row3)
        tr_body.addLayout(tr_row4)
        tr_body.addWidget(self.tr_reflect_cb)
        tr_body.addWidget(self.tr_sat_cb)
        tr_body.addWidget(self.tr_confirm_btn)
        tr_body.addWidget(self.tr_prog)
        tr_body.addWidget(self.tr_metrics)
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

    def _wire_shortcuts(self):
        """Single-key and command shortcuts. They work in both modes; what a
        key MEANS is decided by the handler, not by the binding."""
        # image clicks label particles (only acted on while Training mode is on)
        self.view.clicked.connect(self._on_view_click)
        self.view.right_clicked.connect(self._on_view_right_click)
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
        # ⌘0 puts the image back to fit. Double-click used to do this and was
        # taken away (labelling clicks the same particle twice all the time), but
        # nothing replaced it — so a zoom you didn't mean had no way back and the
        # app had to be restarted. The zoom itself is now bounded (see
        # ImageView.MIN_ZOOM/MAX_ZOOM); this is the deliberate way out.
        sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        sc.activated.connect(self._fit_image)
        # ⌘Z steps back through the corrections of whichever mode is active:
        # training labels, or the normal-mode class/measure edits.
        self._train_shortcuts = []
        sc = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        sc.activated.connect(self._undo)
