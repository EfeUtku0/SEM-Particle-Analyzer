"""The IMAGES panel: the tree itself, and the hand-drawn rows it is made of.

Rows are painted by LibraryDelegate rather than styled with QSS, and that is
not a preference — the indent rules, the selection pill, the chevron and the
status dot cannot be expressed as ::item rules, and mixing the two makes them
fight. See the module's QSS block in ui_theme for the matching warning.

The per-row state lives in Qt item roles defined here (ROLE_*), so the window
and the delegate cannot disagree about what a row is claiming.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from image_files import IMG_EXT

ROLE_PATH = QtCore.Qt.UserRole        # image item -> file path (never mutated)
ROLE_KIND = QtCore.Qt.UserRole + 1    # node kind -> "folder" or "image"
ROLE_MISSING = QtCore.Qt.UserRole + 2     # image whose file is gone from disk
# How fresh this row's analysis is (see MainWindow._refresh_analysis_state):
#   0 not analysed · 1 analysed with the current rules+model · 2 analysed earlier
ROLE_FRESH = QtCore.Qt.UserRole + 3
FRESH_NONE, FRESH_CURRENT, FRESH_STALE = 0, 1, 2
# Where this row stands in the TRAINING SET (shown instead of the freshness dot
# while Training mode is on — see MainWindow._train_state). The whole point is
# that "have I taught the app this photo yet?" is answerable at a glance, which
# the freshness dot never said anything about:
#   0 nothing to say · 1 not in the training set yet · 2 in it, as saved
#   3 in it, but edited since (the saved version and the screen differ)
ROLE_TRAIN = QtCore.Qt.UserRole + 4
TR_NONE, TR_NEW, TR_SAVED, TR_EDITED = 0, 1, 2, 3


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

    # Freshness dot on the right of the row: analysed under the current rules and
    # model, or under older ones. Deliberately unbalanced in weight — a row that
    # is up to date should read as calm (a pale green that the eye skips over),
    # while the ones worth re-running are the only thing that catches it. A row
    # that was never analysed carries nothing at all.
    DOT_R = 3.4        # radius, px
    DOT_PAD = 10       # centre distance from the row's right edge
    DOT_ROOM = 16      # width the text gives up when a dot is present
    DOT_CURRENT = "#a9d7bd"
    DOT_STALE = "#e0a24d"

    # In Training mode the same lane carries the TRAINING state instead, so the
    # panel answers the question that mode is about ("did I already teach it
    # this photo?") rather than the analysis question, which is meaningless
    # while labelling. These read louder than the freshness pair on purpose —
    # here the mark IS the information, not a background hint.
    TDOT = {TR_NEW: "#e8963c",        # not in the training set yet
            TR_SAVED: "#3fae74",      # in it, exactly as it is on screen
            TR_EDITED: "#2b6fff"}     # in it, but changed since — blue is this
                                      # app's colour for "your edit" everywhere
    TDOT_R = 4.2                      # a touch larger than the freshness dot

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
            #
            # It must NEVER go past r.right(). Painting outside the item's own
            # rectangle is what produced the stray blue blocks: on a selection
            # change Qt repaints only the rects of the rows involved, so anything
            # a delegate drew beyond them is left on screen as a smear. The row
            # rect is kept as wide as the panel by the Stretch resize mode set in
            # LibraryTree, so clamping here costs nothing and closes the hole for
            # good.
            right = min(r.right(), self.tree.viewport().width())
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
        fresh = item.data(0, ROLE_FRESH) or FRESH_NONE
        if self.tree.train_marks:
            fresh = item.data(0, ROLE_TRAIN) or TR_NONE
        painter.setPen(QtGui.QColor(fg))
        tx = r.left() + self._text_left(item)
        # the dot's lane is taken out of the text's width, so a long name elides
        # into "…" before it can run underneath the dot
        tw = max(0, r.right() - tx - 4 - (self.DOT_ROOM if fresh else 0))
        fm = QtGui.QFontMetrics(painter.font())
        painter.drawText(QtCore.QRect(tx, r.top(), tw, r.height()),
                         QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                         fm.elidedText(self.display_name(item),
                                       QtCore.Qt.ElideRight, tw))

        if fresh:
            # clamped to the row's own rect for the same reason the pill is (see
            # above): anything painted past it survives as a smear on screen
            right = min(r.right(), self.tree.viewport().width())
            cx = right - self.DOT_PAD
            cy = r.center().y() + 0.5
            painter.setPen(QtCore.Qt.NoPen)
            if self.tree.train_marks:
                col, rad = self.TDOT[fresh], self.TDOT_R
            else:
                col = (self.DOT_CURRENT if fresh == FRESH_CURRENT else self.DOT_STALE)
                rad = self.DOT_R
            painter.setBrush(QtGui.QColor(col))
            painter.drawEllipse(QtCore.QPointF(cx, cy), rad, rad)
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


class SearchBox(QtWidgets.QLineEdit):
    """The IMAGES search field. Esc gives up on the search, Down steps into the
    results so the whole find-and-jump can be done from the keyboard."""
    escaped = QtCore.Signal()
    down = QtCore.Signal()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.escaped.emit(); return
        if e.key() in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Tab) and self.text():
            self.down.emit(); return
        super().keyPressEvent(e)


class SearchDelegate(QtWidgets.QStyledItemDelegate):
    """One search hit per row: the file name, and under it the folder it lives in.

    Hand-drawn for the same reason the tree is (LibraryDelegate): the selection
    has to be a rounded pill inset from the panel edge, and the second line needs
    its own colour and size — neither is expressible in the stylesheet without
    fighting the widget's own item painting.
    """
    NAME_FG, PATH_FG, SEL_BG, SEL_FG = "#1a2129", "#8a95a1", "#d9e6fb", "#12386e"
    ROW_H = 40

    def sizeHint(self, option, index):
        return QtCore.QSize(10, self.ROW_H)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = option.rect.adjusted(2, 1, -4, -1)
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        if selected:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(self.SEL_BG))
            painter.drawRoundedRect(r, 7, 7)
        name = index.data(QtCore.Qt.DisplayRole) or ""
        where = index.data(QtCore.Qt.UserRole + 9) or ""
        f = QtGui.QFont(option.font)
        f.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(f)
        painter.setPen(QtGui.QColor(self.SEL_FG if selected else self.NAME_FG))
        tr = r.adjusted(8, 3, -6, 0)
        fm = QtGui.QFontMetrics(f)
        painter.drawText(tr, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                         fm.elidedText(name, QtCore.Qt.ElideMiddle, tr.width()))
        f2 = QtGui.QFont(option.font); f2.setPointSizeF(max(8.5, f.pointSizeF() - 2.5))
        painter.setFont(f2)
        painter.setPen(QtGui.QColor(self.PATH_FG))
        fm2 = QtGui.QFontMetrics(f2)
        painter.drawText(tr.adjusted(0, fm.height() + 1, 0, 0),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                         fm2.elidedText(where, QtCore.Qt.ElideLeft, tr.width()))
        painter.restore()


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
        # Training mode swaps the meaning of the right-hand dot (see
        # LibraryDelegate.TDOT); the window flips this when the mode toggles.
        self.train_marks = False
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
        # The single column ALWAYS spans the panel exactly (Stretch), so every
        # row's rect is as wide as the viewport. This is the structural half of
        # the stray-blue-block fix: with a content-width column the rows were
        # 104 px wide inside a 223 px panel, the selection pill was drawn out to
        # the panel edge — 120 px beyond the row's own rect — and Qt, which
        # repaints only the rects of the rows whose selection changed, had no
        # reason to ever clean those 120 px again. The leftover pill stayed on
        # screen next to unrelated rows. (The delegate also clamps to the rect
        # now; either one alone would do, together they make it impossible.)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        # nothing scrolls sideways any more, so a name too long for the panel is
        # ended with an ellipsis instead of being cut through a letter
        self.setTextElideMode(QtCore.Qt.ElideRight)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setItemDelegate(LibraryDelegate(self))
        # --- drag auto-scroll ---
        # Qt's built-in auto-scroll is turned OFF: in ScrollPerPixel mode it
        # crawls a pixel at a time and stalls when the pointer is held still at
        # the very edge, so dragging a row towards the top/bottom of a long list
        # felt stuck and jerky. We drive our own steady scroll from a timer, so
        # holding the row inside the edge band keeps the list moving smoothly.
        self.setAutoScroll(False)
        self._as_timer = QtCore.QTimer(self)
        self._as_timer.setInterval(16)          # ~60 fps
        self._as_timer.timeout.connect(self._auto_scroll_step)
        self._as_speed = 0.0                     # px/tick; sign gives direction
        self._as_accum = 0.0                     # carries fractional pixels

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
        self._update_auto_scroll(e.position().toPoint().y())
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dragLeaveEvent(self, e):
        self._stop_auto_scroll()
        super().dragLeaveEvent(e)

    # A band this deep at the top and bottom edge triggers scrolling; the row has
    # to be dragged well into it, so brushing the edge while aiming for a nearby
    # row doesn't set the list moving. Speed ramps up the closer to the edge you
    # get, capped so it never races.
    AS_MARGIN = 34          # px band at top/bottom that arms the scroll
    AS_MAX_SPEED = 12       # px/tick at the very edge

    def _update_auto_scroll(self, y):
        h = self.viewport().height()
        m = self.AS_MARGIN
        if y < m:
            self._as_speed = -self.AS_MAX_SPEED * min(1.0, (m - y) / m)
        elif y > h - m:
            self._as_speed = self.AS_MAX_SPEED * min(1.0, (y - (h - m)) / m)
        else:
            self._as_speed = 0.0
        if self._as_speed and not self._as_timer.isActive():
            self._as_accum = 0.0
            self._as_timer.start()
        elif not self._as_speed:
            self._stop_auto_scroll()

    def _stop_auto_scroll(self):
        self._as_speed = 0.0
        self._as_timer.stop()

    def _auto_scroll_step(self):
        sb = self.verticalScrollBar()
        self._as_accum += self._as_speed
        step = int(self._as_accum)
        if not step:
            return
        self._as_accum -= step
        new = max(sb.minimum(), min(sb.maximum(), sb.value() + step))
        if new == sb.value():
            return                                # already at the end
        sb.setValue(new)

    def dropEvent(self, e):
        self._stop_auto_scroll()
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
