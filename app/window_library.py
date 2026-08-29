"""The IMAGES panel's behaviour: importing, filing, sorting, searching, and
keeping library.json honest.

Split out of the window because it is a self-contained job with its own
persistence: the panel is a VIRTUAL tree (library.json), independent of the
analysis session, and nothing in it is ever pruned automatically. That rule is
the reason for most of the care here — the three-layer backup in _save_library,
the refusal to write when the file failed to load, and the rename-healing in
_load_library all exist because a library that quietly loses photos is the one
failure the user cannot undo.

The widgets themselves live in widget_library_tree; this is what the window
does with them.
"""
from __future__ import annotations

import os
import re
import sys
import time
import traceback

from PIL import Image
from PySide6 import QtCore, QtWidgets

import analyze
import smartsort
import training_store
from image_files import IMG_EXT, _importable_path
from widget_library_tree import (ROLE_PATH, ROLE_KIND, ROLE_MISSING, ROLE_FRESH,
                                 ROLE_TRAIN, FRESH_NONE, FRESH_CURRENT,
                                 FRESH_STALE, TR_NONE, TR_NEW, TR_SAVED,
                                 TR_EDITED)


def _listdir(d):
    """Sorted directory listing that never raises (permission/IO errors -> [])."""
    try:
        return sorted(os.listdir(d))
    except OSError:
        return []


def _library_path():
    """Where the IMAGES panel's folder tree is persisted — its own file, wholly
    independent of the analysis session, so images and folders survive even when
    they've never been analysed. It is only ever rewritten to reflect an edit the
    user made; nothing here is pruned automatically."""
    from paths import data_dir
    return os.path.join(data_dir(), "library.json")


def _native_pick():
    """One native macOS picker for BOTH kinds of import: pick folders to bring
    them in as folders, or pick image files inside a folder to bring in just
    those. Returns a list of paths (empty if cancelled), or None if the native
    panel can't be used so the caller can fall back to a Qt dialog."""
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSOpenPanel, NSModalResponseOK
    except Exception:
        return None
    panel = NSOpenPanel.openPanel()
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(True)
    panel.setAllowsMultipleSelection_(True)
    panel.setResolvesAliases_(True)
    panel.setTitle_("Import images or folders")
    panel.setMessage_("Pick folders to import whole, or pick individual images.")
    panel.setPrompt_("Import")
    if panel.runModal() != NSModalResponseOK:
        return []
    return [str(u.path()) for u in panel.URLs()]


class LibraryPanel:
    """Mixin: everything MainWindow does with the IMAGES panel.

    Not a widget and never instantiated on its own — it is mixed into
    MainWindow, so `self` here is the window and `self.tree` is the real
    LibraryTree. Kept as a mixin rather than a separate object because these
    methods reach across the whole window (status bar, results panel, session)
    and threading them through an owner object would only move the coupling.
    """

    # ---- files ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        # a drop onto the window background (not the tree) goes to the root level
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        self._on_external_drop(paths, None)

    def import_items(self):
        """One picker for both kinds of import (the real macOS Finder panel, so
        several things can be picked at once): whatever folders you select come
        in as folders, whatever image files you select come in as images. Falls
        back to Qt's dialog if PyObjC isn't available. Nothing on disk is moved
        or renamed — the library only ever references these files."""
        paths = _native_pick()
        if paths is None:                       # PyObjC unavailable
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Import images", "",
                "Images (*.jpeg *.jpg *.png *.tif *.tiff *.bmp)")
        if paths:
            self._on_external_drop(paths, self._drop_target())

    def _on_external_drop(self, paths, target):
        """A Finder drop: folders come in as new virtual folders, loose files as
        images (into the folder they were dropped on, else the root)."""
        dirs = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if not os.path.isdir(p)]
        if dirs:
            self._add_folders_as_groups(dirs)
        if files:
            self._add_paths(files, target)

    # ---- tree helpers ----
    def _collect_images(self, paths):
        """Flatten a mix of files/folders into a de-duplicated list of images.

        Only files that actually exist are taken in — a dead path (a stale alias,
        a file deleted between the pick and the drop) would otherwise be added as
        a permanently-missing row. This does NOT affect rows already in the
        library: those are kept even when their file disappears later."""
        out = []
        for p in paths:
            if os.path.isdir(p):
                out += self._collect_images([os.path.join(p, f) for f in _listdir(p)])
            elif p.lower().endswith(IMG_EXT) and os.path.isfile(p):
                out.append(_importable_path(p))
        return out

    def _iter_items(self, kind=None):
        """Depth-first walk over every node, optionally filtered to one kind."""
        root = self.tree.invisibleRootItem()
        stack = [root.child(i) for i in range(root.childCount() - 1, -1, -1)]
        while stack:
            it = stack.pop()
            for j in range(it.childCount() - 1, -1, -1):
                stack.append(it.child(j))
            if kind is None or it.data(0, ROLE_KIND) == kind:
                yield it

    def _iter_image_items(self):
        yield from self._iter_items("image")

    def _all_paths(self):
        return {it.data(0, ROLE_PATH) for it in self._iter_image_items()}

    def _drop_target(self):
        """The folder a new item should land in: the selected folder, or the
        folder holding the selected image, else None (root)."""
        it = self.tree.currentItem()
        if it is None:
            return None
        return it if it.data(0, ROLE_KIND) == "folder" else it.parent()

    def _style_row(self, it, stamp=None):
        """Row colours/weights live in LibraryDelegate; what's recorded here is
        whether the file is still on disk and how fresh its analysis is (both
        checked once, not on every repaint). A vanished file is only greyed out —
        it is KEPT in the list until the user removes it."""
        if it.data(0, ROLE_KIND) == "folder":
            return
        path = it.data(0, ROLE_PATH)
        gone = bool(path) and not os.path.exists(path)
        it.setData(0, ROLE_MISSING, gone)
        fresh = self._fresh_state(path, stamp)
        it.setData(0, ROLE_FRESH, fresh)
        tr = self._train_state(path)
        it.setData(0, ROLE_TRAIN, tr)
        # Tooltips are rationed in this panel (the old full-path one was pure
        # noise): a row only explains itself when it is showing a mark.
        if gone:
            tip = "File not found on disk — kept until you remove it"
        elif self.train_mode:
            tip = {TR_NEW: "Not in the training set yet — label it, then "
                           "“Add photo to training set”",
                   TR_SAVED: "In the training set, exactly as it is on screen",
                   TR_EDITED: "In the training set, but edited since — add it "
                              "again, or press Restore saved version",
                   }.get(tr, "")
        elif fresh == FRESH_STALE:
            tip = ("Analysed with earlier settings or an earlier model — "
                   "press Analyze to bring it up to date")
        elif fresh == FRESH_CURRENT:
            tip = "Analysed with the current settings"
        else:
            tip = ""
        it.setToolTip(0, tip)

    def _fresh_state(self, path, stamp=None):
        """Whether this image's saved analysis came from the rules and model the
        app runs right now (see analyze.pipeline_stamp)."""
        a = self.results.get(path) if path else None
        if a is None:
            return FRESH_NONE
        stamp = stamp or analyze.pipeline_stamp()
        return (FRESH_CURRENT if getattr(a, "pipeline", "") == stamp
                else FRESH_STALE)

    def _train_state(self, path):
        """Where this image stands in the training set (TR_* / the panel dot).

        Only computed while Training mode is on — outside it the marks aren't
        drawn, and reading a labels file per row would be work for nothing.

        "Edited" is decided by comparing what would be SAVED right now against
        the fingerprint stored when the photo was added ([[training_store.signature]]):
        clicks, cleared classes and a re-analysis that changed the model's
        pre-fill all move the photo into that state, while re-clicking a class
        onto the value it already had does not.
        """
        if not (self.train_mode and path):
            return TR_NONE
        try:
            saved = training_store.saved(path)
        except Exception:
            return TR_NONE
        if saved is None:
            return TR_NEW
        a = self.results.get(path)
        if a is None or not getattr(a, "classifiable", False):
            # nothing on screen to compare against (not analysed this session) —
            # the saved version is all there is, so it is simply "in the set"
            return TR_SAVED
        cur = self._train_signature(path, a)
        return TR_SAVED if cur == saved["signature"] else TR_EDITED

    def _train_signature(self, path, a):
        """The fingerprint of what "Add photo to training set" would write for
        this photo right now."""
        eff = self._train_effective(a, path)
        excluded = [pid for pid, cls in self.train_labels.get(path, {}).items()
                    if cls == "exclude"]
        return training_store.signature(eff, excluded)

    def _restyle_rows(self):
        """Re-check every row's on-disk state and analysis freshness (after a
        load, a move, a rename, an analysis or a retrain)."""
        stamp = analyze.pipeline_stamp()       # once per pass, not once per row
        self.tree.blockSignals(True)
        for it in self._iter_items():
            self._style_row(it, stamp)
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            self._roll_fresh(root.child(i))
            self._roll_train(root.child(i))
        self.tree.blockSignals(False)
        self.tree.viewport().update()

    def _roll_train(self, it):
        """Same idea as _roll_fresh for the training marks, with the priority the
        mode cares about: a folder shouts about the photo that needs attention
        first — edited beats not-yet-added, which beats saved."""
        if it.data(0, ROLE_KIND) != "folder":
            return it.data(0, ROLE_TRAIN) or TR_NONE
        state = TR_NONE
        rank = {TR_NONE: 0, TR_SAVED: 1, TR_NEW: 2, TR_EDITED: 3}
        for i in range(it.childCount()):
            c = self._roll_train(it.child(i))
            if rank[c] > rank[state]:
                state = c
        it.setData(0, ROLE_TRAIN, state)
        return state

    def _roll_fresh(self, it):
        """Give folders the state of what they hold, so a COLLAPSED folder still
        says "something in here wants re-running": stale wins over current, and a
        folder with nothing analysed inside it stays bare."""
        if it.data(0, ROLE_KIND) != "folder":
            return it.data(0, ROLE_FRESH) or FRESH_NONE
        state = FRESH_NONE
        for i in range(it.childCount()):
            c = self._roll_fresh(it.child(i))
            if c == FRESH_STALE:
                state = FRESH_STALE
            elif c == FRESH_CURRENT and state != FRESH_STALE:
                state = FRESH_CURRENT
        it.setData(0, ROLE_FRESH, state)
        return state

    def _style_row_for(self, path):
        """Refresh the marks on the rows showing one image (and the folders above
        them). Used after a single analysis — walking the whole tree once per
        image of a 40-photo batch would be wasted work."""
        stamp = analyze.pipeline_stamp()
        self.tree.blockSignals(True)
        tops = []
        for it in self._iter_items("image"):
            if it.data(0, ROLE_PATH) != path:
                continue
            self._style_row(it, stamp)
            top = it
            while top.parent() is not None:
                top = top.parent()
            if top is not it and top not in tops:
                tops.append(top)
        for top in tops:
            self._roll_fresh(top)
            self._roll_train(top)
        self.tree.blockSignals(False)
        self.tree.viewport().update()

    _FOLDER_FLAGS = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                     | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
                     | QtCore.Qt.ItemIsEditable)
    # images are NOT drop-enabled, so nothing can ever be nested *under* an image
    _IMAGE_FLAGS = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsEditable)

    def _make_folder_item(self, name):
        it = QtWidgets.QTreeWidgetItem([name])
        it.setData(0, ROLE_KIND, "folder")
        it.setFlags(self._FOLDER_FLAGS)
        self._style_row(it)
        return it

    def _make_image_item(self, path, name=None):
        it = QtWidgets.QTreeWidgetItem([name or os.path.basename(path)])
        it.setData(0, ROLE_KIND, "image")
        it.setData(0, ROLE_PATH, path)
        it.setFlags(self._IMAGE_FLAGS)
        self._style_row(it)
        return it

    def _add_paths(self, paths, target=None):
        """Add image files as references, into `target` folder (else the root).
        Already-present paths are skipped, so re-adds and session restores never
        duplicate a row."""
        files = self._collect_images(paths)
        existing = self._all_paths()
        parent = target if (target is not None
                            and target.data(0, ROLE_KIND) == "folder") else None
        holder = parent or self.tree.invisibleRootItem()
        added = []
        self.tree.blockSignals(True)
        for path in files:
            if path in existing:
                continue
            it = self._make_image_item(path)
            holder.addChild(it)
            existing.add(path); added.append(it)
        self.tree.blockSignals(False)
        if added:
            self._restyle_rows()
            if parent is not None:
                parent.setExpanded(True)
            if self.current is None:
                self.tree.setCurrentItem(added[0])
            self._save_library()
            self._refresh_results()
        return added

    def _add_folders_as_groups(self, dir_paths):
        """Import each disk folder as a new top-level virtual folder holding its
        images (recursively found, flattened). The files on disk are untouched."""
        existing = self._all_paths()
        created = []
        self.tree.blockSignals(True)
        for d in dir_paths:
            imgs = [p for p in self._collect_images([d]) if p not in existing]
            if not imgs:
                continue
            folder = self._make_folder_item(os.path.basename(d.rstrip("/\\")) or "Folder")
            self.tree.addTopLevelItem(folder)
            for p in imgs:
                folder.addChild(self._make_image_item(p)); existing.add(p)
            folder.setExpanded(True); created.append(folder)
        self.tree.blockSignals(False)
        if created:
            self._restyle_rows()
            if self.current is None:
                first = next(self._iter_image_items(), None)
                if first is not None:
                    self.tree.setCurrentItem(first)
            self._save_library()
            self._refresh_results()

    # ---- new folder / rename ----
    def _new_folder(self):
        """New Folder with rows selected groups THOSE rows into it (the natural
        'put these together' gesture); with nothing selected it just makes an
        empty folder beside the current one."""
        sel = [it for it in self.tree.selectedItems()]
        # keep only the top-most of a nested selection, so moving a folder
        # doesn't also try to move a child that travelled with it
        selset = set(sel)

        def nested(it):
            p = it.parent()
            while p is not None:
                if p in selset:
                    return True
                p = p.parent()
            return False

        tops = [it for it in sel if not nested(it)]
        if not tops:
            self._new_folder_in(self._drop_target())
            return
        # the new folder takes the place of the first selected row
        anchor = tops[0]
        parent = anchor.parent()
        idx = ((parent or self.tree.invisibleRootItem()).indexOfChild(anchor))
        folder = self._make_folder_item("New Folder")
        self.tree.blockSignals(True)
        for it in tops:                       # detach, then re-home under folder
            (it.parent() or self.tree.invisibleRootItem()).removeChild(it)
        (parent or self.tree.invisibleRootItem()).insertChild(idx, folder)
        for it in tops:
            folder.addChild(it)
        self.tree.blockSignals(False)
        folder.setExpanded(True)
        self._restyle_rows()
        self._save_library()
        self.tree.setCurrentItem(folder)
        self.tree.editItem(folder, 0)         # name it straight away

    # ---- smart sort ---------------------------------------------------------
    @staticmethod
    def _natural_key(s):
        """Sort "Janus 2" before "Janus 10" — plain text sorting puts 10 first."""
        return [int(t) if t.isdigit() else smartsort.fold(t)
                for t in re.split(r"(\d+)", s)]

    def _smart_sort(self):
        """Re-file the whole library from the file names, after showing the plan.

        Nothing on disk moves — the panel is a virtual tree — but this DOES
        replace the arrangement the user may have built by hand, so it always
        asks first and keeps the previous library.json as a separate backup.
        """
        items = list(self._iter_image_items())
        if not items:
            self.status.showMessage("Nothing to sort — import some images first.")
            return
        names = {it.data(0, ROLE_PATH): it.text(0) for it in items}
        rows, unsorted_ = smartsort.plan([it.data(0, ROLE_PATH) for it in items])
        outline = smartsort.outline(smartsort.tree_of(rows))
        note = (f"<p style='color:#8a95a1;'>{len(unsorted_)} image(s) could not be "
                f"read from their name — they go to “Unsorted”.</p>"
                if unsorted_ else "")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Smart sort")
        body = QtWidgets.QVBoxLayout(dlg)
        head = QtWidgets.QLabel(
            f"<b>{len(items)} images</b> will be filed like this. "
            f"Your files on disk are not touched.{note}")
        head.setWordWrap(True)
        txt = QtWidgets.QPlainTextEdit("\n".join(outline))
        txt.setReadOnly(True)
        txt.setMinimumSize(360, 380)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        ok = btns.addButton("Sort", QtWidgets.QDialogButtonBox.AcceptRole)
        ok.setObjectName("primary")
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        body.addWidget(head); body.addWidget(txt, 1); body.addWidget(btns)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._backup_library("library.presort.json")
        self._apply_sort(rows, names)

    def _apply_sort(self, rows, names):
        """Rebuild the tree from a plan: folders alphabetically, images naturally."""
        by_folder = {}
        for path, folders in rows:
            by_folder.setdefault(tuple(folders), []).append(path)
        self.tree.blockSignals(True)
        self.tree.clear()
        cache = {(): None}                       # folder path -> item (None = root)

        def folder_for(key):
            if key in cache:
                return cache[key]
            parent = folder_for(key[:-1])
            it = self._make_folder_item(key[-1])
            (parent.addChild if parent is not None else self.tree.addTopLevelItem)(it)
            cache[key] = it
            return it

        for key in sorted(by_folder, key=lambda k: [self._natural_key(x) for x in k]):
            parent = folder_for(key)
            for path in sorted(by_folder[key],
                               key=lambda p: self._natural_key(names.get(p, p))):
                it = self._make_image_item(path, names.get(path))
                (parent.addChild if parent is not None else
                 self.tree.addTopLevelItem)(it)
        self.tree.blockSignals(False)
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)   # samples open, rest closed
        self._restyle_rows()
        self._save_library()
        n_folders = len(cache) - 1
        self.status.showMessage(
            f"Sorted {sum(len(v) for v in by_folder.values())} images into "
            f"{n_folders} folders. Undo isn't wired to the tree — the previous "
            f"arrangement is saved as library.presort.json.")

    def _backup_library(self, name):
        import shutil
        p = _library_path()
        if os.path.exists(p):
            try:
                shutil.copyfile(p, os.path.join(os.path.dirname(p), name))
            except OSError:
                pass

    def _new_folder_in(self, parent):
        it = self._make_folder_item("New Folder")
        if parent is not None and parent.data(0, ROLE_KIND) == "folder":
            parent.addChild(it); parent.setExpanded(True)
        else:
            self.tree.addTopLevelItem(it)
        self._save_library()
        self.tree.setCurrentItem(it)
        self.tree.editItem(it, 0)             # drop straight into rename

    def _rename_current(self):
        it = self.tree.currentItem()
        if it is not None:
            self.tree.editItem(it, 0)

    def _item_renamed(self, item, _col):
        """An in-place rename committed. For an image this changes only its
        display name — the file path it points to is left exactly as it was."""
        if not item.text(0).strip():          # refuse an empty name
            path = item.data(0, ROLE_PATH)
            fallback = os.path.basename(path) if path else "Folder"
            self.tree.blockSignals(True)
            item.setText(0, fallback)
            self.tree.blockSignals(False)
        self._save_library()

    def _on_expand_toggle(self, _item=None):
        # the chevron itself animates in LibraryTree; nothing to redraw here
        self._save_library()                  # remember which folders are open

    def _after_move(self):
        """A drag-move finished. Qt's internal move recreates the dragged rows
        with DEFAULT flags/icons, so re-assert them — most importantly, images
        must stay non-drop-enabled so nothing can be nested under them."""
        self.tree.blockSignals(True)
        for it in self._iter_items():
            it.setFlags(self._FOLDER_FLAGS if it.data(0, ROLE_KIND) == "folder"
                        else self._IMAGE_FLAGS)
        self.tree.blockSignals(False)
        self._restyle_rows()                  # icons/colours are reset by the move
        self._save_library()
        self._refresh_results()

    # ---- persistence: the folder tree, independent of the analysis session ----
    def _serialize_node(self, it):
        if it.data(0, ROLE_KIND) == "image":
            return {"type": "image", "path": it.data(0, ROLE_PATH), "name": it.text(0)}
        return {"type": "folder", "name": it.text(0), "expanded": it.isExpanded(),
                "children": [self._serialize_node(it.child(j))
                             for j in range(it.childCount())]}

    @staticmethod
    def _count_images(data):
        """How many image rows a serialised tree holds."""
        def walk(nodes):
            n = 0
            for x in nodes:
                if x.get("type") == "image":
                    n += 1
                else:
                    n += walk(x.get("children", []))
            return n
        return walk(data.get("tree", []))

    SHRINK_KEEP = 4

    def _keep_shrink_backup(self, p):
        """Snapshot `p` (the library as it stands BEFORE a save that loses rows).

        Dated, and the FULLEST snapshot is never pruned. This used to be one
        rolling `library.shrink.json`, which turned out to protect nothing: on
        2026-08-08 a user with 44 photos to recover spent the good 178-row copy
        by deleting and re-adding a single image a few times, because every one
        of those routine removals is a "shrink" and overwrote it with the
        already-damaged 134-row state. A snapshot is 30 KB; the recovery it
        buys is hours of analysis. Never raises — a failed backup must not stop
        the save that follows it.
        """
        import glob
        import shutil
        d = os.path.dirname(p)
        try:
            # the stamp is only second-resolution, and several removals inside
            # one second is exactly the churn this guards against — so a taken
            # name gets a suffix rather than silently overwriting the snapshot
            # that came a moment before it
            stem = os.path.join(d, "library.shrink-" + time.strftime("%Y%m%d-%H%M%S"))
            dst, i = stem + ".json", 1
            while os.path.exists(dst):
                i += 1
                dst = f"{stem}-{i}.json"
            shutil.copyfile(p, dst)
            snaps = sorted(glob.glob(os.path.join(d, "library.shrink-*.json")))
            if len(snaps) > self.SHRINK_KEEP:
                fullest = max(snaps, key=self._snapshot_images)
                # newest few, plus whichever holds the most rows
                for old in snaps[:-self.SHRINK_KEEP]:
                    if old != fullest:
                        os.remove(old)
        except OSError:
            pass

    @staticmethod
    def _snapshot_images(path):
        import json
        try:
            with open(path, encoding="utf-8") as f:
                return LibraryPanel._count_images(json.load(f))
        except Exception:
            return -1

    def _save_library(self):
        """Write the whole tree to library.json — atomically, and behind several
        layers of backup, because losing this file loses the user's organisation.

        Layers, in order of age:
          library.json.bak            previous save (rolls every time)
          library.startup.json        how the library looked when the app opened
          library.shrink-<stamp>.json dated snapshots taken whenever a save loses
                                      rows; the newest few are kept, and the
                                      FULLEST one is never pruned

        Together these mean an accidental delete is recoverable even after the
        user has kept working — a single rolling .bak would already be gone.
        """
        import json
        import shutil
        # If the file could not be READ at startup (corrupt/unreadable), never
        # write over it: the tree in memory is empty only because loading failed,
        # and saving it would destroy a library the user can still recover from.
        # (This is the lesson from the lost session.pkl.)
        if not getattr(self, "_lib_ok", True):
            return
        try:
            root = self.tree.invisibleRootItem()
            data = {"version": 1,
                    "tree": [self._serialize_node(root.child(j))
                             for j in range(root.childCount())]}
            n_now = self._count_images(data)
            p = _library_path()
            if os.path.exists(p):
                try:
                    shutil.copyfile(p, p + ".bak")
                    # first save of this run: keep the state we opened with
                    if not getattr(self, "_startup_backup_done", False):
                        shutil.copyfile(p, os.path.join(
                            os.path.dirname(p), "library.startup.json"))
                        self._startup_backup_done = True
                    # the row count is going DOWN — keep the fuller version too
                    if n_now < getattr(self, "_lib_count", n_now):
                        self._keep_shrink_backup(p)
                except OSError:
                    pass
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, p)
            self._lib_count = n_now
            self._save_failed = False
        except Exception:
            traceback.print_exc()             # never let saving break the app
            # …but do tell the user once, or they'd lose work silently
            if not getattr(self, "_save_failed", False):
                self._save_failed = True
                self.status.showMessage(
                    "Could not save the image library — check disk space and "
                    "permissions for the app's data folder.", 15000)

    def _load_library(self):
        """Rebuild the tree from library.json. Missing files are kept (greyed),
        never dropped — the user's organisation always survives a restart.

        On any failure `_lib_ok` goes False, which stops _save_library from
        overwriting a file we could not read (see the note there)."""
        import json
        self._lib_ok = True
        self._relink_map = {}
        try:
            p = _library_path()
            if not os.path.exists(p):
                # first launch ever: start on the example micrographs the app
                # ships, so the first Analyze runs on material the networks were
                # actually trained for (see sample_images)
                import sample_images
                if not sample_images.seed_library(p):
                    return
            data = json.load(open(p, encoding="utf-8"))
            if data.get("version") != 1:
                self._lib_ok = False          # unknown format: leave it alone
                return
            # remember how many images we started with, so the very first delete
            # of the session still triggers the "shrink" backup
            self._lib_count = self._count_images(data)

            # every path that still resolves on disk, so the rename-heal below
            # never re-points a row at a file another row already owns
            claimed = set()

            def gather(nodes):
                for n in nodes:
                    if n.get("type") == "image":
                        q = n.get("path")
                        if q and os.path.exists(q):
                            claimed.add(q)
                    else:
                        gather(n.get("children", []))
            gather(data.get("tree", []))

            def build(node, parent):
                if node.get("type") == "image":
                    path = node.get("path")
                    if not path:
                        return
                    # A file that was renamed on disk after import would otherwise
                    # be lost (greyed, its cached analysis dropped). Try to find it
                    # again in the SAME folder and re-point the row at it; the map
                    # is handed to _load_session so the analysis follows.
                    if not os.path.exists(path):
                        new = self._find_renamed(path, claimed)
                        if new:
                            self._relink_map[path] = new
                            claimed.add(new)
                            path = new
                    parent.addChild(self._make_image_item(path, node.get("name")))
                else:
                    it = self._make_folder_item(node.get("name") or "Folder")
                    parent.addChild(it)
                    for ch in node.get("children", []):
                        build(ch, it)
                    it.setExpanded(bool(node.get("expanded", True)))

            self.tree.blockSignals(True)
            root = self.tree.invisibleRootItem()
            for node in data.get("tree", []):
                build(node, root)
            self.tree.blockSignals(False)
            self._restyle_rows()
            if self.current is None:
                first = next(self._iter_image_items(), None)
                if first is not None:
                    self.tree.setCurrentItem(first)
            # persist the healed paths so the next launch is already clean (the
            # usual backup layers in _save_library still guard the old file)
            if self._relink_map:
                self._save_library()
        except Exception:
            self._lib_ok = False              # never write over what we can't read
            traceback.print_exc()
            self.status.showMessage(
                "Could not read the image library — it has been left untouched "
                "(a copy is in library.json.bak).")

    @staticmethod
    def _stem_num(stem):
        """The trailing integer of a filename stem (e.g. '… Janus 3' -> '3'), or
        None. Numbered SEM series are matched on this so a heal keeps the number."""
        import re
        m = re.search(r"(\d+)\s*$", stem)
        return m.group(1) if m else None

    def _find_renamed(self, old_path, claimed):
        """Best-effort locate a file that was renamed on disk after import.

        Looks only inside the row's own recorded folder (so it can't wander off to
        an unrelated image), among files no other row already points at. A match
        must share the same extension, the same trailing number, and a real chunk
        of the original name — otherwise the row is left greyed rather than
        risking a wrong link. Returns the new absolute path, or None."""
        d = os.path.dirname(old_path)
        if not os.path.isdir(d):
            return None
        o_stem, o_ext = os.path.splitext(os.path.basename(old_path))
        o_ext = o_ext.lower()
        o_num = self._stem_num(o_stem)
        best, best_lcp, ties = None, 3, 0     # require >3 shared leading chars
        for f in _listdir(d):
            fp = os.path.join(d, f)
            if fp in claimed or not f.lower().endswith(IMG_EXT) \
                    or not os.path.isfile(fp):
                continue
            s, e = os.path.splitext(f)
            if e.lower() != o_ext:
                continue
            if o_num is not None and self._stem_num(s) != o_num:
                continue                      # a numbered series must keep its number
            lcp = len(os.path.commonprefix([o_stem, s]))
            if lcp > best_lcp:
                best, best_lcp, ties = fp, lcp, 1
            elif lcp == best_lcp:
                ties += 1
        return best if ties == 1 else None    # ambiguous -> don't guess

    def _boot(self):
        """Startup order: rebuild the saved folder tree first, then restore the
        analysis session (which only re-attaches analyses to images already here,
        adding any stray analysed image that isn't in the tree)."""
        self._load_library()
        self._load_session()
        # If any file was healed, the library was rewritten with the new paths
        # during load; write the session out now too, so its keys match — a crash
        # before the next clean close then can't strand the moved analyses.
        if getattr(self, "_relink_map", None):
            self._save_session()
        n = getattr(self, "_ghosts_cleaned", 0)
        if n:
            # say it once, so the changed counts aren't a mystery
            self.status.showMessage(
                f"Removed {n} speckle detections that were too small to be real "
                f"particles (a few pixels across). Counts and mean sizes on the "
                f"coarser photos have changed accordingly.")

    def _selected_paths(self):
        """Selected image paths (the analysis / results set). Selecting a folder
        counts as selecting every image anywhere beneath it."""
        paths = []

        def collect(it):
            if it.data(0, ROLE_KIND) == "image":
                paths.append(it.data(0, ROLE_PATH))
            else:
                for j in range(it.childCount()):
                    collect(it.child(j))

        for it in self.tree.selectedItems():
            collect(it)
        return list(dict.fromkeys(paths))     # de-dup, keep order

    # ---- search ------------------------------------------------------------
    _SEARCH_WHERE = QtCore.Qt.UserRole + 9      # results row -> folder path text

    def _folder_trail(self, it):
        """"05 / Pattern / Janus" for a row — where the search hit lives."""
        parts = []
        p = it.parent()
        while p is not None:
            parts.append(p.text(0))
            p = p.parent()
        return "  /  ".join(reversed(parts)) or "top level"

    def _on_search(self, text):
        """Live results. Matching goes through smartsort.fold, so "karisik" finds
        "Karışık" and case/diacritics never decide whether a photo shows up."""
        q = smartsort.fold(text)
        if not q:
            self.list_stack.setCurrentWidget(self.tree)
            self.results_list.clear()
            return
        terms = q.split()
        self.results_list.clear()
        hits = 0
        for it in self._iter_image_items():
            name = it.text(0)
            hay = smartsort.fold(name + " " + self._folder_trail(it))
            if not all(t in hay for t in terms):     # every word must appear
                continue
            row = QtWidgets.QListWidgetItem(os.path.splitext(name)[0])
            row.setData(self._SEARCH_WHERE, self._folder_trail(it))
            row.setData(ROLE_PATH, it.data(0, ROLE_PATH))
            self.results_list.addItem(row)
            hits += 1
        self.list_stack.setCurrentWidget(self.results_list)
        if hits:
            self.results_list.setCurrentRow(0)
        self.status.showMessage(
            f"{hits} image{'' if hits == 1 else 's'} match “{text}”"
            + ("  ·  double-click one to see where it is" if hits else ""))

    def _search_cancel(self):
        """Esc: drop the search and put the focus back on the tree."""
        self.search.clear()
        self.tree.setFocus()

    def _search_preview(self, row):
        """Single click: show the image, without leaving the results."""
        path = row.data(ROLE_PATH)
        if not path:
            return
        self.current = path
        if path in self.results:
            self._rerender()
        else:
            try:
                self.view.set_image(Image.open(path))
            except Exception:
                self.view.clear_image()
        self._update_class_controls(self.results.get(path))

    def _search_reveal_current(self):
        row = self.results_list.currentItem()
        if row is not None:
            self._search_reveal(row)

    def _search_reveal(self, row):
        """Double click / Enter: back to the tree, with the image revealed."""
        path = row.data(ROLE_PATH)
        self.search.clear()                      # also switches the stack back
        if path:
            self._reveal_path(path)

    def _reveal_path(self, path):
        """Open every folder above the image, select it and scroll it into view."""
        for it in self._iter_image_items():
            if it.data(0, ROLE_PATH) != path:
                continue
            p = it.parent()
            while p is not None:
                p.setExpanded(True)
                p = p.parent()
            self.tree.setCurrentItem(it)
            self.tree.clearSelection()
            it.setSelected(True)
            self.tree.scrollToItem(it, QtWidgets.QAbstractItemView.PositionAtCenter)
            self.tree.setFocus()
            self.status.showMessage(
                f"{os.path.basename(path)}  ·  {self._folder_trail(it)}")
            return

    def _current_changed(self, cur, _prev):
        if cur is None or cur.data(0, ROLE_KIND) != "image":
            return
        self.current = cur.data(0, ROLE_PATH)
        if self.current in self.results:
            self._rerender()
        else:
            try:
                self.view.set_image(Image.open(self.current))
            except Exception:
                self.view.clear_image()
                self.status.showMessage(f"Could not open: {os.path.basename(self.current)}")
        self._update_class_controls(self.results.get(self.current))
        if self.train_mode:
            self._train_update_panel()

    # ---- tree edit: remove folders/images (Delete key or right-click) ----
    #  This is the ONLY path that ever removes anything from the panel, and it
    #  only runs on a deliberate user action. The files on disk are never
    #  touched, and — since 2026-08-08 — NEITHER ARE THE ANALYSES.
    #
    #  It used to `self.results.pop(p)` every removed image. That made this the
    #  one action in the app that could destroy hours of work with no undo: a
    #  user lost 44 analysed photos to it, and the confirmation dialog said only
    #  "the files on disk are not deleted" — reassuring about the one thing that
    #  was never at risk while saying nothing about the one thing that was.
    #  Now the analysis stays in `self.results` (and so in session.pkl), where
    #  _load_session keeps it for as long as the image file exists on disk. Put
    #  the photo back in the library and its analysis is simply there again.
    def _delete_current(self):
        items = self.tree.selectedItems()
        if not items and self.tree.currentItem() is not None:
            items = [self.tree.currentItem()]
        if items:
            self._remove_items(items)

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        if item is None:                       # empty space -> just offer a folder
            menu.addAction("📁   New Folder", self._new_folder)
            menu.exec(self.tree.viewport().mapToGlobal(pos))
            return
        kind = item.data(0, ROLE_KIND)
        sel = [it for it in self.tree.selectedItems()] or [item]
        menu.addAction("✏️   Rename", lambda: self.tree.editItem(item, 0))
        if kind == "folder":
            menu.addAction("📁   New subfolder", lambda: self._new_folder_in(item))
        menu.addSeparator()
        if len(sel) > 1:
            menu.addAction(f"🗑   Remove {len(sel)} selected",
                           lambda: self._remove_items(sel))
        else:
            label = "🗑   Remove folder" if kind == "folder" else "🗑   Remove image"
            menu.addAction(label, lambda: self._remove_items([item]))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _remove_items(self, items):
        """Remove the given rows (folders take their whole subtree). Confirms
        first when a folder or several rows are involved, so nothing substantial
        goes on a stray keypress."""
        itemset = set(items)

        def has_ancestor_in_set(it):
            p = it.parent()
            while p is not None:
                if p in itemset:
                    return True
                p = p.parent()
            return False

        # keep only the top-most of any nested selection, so we never remove a
        # parent and then try to remove a child that went with it
        tops = [it for it in items if not has_ancestor_in_set(it)]
        has_folder = any(it.data(0, ROLE_KIND) == "folder" for it in tops)

        # gather the image paths going away (to count them for the prompt, and to
        # pick a nearby replacement preview). Their ANALYSES are deliberately left
        # alone — see the note above this method.
        removed = set()
        for it in tops:
            if it.data(0, ROLE_KIND) == "image":
                removed.add(it.data(0, ROLE_PATH))
            else:
                for c in self._descendant_images(it):
                    removed.add(c.data(0, ROLE_PATH))

        if has_folder or len(tops) > 1:
            # say the numbers. "the selected item(s)" hid the fact that a folder
            # takes a whole subtree with it, which is exactly how a removal turns
            # out bigger than the user pictured.
            n_an = sum(1 for p in removed if p in self.results)
            what = (f"{len(removed)} image(s)" if removed
                    else f"{len(tops)} item(s)")
            if QtWidgets.QMessageBox.question(
                    self, "Remove",
                    f"Remove {what} from the library?\n\n"
                    "The files on disk are not deleted"
                    + (f", and the {n_an} analysis/analyses among them are kept "
                       "— put the photos back and their analyses come with them"
                       if n_an else "")
                    + ".\n\n⌘Z undoes this.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return

        # remember the exact rows and where they sat, so ⌘Z can put them back
        self._lib_undo = (time.monotonic(), [
            (self._folder_trail(it.parent()),
             (it.parent() or self.tree.invisibleRootItem()).indexOfChild(it),
             self._serialize_node(it))
            for it in tops])

        # line up a NEARBY replacement for the previewed image before mutating,
        # so the selection lands next to where the user was, not at the top
        next_path = None
        if self.current in removed:
            flat = [it.data(0, ROLE_PATH) for it in self._iter_image_items()]
            if self.current in flat:
                i = flat.index(self.current)
                after = next((p for p in flat[i + 1:] if p not in removed), None)
                before = next((p for p in reversed(flat[:i]) if p not in removed), None)
                next_path = after or before

        self.tree.blockSignals(True)
        for it in tops:
            parent = it.parent() or self.tree.invisibleRootItem()
            parent.removeChild(it)
        self.tree.blockSignals(False)

        if self.current in removed:
            self.current = None
            cur = None
            if next_path is not None:
                cur = next((it for it in self._iter_image_items()
                           if it.data(0, ROLE_PATH) == next_path), None)
            if cur is None:
                cur = next(self._iter_image_items(), None)
            if cur is not None:
                self.tree.setCurrentItem(cur)
                self._current_changed(cur, None)   # signal may not fire after edits
            else:
                self.view.clear_image()
                self.result.clear_img()
        if removed:
            self._restyle_rows()      # the folders above lost analyses
        self._save_library()
        self._refresh_results()

    # ---- undoing a removal (⌘Z) -----------------------------------------
    def _folder_trail(self, item):
        """The folder names from the root down to `item`, or None for the root.

        Stored instead of the QTreeWidgetItem itself: a reference to a row that
        has since been deleted is a dangling C++ object, and touching one is a
        hard crash rather than a failed undo."""
        if item is None:
            return None
        trail = []
        while item is not None:
            trail.append(item.text(0))
            item = item.parent()
        return list(reversed(trail))

    def _resolve_trail(self, trail):
        """Find the folder a trail names, or the root if it is gone (restoring
        at the top beats refusing to restore)."""
        node = self.tree.invisibleRootItem()
        for name in trail or []:
            nxt = None
            for j in range(node.childCount()):
                c = node.child(j)
                if c.data(0, ROLE_KIND) == "folder" and c.text(0) == name:
                    nxt = c
                    break
            if nxt is None:
                return self.tree.invisibleRootItem()
            node = nxt
        return node

    def _rebuild_node(self, node, parent, idx=None):
        """Re-create a serialised row (and its subtree) under `parent`."""
        if node.get("type") == "image":
            it = self._make_image_item(node.get("path"), node.get("name"))
        else:
            it = self._make_folder_item(node.get("name") or "Folder")
        if idx is None or idx > parent.childCount():
            parent.addChild(it)
        else:
            parent.insertChild(idx, it)
        if node.get("type") != "image":
            for ch in node.get("children", []):
                self._rebuild_node(ch, it)
            it.setExpanded(bool(node.get("expanded", True)))
        return it

    def _lib_undo_last(self):
        """Put back the rows the last removal took, where they were."""
        if not self._lib_undo:
            self.status.showMessage("Nothing to undo.")
            return
        _, entries = self._lib_undo
        self._lib_undo = None
        self.tree.blockSignals(True)
        first = None
        n = 0
        for trail, idx, node in entries:
            it = self._rebuild_node(node, self._resolve_trail(trail), idx)
            first = first or it
            n += 1 if node.get("type") == "image" else \
                sum(1 for _ in self._descendant_images(it))
        self.tree.blockSignals(False)
        self._restyle_rows()
        if first is not None:
            self.tree.setCurrentItem(first)
            self._current_changed(first, None)
        self._save_library()
        self._refresh_results()
        self.status.showMessage(f"Restored {n} image(s) to the library.")

    def _descendant_images(self, folder):
        for j in range(folder.childCount()):
            c = folder.child(j)
            if c.data(0, ROLE_KIND) == "image":
                yield c
            else:
                yield from self._descendant_images(c)
