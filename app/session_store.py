"""Saving and restoring the analysis session (session.pkl).

The one file whose failure modes are unforgiving, so the rules here are hard:

  * NEVER drop a session because the schema moved on. A version bump that
    silently discarded old pickles once destroyed every analysis the user had,
    unrecoverably. Old versions are MIGRATED (see _load_session); v1 files
    still load today.
  * NEVER let an empty run overwrite a full file, and always leave the previous
    file behind as .bak. The backup is made by RENAMING the old file into
    place, not copying it — the session runs to hundreds of megabytes and a
    copy on every save was punishing on a small machine.
  * WRITE ATOMICALLY (2026-09-22). The new session goes to a temporary file
    first and only replaces session.pkl once it is complete on disk. It used
    to be written straight into place: a crash or power cut mid-save left a
    half-written session.pkl beside a good .bak.
  * A SESSION THAT WILL NOT OPEN IS NOT THE END. The loader falls back to .bak
    and says so. It used to come up empty without a word — and the next save
    then rotated the damaged file over the good .bak, losing both. A damaged
    file is set aside as session.pkl.damaged-<time>, never rotated into .bak.

Analyses are pickled objects, so `Particle` and `Analysis` are stored by their
module path (`analyze`). Renaming that module would make every existing
session unreadable; it must keep its name.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

from PySide6 import QtCore, QtWidgets

import analyze

SESSION_VERSION = 3   # v1 and v2 sessions still load — never drop a user's
                      # analyses. v3 (2026-09-27) stores each analysis compact
                      # (zlib-packed mask, no micrograph: analyze.Analysis.
                      # __getstate__), 566 MB -> ~45 MB for 143 photos. An older
                      # app refuses v3 at the version check instead of
                      # half-reading it, and the last v2 file is kept aside as
                      # session.pkl.pre-compact for going back to it.
READABLE_VERSIONS = (1, 2, 3)


def _session_path():
    """Where the last session (analyses, thresholds…) is persisted."""
    from paths import data_dir
    return os.path.join(data_dir(), "session.pkl")


class SessionStore:
    """Mixin: session persistence and an orderly shutdown. `self` is the window."""

    # ---- hide instead of close (macOS) ----
    def _hide_window(self):
        # written now rather than at quit: a hidden app can sit for days and
        # still be ended by a shutdown that never asks it anything
        self._save_library()
        self._save_session()
        if self.isFullScreen():
            # a full-screen window lives in its own Space; hiding it there
            # leaves that Space black (the trap Artitect fell into, 2026-09-21)
            # — leave full screen first, hide once the animation has finished
            self._hide_after_fullscreen = True
            self.showNormal()
        else:
            self.hide()

    def changeEvent(self, e):
        if (e.type() == QtCore.QEvent.WindowStateChange
                and getattr(self, "_hide_after_fullscreen", False)
                and not self.isFullScreen()):
            self._hide_after_fullscreen = False
            QtCore.QTimer.singleShot(700, self.hide)
        super().changeEvent(e)

    def _app_state_changed(self, state):
        """A click on the Dock icon arrives as 'application active' — Qt
        forces it through even when the app already is — so that is where a
        hidden window comes back."""
        if (state == QtCore.Qt.ApplicationActive and not self.isVisible()
                and not getattr(self, "_quitting", False)):
            self.showNormal()
            self.raise_()
            self.activateWindow()

    # ---- session persistence ----
    # Which switches ride along in the session file. Deliberately a tiny, named
    # list rather than "everything checkable": a session that restores state the
    # user cannot see they changed is worse than one that forgets.
    # (tr_score_cb was dropped 2026-08-05 with the cross-validation it switched
    # off; `_apply_prefs` ignores names that no longer exist, so old sessions
    # carrying it still load.)
    PREF_BOXES = ("tr_sat_cb", "tr_reflect_cb")

    def _prefs(self):
        prefs = {name: bool(getattr(self, name).isChecked())
                 for name in self.PREF_BOXES if hasattr(self, name)}
        if hasattr(self, "body_split"):       # the dragged IMAGES panel width
            prefs["left_w"] = int(self.body_split.want)
        return prefs

    def _apply_prefs(self, prefs):
        w = (prefs or {}).get("left_w")
        if isinstance(w, int) and hasattr(self, "body_split"):
            self.body_split.set_left(max(self.LEFT_MIN_W, min(self.LEFT_MAX_W, w)))
        for name, val in (prefs or {}).items():
            box = getattr(self, name, None)
            if name in self.PREF_BOXES and box is not None:
                box.setChecked(bool(val))

    def _save_session(self):
        """Persist the analyses so closing the app doesn't lose them."""
        import pickle
        tmp = None
        try:
            p = _session_path()
            # never let an empty run wipe a session that still holds analyses
            try:
                if (os.path.exists(p) and not self.results
                        and os.path.getsize(p) > 200):
                    return
            except OSError:
                pass
            data = {"version": SESSION_VERSION,
                    "results": self.results,
                    "chosen": self.chosen,
                    # view-only per-particle corrections, so labelled photos keep
                    # their applied labels across restarts (sets aren't picklable
                    # as-is via defaults, so store lists)
                    "class_overrides": self.class_overrides,
                    "view_excluded": {k: list(v) for k, v in self.view_excluded.items()},
                    "measure_include": {k: list(v) for k, v in self.measure_include.items()},
                    "measure_exclude": {k: list(v) for k, v in self.measure_exclude.items()},
                    "train_labels": self.train_labels,
                    "train_blank": list(self.train_blank),
                    "review_stats": {k: {n: list(s) for n, s in v.items()}
                                     for k, v in self.review_stats.items()},
                    "review_meta": self.review_meta,
                    # small UI choices worth surviving a restart — see _prefs
                    "prefs": self._prefs()}
            # 1) the whole new session goes to a temporary file first
            tmp = p + ".writing"
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            # 2) only now does the old file step aside. RENAMED into .bak, not
            #    copied: the session is hundreds of MB, and a copy meant reading
            #    and writing all of it on every save — painful on an 8 GB
            #    machine. A file that failed to load is NOT the last good
            #    session: it goes aside under its own name and .bak is kept.
            if os.path.exists(p):
                if getattr(self, "_session_damaged", False):
                    os.replace(p, p + time.strftime(".damaged-%Y%m%d-%H%M%S"))
                elif (getattr(self, "_session_loaded_version", SESSION_VERSION)
                        < 3 and not os.path.exists(p + ".pre-compact")):
                    # the first compact save: the last file an older app can
                    # read steps aside for good instead of into the rolling .bak
                    os.replace(p, p + ".pre-compact")
                else:
                    os.replace(p, p + ".bak")
            self._session_loaded_version = SESSION_VERSION
            # 3) the complete new file takes its place
            os.replace(tmp, p)
            tmp = None
            self._session_damaged = False
        except Exception:
            traceback.print_exc()      # never let saving break the app
        finally:
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _load_session(self):
        """Restore the previous session. Anything incompatible (old app
        version, moved files, unpicklable data) is silently discarded."""
        import pickle
        try:
            p = _session_path()
            if not os.path.exists(p):
                return
            data, restored = None, False
            for cand in (p, p + ".bak"):
                if not os.path.exists(cand):
                    continue
                try:
                    with open(cand, "rb") as f:
                        data = pickle.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("not a session")
                    restored = cand != p
                    break
                except Exception:
                    traceback.print_exc()
                    data = None
                    if cand == p:
                        # remembered for _save_session: this file must not be
                        # rotated over the good .bak
                        self._session_damaged = True
            if data is None:
                QtWidgets.QMessageBox.warning(
                    self, "Session could not be opened",
                    "The saved session and its backup could not be read, so the "
                    "app started without your previous analyses.\n\n"
                    f"Both files were kept untouched in:\n{os.path.dirname(p)}")
                return
            ver = data.get("version")
            if ver not in READABLE_VERSIONS:
                return
            self._session_loaded_version = ver
            # before anything that can bail out: the switches are independent of
            # whether any analysis survived the restore
            self._apply_prefs(data.get("prefs"))
            # translate any path the library healed (a file renamed on disk) so
            # its analysis and corrections follow it to the new name
            rmap = getattr(self, "_relink_map", {})
            def rk(k):
                return rmap.get(k, k)
            # Keep a result when its (healed) path resolves on disk OR is still a
            # row in the tree; only truly orphaned analyses (image removed by the
            # user) are dropped. (The micrograph is no longer stored with it —
            # Analysis.gray() re-reads the file — and the overlay always drew
            # from the file anyway.)
            lib_paths = self._all_paths()
            results = {}
            for k, v in data.get("results", {}).items():
                nk = rk(k)
                if os.path.exists(nk) or nk in lib_paths:
                    results[nk] = v
            if not results:
                return
            # Sessions written before the label masks were compacted hold them as
            # int64 — 12.7 MB each, four times what they need (591 MB of session
            # file, 614 MB resident for 43 analyses). Shrink on the way in, so the
            # memory drops now and the next save writes the small form.
            n_ghosts = 0
            for v in results.values():
                # only masks that arrived UNPACKED (v1/v2 files) can be wide;
                # a packed one was compacted before it was packed, and asking
                # for label_mask here would open all of them (see analyze._lm_get)
                m = v.__dict__.get("_lm")
                if m is not None:
                    c = analyze._compact_labels(m)
                    if c is not m:
                        v.label_mask = c
                # Bring old analyses up to the current MIN_COUNT floors. Purely a
                # filter over masks that do not change, so it gives exactly what a
                # fresh run would — without re-segmenting 36 photos.
                n_ghosts += analyze.apply_count_floors(v)
            if n_ghosts:
                self._ghosts_cleaned = n_ghosts
            self.results.update(results)
            self.chosen.update({rk(k): v for k, v in data.get("chosen", {}).items()})
            # Restore the user's hand corrections for any photo still KNOWN to the
            # app — its analysis came back, or the file is on disk, or it is simply
            # still a row in the library. They used to be kept only when the
            # analysis came back, which quietly threw away hours of clicking the
            # moment a photo's analysis was missing for any reason: the labels are
            # a few hundred bytes and the particle ids they key on come back
            # unchanged as soon as the photo is analysed again (same masks, from
            # the cache), so there is nothing to gain by dropping them and a lot
            # to lose. Same rule train_labels has always had, for the same reason.
            def keep(k):
                return k in results or k in lib_paths or os.path.exists(k)
            self.class_overrides.update({rk(k): v for k, v in
                                         data.get("class_overrides", {}).items()
                                         if keep(rk(k))})
            self.view_excluded.update({rk(k): set(v) for k, v in
                                       data.get("view_excluded", {}).items()
                                       if keep(rk(k))})
            self.measure_include.update({rk(k): set(v) for k, v in
                                         data.get("measure_include", {}).items()
                                         if keep(rk(k))})
            self.measure_exclude.update({rk(k): set(v) for k, v in
                                         data.get("measure_exclude", {}).items()
                                         if keep(rk(k))})
            self.train_labels.update({rk(k): v for k, v in
                                      data.get("train_labels", {}).items()
                                      if rk(k) in results or os.path.exists(rk(k))})
            self.train_blank.update(rk(q) for q in data.get("train_blank", [])
                                    if rk(q) in results or os.path.exists(rk(q)))
            self.review_stats.update(
                {rk(k): {n: set(s) for n, s in v.items()}
                 for k, v in data.get("review_stats", {}).items()
                 if rk(k) in results or os.path.exists(rk(k))})
            self.review_meta.update(
                {rk(k): v for k, v in data.get("review_meta", {}).items()
                 if rk(k) in results or os.path.exists(rk(k))})
            self._add_paths(list(results))
            # _add_paths only restyles when it actually adds a row, and on a normal
            # boot the library already holds every one of them — so the freshness
            # dots have to be asked for explicitly here.
            self._restyle_rows()
            msg = f"Restored {len(results)} analysis(es) from the last session."
            if restored:
                msg = ("The session file was damaged — restored "
                       f"{len(results)} analysis(es) from its backup.")
            if rmap:            # the local, not self. — the attribute only
                                # exists once _load_library has run, and this
                                # bare access aborted the tail of the restore
                msg += f"  Reconnected {len(rmap)} renamed file(s)."
            self.status.showMessage(msg)
        except Exception:
            traceback.print_exc()

    def closeEvent(self, e):
        # The red button and ⌘W HIDE the window; the app keeps running and the
        # Dock icon brings it back (user request, 2026-09-27: "like Artitect —
        # ⌘Q quits, the red button must not"). Only a real quit — ⌘Q, Quit in
        # the Dock or the app menu, logging out — gets past this; the window
        # marks those when the application's Quit event goes by (gui.eventFilter).
        if sys.platform == "darwin" and not getattr(self, "_quitting", False):
            e.ignore()
            self._hide_window()
            return
        # Never let the interpreter tear down a QThread that is still running —
        # that aborts the process on the way out (see _retire_worker). Stop
        # feeding the queue and join whatever is in flight; an analysis cannot be
        # interrupted mid-Cellpose, so closing during one waits for that image.
        self.queue = []
        for attr in ("worker", "_train_worker", "_sat_worker", "_scale_worker"):
            if getattr(self, attr, None) is not None:
                self.status.showMessage("Finishing the current run before closing…")
                QtWidgets.QApplication.processEvents(
                    QtCore.QEventLoop.ExcludeUserInputEvents)
                self._retire_worker(attr)
        self._save_session()
        self._save_library()
        super().closeEvent(e)
