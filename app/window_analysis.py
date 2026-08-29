"""Running Analyze: turning a selection into finished analyses.

The queue is deliberately one-image-at-a-time. Cellpose saturates the machine
on a single photo, so running two in parallel is slower AND makes the progress
line meaningless.

The worker handover is the part not to improvise on: a QThread whose last
Python reference is dropped while it is still running aborts the whole process
(three recorded crashes). Everything goes through _retire_worker, which joins
first and releases second.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets

import analyze
import training_store
from analyze import DEFAULT_FACET_THRESH
from background_workers import Worker


class AnalysisFlow:
    """Mixin: the Analyze queue and its results. `self` is the window."""

    # ---- analysis ----
    def analyze_selected(self):
        sel = self._selected_paths()
        if not sel and self.current:   # nothing ticked -> just the previewed image
            sel = [self.current]
        # pressing Analyze always re-runs the selected image(s) from scratch, even
        # if already analysed — that's how the user sees the effect of changed
        # rules/thresholds without having to delete and re-add the image. (The
        # Cellpose masks are cached, so a re-run only redoes the fast CNN steps.)
        if sel:
            if not self._confirm_discard_edits(sel):
                return
            self._run(sel)
        elif self.current in self.results:
            self._rerender()

    _EDIT_LABELS = [("class_overrides", "class corrections"),
                    ("view_excluded", "excluded particles"),
                    ("measure_include", "forced-in measurements"),
                    ("measure_exclude", "dropped measurements"),
                    ("train_labels", "training-mode labels on screen")]

    def _confirm_discard_edits(self, paths):
        """Re-analysing throws away every correction pinned on these photos —
        view-only AND training-mode alike (user rule, 2026-08-04: a stray
        "exclude everything" in review used to survive re-analysis, because the
        segmentation is fresh but particle ids restart from 1 too, so the old
        per-id labels silently landed on whatever new particle now holds that
        id — indistinguishable from the app being stuck). Say so first when
        there are any; silent loss of an evening's clicking is exactly the kind
        of thing this app has been bitten by before.

        Labels already CONFIRMED into the training set are the one thing that
        stays safe — they live in the training folder on disk, not in these
        in-memory stores, and "Restore saved version" brings them back onto
        the new segmentation afterwards."""
        counts, photos = {}, set()
        for attr, label in self._EDIT_LABELS:
            store = getattr(self, attr)
            n = 0
            for p in paths:
                v = store.get(p)
                if v:
                    n += len(v)
                    photos.add(p)
            if n:
                counts[label] = n
        if not counts:
            return True
        detail = "\n".join(f"   •  {n} {label}" for label, n in counts.items())
        n_conf = sum(1 for p in paths if training_store.saved(p))
        keep = (f"\n\nPhotos already confirmed into the training set are safe "
                f"either way ({n_conf} of these) — their saved labels stay on "
                f"disk; use \"Restore saved version\" afterwards to bring them "
                f"back."
                if n_conf else "")
        r = QtWidgets.QMessageBox.question(
            self, "Re-analyze",
            f"Re-analysing decides {'this photo' if len(photos) == 1 else f'these {len(photos)} photos'} "
            f"again from scratch, so the corrections you made by hand on "
            f"{'it' if len(photos) == 1 else 'them'} are discarded:\n\n{detail}{keep}\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes)
        return r == QtWidgets.QMessageBox.Yes

    def _run(self, paths, do_class=None, do_pattern=None):
        if not paths or self.worker is not None:
            return
        # Analyze evaluates everything; callers may still switch a step off —
        # loading labelled photos skips the pattern CNN because the saved labels
        # override it anyway.
        self._do_class = True if do_class is None else do_class
        self._do_pattern = True if do_pattern is None else do_pattern
        self.queue = list(paths)
        self._prog_total = len(paths)
        self._prog_done = 0
        self._scale_warnings = []
        self._empty_warnings = []
        self._next()

    def _next(self):
        if not self.queue:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("🔬  Analyze")
            self.busy.setVisible(False)
            self.status.showMessage("Analysis complete.")
            # A finished batch is the moment fresh, labelled particles exist, so
            # it is the moment the model-space thresholds can be re-derived (see
            # calibrate.py). Nothing is re-inferred — it counts over what was
            # just computed — and it goes to the status bar, not a dialog.
            msg = self._recalibrate_thresholds(quiet=True)
            if msg:
                self.status.showMessage(msg)
            self._refresh_results()
            self._save_session()          # keep results across restarts
            warns = []
            if self._scale_warnings:
                warns.append("Scale bar could not be read (sizes shown as 0 nm):\n  "
                             + ", ".join(self._scale_warnings))
            if self._empty_warnings:
                warns.append("No particles detected:\n  " + ", ".join(self._empty_warnings))
            if warns:
                QtWidgets.QMessageBox.warning(self, "Warning", "\n\n".join(warns))
            return
        path = self.queue.pop(0)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳  Analyzing…")
        # The AI model loads on the very first analysis of a session, and on a new
        # computer that first load compiles GPU kernels — it can take a minute or
        # two while nothing else appears to happen. Say so LOUDLY (a moving bar +
        # an explicit note), or a new user thinks the app has frozen and gives up.
        first_run = analyze._model is None
        self.busy.setVisible(True)
        if first_run:
            self.busy.setFormat("Loading the AI model — first run, please wait…")
            self.status.showMessage(
                "Loading the AI model. On a new computer the FIRST analysis can "
                "take 1–2 minutes; every analysis after this is fast. Please wait…")
        else:
            self.busy.setFormat(
                f"Analyzing {self._prog_done + 1} / {self._prog_total}…")
            self.status.showMessage(
                f"Analyzing: {os.path.basename(path)}  "
                f"({self._prog_done + 1}/{self._prog_total})")
        self.worker = Worker(path, self._do_class, self._do_pattern)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _retire_worker(self, attr="worker"):
        """Drop a finished QThread — but JOIN it first.

        Letting the last Python reference go while the thread object is still
        alive is what killed the app at random ("beni ara ara uygulamadan
        atıyor"): PySide then destroys a QThread that has not finished, Qt calls
        qFatal("QThread: Destroyed while thread is still running") and the process
        aborts (SIGABRT). It is a race — the slot runs on the GUI thread while
        run() is still returning — which is why it only bit now and then. Two
        crash reports (2026-07-29 22:35, 2026-07-30 03:40) have exactly this
        stack. wait() returns immediately here (the signal is emitted from the
        last line of run()), it just makes the hand-over ordered.
        """
        w = getattr(self, attr, None)
        setattr(self, attr, None)
        if w is not None:
            w.wait()

    def _on_done(self, path, analysis):
        analysis.reclassify(DEFAULT_FACET_THRESH)
        self.results[path] = analysis
        self.chosen.setdefault(path, [])   # measurements are click-picked now
        self._forget_undo(path)            # fresh segmentation = fresh particle ids
        # Re-analysing means "decide this photo again", and it means it for
        # EVERYTHING the user had pinned on top of the model (user rule,
        # 2026-07-31, EXTENDED 2026-08-04). It used to keep the hand-made CLASS
        # corrections while resetting the measurement ones, and that split is
        # what let a photo drift into showing two contradictory answers at
        # once: UÖ-15 Stripe 17 carried 551 normal-mode class corrections that
        # disagreed with the user's own TRAINING labels on 142 particles (72 of
        # them "stripe in training, janus on screen"), so the screen showed 148
        # stripes while the model — trained on those very training labels —
        # said 247, and the model looked broken when it was simply displaying
        # an older opinion.
        #
        # train_labels is dropped too now (2026-08-04: "review modunda
        # yanlışlıkla bütün parçacıkları exclude ettim, tekrar analyze edince
        # hepsi exclude olarak kaldı"). Segmentation ids restart from 1 on a
        # fresh Analyze, so old per-id labels don't just go stale, they land on
        # whatever DIFFERENT particle now holds that id — a silent mislabel,
        # not a safe no-op. Already-CONFIRMED labels are unaffected: they live
        # in the training folder on disk (training_store), not here, and
        # "Restore saved version" reapplies them onto the new segmentation
        # (matching by id, warning if some no longer exist — see
        # TrainingMode._train_restore).
        self.measure_include.pop(path, None)
        self.measure_exclude.pop(path, None)
        self.class_overrides.pop(path, None)
        self.view_excluded.pop(path, None)
        self.train_labels.pop(path, None)
        self.train_blank.discard(path)
        self.review_stats.pop(path, None)
        self.review_meta.pop(path, None)
        self._train_undo = [u for u in self._train_undo if u[0] != path]
        self._retire_worker()
        self._prog_done += 1
        if not analysis.nm_per_px:            # scale bar could not be read
            self._scale_warnings.append(os.path.basename(path))
        if not analysis.particles:            # nothing segmented
            self._empty_warnings.append(os.path.basename(path))
        if path == self.current:
            self._rerender(); self._update_class_controls(analysis)
        self._style_row_for(path)             # its freshness dot turns green
        self._refresh_results()
        # Checkpoint a long batch. The session used to be written only when the
        # queue ran dry, so anything that ended the process mid-run (the crashes
        # above, or macOS reclaiming memory on an 8 GB machine) threw away an
        # hour of segmentation. Every few images costs well under a second.
        if self._prog_done % 5 == 0 and self.queue:
            self._save_session()
        self._next()

    def _on_failed(self, path, tb):
        self._retire_worker()
        self._prog_done += 1
        QtWidgets.QMessageBox.critical(self, "Error", f"{os.path.basename(path)}:\n\n{tb}")
        self._next()
