"""Driving the contact-sheet review, and clearing corrections again.

The dialog itself is dialog_review_sheet; this is what the window does around
it — which particles to queue, what to do with the answers, and how to report
what the round revealed.

The reporting is the point that is easy to miss: the queue is ordered
least-confident first, so corrections should cluster at the START. Corrections
still arriving at the END mean the model is not merely unsure at the edges, it
is wrong systematically — which is exactly what the second instrument's photos
turned out to be. _review_rank_note is what says so out loud.

Review answers are also written straight through to the training folder. They
used to stop at the session, and a photo with 327 hand-corrections never
reached the model.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

import review_queue
import training_store
import golden_store
from chart_data import CLASS_ORDER, CLASS_LABELS, CLASS_COLORS as COLORS
from dialog_review_sheet import ReviewDialog
from widget_tiles import CheckMenu


class ReviewFlow:
    """Mixin: the review round and the Clear menu. `self` is the window."""


    _CLEAR_CLASSES = ["janus", "stripe", "composite", "lamellar",
                      "undercooled", "exclude"]

    def _build_clear_menu(self):
        """Populate the Temizle menu on demand: clear one class's labels, or all
        (all also blanks the model's pre-fill so the photo can be labelled from
        scratch — for a golden test set)."""
        m = self._clear_menu
        m.clear()
        ov = self.train_labels.get(self.current, {})
        per = {}
        for c in ov.values():
            per[c] = per.get(c, 0) + 1
        for cls in self._CLEAR_CLASSES:
            n = per.get(cls, 0)
            lab = "Exclude" if cls == "exclude" else cls.capitalize()
            act = m.addAction(f"Clear {lab} labels  ({n})")
            act.setEnabled(n > 0)
            act.triggered.connect(lambda _=False, c=cls: self._clear_class(c))
        m.addSeparator()
        blank = self.current in self.train_blank
        if blank:
            act = m.addAction("↩  Restore model predictions")
            act.triggered.connect(self._restore_model_view)
        else:
            act = m.addAction("🧹  Clear all · label from scratch")
            act.triggered.connect(self._clear_all)

    # ---- contact-sheet review ------------------------------------------------
    def _open_review(self):
        """Review every particle the model gave one class, across the photos
        selected in IMAGES (or all analysed ones if the selection has none)."""
        paths = [p for p in self._selected_paths() if p in self.results]
        analyses = {p: self.results[p] for p in paths} or dict(self.results)
        analyses = {p: a for p, a in analyses.items()
                    if getattr(a, "classifiable", False)}
        if not analyses:
            self.status.showMessage("Analyze an image first."); return
        # Two counts, because the menu offers two different rounds: what is
        # still waiting, and everything the class holds including what has
        # already been confirmed once (see _pick_review_class).
        skip = {p: set(v) for p, v in (self.train_labels or {}).items()}
        pending = self._review_counts(analyses, skip)
        every = self._review_counts(analyses)
        if not every:
            self.status.showMessage("Nothing classified to review yet."); return
        # Offer the classes that exist, commonest label first, with their counts.
        order = [c for c in CLASS_ORDER if c in every] + \
                [c for c in every if c not in CLASS_ORDER]
        chosen, again = self._pick_review_class(order, pending, every,
                                                len(analyses))
        if chosen is None:
            return
        # the review now edits the normal-mode stores too, so it has to be one
        # step on the ⌘Z stack like any other correction
        self._push_undo()
        dlg = ReviewDialog(self, analyses, chosen, include_done=again)
        dlg.exec()
        self._rerender()
        self._train_state_line()
        n, changed, nph = getattr(dlg, "_written", (0, 0, 0))
        if n:
            self._review_finished(dlg, n, changed, nph)

    @staticmethod
    def _review_counts(analyses, skip=None):
        """How many particles each class holds, optionally minus the ones the
        user has already given a label to. `skip` is {path: {pid}}."""
        counts = {}
        for path, a in analyses.items():
            done = (skip or {}).get(path) or set()
            for p in a.particles:
                if p.id in done:
                    continue
                c = review_queue.particle_class(p)
                if c:
                    counts[c] = counts.get(c, 0) + 1
        return counts

    def _pick_review_class(self, order, pending, every, n_photos):
        """Which of the model's answers to check — as a menu that opens ON the
        Review button, not a dialog in the middle of the screen.

        The old QInputDialog appeared centred, so every review began with a trip
        across the window and back (user, 2026-08-04: "mouse'u oraya kadar
        götürmeyeyim"). A popup anchored to the button puts the choice under the
        cursor that just clicked it, and each row is a full-width menu item —
        a bigger target than a list row plus an OK button.

        Returns (class key, include_done), or (None, False) if dismissed.
        """
        menu, acts, again = self._build_review_menu(order, pending, every,
                                                    n_photos)
        picked = menu.exec(self._review_menu_pos(menu))
        return acts.get(picked), bool(again.isChecked())

    def _build_review_menu(self, order, pending, every, n_photos):
        """The menu itself — built apart from opening it, so its contents can be
        checked without a modal loop that no test can answer.

        The first row is a CHECKABLE switch, not a class: "also the ones I have
        already confirmed". A review used to be one-way — once a particle
        carried a label it never came back, so a decision made too fast could
        not be revisited (user, 2026-08-04: "review'ları sonradan düzenlemek
        isterim"). Ticking it puts every particle of the class back in the
        queue, showing YOUR last answer on each tile rather than the model's.
        It is a CheckMenu so ticking it does not close the menu — the counts
        update in place and the next click is the class you want.
        """
        menu = CheckMenu(self)
        head = menu.addAction(f"{n_photos} photo(s) — check which answer?")
        head.setEnabled(False)
        again = menu.addAction("↻   Also the ones I already confirmed")
        again.setCheckable(True)
        menu.addSeparator()
        acts = {}
        for c in order:
            a = menu.addAction("")
            col = COLORS.get(c)          # the class's own colour, as the overlay
            if col:
                pm = QtGui.QPixmap(11, 11)
                pm.fill(QtGui.QColor(*col) if isinstance(col, tuple)
                        else QtGui.QColor(col))
                a.setIcon(QtGui.QIcon(pm))
            acts[a] = c

        def relabel():
            """Row text follows the switch, so the number on the row is always
            the number of tiles the click would actually open."""
            src = every if again.isChecked() else pending
            for act, c in acts.items():
                n = src.get(c, 0)
                done = every.get(c, 0) - pending.get(c, 0)
                extra = (f"   · {done} confirmed" if done and not again.isChecked()
                         else "")
                act.setText(f"{CLASS_LABELS.get(c, c.title())}   ({n}){extra}")
                act.setEnabled(n > 0)

        again.toggled.connect(lambda _=False: relabel())
        relabel()
        return menu, acts, again

    def _review_menu_pos(self, menu):
        """Where it opens: bottom-left of the Review button, growing UPWARDS.

        Downwards would put it under the review sheet's own window, which opens
        over the middle of the screen.
        """
        btn = getattr(self, "tr_review_btn", None)
        if btn is None:
            return QtGui.QCursor.pos()
        pos = btn.mapToGlobal(QtCore.QPoint(0, 0))
        pos.setY(pos.y() - menu.sizeHint().height() - 2)
        return pos

    def _review_finished(self, dlg, n, changed, nph):
        """Close out a review: say what it found, then get it onto disk.

        Saving every reviewed photo is the load-bearing half, and it happens
        WITHOUT asking (user rule, 2026-08-04: "Training set'e ekle uyarısı hiç
        sorulmasın"). Until 2026-08-04 this message just told the user to press
        "Add photo to training set" — a button that saves only the photo on
        screen — after a review that had labelled up to thirty. Anything they
        didn't then visit one by one never reached the training folder and
        `Train model` never saw it, silently; a 912-particle review of MU 2 was
        lost that way. A confirmation prompt was the first fix and was wrong
        too: the user reviewed the page in order to teach the model, so saving
        is the thing they already asked for, and a dialog that can be dismissed
        is one more way to lose the work."""
        paths = [p for p in dlg.accepted if p in self.results]
        ok, crops, failed, golden = 0, 0, [], []
        for p in paths:
            try:
                res = self._save_training_photo(p)
            except golden_store.GoldenPhotoError:
                # protected, not broken: the review itself still stands (the
                # corrections are in train_labels and on screen), it simply
                # does not become training data
                golden.append(os.path.basename(p)); continue
            except Exception:
                failed.append(os.path.basename(p)); continue
            if res:
                ok += 1
                crops += res[0]
        self._train_refresh_confirmed()
        self._train_state_line()
        for p in paths:
            self._style_row_for(p)
        # NO dialog (user, 2026-08-04: "işlemden sonra şu da çıkmasın"). The
        # report still has to exist — silence is what let MU 2 go missing — so it
        # is written into the training panel's own readout, where the training
        # run's results land too. Nothing to dismiss, nothing to click through
        # on the way to the next review, and it stays on screen instead of
        # vanishing with an OK.
        saved = (f"<br>Saved to the training set: <b>{ok}</b> photo(s), "
                 f"{crops} pattern crops.")
        if golden:
            saved += ("<br><span style='color:#7a3517;'>Left out of the "
                      "training set (golden — never trained on, so the accuracy "
                      "figure stays honest): " + ", ".join(golden[:8]) + "</span>")
        if failed:
            saved += ("<br><span style='color:#c2410c;'>Could NOT save: "
                      + ", ".join(failed[:8]) + "</span>")
        note = self._review_rank_note(dlg)
        panel = getattr(self, "tr_metrics", None)
        if panel is not None:
            # left on AutoText on purpose: this message is HTML and renders as
            # such, while the training run's later plain-text lines keep their
            # newlines. Forcing RichText here would flatten those into one line.
            panel.setText(
                f"<b>{n}</b> particle(s) confirmed across <b>{nph}</b> photo(s), "
                f"of which <b>{changed}</b> were corrected." + note + saved)
        self.status.showMessage(
            f"Added {ok} photo(s), {crops} pattern crops → "
            f"{training_store.train_dir()}")

    @staticmethod
    def _review_rank_note(dlg):
        """Where the corrections fell in the confidence ordering.

        The list is least-confident-first precisely so mistakes cluster at the
        front and the user can stop when they dry up. Corrections still coming
        in the last quarter mean the model is wrong where it was SUREST, which
        is a different and much worse illness than being wrong at the margin —
        and the one thing this dialog was never reporting."""
        v = [x for per in dlg.accepted.values() for x in per.values()]
        if len(v) < 40:
            return ""                    # too few to split into quarters
        v.sort(key=lambda x: x["rank"])
        q = len(v) // 4
        first, last = v[:q], v[-q:]
        rate = lambda s: 100.0 * sum(1 for x in s if x["cls"] != x["model_cls"]) / len(s)
        f, l = rate(first), rate(last)
        note = (f"<br><br>By confidence: <b>{f:.0f}%</b> corrected in the least "
                f"confident quarter, <b>{l:.0f}%</b> in the most confident one.")
        if l >= 15.0 and l >= 0.5 * f:
            note += ("<br><span style='color:#c2410c;'>The model is still wrong "
                     "where it was surest — that points at a systematic problem "
                     "(wrong instrument calibration, a class it has never seen), "
                     "not at ordinary borderline cases. Worth reviewing further "
                     "rather than stopping here.</span>")
        return note

    def _apply_review(self, accepted, save=False):
        """Write reviewed pages in as USER labels.

        `train_blank` is set on every photo touched, and that is the load-bearing
        part: without it `_train_effective` would top the reviewed labels up with
        the model's own answer for every particle nobody looked at, which is the
        self-confirmation loop this whole feature exists to avoid.

        The same verdicts are ALSO written to the normal-mode stores
        (class_overrides / view_excluded) so the correction shows up where the
        user is looking: the RESULTS counts and the overlay. Training labels and
        the display would otherwise disagree silently, which is the exact
        failure the 2026-07-31 audit spent a round diagnosing ("the model has
        ruined stripe" turned out to be the screen showing one store while
        training held another). Writing both keeps them in step from the start.

        `save` only on the way out: session.pkl is a couple of hundred MB, so
        writing it once per accepted page would put a visible stall between every
        page of a review that is supposed to be fast.
        """
        n = changed = 0
        for path, per in accepted.items():
            ov = self.train_labels.setdefault(path, {})
            disp = self.class_overrides.setdefault(path, {})
            exc = self.view_excluded.setdefault(path, set())
            meta = self.review_meta.setdefault(path, {})
            for pid, v in per.items():
                cls, model_cls = v["cls"], v["model_cls"]
                ov[pid] = cls
                n += 1
                changed += (cls != model_cls)
                # what the model said and how sure it was, kept for the label
                # file (see training_store.save_confirmed) — the verdict alone can't
                # be re-derived once the model moves
                meta[pid] = {k: v[k] for k in ("model_cls", "conf", "rank",
                                               "n_items")}
                # mirror into the view stores, whose vocabulary differs: an
                # exclude lives in its own set, everything else is a class name
                if cls == "exclude":
                    exc.add(pid)
                    disp.pop(pid, None)
                else:
                    disp[pid] = cls
                    exc.discard(pid)
            self.train_blank.add(path)
            # Sets, not counters: this runs again for every page accepted, each
            # time with the CUMULATIVE dict, so anything additive would count
            # page 1 once more on every page that follows. Sets also make the
            # totals survive a second review pass over the same photo.
            st = self.review_stats.setdefault(path, {"seen": set(), "fixed": set()})
            for pid, v in per.items():
                st["seen"].add(pid)
                (st["fixed"].add if v["cls"] != v["model_cls"]
                 else st["fixed"].discard)(pid)
        if save and n:
            self._save_session()
        return n, changed

    def _clear_class(self, cls):
        ov = self.train_labels.get(self.current)
        if not ov:
            return
        gone = [pid for pid, c in ov.items() if c == cls]
        for pid in gone:
            ov.pop(pid, None)
        self._train_undo = [u for u in self._train_undo
                            if not (u[0] == self.current and u[1] in gone)]
        self._rerender()
        self.status.showMessage(f"Cleared {len(gone)} “{cls}” label(s).")

    def _restore_model_view(self):
        self.train_blank.discard(self.current)
        self._rerender()
        self.status.showMessage("Model predictions restored for this photo.")
    # ---- review: where does the model still disagree with the ground truth? --
    # The model's own answer per particle in words the panel can print. "solid"
    # is the model saying "crystalline, but I can't read a pattern"; "excluded"
    # is it dropping the particle as too dim/unclear to judge.
    _REVIEW_NAME = {"solid": "no pattern", "excluded": "dropped",
                    "exclude": "excluded", "": "unlabelled"}

    def _review_truth(self, path):
        """(truth {pid: class}, in_set) — what this photo is judged against.

        The base is the labels the photo was ADDED to the training set with:
        those are the ones the model was actually taught, and they don't drift
        as the model is retrained. The user's own clicks on the photo are laid
        on top, so a particle they have just re-labelled (or accepted the
        model's answer on) stops being a disagreement immediately instead of
        arguing with a version they have already moved past.

        Outside the training set there is still something worth comparing — the
        clicks alone — so the review works before a photo is added too."""
        saved = training_store.saved(path) if path else None
        truth = {}
        if saved:
            truth.update(saved["labels"])
            truth.update({pid: "exclude" for pid in saved["excluded"]})
        truth.update(self.train_labels.get(path, {}))
        return truth, bool(saved)

    def _review_pairs(self, a, path=None):
        """{pid: (your class, the model's class)} for every labelled particle the
        model now answers differently — the whole point of the review_queue."""
        path = path or self.current
        truth, _in_set = self._review_truth(path)
        if not truth:
            return {}
        out = {}
        for p in a.particles:
            t = truth.get(p.id)
            if t is None:
                continue                      # never labelled -> nothing to check
            m = self._model_class(p)
            if t == "exclude" and m == "excluded":
                continue                      # both say "don't use this one"
            if t != m:
                out[p.id] = (t, m)
        return out

    def _review_report(self, x, y):
        """Read out one particle's comparison in the status bar."""
        a, pid = self._particle_at(x, y)
        if a is None or pid is None:
            return
        truth, _in_set = self._review_truth(self.current)
        t = truth.get(pid)
        m = self._model_class(next((p for p in a.particles if p.id == pid), None))

        def name(c):
            return self._REVIEW_NAME.get(c, c or "unlabelled").capitalize()
        if t is None:
            self.status.showMessage(
                f"#{pid} — not one of your labels · model: {name(m)}")
        elif t == m or (t == "exclude" and m == "excluded"):
            self.status.showMessage(f"#{pid} — you and the model agree: {name(t)}")
        else:
            self.status.showMessage(
                f"#{pid} — you: {name(t)}   ·   model: {name(m)}")

    def _toggle_review(self, on):
        if not self.train_mode:
            return
        self.review_mode = on
        self.tr_accept_cb.setVisible(on)
        if not on and self.tr_accept_cb.isChecked():
            self.tr_accept_cb.setChecked(False)     # -> _toggle_accept
        if on:
            self.status.showMessage(
                "Review — the painted particles are the ones the model answers "
                "differently from your labels; the colour is the MODEL's answer. "
                "Click one to read the comparison, or label it to correct it.")
        self._rerender()

    def _toggle_accept(self, on):
        self.accept_mode = on
        if on:
            self.status.showMessage(
                "Accept mode — click a painted particle to take the model's "
                "answer as your label. Add the photo to the training set again "
                "when you're done.")

    def _review_accept(self, x, y):
        """Take the model's answer on one particle as the user's own label."""
        a, pid = self._particle_at(x, y)
        if a is None or pid is None:
            return
        truth, _in_set = self._review_truth(self.current)
        t = truth.get(pid)
        m = self._model_class(next((p for p in a.particles if p.id == pid), None))

        def name(c):
            return self._REVIEW_NAME.get(c, c or "unlabelled").capitalize()
        if m == "solid":
            # "crystalline, pattern unreadable" is not a class the net can be
            # taught (see _train_click) — there is nothing here to accept
            self.status.showMessage(
                f"#{pid} — the model reads no pattern here, so there is nothing "
                f"to accept. Name it yourself (1–4) or leave your label.")
            return
        cls = "exclude" if m == "excluded" else m
        if t == cls or (t == "exclude" and m == "excluded"):
            self.status.showMessage(f"#{pid} — already {name(cls)}; nothing to change.")
            return
        ov = self.train_labels.setdefault(self.current, {})
        self._train_undo.append((self.current, pid, ov.get(pid)))
        ov[pid] = cls
        self._rerender()
        self.status.showMessage(
            f"#{pid} — took the model's answer: {name(t)} → {name(cls)}. "
            f"Add the photo to the training set to save it.")

    def _clear_all(self):
        if not self.train_mode:
            return
        a = self.results.get(self.current)
        if a is None or not a.classifiable:
            self.status.showMessage("Analyze a CBS image first.")
            return
        ov = self.train_labels.get(self.current, {})
        if ov and QtWidgets.QMessageBox.question(
                self, "Clear all",
                f"Delete all {len(ov)} labels on this photo and hide the model's "
                "predictions (to label from scratch)?") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.train_labels.pop(self.current, None)
        self._train_undo = [u for u in self._train_undo if u[0] != self.current]
        self.train_blank.add(self.current)
        self._rerender()
        self.status.showMessage("Cleared — label from scratch; only your clicks are saved.")
