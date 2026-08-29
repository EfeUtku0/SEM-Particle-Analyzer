"""Training mode: teaching the model, and everything the right-hand panel says
about how far that has got.

The difference from window_particle_edit is the whole point: a click here is
GROUND TRUTH. It is written to the visible training folder on the Desktop and
the next "Train model" run learns from it, so the panel has to be honest about
what is saved, what is only on screen, and what the model would have said
instead.

Two invariants worth keeping:
  * pre-filling from the model honours the measurement gate. Blue particles
    (analyze.measurable() is False) are not labelled and not shown as labelled
    — the user should never have to judge a particle the app will not measure.
    An explicit click still overrides the gate.
  * the saved version and the screen are compared by signature, not by
    guesswork, so "in the training set, but edited since" is a fact rather
    than an assumption. See training_store.signature.
"""
from __future__ import annotations

import os
import traceback

from PySide6 import QtCore, QtWidgets

import analyze
import review_queue
import training_store
import golden_store
from overlay_draw import PATTERN_COLORS, TRAIN_COLORS, GREEN, UNMEASURED
from window_pattern_size import ClassSizeWindow
from background_workers import TrainWorker, SaturationWorker
from widget_library_tree import (FRESH_STALE, TR_NONE, TR_NEW, TR_SAVED,
                                 TR_EDITED)


class TrainingMode:
    """Mixin: labelling, the training panel, and running a retrain."""


    def _toggle_train_mode(self, on):
        self.train_mode = on
        self.right_stack.setCurrentIndex(1 if on else 0)
        for sc in self._train_shortcuts:
            sc.setEnabled(on)
        self._sync_pattern_enable()          # View pattern filters work in training
        # the IMAGES panel now answers a different question (see LibraryDelegate)
        self.tree.train_marks = on
        if not on:
            # _toggle_review bails out once train_mode is already False, so the
            # review's own state is cleared here rather than left hanging
            self.review_mode = self.accept_mode = False
            self.tr_review_cb.setChecked(False)
            self.tr_accept_cb.setChecked(False)
            self.tr_accept_cb.setVisible(False)
        self._restyle_rows()
        if on:
            self._train_refresh_confirmed()
            self._train_update_panel()
            self.status.showMessage(
                "Training mode — pick a class in Adjustments (1–6, 0 = exclude), "
                "click particles; the View checkboxes filter classes; "
                "Space hides the overlay; ⌘Z undoes.")
        else:
            self.status.showMessage("Training mode off.")
            self._refresh_results()
        self._rerender()

    def _train_effective(self, a, path=None):
        """{pid: (class, from_user)}: the model's current view of every particle,
        overridden by the user's clicks. 'exclude' drops the particle entirely.

        The prefill is limited to particles the app MEASURES, which is what makes
        the two modes agree (user report, 2026-08-04: "aynı görselde normal modda
        644 parçacık ölçülmüş, training modda 751 labelled var. Neden?"). On MU 5
        the 107 extra were all particles the size gate had thrown out — 91 bitten
        or buried under a neighbour, 12 sliced by the frame, 4 thin slivers — and
        labelling the pattern of a particle that is not fully in view is exactly
        the thing the gate exists to prevent. The pattern gate is deliberately
        looser than the size gate (see analyze.measurable), so this bites: 8% of
        the prefill across the library. That is the point.

        What remains is an exact, explainable identity: measured = prefilled +
        solid-with-no-pattern (28793 = 26678 + 2115 over the whole library), so
        the panels can no longer disagree without a reason that can be named.

        The user's OWN clicks below are applied unconditionally: an explicit
        label always outranks the gate, exactly as `measurable` itself honours
        `user_measurable`.

        `path` defaults to the photo on screen; it is passed explicitly when the
        panel needs another photo's state (the training marks in the IMAGES
        tree)."""
        path = path or self.current
        out = {}
        if a.classifiable and path not in self.train_blank:
            for p in a.particles:
                if getattr(p, "excluded", False):
                    continue                       # model dropped it -> unlabelled
                if not analyze.measurable(p):
                    continue                       # not fully in view -> no label
                if p.is_solid and p.pattern:
                    out[p.id] = (p.pattern, False)
                elif not p.is_solid:
                    out[p.id] = ("undercooled", False)
        for pid, cls in self.train_labels.get(path, {}).items():
            if cls == "exclude":
                out.pop(pid, None)
            else:
                out[pid] = (cls, True)
        return out

    # ---- training labels -> normal mode (one way only) -------------------
    def _reflect_training(self, path):
        """Copy this photo's TRAINING labels onto its normal-mode view.

        One way, deliberately. Training labels are the user's ground truth —
        "analizleri hep normal moddan yapıyorum, doğru veriyle yapmış olayım" —
        so they should reach the panel they actually work in. Normal-mode
        corrections must NOT flow back: they are transient view edits (see
        _view_analysis) and letting them into training data would teach the
        model from clicks the user never meant as truth.

        Only the particles TRAINING actually names are copied. The rest of the
        photo is left alone, so a normal-mode correction on some other particle
        survives — and there is nothing to copy for the untouched ones anyway,
        since the training prefill is just the model's answer, which is what
        normal mode already shows.

        Returns how many particles were written.
        """
        labels = self.train_labels.get(path) or {}
        if not labels:
            return 0
        ov = self.class_overrides.setdefault(path, {})
        exc = self.view_excluded.setdefault(path, set())
        n = 0
        for pid, cls in labels.items():
            if cls == "exclude":
                # the training word for "leave this one out" is the normal-mode
                # exclude: out of the counts, still drawn as not-measured
                exc.add(pid)
                ov.pop(pid, None)
            else:
                exc.discard(pid)
                ov[pid] = cls
            n += 1
        return n

    def _reflect_training_now(self):
        """The Training panel's button: apply this photo's labels to normal mode
        without waiting for it to be confirmed into the training set."""
        if self.current is None:
            return
        self._push_undo()          # ⌘Z, like every other edit to these stores
        n = self._reflect_training(self.current)
        if not n:
            self.status.showMessage("No training labels on this photo yet.")
            return
        self._rerender()
        self._refresh_results()
        self.status.showMessage(
            f"{n} training label(s) applied to normal mode for this photo. "
            "Normal-mode edits still never travel back into training.")

    # ---- Class × Size distribution ----
    @staticmethod
    def _classsize_pairs_model(target):
        """(class, diam_nm) for every classified particle in the model's output.
        Only CBS images are classified; ETD ones are skipped."""
        analyses = getattr(target, "analyses", None) or [target]
        pairs = []
        for a in analyses:
            if not getattr(a, "classifiable", False):
                continue
            for p in a.particles:
                if not getattr(p, "diam_nm", 0):
                    continue
                if getattr(p, "excluded", False):
                    continue               # dim/unreliable -> not in the distribution
                if not p.is_solid:
                    pairs.append(("undercooled", p.diam_nm))
                elif p.pattern:
                    pairs.append((p.pattern, p.diam_nm))
                else:                      # solid but no pattern (edge/occluded/small)
                    pairs.append(("solid", p.diam_nm))
        return pairs

    def _open_classsize(self):
        # In training mode the top Class × Size button reflects the TRAINING
        # result (the user's corrected ground-truth labels for the current
        # photo); otherwise it's the model's output over the selected images.
        if self.train_mode:
            a = self.results.get(self.current)
            if a is None:
                self.status.showMessage("Analyze this image first.")
                return
            eff = self._train_effective(a)
            diam = {p.id: p.diam_nm for p in a.particles}
            pairs = [(cls, diam[pid]) for pid, (cls, _) in eff.items()
                     if pid in diam and diam[pid]]
            if not pairs:
                self.status.showMessage("No labelled particles in this photo yet.")
                return
            title = os.path.basename(getattr(a, "image", "") or self.current or "photo")
            self._cs_win = ClassSizeWindow(pairs, title, ground_truth=True, parent=self)
            self._cs_win.show()
            return
        target = getattr(self, "_result_target", None) or self.results.get(self.current)
        if target is None:
            self.status.showMessage("Analyze an image first.")
            return
        pairs = self._classsize_pairs_model(target)
        if not pairs:
            self.status.showMessage("No classified (CBS) particles in the selection.")
            return
        title = os.path.basename(getattr(target, "image", "") or "selection")
        self._cs_win = ClassSizeWindow(pairs, title, ground_truth=False, parent=self)
        self._cs_win.show()
    # ---- training set: saved version vs. what is on screen -------------------
    def _train_restore(self):
        """Put back exactly the labels this photo was added to the training set
        with. The safety net for the mis-click: a stray class key on a photo that
        was already confirmed is otherwise invisible until the next retrain.

        The saved labels are re-applied as explicit clicks on a blanked photo
        (train_blank), NOT as "model pre-fill + the clicks that were saved". That
        is what makes the restore exact even when the pre-fill itself has moved
        since — a retrained model predicts differently, and rebuilding from it
        would quietly hand back a different set than the one that was saved.
        """
        path = self.current
        saved = training_store.saved(path) if path else None
        if not saved:
            self.status.showMessage("This photo isn't in the training set yet.")
            return
        a = self.results.get(path)
        if a is None or not a.classifiable:
            self.status.showMessage("Analyze this image first.")
            return
        cur = self._train_signature(path, a)
        if cur == saved["signature"]:
            self.status.showMessage(
                "Already showing the version in the training set.")
            return
        ids = {p.id for p in a.particles}
        missing = [pid for pid in list(saved["labels"]) + list(saved["excluded"])
                   if pid not in ids]
        n_saved = len(saved["labels"]) + len(saved["excluded"])
        if missing and n_saved:
            # The particle ids come from the segmentation, so they only mean the
            # same thing while it does. If the photo has been re-segmented since,
            # say so rather than silently restoring labels onto other particles.
            if QtWidgets.QMessageBox.question(
                    self, "Restore saved version",
                    f"{len(missing)} of {n_saved} saved labels point at particles "
                    f"this photo no longer has — it was re-analysed with a "
                    f"different segmentation since it was added.\n\n"
                    f"Restore the {n_saved - len(missing)} that still match?") \
                    != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        elif QtWidgets.QMessageBox.question(
                self, "Restore saved version",
                f"Replace the labels on screen with the {n_saved} this photo was "
                f"added to the training set with"
                + (f" on {saved['confirmed']}?" if saved["confirmed"] else "?")) \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        ov = {pid: cls for pid, cls in saved["labels"].items() if pid in ids}
        ov.update({pid: "exclude" for pid in saved["excluded"] if pid in ids})
        self.train_labels[path] = ov
        self.train_blank.add(path)              # only these labels, nothing else
        self._train_undo = [u for u in self._train_undo if u[0] != path]
        self._rerender()
        self.status.showMessage(
            f"Restored the saved version — {len(ov)} labels"
            + (f" ({len(missing)} skipped)" if missing else "") + ".")
    def _train_click(self, x, y):
        if not self.train_mode:
            return
        if self.review_mode and self.accept_mode:
            # Accept mode owns the click while it is on — that is the whole
            # point of it (most of the model's "mistakes" turn out to be right,
            # and picking the matching class tool for each one is the work it
            # exists to remove). Untick it to go back to hand-labelling.
            self._review_accept(x, y)
            return
        cls = self.click_tool
        if cls in (None, "measure"):
            if self.review_mode:
                # with no class picked, a click in review mode ASKS instead of
                # labelling: "what did I say here, and what does the model say?"
                self._review_report(x, y)
                return
            self.status.showMessage(
                "Pick a class in Adjustments (1–6, 0 = exclude), then click a particle.")
            return
        a = self.results.get(self.current)
        if a is None:
            self.status.showMessage("Analyze this image first, then label it.")
            return
        if not a.classifiable:
            self.status.showMessage("Pattern labels need a CBS image.")
            return
        masks = a.label_mask
        iy, ix = int(y), int(x)
        if masks is None or not (0 <= iy < masks.shape[0] and 0 <= ix < masks.shape[1]):
            return
        pid = int(masks[iy, ix])
        if pid == 0:
            return                                    # background click
        if cls == "solid":
            # 'Solid' isn't a training class: resolve it to the pattern the model
            # would assign this now-solid particle.
            cls = self._resolve_solid(pid)
            if cls is None:
                return
            if cls == "solid":
                # ...and it couldn't read one. "Solid, pattern unknown" teaches the
                # net nothing, so it is a view-only state (normal mode), never a
                # training label — the user has to name the pattern themselves.
                self.status.showMessage(
                    "The model can't read a pattern here — pick the pattern "
                    "yourself (1–4), or 0 to exclude it from training.")
                return
        # Any SEGMENTED particle can be labelled — the user's judgement overrides
        # the model. (The model itself won't auto-assign a pattern to edge/occluded
        # particles — see analyze.pattern_eligible — but if the user sees a clear
        # pattern the model missed, they set it here. 0 = exclude drops it.)
        ov = self.train_labels.setdefault(self.current, {})
        self._train_undo.append((self.current, pid, ov.get(pid)))
        self._mark_undo_push()        # keeps ⌘Z ordering right against a removal
        ov[pid] = cls
        self._rerender()

    def _train_undo_last(self):
        if not self._train_undo:
            return
        path, pid, prev = self._train_undo.pop()
        ov = self.train_labels.setdefault(path, {})
        if prev is None:
            ov.pop(pid, None)
        else:
            ov[pid] = prev
        if path == self.current:
            self._rerender()

    _DISP = ["undercooled", "janus", "stripe", "lamellar", "composite"]

    @staticmethod
    def _model_class(p):
        if p is None:
            return ""            # mask id with no particle behind it (filtered out)
        if getattr(p, "excluded", False):
            return "excluded"
        if not p.is_solid:
            return "undercooled"
        if p.pattern:
            return p.pattern
        return "solid"

    # ---- training panel ------------------------------------------------------
    _TR_STATE_STYLE = {
        TR_NEW: ("#e8963c", "Not in the training set yet",
                 "label it, then “Add photo to training set”"),
        TR_SAVED: ("#3fae74", "In the training set", ""),
        TR_EDITED: ("#2b6fff", "Edited since you added it",
                    "add it again to save the change, or restore the saved one"),
    }

    def _train_state_line(self):
        """The state chip at the top of the training card + the buttons that
        depend on it (Restore only exists on a photo that has drifted)."""
        state = self._train_state(self.current)
        self.tr_restore_btn.setVisible(state == TR_EDITED)
        self.tr_accept_cb.setVisible(self.review_mode)
        # A golden photo says so ON the button, before it is pressed. The save
        # refuses either way (training_store.save_confirmed), but a refusal
        # after the click is a worse way to learn it than a label that was
        # never offering in the first place.
        gold = bool(self.current) and golden_store.is_golden(self.current)
        self.tr_confirm_btn.setEnabled(not gold)
        self.tr_confirm_btn.setText(
            "🔒   Golden photo — never trained on" if gold
            else "✓   Add photo to training set")
        self.tr_confirm_btn.setToolTip(
            "This photo measures the model's accuracy, which only works while "
            "the model has never seen it. Move it out of the golden folder to "
            "train on it." if gold else "")
        if state == TR_NONE:
            self.tr_state.setText("")
            self.tr_state.setVisible(False)
            return state
        col, title, hint = self._TR_STATE_STYLE[state]
        saved = training_store.saved(self.current) if state != TR_NEW else None
        when = (saved or {}).get("confirmed") or ""
        if state == TR_SAVED and when:
            hint = f"added {when}"
        self.tr_state.setVisible(True)
        # one table cell, not a styled div: Qt's rich text paints a div's
        # background per line box, so a two-line chip came out as two bands
        self.tr_state.setText(
            f"<table width='100%' cellspacing='0' cellpadding='0'>"
            f"<tr><td bgcolor='#f1f4f8' style='padding:7px 10px;'>"
            f"<span style='color:{col};font-size:14px;'>●</span>&nbsp;&nbsp;"
            f"<span style='color:#2c3442;font-size:11.5px;font-weight:800;'>"
            f"{title}</span>"
            + (f"<div style='color:#8a95a1;font-size:11px;font-weight:600;"
               f"margin:1px 0 0 19px;'>{hint}</div>" if hint else "")
            + self._review_progress_html()
            + "</td></tr></table>")
        return state

    def _review_progress_html(self):
        """How much of this photo has been through the bulk review, and how
        often the model was wrong on the part that was checked.

        The correction rate is the useful half. It is the model's measured error
        rate on this photo over particles a human actually looked at — an honest
        number, unlike anything computed against the model's own output, and the
        signal for whether it is worth carrying on reviewing this photo at all.
        """
        st = self.review_stats.get(self.current)
        if not st or not st.get("seen"):
            return ""
        a = self.results.get(self.current)
        total = sum(1 for p in (a.particles if a else [])
                    if review_queue.particle_class(p))
        seen, fixed = len(st["seen"]), len(st["fixed"])
        pct = f" ({100.0 * fixed / seen:.0f}%)" if seen else ""
        of = f" of {total}" if total else ""
        return (f"<div style='color:#8a95a1;font-size:11px;font-weight:600;"
                f"margin:3px 0 0 19px;'>▦&nbsp; reviewed {seen}{of} &nbsp;·&nbsp; "
                f"<span style='color:#2b6fff;'>{fixed} corrected{pct}</span>"
                f"{self._review_sure_html(st)}</div>")

    def _review_sure_html(self, st):
        """The half of the correction rate that says whether the model is merely
        imprecise or actually broken: how often it was wrong on the particles it
        was MOST sure about. Blank until enough of the photo has been reviewed
        for the top half to mean anything."""
        meta = self.review_meta.get(self.current) or {}
        v = [(pid, m) for pid, m in meta.items()
             if pid in st["seen"] and "rank" in m]
        if len(v) < 20:
            return ""
        v.sort(key=lambda kv: kv[1]["rank"])
        top = v[len(v) // 2:]                  # the more-confident half
        wrong = sum(1 for pid, _ in top if pid in st["fixed"])
        r = 100.0 * wrong / len(top)
        col = "#c2410c" if r >= 15 else "#8a95a1"
        return (f"<br><span style='color:{col};'>&nbsp;&nbsp;&nbsp;&nbsp;"
                f"{r:.0f}% wrong where the model was most confident</span>")

    def _train_update_panel(self):
        self._train_state_line()
        a = self.results.get(self.current)
        if a is None:
            self.tr_info.setText("Analyze an image, then click particles to label.")
            return
        if not a.classifiable:
            self.tr_info.setText("This image isn't CBS — patterns can't be labelled here.")
            return
        if self.review_mode:
            self._review_update_panel(a)
            return
        blank = self.current in self.train_blank
        ov = self.train_labels.get(self.current, {})
        eff = self._train_effective(a)

        # model's own prediction per particle, and the final (effective) labels
        model = {k: 0 for k in self._DISP + ["solid", "excluded"]}
        for p in a.particles:
            model[self._model_class(p)] += 1
        final = {k: 0 for k in self._DISP}
        for cls, _fu in eff.values():
            if cls in final:
                final[cls] += 1

        user_excl = sum(1 for c in ov.values() if c == "exclude")
        # dim particles the model dropped that the user re-labelled as a class
        rescued = {}
        for p in a.particles:
            if getattr(p, "excluded", False):
                c = ov.get(p.id)
                if c and c != "exclude":
                    rescued[c] = rescued.get(c, 0) + 1
        n_rescued = sum(rescued.values())

        COL = {"undercooled": GREEN, **PATTERN_COLORS}

        def chip(c):
            r, g, b = COL[c]
            return f"<span style='color:rgb({r},{g},{b});'>■</span>"

        rows = ""
        for c in self._DISP:
            m, f = model[c], final[c]
            if m == f:
                you = f"<span style='color:#1a2129;font-weight:700;'>{f}</span>"
            else:
                you = (f"<span style='color:#1a2129;font-weight:700;'>{f}</span>"
                       f"&nbsp;<span style='color:#2b6fff;font-weight:700;'>"
                       f"{f-m:+d}</span>")
            rows += (
                f"<tr>"
                f"<td style='padding:4px 0;'>{chip(c)}&nbsp;&nbsp;"
                f"<span style='color:#3a4351;font-weight:600;'>{c.capitalize()}</span></td>"
                f"<td align='right' style='padding:4px 10px;color:#aab2bc;'>{m}</td>"
                f"<td align='right' style='padding:4px 0;'>{you}</td>"
                f"</tr>")

        blank_badge = ""
        if blank:
            blank_badge = ("&nbsp;&nbsp;<span style='color:#c08a30;"
                           "font-weight:700;'>· from scratch</span>")

        def stat(label, value, color="#1a2129"):
            return (f"<tr>"
                    f"<td style='padding:3px 0;color:#7a8492;'>{label}</td>"
                    f"<td align='right' style='padding:3px 0;color:{color};"
                    f"font-weight:700;'>{value}</td></tr>")

        # The blue ones, named and counted. They are the answer to "why does this
        # panel say fewer than the photo has?" — the app does not measure them,
        # so it does not label them and the review never offers them, and the
        # user does not have to judge a pattern they cannot see (2026-08-04).
        n_meas = sum(1 for p in a.particles if analyze.measurable(p))
        # EXACTLY the set painted slate blue by both overlays — `not measurable`,
        # nothing added or taken away. The count and the picture have to be the
        # same thing or the number is worse than no number at all. The
        # model-dropped ones are a SUBSET, shown indented rather than as a second
        # figure that appears to add up with this one.
        n_unmeas = sum(1 for p in a.particles if not analyze.measurable(p))
        ur, ug, ub = UNMEASURED
        blue_chip = f"<span style='color:rgb({ur},{ug},{ub});'>■</span>&nbsp;&nbsp;"
        sub = "<span style='color:#aab2bc;'>&nbsp;&nbsp;&nbsp;&nbsp;↳&nbsp;</span>"
        secondary = (
            f"<table width='100%' cellspacing='0' cellpadding='0' "
            f"style='font-size:11.5px;'>"
            + stat(blue_chip + "Not measured — skip these", n_unmeas)
            + stat(sub + "too dim to judge", model['excluded'], "#7a8492")
            + stat("Pattern unclear", model['solid'])
            + stat("Excluded by you", user_excl)
            + "</table>")

        rescue_line = ""
        if n_rescued:
            parts = ", ".join(f"{k.capitalize()} {v}" for k, v in rescued.items())
            rescue_line = (
                f"<div style='margin-top:12px;padding:8px 10px;"
                f"background:#e7f2ec;border-radius:8px;color:#2f6b48;"
                f"font-size:11.5px;font-weight:600;'>"
                f"Rescued from model &nbsp;<span style='font-weight:800;'>"
                f"{n_rescued}</span>"
                f"<span style='color:#5f9c78;font-weight:600;'> &nbsp;·&nbsp; "
                f"{parts}</span></div>")

        html = (
            f"<div style='font-size:20px;font-weight:800;color:#1a2129;'>{a.n}"
            f"<span style='font-size:12px;font-weight:600;color:#8a95a1;'>"
            f"&nbsp;particles</span></div>"
            f"<div style='font-size:11.5px;color:#8a95a1;font-weight:600;"
            f"margin-top:1px;'>{n_meas} measured &nbsp;·&nbsp; "
            f"{len(eff)} labelled &nbsp;·&nbsp; "
            f"{len(ov)} edits{blank_badge}</div>"
            # The CLASS / MODEL / YOU table used to sit here. Removed on the
            # user's call (2026-08-04, "sağ taraf çok kalabalık oldu"): it was
            # the tallest block in the panel and the same comparison is on the
            # micrograph itself, in colour, while labelling. `rows` is still
            # built above — putting it back is one line — because the counts
            # behind it feed the rest of this panel.
            f"<div style='border-top:1px solid #dce1e7;margin:12px 0 10px;'></div>"
            f"{secondary}"
            f"{rescue_line}")
        self.tr_info.setText(html)

    def _review_update_panel(self, a):
        """The review panel: how far the model's current answer is from the
        ground truth this photo carries, and — the useful part — WHICH way it is
        wrong, so a systematic mistake ("calls undercooled particles janus")
        shows up as one line instead of having to be noticed particle by
        particle."""
        truth, in_set = self._review_truth(self.current)
        if not truth:
            self.tr_info.setText(
                "<div style='color:#6a7484;font-size:12px;'>Nothing to compare "
                "yet — label some particles, or add this photo to the training "
                "set, then turn the review back on.</div>")
            return
        pairs = self._review_pairs(a)
        checked = sum(1 for p in a.particles if p.id in truth)
        agree = checked - len(pairs)
        pct = (agree / checked) if checked else 0.0
        # A second reading the user asked for: the model REFUSING to name a
        # pattern is not the same kind of miss as it naming the wrong one — a
        # blank costs nothing downstream, a wrong pattern does. So the same score
        # is also given with those dropped from both sides, which is the number
        # that says how often it is wrong when it does commit.
        nopat = sum(1 for _t, m in pairs.values() if m == "solid")
        checked2 = checked - nopat
        pct2 = (agree / checked2) if checked2 else 0.0

        counts = {}
        for t, m in pairs.values():
            counts[(t, m)] = counts.get((t, m), 0) + 1

        def chip(c):
            r, g, b = TRAIN_COLORS.get(c, (150, 158, 168))
            name = self._REVIEW_NAME.get(c, c).capitalize()
            return (f"<span style='color:rgb({r},{g},{b});'>■</span>&nbsp;"
                    f"<span style='color:#3a4351;font-weight:600;'>{name}</span>")

        rows = ""
        for (t, m), n in sorted(counts.items(), key=lambda kv: -kv[1]):
            rows += (f"<tr>"
                     f"<td style='padding:4px 0;'>{chip(t)}</td>"
                     f"<td align='center' style='padding:4px 6px;color:#aab2bc;'>→</td>"
                     f"<td style='padding:4px 0;'>{chip(m)}</td>"
                     f"<td align='right' style='padding:4px 0;color:#1a2129;"
                     f"font-weight:700;'>{n}</td></tr>")
        if not rows:
            rows = ("<tr><td style='padding:6px 0;color:#3fae74;font-weight:700;'>"
                    "The model agrees with every label on this photo.</td></tr>")

        notes = []
        if not in_set:
            notes.append("This photo isn't in the training set — compared "
                         "against your own clicks on it.")
        if self._fresh_state(self.current) == FRESH_STALE:
            notes.append("Analysed with an earlier model — press Analyze to "
                         "compare against the current one.")
        note = "".join(
            f"<div style='margin-top:10px;padding:7px 9px;background:#fdf4e6;"
            f"border-radius:8px;color:#8a6224;font-size:11px;font-weight:600;'>"
            f"{n}</div>" for n in notes)

        self.tr_info.setText(
            f"<div style='font-size:20px;font-weight:800;color:#1a2129;'>"
            f"{len(pairs)}"
            f"<span style='font-size:12px;font-weight:600;color:#8a95a1;'>"
            f"&nbsp;disagreements</span></div>"
            f"<div style='font-size:11.5px;color:#8a95a1;font-weight:600;"
            f"margin-top:1px;'>{agree} of {checked} labels match &nbsp;·&nbsp; "
            f"<span style='color:{'#3fae74' if pct >= 0.9 else '#c08a30'};"
            f"font-weight:700;'>{pct:.0%}</span></div>"
            + (f"<div style='font-size:11.5px;color:#8a95a1;font-weight:600;"
               f"margin-top:2px;'>{agree} of {checked2} ignoring the "
               f"{nopat} it left without a pattern &nbsp;·&nbsp; "
               f"<span style='color:{'#3fae74' if pct2 >= 0.9 else '#c08a30'};"
               f"font-weight:700;'>{pct2:.0%}</span></div>" if nopat else "")
            + f"<div style='margin:14px 0 4px;'>"
            f"<table width='100%' cellspacing='0' cellpadding='0' "
            f"style='font-size:12.5px;'>"
            f"<tr style='color:#a0a9b4;font-size:10.5px;font-weight:700;'>"
            f"<td style='padding-bottom:4px;'>YOU</td><td></td>"
            f"<td style='padding-bottom:4px;'>MODEL SAYS</td>"
            f"<td align='right' style='padding-bottom:4px;'>N</td></tr>"
            f"{rows}</table></div>"
            f"<div style='color:#8a95a1;font-size:11px;font-weight:600;"
            f"margin-top:10px;'>The painted particles are the disagreements; "
            f"the colour is what the MODEL says. Click one to read it, or label "
            f"it (1–4, 5, 0) to correct it.</div>"
            f"{note}")

    def _save_training_photo(self, path):
        """Write one photo's labels into the training folder.

        Split out of `_train_confirm` because the bulk review labels MANY photos
        at once while that button only ever saved the one on screen — labels for
        every other photo stayed in session.pkl, invisible to `Train model`. A
        912-particle review of MU 2 was lost that way (found 2026-08-04), which
        is the whole reason this takes a path.

        Returns (n_pattern_crops, n_labels, n_excluded), or None when there is
        nothing labelled on it. Raises whatever the save raised."""
        a = self.results.get(path)
        if a is None or not a.classifiable:
            return None
        eff = self._train_effective(a, path)
        excluded = [pid for pid, cls in self.train_labels.get(path, {}).items()
                    if cls == "exclude"]
        if not eff and not excluded:
            return None
        ncrops, ntot = training_store.save_confirmed(
            a, eff, excluded=excluded, meta=self.review_meta.get(path))
        return ncrops, ntot, len(excluded)

    def _train_confirm(self):
        a = self.results.get(self.current)
        if a is None or not a.classifiable:
            self.status.showMessage("Analyze a CBS image first."); return
        try:
            res = self._save_training_photo(self.current)
        except golden_store.GoldenPhotoError as exc:
            # Not an error the user made — say what the photo is FOR, and what
            # to do if they meant it, rather than just refusing.
            QtWidgets.QMessageBox.information(
                self, "Golden photo",
                f"{exc}\n\nGolden photos are the only honest measure of the "
                f"model's accuracy: it is scored on them precisely because it "
                f"has never trained on them. Training on this one would spend "
                f"that.\n\nIf you do want it in the training set, move its "
                f"files from the golden folder to the training folder — then "
                f"it stops counting towards the score.")
            return
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Save failed", traceback.format_exc())
            return
        if res is None:
            self.status.showMessage("Nothing labelled on this image yet."); return
        ncrops, ntot, nexc = res
        exmsg = f", {nexc} excluded" if nexc else ""
        # confirming is the moment these labels BECOME the ground truth for this
        # photo, so it is the moment to put them in front of the panel the user
        # analyses from (see _reflect_training)
        nref = 0
        if self.tr_reflect_cb.isChecked():
            nref = self._reflect_training(self.current)
        refmsg = f", {nref} applied to normal mode" if nref else ""
        self.status.showMessage(
            f"Added {a.image}: {ntot} labels{exmsg}{refmsg}, {ncrops} pattern "
            f"crops → {training_store.train_dir()}")
        self._train_refresh_confirmed()
        # the row's dot goes green (and the state chip with it) the moment the
        # save lands — that IS the confirmation the user was missing
        self._train_state_line()
        self._style_row_for(self.current)
        # visible confirmation: flash the button, so the user isn't left
        # guessing whether the click registered. AFTER _train_state_line, which
        # also writes this button's label now (the golden lock) and would
        # otherwise wipe the flash the moment it was set.
        self.tr_confirm_btn.setText("✓   Added to training set")
        QtCore.QTimer.singleShot(1900, lambda: self.tr_confirm_btn.setText(
            "✓   Add photo to training set"))

    def _train_refresh_confirmed(self):
        """Update just the Train button's ready-state / photo count (the confirmed
        photos list was removed — the training folder button opens them)."""
        n = len(training_store.confirmed())
        need = training_store.TRAIN_MIN_PHOTOS
        self.tr_train_btn.setEnabled(n >= need and self._train_worker is None)
        # the button sits in a narrow row now, so the photo count that used to be
        # in the label lives in the tooltip until it is the thing standing in the
        # way (n < need), when it goes back on the button
        self.tr_train_btn.setText("🧠  Train model" if n >= need
                                  else f"🧠  Train  ({n}/{need})")
        self.tr_train_btn.setToolTip(
            f"Retrain the pattern model on your {n} confirmed photos"
            if n >= need else
            f"{n} of {need} photos added — add {need - n} more to train")
        # The golden set's size on the report button, because a ruler nobody
        # knows exists is a ruler nobody keeps: without this the panel gives no
        # sign that the accuracy is measured on anything at all.
        try:
            import golden_store
            ng, by = golden_store.summary()
        except Exception:
            ng, by = 0, {}
        self.tr_report_btn.setText(
            f"📋  Model raporu  ({ng} golden)" if ng else "📋  Model raporu")
        if ng:
            from dialog_train_report import _INSTRUMENT
            mix = ", ".join(f"{v} × {_INSTRUMENT.get(k, k)}"
                            for k, v in sorted(by.items()))
            self.tr_report_btn.setToolTip(
                f"Doğruluk {ng} golden fotoğrafta ölçülüyor ({mix}) — model "
                f"bunları hiç görmez. Sınıf sınıf recall / precision / F1, "
                f"karışıklık matrisi ve geçmiş eğitimler sekme sekme.")

    def _recalibrate_thresholds(self, quiet=False):
        """Re-derive the probability-space thresholds from the labelled photos.

        Deliberately NOT per-instrument: one set of numbers serves every
        microscope. What makes that possible is the data, not a special case —
        a gate trained on both machines wants nearly the same cut for both,
        while one trained on a single machine does not (see thresholds.py).

        Only FRESH analyses are used: a particle's P(solid) came from whichever
        model analysed it, and fitting the current model's threshold against an
        older model's numbers is the precise mistake this whole mechanism exists
        to prevent.
        """
        try:
            import calibrate
            import analyze

            def labels_of(path):
                saved = training_store.saved(os.path.basename(path))
                if not saved:
                    return {}
                out = {}
                for r in saved.get("labels", []):
                    try:
                        out[int(r["id"])] = r.get("class")
                    except (KeyError, TypeError, ValueError):
                        continue
                return out

            before = analyze.pipeline_stamp()
            values, report = calibrate.run(self.results, labels_of, before)
            if values:
                self._restamp_after_calibration(before)
        except Exception as exc:
            return f"Threshold calibration skipped: {exc}"
        if quiet and not values:
            return ""
        head = ("Thresholds recalibrated: " if values
                else "Thresholds checked against your labels: ")
        return head + "  ·  ".join(report)

    def _restamp_after_calibration(self, old_stamp):
        """Bring the analyses that were fresh a moment ago back up to date.

        New thresholds change the pipeline stamp, so without this every analysis
        the calibration was just fitted ON would be marked stale by the act of
        fitting — and re-running Analyze would produce a new calibration, which
        would stale them again. There is nothing to re-infer: reclassify() reads
        the per-particle numbers already stored and applies the new cuts, which
        is exactly what a fresh run would do.
        """
        import analyze
        new_stamp = analyze.pipeline_stamp()
        for path, a in self.results.items():
            if getattr(a, "pipeline", "") == old_stamp:
                a.reclassify(analyze.DEFAULT_FACET_THRESH)
                a.pipeline = new_stamp
        self._restyle_rows()
        if self.current in self.results:
            self._rerender()

    # ------------------------------------------------------------ saturation

    def _open_saturation(self):
        """Show the learning curve. Opens on whatever was last measured — the
        window is useful with nothing in it too, since it is where the
        measurement is started from."""
        from dialog_saturation import SaturationWindow
        w = getattr(self, "_sat_win", None)
        if w is None:
            w = self._sat_win = SaturationWindow(self, on_measure=self._sat_go)
        else:
            w.refresh()
        w.show(); w.raise_(); w.activateWindow()
        if getattr(self, "_sat_worker", None) is not None:
            w.measuring(True, 0, 1)

    def _sat_go(self):
        """Measure the curve without retraining. Refuses to run beside a
        training run: both want the GPU, and two of these at once is the slowest
        possible way to get either."""
        if getattr(self, "_sat_worker", None) is not None:
            return
        if self._train_worker is not None:
            QtWidgets.QMessageBox.information(
                self, "Training is running",
                "Wait for the training run to finish — the curve is measured on "
                "the same device and the two would only slow each other down.")
            return
        self.tr_sat_btn.setEnabled(False)
        w = SaturationWorker(); self._sat_worker = w
        w.progress.connect(self._sat_progress)
        w.done.connect(self._sat_done)
        w.failed.connect(self._sat_failed)
        w.start()
        if getattr(self, "_sat_win", None):
            self._sat_win.measuring(True, 0, 1)

    def _sat_progress(self, done, total):
        if getattr(self, "_sat_win", None):
            self._sat_win.measuring(True, done, total)

    def _sat_done(self, rec):
        self._retire_worker("_sat_worker")
        self.tr_sat_btn.setEnabled(True)
        if getattr(self, "_sat_win", None):
            self._sat_win.measuring(False)
            self._sat_win.refresh(rec)
        else:
            self._open_saturation()

    def _sat_failed(self, msg):
        self._retire_worker("_sat_worker")
        self.tr_sat_btn.setEnabled(True)
        if getattr(self, "_sat_win", None):
            self._sat_win.measuring(False)
            self._sat_win.status.setText(f"Measurement failed: {msg}")
        else:
            QtWidgets.QMessageBox.critical(self, "Data saturation", msg)

    def _train_go(self):
        if self._train_worker is not None:
            return
        if getattr(self, "_sat_worker", None) is not None:
            QtWidgets.QMessageBox.information(
                self, "Saturation measurement is running",
                "Wait for it to finish — it is using the same device.")
            return
        self.tr_train_btn.setEnabled(False)
        self.tr_cancel_btn.setVisible(True); self.tr_cancel_btn.setEnabled(True)
        self.tr_prog.setVisible(True); self.tr_prog.setValue(0)
        self.tr_metrics.setText("Preparing dataset…")
        w = TrainWorker(saturation=self.tr_sat_cb.isChecked())
        self._train_worker = w
        w.progress.connect(self._train_progress)
        w.done.connect(self._train_done)
        w.failed.connect(self._train_failed)
        w.cancelled.connect(self._train_cancelled)
        w.start()

    def _train_cancel(self):
        """The 'X' next to Train model — for a click that was a mistake.
        Cooperative: the worker only checks between epochs, so this can take a
        moment to actually land, and the button says so rather than looking
        stuck."""
        if self._train_worker is not None:
            self._train_worker.cancel()
            self.tr_cancel_btn.setEnabled(False)
            self.tr_metrics.setText("Cancelling — finishing the current epoch…")

    def _train_progress(self, done, total, phase):
        self.tr_prog.setMaximum(total); self.tr_prog.setValue(done)
        if phase.startswith("golden"):        # not epochs — a pass over the photos
            n = phase.split(":", 1)[1] if ":" in phase else ""
            self.tr_metrics.setText(
                f"Model saved. Scoring it on the golden set{' — ' + n if n else ''}…")
        elif phase == "saturation":
            # the model is already trained and saved by now; say so, or this
            # last stretch reads as "still training" and invites a cancel
            self.tr_metrics.setText(
                "Model saved. Measuring data saturation — training small models "
                "on 20 – 100 % of your photos…")
        else:
            self.tr_metrics.setText(f"Training ({phase})…  {done}/{total} epochs")

    def _train_done(self, res):
        self._retire_worker("_train_worker")
        self.tr_prog.setVisible(False)
        self.tr_cancel_btn.setVisible(False)
        import model_pattern
        model_pattern.reload()
        # existing analyses hold the OLD model's predictions; dropping the flag
        # makes the next "Start analysis" genuinely re-run them with the new one
        for a in self.results.values():
            a.evaluated_pattern = False
        # ...and the new weights change analyze.pipeline_stamp(), so every row in
        # the panel now shows the amber "analysed with an earlier model" dot until
        # it is re-run. That mark is the only thing that made this visible: the
        # flag above has never been shown anywhere.
        self._restyle_rows()
        # The headline is the golden score — photos this model can never have
        # seen. When there is no golden set the panel says so plainly instead of
        # falling back to a number measured on the training labels, which is the
        # number this whole change exists to stop reporting.
        g = ((res.get("golden") or {}).get("combined") or {})
        if g.get("macro_f1") is not None:
            gs = res["golden"]
            head = (f"Done — golden set macro-F1 {g['macro_f1']:.1%}, accuracy "
                    f"{g['acc']:.1%} on {gs['photos']} photos it has never seen")
        elif res.get("golden_error"):
            head = f"Done — model retrained (golden score failed: {res['golden_error']})"
        else:
            head = ("Done — model retrained. No golden set yet, so its accuracy "
                    "is unmeasured; see the report for how to make one.")
        line2 = "New analyses use the retrained model; re-run Analyze to see it."
        if res.get("golden_conflicts"):
            line2 = ("⚠ In BOTH folders, so the score is flattered: "
                     + ", ".join(res["golden_conflicts"]) + "\n" + line2)
        # The thresholds are cuts through THIS model's output, so a retrain
        # invalidates them (see thresholds.py). Re-derive them from the user's
        # own labels now, while the panel is already reporting — it costs no
        # inference, only counting over particles already analysed.
        cal = self._recalibrate_thresholds()
        if cal:
            line2 += "\n\n" + cal
        if res.get("saturation"):
            import model_pattern_curve as curve
            v = curve.verdict(res["saturation"])
            line2 += f"\n\nData saturation: {v['text']}"
            if getattr(self, "_sat_win", None):
                self._sat_win.refresh(res["saturation"])
        elif res.get("saturation_error"):
            line2 += (f"\n\nData saturation could not be measured: "
                      f"{res['saturation_error']}")
        self.tr_metrics.setText(
            f"{head}  ·  {res['n']} particles total.\n{line2}")
        self._train_refresh_confirmed()
        # The full table, automatically: the panel has room for one sentence,
        # and the per-class numbers are the reason the run was scored at all.
        self._open_train_report()

    # -------------------------------------------------------- model report

    def _open_train_report(self):
        """The model report window, rebuilt from the stored history each time so
        it can never show a run that has since been pruned."""
        from dialog_train_report import TrainReportWindow
        w = getattr(self, "_report_win", None)
        if w is None:
            w = self._report_win = TrainReportWindow(self)
        else:
            w.refresh()
        w.show(); w.raise_(); w.activateWindow()

    def _train_failed(self, msg):
        self._retire_worker("_train_worker")
        self.tr_prog.setVisible(False)
        self.tr_cancel_btn.setVisible(False)
        self.tr_metrics.setText(f"Training failed: {msg}")
        self._train_refresh_confirmed()

    def _train_cancelled(self):
        """Nothing was written to disk (TrainCancelled fires before the model is
        saved) — the model on disk, and the panel state, are exactly as they
        were before this run started."""
        self._retire_worker("_train_worker")
        self.tr_prog.setVisible(False)
        self.tr_cancel_btn.setVisible(False)
        self.tr_metrics.setText("Training cancelled — nothing was changed.")
        self._train_refresh_confirmed()
