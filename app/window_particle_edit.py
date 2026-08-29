"""Correcting one particle at a time, on the image itself.

These are VIEW-ONLY corrections: clicking a class key repaints that particle
and changes what the RESULTS panel counts, but it teaches the model nothing —
that is what Training mode is for, and the Guide says so explicitly.

Two rules here are easy to break and were argued out with the user:
  * class and measurement are SEPARATE. Naming a particle never drags it into
    the size statistics, and dropping one from the statistics never costs it
    its class. The only way in or out of the statistics is the Measure tool:
    left-click admits, right-click drops.
  * undo is a SNAPSHOT, not an inverse. One class click can touch three stores
    at once, so ctrl-Z restores a copy of the whole per-photo edit state.
"""
from __future__ import annotations

import os
import dataclasses

from PySide6 import QtWidgets

import analyze
from overlay_draw import PATTERN_COLORS, GREEN, RED
from dialog_guide import GuideDialog


class ParticleEditor:
    """Mixin: the click tools and per-particle corrections. `self` is the window."""


    # ---- normal-mode click tools (measure / class-paint), view-only ----
    def _typing(self):
        """True while the user is in a text box — the single-key shortcuts (class
        digits, M, C, Space) and ⌘Z must stay out of the way then. Checks the
        search box by name as well as the focus widget, so a shortcut can never
        hijack a keystroke meant for it."""
        return (isinstance(QtWidgets.QApplication.focusWidget(), QtWidgets.QLineEdit)
                or getattr(self, "search", None) is not None and self.search.hasFocus())

    def _key_class(self, cls):
        """Class/measure keyboard shortcut, routed by mode."""
        if self._typing():
            return                                 # typing digits in a box
        # both modes use the Adjustments tool selector; only the click ACTION
        # differs (train label vs view-only correction), handled in _on_view_click
        self._toggle_click_tool(cls)

    def _toggle_click_tool(self, key):
        """Clicking the active tool again turns it off (no tool selected)."""
        self._set_click_tool(None if key == self.click_tool else key)

    def _set_click_tool(self, key):
        self.click_tool = key
        # reflect the single active tool (or none) in the button row
        for k, b in self._tool_btns.items():
            b.setChecked(k == key)
        if key is None:
            self.status.showMessage("No tool selected — clicks on the image do nothing.")
        elif key == "measure":
            self.status.showMessage(
                "Measure: click a particle to show its size, click again to hide. "
                "Blue particles are left out of the size statistics (buried or "
                "frame-cut) — left-click puts one back in, right-click drops a "
                "measured one out. Naming a class never changes this.")
        elif key == "solid":
            self.status.showMessage(
                "Solid: click a particle to force it solid — the model then picks "
                "its pattern automatically, or leaves it solid without one if no "
                "pattern is clear. It is never excluded.")
        elif key == "certainty":
            self.status.showMessage(
                "Certainty: click a particle to show the model's confidence in its "
                "current class; click again to hide it.")
        elif self.train_mode:
            self.status.showMessage(
                f"Click a particle to label it “{key}” (training).")
        else:
            self.status.showMessage(
                f"Click a particle to mark it “{key}” (view-only correction).")
        self._rerender()

    def _particle_at(self, x, y):
        a = self.results.get(self.current)
        if a is None:
            return None, None
        masks = a.label_mask
        iy, ix = int(y), int(x)
        if masks is None or not (0 <= iy < masks.shape[0] and 0 <= ix < masks.shape[1]):
            return a, None
        pid = int(masks[iy, ix])
        return a, (pid if pid else None)

    def _on_view_click(self, x, y):
        # Certainty is an inspection tool: it reads the model's confidence and
        # works the same in both modes (never writes a label / correction).
        if self.click_tool == "certainty":
            a, pid = self._particle_at(x, y)
            if a is not None and pid is not None:
                self._toggle_certainty(pid)
            return
        if self.train_mode:
            # Measure works in training mode too: pick the center Measure tool and
            # clicks measure size; otherwise clicks label with the training class.
            if self.click_tool == "measure":
                a, pid = self._particle_at(x, y)
                if a is not None and pid is not None:
                    self._toggle_measure(a, pid)
            else:
                self._train_click(x, y)
            return
        tool = self.click_tool
        if tool is None:
            return
        a, pid = self._particle_at(x, y)
        if a is None or pid is None:
            return
        if tool == "measure":
            self._toggle_measure(a, pid)
        else:
            self._set_particle_class(pid, tool)

    # ---- undo (⌘Z) ----------------------------------------------------------
    # Every store that holds a view-only correction for one photo. They are
    # snapshotted together, because a single click can write to several of them.
    _EDIT_STORES = ("class_overrides", "view_excluded", "measure_include",
                    "measure_exclude", "certainty")
    _EDIT_UNDO_MAX = 100          # deep enough for a labelling session, bounded

    def _edit_snapshot(self, path):
        snap = {}
        for name in self._EDIT_STORES:
            v = getattr(self, name).get(path)
            snap[name] = None if v is None else (dict(v) if isinstance(v, dict)
                                                 else set(v))
        lst = self.chosen.get(path)
        snap["chosen"] = None if lst is None else list(lst)
        return snap

    def _push_undo(self):
        """Remember the current photo's edit state, so ⌘Z can come back to it."""
        if self.current is None:
            return
        self._edit_undo.append((self.current, self._edit_snapshot(self.current)))
        if len(self._edit_undo) > self._EDIT_UNDO_MAX:
            self._edit_undo.pop(0)
        self._mark_undo_push()

    def _mark_undo_push(self):
        """Timestamp the newest per-photo undo step. ⌘Z compares this against the
        library's removal slot so the MOST RECENT action is the one undone —
        remove a folder, click three particles, and ⌘Z still walks back through
        the clicks first."""
        import time
        self._undo_mark = time.monotonic()

    def _undo(self):
        """⌘Z: the most recent undoable thing — a library removal, a training
        label, or a class / exclude / measure correction."""
        if self._typing():
            return                                # typing in a text box
        # A removal is the most destructive action in the app, so it is checked
        # first rather than living behind a mode. It wins when it is the newest
        # action, and ALSO when the per-photo stack has run dry — otherwise
        # ⌘Z would keep answering "nothing to undo" with a removal still
        # sitting there waiting to be taken back.
        if self._lib_undo is not None:
            stack = self._train_undo if self.train_mode else self._edit_undo
            if self._lib_undo[0] > self._undo_mark or not stack:
                self._lib_undo_last()
                return
        if self.train_mode:
            self._train_undo_last()
        else:
            self._edit_undo_last()

    def _edit_undo_last(self):
        if not self._edit_undo:
            self.status.showMessage("Nothing to undo.")
            return
        path, snap = self._edit_undo.pop()
        for name in self._EDIT_STORES:
            store = getattr(self, name)
            if snap[name] is None:
                store.pop(path, None)
            else:
                store[path] = snap[name]
        if snap["chosen"] is None:
            self.chosen.pop(path, None)
        else:
            self.chosen[path] = snap["chosen"]
        if path == self.current:
            self._rerender()
            note = ""
        else:
            note = f" on {os.path.basename(path)}"
        self._refresh_results()
        left = len(self._edit_undo)
        self.status.showMessage(
            f"Undone{note}. {left} step{'' if left == 1 else 's'} left.")

    def _forget_undo(self, path):
        """Drop a photo's history — its particle ids are about to change."""
        self._edit_undo = [u for u in self._edit_undo if u[0] != path]

    def _toggle_measure(self, a, pid):
        self._push_undo()
        lst = self.chosen.setdefault(self.current, [])
        inc = self.measure_include.setdefault(self.current, set())
        if any(p.id == pid for p in lst):
            self.chosen[self.current] = [p for p in lst if p.id != pid]
            inc.discard(pid)
        else:
            p = next((q for q in a.particles if q.id == pid), None)
            if p is not None:
                lst.append(p)
                # left-clicking a particle to measure it also undoes any manual
                # right-click exclusion on it (the two are opposites)
                self.measure_exclude.get(self.current, set()).discard(pid)
                # measuring a greyed-out (occluded / frame-cut) particle is the
                # user overruling the gate -> it counts towards the statistics
                if not analyze.measurable(p):
                    inc.add(pid)
        self._rerender()
        self._refresh_results()

    def _on_view_right_click(self, x, y):
        """Right-click a particle to flip its measurement status — measured
        becomes not, and (2026-07-30, user rule: "sağ tık değişimli çalışsın")
        the light-blue not-measured ones become measured too. A true toggle in
        both directions, not just the drop-out half of it."""
        if self.train_mode:
            return
        a, pid = self._particle_at(x, y)
        if a is None or pid is None:
            return
        self._toggle_measure_exclude(a, pid)

    def _toggle_measure_exclude(self, a, pid):
        exc = self.measure_exclude.setdefault(self.current, set())
        inc = self.measure_include.setdefault(self.current, set())
        p = next((q for q in a.particles if q.id == pid), None)
        # the particle's CURRENT effective state, in the same priority order
        # measurable() itself uses: an explicit override wins, else the gate.
        currently_measured = (pid not in exc
                              and (pid in inc or (p is not None and analyze.measurable(p))))
        self._push_undo()
        if currently_measured:
            if pid in inc:
                inc.discard(pid)                 # was force-included -> just revert
            else:
                exc.add(pid)                     # the gate accepts it -> force it out
            # either way it's leaving the measurement: drop any size line for it
            lst = self.chosen.get(self.current)
            if lst:
                self.chosen[self.current] = [q for q in lst if q.id != pid]
            note = "excluded from the measurement"
        else:
            if pid in exc:
                exc.discard(pid)                 # was manually excluded -> restore it
            else:
                inc.add(pid)                     # the gate rejects it -> force it in
            note = "included in the measurement"
        if not exc:
            self.measure_exclude.pop(self.current, None)
        if not inc:
            self.measure_include.pop(self.current, None)
        self._rerender()
        self._refresh_results()
        self.status.showMessage(f"Particle {note}.")

    # ---- in-app guide ----
    def _open_guide(self):
        """A compact manual of the non-obvious controls (built once, reused)."""
        if getattr(self, "_guide_dlg", None) is None:
            self._guide_dlg = GuideDialog(self)
        self._guide_dlg.show()
        self._guide_dlg.raise_()
        self._guide_dlg.activateWindow()

    # ---- Certainty tool (inspect the model's confidence, both modes) ----
    def _overlay_modes(self):
        """(show_pattern, show_class): which normal-mode overlay is active — mirrors
        _rerender, so Certainty reports the class the particle is coloured as."""
        if self._overlay_hidden:
            return False, False
        show_pat = any(cb.isChecked() for cb in self._pat_cbs.values())
        show_cls = ((self.cb_solid.isChecked() or self.cb_under.isChecked())
                    and not show_pat)
        return show_pat, show_cls

    def _certainty_label(self, pid):
        """The class particle `pid` is currently shown as (so its certainty matches
        the colour on screen), or None when it carries no usable label."""
        a = self.results.get(self.current)
        if a is None:
            return None
        if self.train_mode:
            eff = self._train_effective(a)
            if pid in eff:
                return eff[pid][0]
            p = next((q for q in a.particles if q.id == pid), None)
            if p is None or getattr(p, "excluded", False):
                return None
            if p.is_solid and p.pattern:
                return p.pattern
            return "undercooled" if not p.is_solid else "solid"
        va = self._view_analysis(a, self.current)
        vp = next((q for q in va.particles if q.id == pid), None)
        if (vp is None or getattr(vp, "excluded", False)
                or getattr(vp, "unclassified", False)):
            return None      # excluded, or "class unclear" -> nothing to be sure of
        show_pat, show_cls = self._overlay_modes()
        if show_cls:                             # Solid / Undercooled overlay
            return "solid" if vp.is_solid else "undercooled"
        if vp.is_solid and vp.pattern:           # pattern overlay / plain view
            return vp.pattern
        return "solid" if vp.is_solid else "undercooled"

    def _certainty_value(self, pid, label):
        """Model confidence in [0,1] for `label`: solidnet P(solid) for solid /
        undercooled, patternnet softmax for a pattern class."""
        a = self.results.get(self.current)
        p = next((q for q in a.particles if q.id == pid), None) if a else None
        if p is None or label is None:
            return None
        if label == "undercooled":
            return 1.0 - float(getattr(p, "facet_frac", 0.0))
        if label == "solid":
            return float(getattr(p, "facet_frac", 0.0))
        if label in analyze.PATTERN_CLASSES:
            pp = getattr(p, "pattern_probs", ())
            if pp:
                return float(pp[analyze.PATTERN_CLASSES.index(label)])
        return None

    def _fit_image(self):
        """⌘0 — put the micrograph back to fit, whatever the zoom got up to."""
        self.view.fit()
        self.status.showMessage("Image fitted to the panel.")

    def _class_rule_note(self, pid, label):
        """Why the colour on screen can disagree with the number beside it.

        A few classes are decided by a RULE, not by the net whose probability the
        badge shows — and then the app looks like it is contradicting itself
        ("5% undercooled" written on an undercooled particle, the complaint that
        led to this). The badge stays the model's own reading; this sentence says
        which rule took the decision out of its hands, and how to overrule it.
        """
        if label != "undercooled":
            return ""
        a = self.results.get(self.current)
        p = next((q for q in a.particles if q.id == pid), None) if a else None
        if p is None:
            return ""
        ff = float(getattr(p, "facet_frac", 0.0))
        if ff < analyze.DEFAULT_FACET_THRESH:
            return ""                     # the model itself says undercooled
        pct = f"{ff * 100:.0f}%"
        if p.diam_nm < analyze.SOLID_MIN_DIAM_NM:
            return (f"  ⓘ By rule, not by the model: it reads {pct} crystalline, "
                    f"but everything below {analyze.SOLID_MIN_DIAM_NM:.0f} nm is "
                    f"counted undercooled.")
        pat = (analyze.size_aware_pattern(p.diam_nm, p.pattern_probs)
               if len(getattr(p, "pattern_probs", ())) else "")
        if pat == "janus" and ff < analyze.JANUS_MIN_SOLID:
            return (f"  ⓘ By rule, not by the model: its only pattern guess is "
                    f"janus, which needs a clearer solid reading than {pct}. "
                    f"Press 6 to overrule.")
        if not pat:
            return (f"  ⓘ By rule, not by the model: it reads {pct} crystalline, "
                    f"but no pattern is credible at {p.diam_nm:.0f} nm (janus "
                    f"needs ≥{analyze.JANUS_MIN_NM:.0f} nm, stripe "
                    f"≥{analyze.STRIPE_SMALL_MIN_CONF * 100:.0f}% confidence), so "
                    f"it is counted as liquid. Press 6 to overrule.")
        return ""

    @staticmethod
    def _certainty_color(label):
        if label == "undercooled":
            return GREEN
        if label == "solid":
            return RED
        return PATTERN_COLORS.get(label, (200, 200, 200))

    def _toggle_certainty(self, pid):
        self._push_undo()
        store = self.certainty.setdefault(self.current, {})
        if pid in store:                         # click again -> hide this badge
            del store[pid]
            self._rerender()
            return
        label = self._certainty_label(pid)
        val = self._certainty_value(pid, label)
        if label is None or val is None:
            self.status.showMessage(
                "This particle is excluded / not classified — no certainty to show.")
            return
        store[pid] = (f"{val * 100:.0f}%", self._certainty_color(label))
        self.status.showMessage(
            f"{label.capitalize()} certainty: {val * 100:.0f}%"
            + self._class_rule_note(pid, label))
        self._rerender()

    def _certainty_overlay(self):
        """Badges (cx, cy, text, rgb) for the Certainty tool, or None. Only drawn
        while the tool is active, so the numbers don't clutter other views."""
        if self.click_tool != "certainty":
            return None
        store = self.certainty.get(self.current)
        a = self.results.get(self.current)
        if not store or a is None:
            return None
        pos = {p.id: (p.cx, p.cy) for p in a.particles}
        out = [(pos[pid][0], pos[pid][1], text, rgb)
               for pid, (text, rgb) in store.items() if pid in pos]
        return out or None

    def _resolve_solid(self, pid):
        """The manual 'Solid' tool: treat particle `pid` as solid and let the model
        pick its pattern — or plain "solid" (solid, pattern unknown) when it can't
        recognise one. Returns the resolved class name, or None when patterns
        weren't evaluated for this run."""
        a = self.results.get(self.current)
        if a is None:
            return None
        p = next((q for q in a.particles if q.id == pid), None)
        if p is None:
            return None
        if not len(getattr(p, "pattern_probs", ())):
            # No pattern prediction for this one (too small for the pattern crop,
            # or the run skipped the pattern step). It is STILL solid if the user
            # says so — "solid diyorsam solid olsun, bir şey atayamıyorsan
            # sıkıntı yok" (user rule, 2026-07-30). Refusing here was why the
            # Solid tool looked like it did nothing on some particles.
            return "solid"
        return analyze.solidify_pattern(p)

    # Where a second click on the same class lands. Values stored in
    # class_overrides: a pattern name, "undercooled", "solid" (solid, pattern
    # unknown), "solid:<pattern>" (the same thing, reached by un-toggling that
    # pattern — kept distinct so the third click knows what it is undoing), or
    # "unknown" (no class at all, still measured). "solid" has no unsure state:
    # it already IS "solid, pattern unknown", so a second click just undoes it.
    _UNSURE_OF = {"undercooled": "unknown", "janus": "solid:janus",
                  "stripe": "solid:stripe", "lamellar": "solid:lamellar",
                  "composite": "solid:composite"}

    def _set_particle_class(self, pid, cls):
        ov = self.class_overrides.setdefault(self.current, {})
        exc = self.view_excluded.setdefault(self.current, set())
        if cls == "solid":
            cls = self._resolve_solid(pid)
            if cls is None:
                return
            msg = ("solid (no clear pattern)" if cls == "solid"
                   else f"solid → {cls}")
            self.status.showMessage(f"Particle re-analysed as {msg}.")
        self._push_undo()
        if cls == "exclude":
            exc.discard(pid) if pid in exc else exc.add(pid)  # click again = undo
            ov.pop(pid, None)
            if pid in exc:            # an excluded particle keeps no size line
                lst = self.chosen.get(self.current)
                if lst:
                    self.chosen[self.current] = [q for q in lst if q.id != pid]
        else:
            exc.discard(pid)
            # Clicking the SAME class a second time means "actually I can't tell
            # what this is" (user rule, 2026-07-30) — the particle keeps its size
            # but drops out of the classes, and a third click hands it back to the
            # model. Un-toggling a pattern leaves it SOLID with no pattern; only
            # un-toggling undercooled leaves it class-less altogether, since
            # "not undercooled" says nothing about solid.
            unsure = self._UNSURE_OF.get(cls)
            prev = ov.get(pid)
            if prev == cls and unsure:
                ov[pid] = unsure
                self.status.showMessage(
                    "Class cleared — the particle keeps its size but counts in no "
                    "class. Click again to hand it back to the model."
                    if unsure == "unknown" else
                    "Pattern cleared — still solid and still measured, but in no "
                    "pattern class. Click again to hand it back to the model.")
            elif prev == cls or (unsure and prev == unsure):
                ov.pop(pid, None)                       # back to the model's call
                self.status.showMessage("Back to the model's own classification.")
            else:
                ov[pid] = cls
        # NOTE: a hand-assigned class no longer changes WHAT IS MEASURED (user
        # rule, 2026-07-30, reversing the 2026-07-25 auto-include). Class and size
        # are separate judgements: telling a slate-blue particle it is lamellar
        # leaves it unmeasured, and the Measure tool (left-click / right-click)
        # stays the only way to move a particle in or out of the statistics.
        # _view_analysis freezes the gate's verdict for hand-classified particles.
        self._rerender()
        self._refresh_results()

    def _view_analysis(self, a, path):
        """Return `a` with the user's view-only corrections applied (class paints
        + excludes). The original analysis is never mutated, so corrections are
        transient and never leak into training data or the saved model."""
        ov = self.class_overrides.get(path, {})
        exc = self.view_excluded.get(path, set())
        inc = self.measure_include.get(path, set())
        mexc = self.measure_exclude.get(path, set())
        if not ov and not exc and not inc and not mexc:
            return a
        parts = []
        for p in a.particles:
            if p.id in exc:
                # Excluded by hand: the particle STAYS in the analysis (marked the
                # way the model's own dim-excludes are), so the Measure tool still
                # paints it slate-blue = "not measured". It carries no class, no
                # size, and counts under n_excluded rather than vanishing.
                parts.append(dataclasses.replace(
                    p, excluded=True, is_solid=False, pattern="",
                    user_measurable=False))
                continue
            if p.id in mexc:
                p = dataclasses.replace(p, user_measurable=False)
            elif p.id in inc:
                p = dataclasses.replace(p, user_measurable=True)
            elif p.id in ov:
                # A hand-assigned class must not silently change what is measured
                # (the gate's `if p.pattern: return True` rule would otherwise
                # start measuring an occluded particle the moment it is named).
                # Freeze the gate's own verdict on the untouched particle.
                p = dataclasses.replace(p, user_measurable=analyze.measurable(p))
            if p.id in ov:
                # class display only; whether it counts towards the size stats is
                # governed by the Measure tool (inc / mexc) above
                c = ov[p.id]
                if c == "unknown":       # measured, but in no class at all
                    p = dataclasses.replace(p, is_solid=False, pattern="",
                                            unclassified=True, excluded=False)
                elif c == "undercooled":
                    p = dataclasses.replace(p, is_solid=False, pattern="",
                                            unclassified=False, excluded=False)
                elif c == "solid" or c.startswith("solid:"):
                    # user_solid: their own call, so the overlay shows it even
                    # though there is no pattern to colour it with
                    p = dataclasses.replace(p, is_solid=True, pattern="",
                                            unclassified=False, excluded=False,
                                            user_solid=True)
                else:
                    p = dataclasses.replace(p, is_solid=True, pattern=c,
                                            unclassified=False, excluded=False)
            parts.append(p)
        return dataclasses.replace(a, particles=parts)

    def _flip_overlay(self):
        """Space: hide the overlay, then restore whatever was showing."""
        if self._typing():
            return                                  # typing in a size/search box
        if self.train_mode:
            self.train_show_overlay = not self.train_show_overlay
        else:
            self._overlay_hidden = not self._overlay_hidden
        self._rerender()
