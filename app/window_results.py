"""What the RESULTS panel and the image preview draw.

Everything the right-hand side shows for the current selection: the class
filter chips, the size histogram and its size-range readout, the cumulative
curve, the Solid / Liquid split, Pattern x Size, the stat tiles — plus the
overlay redraw for the micrograph itself, because the two always change
together (ticking a view toggle or a chip has to repaint both).

One rule holds this together: the panel and an exported file must never
disagree. Both go through report.py's render_* functions and report's
class_counts(), so a number on screen and the same number in a saved figure
come from one calculation, not two.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

import chart_data
import charts
from analyze import Aggregate, ensure_perimeter
from overlay_draw import render, render_training
from chart_data import class_diams, solid_split, CLASS_LABELS, CLASS_COLORS
from charts import render_report
from export_files import export
from window_pattern_size import _Plot as PatternSizePlot


class ResultsPanel:
    """Mixin: the RESULTS panel and preview rendering. `self` is the window."""


    def _open_save_menu(self):
        """Drop the save menu under the Save button, right-aligned to it."""
        m = self._save_menu
        x = self.save_btn.width() - m.sizeHint().width()
        pos = self.save_btn.mapToGlobal(QtCore.QPoint(x, self.save_btn.height() + 4))
        m.popup(pos)

    def _update_class_controls(self, a):
        ok = bool(a and a.classifiable)
        for wdg in (self.cb_under, self.cb_solid):
            wdg.setEnabled(ok)
        # Class-paint tools need a classifiable (CBS) image; Measure always works.
        for key in ("certainty", "janus", "stripe", "composite", "lamellar",
                    "undercooled", "solid", "exclude"):
            self._tool_btns[key].setEnabled(ok)
        if not ok and self.click_tool not in (None, "measure"):
            self._tool_btns["measure"].setChecked(True)
            self._set_click_tool("measure")
        if a and not ok:
            self.cb_under.setChecked(False)
            self.cb_solid.setChecked(False)
        # The pattern-class toggles are a permanent part of the View bar (the
        # layout never reshuffles); they only *enable* once the current image
        # actually has pattern results — and never together with Solid.
        self._pat_ok = bool(a and a.has_patterns)
        self._sync_pattern_enable()

    def _sync_pattern_enable(self):
        """Solid and the pattern classes are mutually exclusive: a solid
        particle can't be tinted red and janus-orange at once. In training mode
        the pattern checkboxes are always available (they filter labelled classes)."""
        on = self.train_mode or (self._pat_ok and not self.cb_solid.isChecked())
        for cb in self._pat_cbs.values():
            if not on and cb.isChecked():
                cb.setChecked(False)
            cb.setEnabled(on)

    def _view_toggle(self):
        self._overlay_hidden = False   # ticking a class un-hides the overlay
        self._rerender()

    def _rerender(self):
        a = self.results.get(self.current)
        if not a:
            return
        if self.train_mode:
            # the Measure tool draws its size lines here too; chosen is per-photo
            chosen = self.chosen.get(self.current) if self.click_tool == "measure" else None
            # The top View checkboxes mean EXACTLY what they mean in normal mode
            # (user rule, 2026-08-04): ticked classes are painted, nothing ticked
            # paints nothing. They used to fall back to "show everything" when
            # none was ticked, which made the same bar do two different things
            # depending on which mode you were in — you could not turn a class
            # off, because turning the last one off brought them all back.
            sel = {k for k, cb in self._pat_cbs.items() if cb.isChecked()}
            if self.cb_under.isChecked():
                sel.add("undercooled")
            if self.review_mode:
                # Review draws the model's OWN answer on the particles where it
                # disagrees with the labels — see _review_pairs.
                labels = {pid: (mdl, True)
                          for pid, (_truth, mdl) in self._review_pairs(a).items()}
            else:
                labels = self._train_effective(a)
            self.view.set_image(render_training(
                a, labels, show_overlay=self.train_show_overlay,
                chosen=chosen, show_classes=sel,
                # and the slate-blue "not measured" tint follows the Measure
                # tool here too, instead of being permanently on. Same rule as
                # normal mode: it is an answer to a question you asked, and
                # clicking Measure again puts it away.
                mark_unmeasured=(self.click_tool == "measure"),
                certainty=self._certainty_overlay()))
            self._train_update_panel()
            self._style_row_for(self.current)     # the training dot follows edits
            return
        # measurement lines are hidden while a fill/border overlay is shown, so
        # the cyan lines don't clutter those views. Any ticked pattern class
        # switches to the per-class overlay (painting just the ticked classes,
        # plus green undercooled if that's on); otherwise Solid / Undercooled
        # tick the plain red / green fills individually.
        va = self._view_analysis(a, self.current)   # apply view-only class paints
        # The overlay is driven ONLY by the top View checkboxes — a class-paint
        # tool being active no longer forces the full pattern overlay on (the user
        # wants whatever is ticked above to stay; corrections still apply to the
        # ticked classes live).
        if self._overlay_hidden:
            # Space toggled the overlay off: show the plain micrograph (measure
            # lines stay). A second press restores the ticked classes.
            sel, show_pat, cf, show_cls = set(), False, set(), False
        else:
            sel = {k for k, cb in self._pat_cbs.items() if cb.isChecked()}
            show_pat = bool(sel)
            if show_pat and self.cb_under.isChecked():
                sel.add("undercooled")
            if show_pat and self.cb_solid.isChecked():
                # over the pattern overlay, "Solid" means the ones with no pattern
                # of their own (the model couldn't read one) — red, like the plain
                # classification view. Ticking it is opt-in: ~22% of particles are
                # in that state, so they stay plain until asked for.
                sel.add("solid")
            cf = set()
            if self.cb_solid.isChecked():
                cf.add("solid")
            if self.cb_under.isChecked():
                cf.add("undercooled")
            show_cls = bool(cf) and not show_pat
        im = render(va, show_measurements=(self.click_tool == "measure"),
                    show_classification=show_cls,
                    class_filter=cf,
                    show_pattern=show_pat,
                    pattern_classes=sel,
                    # Space blanks the whole overlay — borders included, so one
                    # press leaves the bare micrograph and the next brings back
                    # exactly what the View bar has ticked.
                    show_outlines=(self.cb_outline.isChecked()
                                   and not self._overlay_hidden),
                    # with the Measure tool active, grey out everything that is
                    # not contributing a diameter to the statistics
                    mark_unmeasured=(self.click_tool == "measure"),
                    chosen=self.chosen.get(self.current),
                    certainty=self._certainty_overlay())
        self.view.set_image(im)

    def _refresh_results(self):
        sel = self._selected_paths()
        if not sel and self.current:      # nothing ticked -> show the previewed one
            sel = [self.current]
        # Sphericity needs each particle's perimeter. Analyses saved before that
        # existed get it filled in here, off their stored mask — on the ORIGINAL
        # analysis, not the view copy, so it is measured once and then travels
        # with the session file instead of being recomputed on every click.
        filled = 0
        for q in sel:
            if q in self.results:
                try:
                    filled += ensure_perimeter(self.results[q])
                except Exception:
                    traceback.print_exc()   # a missing shape must not stop the panel
        if filled:
            # ~50 ms a photo, so a whole library selected at once is a few
            # seconds ONCE — say what the pause was for rather than just pausing
            self.status.showMessage(
                f"Measured the roundness of {filled} particles in "
                f"{len(sel)} earlier analysis(es).")
        analyzed = [self._view_analysis(self.results[p], p)
                    for p in sel if p in self.results]
        if not analyzed:
            self._result_target = None
            self._update_chips(None)
            self.result.clear_img(); return
        self._result_target = analyzed[0] if len(analyzed) == 1 else Aggregate(analyzed)
        self._update_chips(self._result_target)
        # Debounced: the matplotlib render takes a noticeable moment, and this
        # fires on EVERY selection change (arrow keys, drag-select, analysis
        # batches). Rendering through the timer coalesces bursts into one draw
        # and keeps the file tree responsive instead of freezing per click.
        self._result_timer.start(120)

    def _update_chips(self, target):
        """One chip per class present in the current selection; the bar hides
        when nothing is classified (ETD, or not analysed). Everything the panel
        needs is cached here so the readouts touch no heavy code."""
        self._res_groups = class_diams(target) if target is not None else {}
        self._res_all = (target.diam_array(whole_only=True) if target is not None
                         else np.array([]))
        # the solid/undercooled split is its own pass over the particles: unlike
        # class_diams it keeps crystalline particles that got no pattern
        self._res_split = solid_split(target) if target is not None else {}
        self._res_stats = target.stats() if target is not None else None
        groups = self._res_groups
        self.chip_bar.setVisible(bool(groups))
        for key, b in self._chip_btns.items():
            if key in ("all",) + self.OVERVIEW_TABS:
                continue
            d = groups.get(key)
            n = int(d.size) if d is not None else 0
            b.setEnabled(n > 0)
            b.setToolTip(f"{n} {CLASS_LABELS[key].lower()} particles" if n else
                         f"no {CLASS_LABELS[key].lower()} particles here")
        self._chip_btns["patternsize"].setEnabled(bool(groups))
        # Cumulative needs measured particles, not classes: it works on an ETD
        # image or a single-class selection where Pattern × Size has nothing to
        # show, so it is gated on the diameters instead.
        self._chip_btns["cumulative"].setEnabled(
            bool(getattr(self, "_res_all", np.array([])).size))
        # Solid / Liquid needs BOTH states present (solid_split returns
        # nothing otherwise) — a stack of one colour is not a comparison
        self._chip_btns["solidsplit"].setEnabled(bool(self._res_split))
        self._chip_btns["solidsplit"].setToolTip(
            "" if self._res_split else
            "needs both solid and undercooled particles in the selection")
        # the active view vanished with the selection -> fall back to All
        rf = self.result_filter
        gone = ((rf == "patternsize" and not groups)
                or (rf == "cumulative" and not self._chip_btns["cumulative"].isEnabled())
                or (rf == "solidsplit" and not self._res_split)
                or (rf not in (None,) + self.OVERVIEW_TABS and rf not in groups))
        if gone:
            self.result_filter = None
            for k, b in self._chip_btns.items():
                b.setChecked(k == "all")
        self._apply_result_view(target)

    CHIP_GAP = 5

    def _layout_chips(self):
        """Deal the chips into as few rows as their labels actually need.

        Two rows if they fit, three if they don't (user rule, 2026-07-30). The
        widths are measured, not assumed, because a chip that is one pixel too
        narrow elides its label — which is what a fixed four-column grid did to
        "Pattern × Size" the moment an eighth chip joined. Rows are balanced
        (the split that keeps the widest row narrowest) and each chip is
        stretched by its own natural width, so a row fills the panel evenly
        instead of ending ragged.

        An extra row makes the chip bar taller; the chart above the tiles is the
        only thing that gives up the space, so nothing below it shifts.
        """
        keys = self._chip_order
        need = [self._chip_btns[k].sizeHint().width() for k in keys]
        avail = max(120, self.chip_bar.width() - 4)

        def split(n_rows):
            """Balanced contiguous split into n_rows; None if a row can't fit."""
            best = None
            n = len(keys)
            cuts = [(i,) for i in range(1, n)] if n_rows == 2 else \
                   [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]
            for cut in cuts:
                bounds = (0,) + cut + (n,)
                rows = [list(range(bounds[i], bounds[i + 1]))
                        for i in range(n_rows)]
                widths = [sum(need[i] for i in r) + self.CHIP_GAP * (len(r) - 1)
                          for r in rows]
                if max(widths) > avail:
                    continue
                if best is None or max(widths) < best[0]:
                    best = (max(widths), rows)
            return best[1] if best else None

        rows = split(2) or split(3)
        if rows is None:
            # Panel too narrow for a balanced 2- or 3-row deal: wrap greedily,
            # filling each row before starting the next (never one chip per row).
            rows, cur, used = [], [], 0
            for i, wdt in enumerate(need):
                add = wdt + (self.CHIP_GAP if cur else 0)
                if cur and used + add > avail:
                    rows.append(cur); cur, used = [], 0
                    add = wdt
                cur.append(i); used += add
            if cur:
                rows.append(cur)
        if len(rows) == self._chip_rowcount and getattr(self, "_chip_shape", None) == rows:
            return                           # nothing to rebuild
        self._chip_shape = rows
        self._chip_rowcount = len(rows)
        while self._chip_rows.count():       # tear the old rows down
            item = self._chip_rows.takeAt(0)
            sub = item.layout()
            if sub is not None:
                while sub.count():
                    w = sub.takeAt(0).widget()
                    if w is not None:
                        w.setParent(self.chip_bar)
                sub.deleteLater()
        for r in rows:
            hb = QtWidgets.QHBoxLayout()
            hb.setSpacing(self.CHIP_GAP); hb.setContentsMargins(0, 0, 0, 0)
            for i in r:
                hb.addWidget(self._chip_btns[keys[i]], need[i])
            self._chip_rows.addLayout(hb)

    def _set_result_filter(self, key):
        """Switch the RESULTS view: All, one class, or the Pattern × Size tab."""
        self.result_filter = None if key == "all" else key
        for k, b in self._chip_btns.items():
            b.setChecked(k == key)
        self._apply_result_view(getattr(self, "_result_target", None))
        if not self._ps_active:
            self._result_timer.start(10)

    OVERVIEW_TABS = ("solidsplit", "patternsize", "cumulative")

    def _apply_result_view(self, target):
        """Show the histogram+tiles, or one of the overview pages, per the chip."""
        ps = (self.result_filter == "patternsize")
        cdf = (self.result_filter == "cumulative")
        spl = (self.result_filter == "solidsplit")
        # _ps_active means "an overview page is up" to the rest of the panel:
        # the histogram and its tiles must not be recomputed while hidden
        self._ps_active = ps or cdf or spl
        self.normal_page.setVisible(not self._ps_active)
        self.ps_page.setVisible(ps)
        self.cdf_page.setVisible(cdf)
        self.split_page.setVisible(spl)
        # Lay the card out NOW. A page that has just been un-hidden still has its
        # placeholder size until the next layout pass, and the figures are
        # rendered AT the view's aspect ratio — so without this the first draw is
        # a squat chart in a half-empty panel, corrected only by a later resize.
        card = self.normal_page.parentWidget()
        for lay in (card.layout() if card is not None else None,
                    self.cdf_page.layout(), self.ps_page.layout(),
                    self.split_page.layout()):
            if lay is not None:
                lay.activate()          # the page's own layout too, not just the card's
        if ps:
            self._refresh_ps()
        elif cdf:
            self._refresh_cdf()
        elif spl:
            self._refresh_split()
        else:
            self.stat_bar.setVisible(target is not None)
            self._update_stats()

    # ---- Cumulative tab (CDF + D-values) ----
    def _refresh_cdf(self):
        """Tiles now, curve through the debounce (it is a matplotlib render)."""
        t = getattr(self, "_result_target", None)
        rows = chart_data.cumulative_tiles(t)
        flat = [c for row in rows for c in row]
        for (cap, val, sub, _accent), key in zip(
                flat, ("d10", "d50", "d90", "span", "measured")):
            self.cdf_tiles[key].set(cap=cap, val=val, sub=sub)
        # 120 ms, not immediately: the page has only just been shown, so its
        # final height arrives with the next layout pass — rendering before that
        # produced a squat figure sitting in a half-empty panel (the figure is
        # drawn AT the view's aspect ratio, so the aspect has to be the real one)
        self._cdf_timer.start(120)

    def _render_cdf(self):
        target = getattr(self, "_result_target", None)
        if target is None or self.result_filter != "cumulative":
            return
        w = max(60, self.cdf_view.width())
        h = max(120, self.cdf_view.height())
        try:
            self.cdf_view.set_image(charts.render_cumulative(target, aspect=w / h))
        except ValueError:
            self.cdf_view.clear_img()          # nothing measured yet
        except Exception:
            traceback.print_exc()

    # ---- Solid / Liquid tab (the stacked size histogram) ----
    def _refresh_split(self):
        """Tiles and the range readout now, the chart through the debounce."""
        g = getattr(self, "_res_split", {}) or {}
        T = self.split_tiles
        if not g:
            for t in T.values():
                t.set(val="—", sub="")
            self._update_split_readout()
            self._split_timer.start(120)
            return
        ns, nu = int(g["solid"].size), int(g["undercooled"].size)
        tot = ns + nu
        T["solid"].set(val=f"{100.0 * ns / tot:.0f}%", sub=f"{ns}",
                       accent=chart_data.STAT_SOLID_COLOR)
        T["undercooled"].set(val=f"{100.0 * nu / tot:.0f}%", sub=f"{nu}",
                             accent=CLASS_COLORS["undercooled"])
        T["meansolid"].set(val=f"{g['solid'].mean():.0f}", sub="nm")
        T["meanunder"].set(val=f"{g['undercooled'].mean():.0f}", sub="nm")
        self._set_sph_tile(T["sphsolid"], state="solid")
        self._set_sph_tile(T["sphunder"], state="undercooled")
        self._update_split_readout()
        # 120 ms like the other pages: the figure is drawn AT the view's aspect,
        # which is only final after the layout pass that follows showing the page
        self._split_timer.start(120)

    def _update_split_readout(self):
        """"Of the particles in the typed range, this share is solid" — the
        number the stacked bars can only be eyeballed for."""
        g = getattr(self, "_res_split", {}) or {}
        if not g or self.size_range is None:
            self.split_pct.setText(""); self.split_cnt.setText("")
            return
        frac, k, tot = chart_data.solid_share(g, *self.size_range)
        if frac is None:                     # nothing of either state in there
            self.split_pct.setText("—"); self.split_cnt.setText("(0 particles)")
            return
        self.split_pct.setText(f"{100.0 * frac:.0f}% solid")
        self.split_cnt.setText(f"({k} / {tot})")

    def _render_split(self):
        if self.result_filter != "solidsplit":
            return
        target = getattr(self, "_result_target", None)
        if target is None:
            self.split_view.clear_img(); return
        w = max(60, self.split_view.width())
        h = max(120, self.split_view.height())
        try:
            self.split_view.set_image(charts.render_solid_split(
                target, aspect=w / h, size_range=self.size_range))
        except ValueError:
            self.split_view.clear_img()      # only one state in this selection
        except Exception:
            traceback.print_exc()

    def _rerender_chart(self):
        """Redraw whichever chart the size range applies to (both highlight it)."""
        if self.result_filter == "solidsplit":
            self._render_split()
        else:
            self._render_result()

    # ---- Pattern × Size tab (embedded size-composition view) ----
    def _ps_pairs(self):
        return [(cls, float(d)) for cls, arr in self._res_groups.items()
                for d in arr]

    def _refresh_ps(self):
        pairs = self._ps_pairs()
        first = self._ps_plot is None
        if first:
            self._ps_plot = PatternSizePlot(pairs)
            self._ps_holder.addWidget(self._ps_plot)
            # dragging a line on the plot IS moving a handle: same range, so
            # the two controls can never show different numbers
            self._ps_plot.rangeChanged.connect(self._ps_range_from_plot)
        else:
            self._ps_plot.set_data(pairs)
        top = self._ps_plot._top_value()
        self.ps_slider.blockSignals(True)
        self.ps_slider.set_bounds(top)
        if first:
            self.ps_slider.set_values(400.0, top)
        self.ps_slider.blockSignals(False)
        self._ps_range_changed(*self.ps_slider.values())
        self.ps_head.setText(f"{self._ps_plot.n_shown} particles")

    def _ps_range_changed(self, lo, hi):
        if self._ps_plot is not None:
            self._ps_plot.set_range(lo, hi)
            self.ps_thlab.setText(self._ps_plot.range_text())

    def _ps_range_from_plot(self, lo, hi):
        self.ps_slider.set_values(lo, hi)
        self.ps_thlab.setText(self._ps_plot.range_text())

    def _filtered_diams(self):
        """The diameters the RESULTS panel is currently about: one class, or
        every measured particle when no chip is active (both cached)."""
        if self.result_filter and self.result_filter not in self.OVERVIEW_TABS:
            return self._res_groups.get(self.result_filter, np.array([]))
        return getattr(self, "_res_all", np.array([]))

    def _relayout_stats(self, mode):
        """Arrange the tile grid for the current view (built once per mode
        change, not every value update)."""
        if self._stat_mode == mode:
            return
        self._stat_mode = mode
        g = self.stat_grid
        while g.count():                       # detach the previous set (reused)
            w = g.takeAt(0).widget()
            if w is not None:
                w.hide()
        if mode == "filtered":
            # SPHERICITY sits where RANGE used to (user request, 2026-08-08) —
            # the same swap chart_data.stat_tiles makes for an exported figure.
            rows = [(self.tile_total, self.tile_n),
                    (self.tile_mean, self.tile_sph)]
        elif mode == "all":
            # MEAN SIZE replaces TOTAL PARTICLES here: the mean was the one
            # number the All view never showed, and the total is already implied
            # by MEASURED + the class counts below it. The third tile was
            # CLASSIFIED until 2026-08-08, when the user asked for sphericity in
            # its place (the class row below already adds up to it).
            rows = [(self.tile_mean, self.tile_measured, self.tile_sph),
                    (self.tile_cls["undercooled"], self.tile_cls["solid"],
                     self.tile_cls["janus"]),
                    (self.tile_cls["stripe"], self.tile_cls["composite"],
                     self.tile_cls["lamellar"])]
        else:                                  # unclassified (ETD): no breakdown
            # sphericity is geometry, not a class, so it belongs here too — in
            # the same slot as in the classified view, with RANGE keeping its own
            rows = [(self.tile_total, self.tile_measured, self.tile_sph),
                    (self.tile_mean, self.tile_range)]
        ncol = max(len(r) for r in rows)
        for c in range(3):
            g.setColumnStretch(c, 1 if c < ncol else 0)
        # the rows share the block height evenly, so on a class view the two tile
        # rows grow to fill the space Size range leaves — no gap under it
        for r in range(3):
            g.setRowStretch(r, 1 if r < len(rows) else 0)
        for r, row in enumerate(rows):
            for c, t in enumerate(row):
                g.addWidget(t, r, c); t.show()

    def _set_sph_tile(self, tile, cls=None, state=None):
        """Fill a sphericity tile from the same helper the exported figure uses,
        so a saved chart can never quote a different number than the panel."""
        target = getattr(self, "_result_target", None)
        cap, val, sub, accent = chart_data.sphericity_tile(
            target, cls=cls, state=state, cap=tile.cap.text())
        tile.set(val=val, sub=sub, accent=accent)
        # the tile shows the app's stretched scale; the tooltip carries the plain
        # circularity behind it, which is the number with a name outside this app
        s = chart_data.sphericity(target, cls=cls, state=state)
        tile.setToolTip(self.SPH_TIP + (
            "" if not s or s["raw"] is None else
            f"\n\nPlain circularity (4πA/P², the ISO / ImageJ definition) for "
            f"these particles: {s['raw']:.3f} — that is the one to quote outside "
            f"this app.\nScored {s['n']} of {s['total']} measured particles."))

    def _update_stats(self):
        """The tiles and the size-range readout, from the cached diameters."""
        if getattr(self, "_res_stats", None) is None or self._ps_active:
            return
        groups = self._res_groups
        cls = self.result_filter
        total = int(self._res_stats["count_total"])
        measured = int(getattr(self, "_res_all", np.array([])).size)
        d = np.asarray(self._filtered_diams(), float)
        n = int(d.size)

        if cls:                                # one class in focus
            self._relayout_stats("filtered")
            tot = sum(int(x.size) for x in groups.values())
            pct = f"{100.0 * n / tot:.0f}%" if tot else ""
            self.tile_total.set(val=f"{total}")
            self.tile_n.set(cap=CLASS_LABELS[cls].upper(), val=f"{n}", sub=pct,
                            accent=CLASS_COLORS[cls])
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            self._set_sph_tile(self.tile_sph, cls=cls)
        elif groups:                           # All, with a class breakdown
            self._relayout_stats("all")
            # ONE definition of these counts, in chart_data.class_counts, so the tiles
            # on screen and the tiles drawn into an exported figure can never
            # disagree again (they had already drifted over "class unclear").
            cc = chart_data.class_counts(self._result_target)
            counts = cc["counts"]
            measured = cc["measured"]
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            # MEASURED's own % is of every DETECTED particle (not the classified
            # ones) — "how much of what I segmented got a trustworthy size".
            self.tile_measured.set(
                val=f"{measured}",
                sub=(f"{100.0 * measured / total:.0f}%" if total else ""))
            # How round they are, over every measured particle; the small number
            # is the share of the measured set it could be taken over (frame
            # particles are cut by the photo, so they are left out).
            self._set_sph_tile(self.tile_sph)
            for k in self._stat_classes:
                c = counts.get(k, 0)
                pct = f"{100.0 * c / measured:.0f}%" if measured else "—"
                self.tile_cls[k].set(val=pct, sub=f"{c}",
                                     accent=self._stat_colors[k])
        else:                                  # All, nothing classified (ETD)
            self._relayout_stats("plain")
            self.tile_total.set(val=f"{total}")
            self.tile_measured.set(
                val=f"{measured}",
                sub=(f"{100.0 * measured / total:.0f}%" if total else ""))
            self._set_sph_tile(self.tile_sph)
            self.tile_mean.set(val=(f"{d.mean():.0f}" if n else "—"),
                               sub="nm" if n else "")
            self.tile_range.set(val=(f"{d.min():.0f} – {d.max():.0f}" if n else "—"),
                                sub="nm" if n else "")
        self._update_range_readout(d)

    def _update_range_readout(self, d=None):
        """How much of what's on the plot sits inside the typed size range."""
        if d is None:
            d = np.asarray(self._filtered_diams(), float)
        lo, hi = self.size_range or (None, None)
        if d.size == 0 or self.size_range is None:
            self.range_pct.setText("")       # nothing typed -> no readout at all
            self.range_cnt.setText("")
            return
        m = np.ones(d.shape, bool)
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
        k = int(m.sum())
        self.range_pct.setText(f"{100.0 * k / d.size:.0f}%")
        self.range_cnt.setText(f"({k})")

    # The chart fills the whole space above the tile block (which is a fixed
    # height, the same in every tab), so the histogram lands in exactly the same
    # rectangle and never shifts when you switch chips — it just reaches down to
    # just above the Size-range row.
    def _render_result(self):
        target = getattr(self, "_result_target", None)
        if target is None or self._ps_active:
            return                              # Pattern × Size page is showing
        w = max(60, self.result.width())
        # no height cap: the figure fills the whole space above the tiles, down to
        # just above the Size-range row, and renders at that box's real aspect so
        # it always fills the width (no side letterboxing)
        h = max(120, self.result.height())
        try:
            cf = (None if self.result_filter in self.OVERVIEW_TABS
                  else self.result_filter)
            self.result.set_image(render_report(target, aspect=w / h,
                                                 size_range=self.size_range,
                                                 cls_filter=cf))
        except Exception:
            traceback.print_exc()

    # ---- size-fraction tool ----
    @staticmethod
    def _parse_num(text):
        text = text.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _range_changed(self, src="main"):
        if self._range_sync:               # we are the ones writing the mirror
            return
        src_boxes = ((self.range_lo, self.range_hi) if src == "main"
                     else (self.srange_lo, self.srange_hi))
        mirror = ((self.srange_lo, self.srange_hi) if src == "main"
                  else (self.range_lo, self.range_hi))
        lo, hi = (self._parse_num(b.text()) for b in src_boxes)
        self.size_range = None if lo is None and hi is None else (lo, hi)
        # keep the other page's boxes showing the same range (one range, two
        # places to type it)
        self._range_sync = True
        try:
            for dst, s in zip(mirror, src_boxes):
                if dst.text() != s.text():
                    dst.setText(s.text())
        finally:
            self._range_sync = False
        self._update_range_readout()   # the numbers answer instantly …
        self._update_split_readout()
        self._range_timer.start(300)   # … the bar highlight re-renders after a beat

    def export_selected(self):
        sel = self._selected_paths()
        # Export what is ON SCREEN: _view_analysis applies the user's own class
        # paints and excludes. Exporting the raw analysis instead was why saved
        # images didn't match the colours in the app.
        analyzed = [self._view_analysis(self.results[p], p)
                    for p in sel if p in self.results]
        if not analyzed and self.current in self.results:
            analyzed = [self._view_analysis(self.results[self.current], self.current)]
        if not analyzed:
            QtWidgets.QMessageBox.information(self, "Info", "Analyze image(s) first.")
            return
        types = [t for cb, t in [(self.ex_line, "line"),
                                 (self.ex_under, "undercooled"),
                                 (self.ex_pattern, "pattern")] if cb.isChecked()]
        cls_filters = [k for k, a in self.ex_cls.items() if a.isChecked()]
        if not types and not cls_filters:
            QtWidgets.QMessageBox.information(self, "Info", "Select at least one export type.")
            return
        title = self.ex_title.text().strip() or None
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if not d:
            return
        try:
            out = export(analyzed, d, types, chosen_map=self.chosen, k=0,
                         size_range=self.size_range,
                         cls_filters=cls_filters, title=title)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Export failed", traceback.format_exc())
            return
        folder = os.path.dirname(next(iter(out.values()))) if out else d
        QtWidgets.QMessageBox.information(
            self, "Exported",
            f"{len(out)} file(s) saved to folder:\n{os.path.basename(folder)}/")
