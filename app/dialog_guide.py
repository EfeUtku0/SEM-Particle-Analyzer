"""The in-app guide: one screen explaining the controls that are not obvious.

Deliberately NOT a full manual — only the things a user cannot discover by
looking (right-click drops a particle from the statistics, Space hides the
overlay, number keys correct the view without teaching the model).
"""
from __future__ import annotations

from PySide6 import QtWidgets


class GuideDialog(QtWidgets.QDialog):
    """A one-screen manual of the controls a first-time user can't guess. Kept
    deliberately short — only the non-obvious things (the obvious buttons speak
    for themselves)."""

    # accent used for each section's heading + rule
    _A = "#2b6fff"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide")
        self.setModal(False)
        self.resize(600, 660)
        self.setStyleSheet("""
            QDialog { background:#ffffff; }
            QLabel#title { color:#1a2129; font-size:20px; font-weight:800; }
            QLabel#sub   { color:#8a95a1; font-size:13px; font-weight:600; }
            QTextBrowser { background:transparent; border:none; }
            QScrollBar:vertical { background:transparent; width:9px; margin:2px; }
            QScrollBar::handle:vertical { background:#d3d9e0; border-radius:4px;
                                          min-height:30px; }
            QScrollBar::handle:vertical:hover { background:#bcc4cd; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
            QPushButton#gotit { background:#2b6fff; border:none; color:white;
                                border-radius:8px; font-size:13px; font-weight:700;
                                padding:8px 20px; }
            QPushButton#gotit:hover { background:#1c5df0; }
        """)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20); lay.setSpacing(4)

        title = QtWidgets.QLabel("Quick guide"); title.setObjectName("title")
        sub = QtWidgets.QLabel("The controls that aren't obvious.")
        sub.setObjectName("sub")
        lay.addWidget(title); lay.addWidget(sub); lay.addSpacing(10)

        body = QtWidgets.QTextBrowser()
        body.setOpenExternalLinks(False)
        body.setHtml(self._html())
        lay.addWidget(body, 1)

        row = QtWidgets.QHBoxLayout(); row.addStretch(1)
        ok = QtWidgets.QPushButton("Got it"); ok.setObjectName("gotit")
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        lay.addLayout(row)

    @classmethod
    def _section(cls, title, items):
        rows = "".join(
            f'<li style="margin-bottom:7px; line-height:145%;">{it}</li>'
            for it in items)
        return (
            f'<p style="margin-top:20px; margin-bottom:2px; color:{cls._A}; '
            f'font-size:14px; font-weight:800; letter-spacing:0.3px;">{title}</p>'
            f'<ul style="margin-left:-18px;">{rows}</ul>')

    @classmethod
    def _html(cls):
        def k(t):   # a control / key, emphasised inline
            return f'<b style="color:#12386e;">{t}</b>'

        blue = ('<span style="color:#3d8bd4; font-weight:700;">light-blue</span>')

        def green(t):
            return f'<span style="color:#5aa77f; font-weight:700;">{t}</span>'

        def amber(t):
            return f'<span style="color:#c9862c; font-weight:700;">{t}</span>'

        def accent(t):      # the app's "your edit" blue (training marks)
            return f'<span style="color:#2b6fff; font-weight:700;">{t}</span>'

        secs = [
            # FIRST, and deliberately not buried: a classifier answers with one of
            # the classes it was trained on whatever you feed it, so a stranger's
            # powder comes back confidently mislabelled rather than refused.
            cls._section("What this is trained for", [
                f"The two networks were trained on {k('BiSn colloidal particles')} "
                f"imaged on two specific SEMs. On another material they still "
                f"produce an answer — a {amber('wrong')} one, with no warning. "
                f"Sizes travel further than classes (they are geometry plus the "
                f"scale bar), but the pattern classes do not travel at all.",
                f"For your own material, {k('🎓 Training')} is the way: label your "
                f"particles and retrain the pattern model on them.",
                f"Patterns are only read on images from a "
                f"{k('composition-contrast')} detector — a topographic image "
                f"cannot show what a particle is made of, so the app reports "
                f"sizes only instead of guessing.",
            ]),
            cls._section("Getting images in", [
                f"Drag images or whole folders onto the left panel, or use "
                f"{k('＋ Import')}. Nothing on disk is moved — the library only "
                f"references your files, and it comes back the next time you open "
                f"the app.",
                f"{k('.tif')} files are converted automatically, so you never "
                f"have to prepare them first.",
                f"The {k('🔍 Search')} box filters the whole library as you type — "
                f"accents and capitals don't matter, so “karisik” finds “Karışık”. "
                f"Click a hit to look at it, {k('double-click')} it (or press "
                f"{k('Enter')}) to jump to where it sits in the tree, {k('Esc')} to "
                f"go back.",
                f"{k('✨ Sort')} reads the file names and files everything by "
                f"sample number, then by region (Alt / Karışık / Üst) or under "
                f"{k('Pattern')} by class. It shows the plan first and only moves "
                f"rows in the panel — never your files.",
                f"Select one or several images (or a folder) and press "
                f"{k('🔬 Analyze')}. With several selected the results are pooled "
                f"into one distribution. Pressing Analyze again re-runs from "
                f"scratch, so any change you made takes effect.",
                f"The small dot on the right of a row says how fresh its analysis "
                f"is: {green('green')} = analysed with the settings and the model "
                f"in force now, {amber('amber')} = analysed earlier, so press "
                f"{k('Analyze')} again to bring it up to date (a folder shows amber "
                f"when anything inside it does). No dot means not analysed yet. "
                f"Re-running is quick — the heavy segmentation step is cached. "
                f"(In {k('🎓 Training')} the same dot switches to the training "
                f"set instead — see below.)",
            ]),
            cls._section("Looking at the image", [
                f"{k('Space')} hides the whole overlay — colours and borders — so "
                f"you see the raw micrograph; press it again to bring back "
                f"exactly what the View bar has ticked.",
                f"Zoom with a trackpad pinch or {k('⌘ + scroll')}, drag to pan, "
                f"and {k('⌘0')} to fit the image back into the view. "
                f"(Double-click does nothing on purpose — labelling clicks the "
                f"same particle twice all the time.)",
            ]),
            cls._section("Measuring — the key idea", [
                f"Particles tinted {blue} are <b>not</b> counted in the size "
                f"statistics: they touch the frame or are too covered for their "
                f"diameter to be trusted.",
                f"With the {k('📏 Measure')} tool, {k('left-click')} a particle to "
                f"show or hide its size line.",
                f"{k('Left-click')} a {blue} particle to <b>force it in</b> to the "
                f"statistics (you're overruling the app).",
                f"{k('Right-click')} <b>flips</b> a particle either way: it drops a "
                f"measured one out (turning it {blue}), and brings a {blue} one back "
                f"in — the same overrule as left-click, from the other button.",
            ]),
            cls._section("Fixing classes", [
                f"The number-key tools ({k('1–6')}, {k('0')} to exclude) repaint a "
                f"particle's class, but only for <b>this view and your exports</b> "
                f"— they do <b>not</b> teach the model.",
                f"Naming a class never changes what is measured: a {blue} "
                f"particle you call lamellar stays out of the size statistics, "
                f"and a measured one you right-click keeps its class but loses "
                f"its size. Only the {k('📏 Measure')} tool moves a particle in "
                f"or out. Note that {k('Analyze')} decides the photo again from "
                f"scratch: every correction you made on it by hand — classes, "
                f"exclusions and measure decisions alike — is discarded, so what "
                f"you see is one consistent answer rather than a mix of old and "
                f"new. It asks first. Labels you made in {k('Training')} are a "
                f"separate thing and are never touched.",
                f"Click the <b>same</b> class a second time when you can't tell "
                f"what the particle is: it keeps its size but counts in no class "
                f"— un-toggling a pattern leaves it plain solid, un-toggling "
                f"{k('undercooled')} leaves it class-less and unpainted. A third "
                f"click hands it back to the model.",
                f"{k('6 Solid')} never excludes a particle: if the model can't "
                f"read a pattern the particle simply stays solid without one — "
                f"shown in red so you can see your own call. Ticking "
                f"{k('Solid')} in the View bar also reddens the pattern-less "
                f"solids the model itself couldn't read.",
                f"{k('⌘Z')} steps back through your corrections — press it "
                f"repeatedly to undo several. It also puts back anything you "
                f"removed from the image list; removing a photo never deletes "
                f"its analysis, so re-adding it brings the analysis along.",
                f"To actually correct the model, turn on {k('🎓 Training')} and "
                f"click the mistakes there.",
                f"{k('🎯 Certainty')} tells you how sure the model is — click a "
                f"particle to see its score.",
            ]),
            cls._section("Training", [
                f"While {k('🎓 Training')} is on, the dot on the right of a row "
                f"stops talking about the analysis and says where the photo "
                f"stands in the <b>training set</b>: {amber('orange')} = not "
                f"added yet, {green('green')} = added exactly as it looks now, "
                f"{accent('blue')} = added, then changed since. A folder shows "
                f"the one that most wants your attention.",
                f"On a blue (changed) photo the panel offers "
                f"{k('↩ Restore saved version')}, which puts back exactly the "
                f"labels it was added with — the way out of a stray key press.",
                f"{k('🔍 Show where the model disagrees')} paints only the "
                f"particles whose class the model now reads differently from "
                f"your labels, in the colour of the <b>model's</b> answer, and "
                f"lists the differences as “you → model”. That is where you see "
                f"it calling an undercooled particle janus. Click a particle "
                f"with no class tool picked to read its comparison; press a "
                f"class key to correct it on the spot.",
                f"Plenty of those disagreements are the model being right. "
                f"Tick {k('Click a particle to accept the model’s answer')} and "
                f"a click adopts its class as your label — no need to find the "
                f"matching class tool each time. Then "
                f"{k('Add photo to training set')} once, at the end.",
                f"The score is also given a second way, with the particles the "
                f"model left <b>without a pattern</b> dropped from both sides: "
                f"a blank is a different kind of miss from a wrong pattern, and "
                f"that number says how often it is wrong when it does commit.",
                f"The comparison starts from the labels the photo was "
                f"<b>added</b> with, with your clicks on it laid over the top. "
                f"Re-run {k('Analyze')} after training a new model to see that "
                f"model's answers.",
                f"{k('📈 Data saturation')} answers the question behind all of "
                f"this: <b>is more labelling still buying accuracy?</b> It "
                f"trains small models on 20 – 100 % of your photos and scores "
                f"each on the same held-out ones, so the curve shows what "
                f"another few hundred labels would be worth. Still rising = "
                f"keep labelling. Gone flat = more of the same data will not "
                f"help, and the next gain has to come from the model itself. "
                f"Measured with each training run while the checkbox next to "
                f"{k('Train model')} is ticked (it adds about a third to the "
                f"time), or on its own with {k('Measure now')} in that window, "
                f"and saved as a chart or a CSV from there.",
                f"When your photos come from <b>two microscopes</b>, the lower "
                f"strip of that chart splits the same held-out particles by "
                f"machine. A machine you have only a few photos of will sit "
                f"well below the other and drag the combined curve flat — that "
                f"is missing coverage of that machine, not saturation, and more "
                f"photos from it is the fix.",
            ]),
            cls._section("Results & saving", [
                f"The chips above the chart filter the distribution to one class; "
                f"{k('Pattern × Size')} shows what each size range is made of.",
                f"{k('Solid / Liquid')} splits every bar of the size "
                f"histogram into the two states and draws the solid share over "
                f"it, so you can see where on the size axis the particles stop "
                f"being undercooled. Type a range into {k('Size range')} under it "
                f"and it tells you exactly what share of the particles that size "
                f"is solid. (The same range also highlights the {k('All')} "
                f"histogram — it is one range, typed in either place.)",
                f"{k('SPHERICITY')} is how round the particles are, averaged over "
                f"them — 1.00 is a perfect circle, 0 is an outline no rounder "
                f"than a square. The class tabs and the {k('Solid / Liquid')} "
                f"page give each class and each state its own figure. The small "
                f"number beside it is the share of the measured particles it "
                f"rests on: a particle touching the edge of the photo is cut by "
                f"the frame rather than by its own shape, so it is left out. "
                f"Hover the tile for the plain circularity (4πA/P²) behind the "
                f"score — that is the number to quote outside this app; the "
                f"scale here is stretched onto the band real particles occupy so "
                f"that photos spread out instead of all reading 0.9-something.",
                f"{k('Cumulative')} reads the distribution the other way round — "
                f"what share of the particles is at or below a given diameter — "
                f"and gives you {k('D10 / D50 / D90')} and the span, the numbers a "
                f"Zetasizer-style report is usually quoted with. The thin coloured "
                f"curves are the individual classes, so you can see at a glance "
                f"which class runs larger.",
                f"{k('💾 Save')} exports the chart and the SEM overlay exactly as "
                f"they look on screen — including every correction you've made.",
            ]),
        ]
        return ('<div style="color:#26313c; font-size:13px;">'
                + "".join(secs) + '</div>')
