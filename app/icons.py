"""Line icons for buttons, drawn from small SVG paths.

They replace the emoji the buttons used to carry (🔬 💾 🎓 …). An emoji is a
colour picture the system font draws, so it never took the button's text
colour, looked different on every OS and turned the toolbar into a sticker
sheet. These are single-stroke outlines on a 24 x 24 grid: they are drawn in
the colour they are given, so a button's icon matches its label, goes white on
the blue primary button and greys out with it when disabled.

set_icon(button, "save") is all a caller needs. The icon is rendered at twice
its size for Retina screens, and a few transparent pixels to its right stand in
for the icon-to-text gap QPushButton has no setting for.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtSvg

# ink colours, the same ones ui_theme's STYLE uses for the matching text
INK = "#1a2129"          # QPushButton text
INK_SIDE = "#4a5560"     # the IMAGES panel's small buttons
INK_MINI = "#5a6472"     # the TRAINING card's small buttons
INK_ON = "#12386e"       # a checked button
INK_OFF = "#aab2bc"      # a disabled button
WHITE = "#ffffff"        # on the blue primary button
WHITE_OFF = "#eef2ff"    # the primary button, disabled

# one stroke-only path set per icon, 24 x 24
PATHS = {
    "analyze": '<circle cx="12" cy="12" r="6.5"/><circle cx="12" cy="12" r="1.4"/>'
               '<path d="M12 2.5v3.2M12 18.3v3.2M2.5 12h3.2M18.3 12h3.2"/>',
    "busy": '<path d="M12 3.5v3M12 17.5v3M3.5 12h3M17.5 12h3M6 6l2.1 2.1M15.9 15.9 18 18'
            'M6 18l2.1-2.1M15.9 8.1 18 6"/>',
    "save": '<path d="M12 3.5v11M7.5 10l4.5 4.5 4.5-4.5"/>'
            '<path d="M4.5 16v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V16"/>',
    "training": '<path d="M2.5 9.5 12 5l9.5 4.5L12 14z"/><path d="M6.5 11.6v4.4c3 2.2 8 2.2 11 0v-4.4"/>'
                '<path d="M21.5 9.5v5"/>',
    "guide": '<path d="M2.5 5.5h6a3.5 3.5 0 0 1 3.5 3.5v11a2.6 2.6 0 0 0-2.6-2.6H2.5z"/>'
             '<path d="M21.5 5.5h-6A3.5 3.5 0 0 0 12 9v11a2.6 2.6 0 0 1 2.6-2.6h6.9z"/>',
    "update": '<circle cx="12" cy="12" r="8.5"/><path d="M12 16.5v-9M8 11.5l4-4 4 4"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "folder": '<path d="M3 7.5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "folder_plus": '<path d="M3 7.5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
                   '<path d="M12 10.5v5M9.5 13h5"/>',
    "sort": '<path d="M3.5 15.5 7 19l3.5-3.5M7 19V5M13 6h8M13 10h6M13 14h4"/>',
    "search": '<circle cx="11" cy="11" r="6"/><path d="M15.5 15.5 20 20"/>',
    "measure": '<path d="M3.5 16.5 16.5 3.5l4 4-13 13z"/><path d="M7.5 12.5l2 2M10.5 9.5l2 2M13.5 6.5l2 2"/>',
    "certainty": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/><circle cx="12" cy="17" r="1.2"/>',
    "restore": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-2.3-5.7L20 8.5"/><path d="M20 3.5v5h-5"/>',
    "clear": '<path d="M8 20.5h12"/><path d="M4.6 14.6 13.4 5.8a2 2 0 0 1 2.8 0l3 3a2 2 0 0 1 0 2.8L11 20H8.4z"/>'
             '<path d="M9 10.2l5.8 5.8"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/>',
    "model": '<rect x="5.5" y="5.5" width="13" height="13" rx="2.5"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/>'
             '<path d="M9.5 2.5v3M14.5 2.5v3M9.5 18.5v3M14.5 18.5v3M2.5 9.5h3M2.5 14.5h3M18.5 9.5h3M18.5 14.5h3"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "apply": '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 20h14"/>',
    "grid": '<rect x="4" y="4" width="6.5" height="6.5" rx="1.2"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.2"/>'
            '<rect x="4" y="13.5" width="6.5" height="6.5" rx="1.2"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.2"/>',
    "bars": '<path d="M3.5 20h17"/><path d="M7 16.5v-5M12 16.5V6M17 16.5V9.5"/>',
    "curve": '<path d="M3.5 20h17M3.5 20V4"/><path d="M6.5 16c2.5-6 5-9 13-10"/>',
    "report": '<rect x="5.5" y="4.5" width="13" height="16.5" rx="2"/><path d="M9 4.5V3h6v1.5"/>'
              '<path d="M8.5 10h7M8.5 13.5h7M8.5 17h4"/>',
}


def _pixmap(name, colour, size, gap, scale=2):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           f'stroke="{colour}" stroke-width="1.9" stroke-linecap="round" '
           f'stroke-linejoin="round">{PATHS[name]}</svg>')
    r = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    img = QtGui.QImage((size + gap) * scale, size * scale,
                       QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    r.render(p, QtCore.QRectF(0, 0, size * scale, size * scale))
    p.end()
    pm = QtGui.QPixmap.fromImage(img)
    pm.setDevicePixelRatio(scale)
    return pm


def icon(name, colour=INK, *, on=None, disabled=INK_OFF, size=16, gap=6):
    """A QIcon of `name` in `colour`; `on` is the colour while the button is
    checked, `disabled` while it is greyed out. `gap` px of empty space follow
    the drawing, so the label doesn't sit against it."""
    ic = QtGui.QIcon()
    ic.addPixmap(_pixmap(name, colour, size, gap), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    ic.addPixmap(_pixmap(name, on or colour, size, gap), QtGui.QIcon.Normal, QtGui.QIcon.On)
    ic.addPixmap(_pixmap(name, disabled, size, gap), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
    ic.addPixmap(_pixmap(name, disabled, size, gap), QtGui.QIcon.Disabled, QtGui.QIcon.On)
    return ic


def set_icon(button, name, colour=INK, *, on=None, disabled=INK_OFF, size=16, gap=6):
    """Put the line icon `name` on a QPushButton, sized to match."""
    button.setIcon(icon(name, colour, on=on, disabled=disabled, size=size, gap=gap))
    button.setIconSize(QtCore.QSize(size + gap, size))


def set_primary_icon(button, name, size=16):
    """The white version, for the blue `primary` buttons."""
    set_icon(button, name, WHITE, disabled=WHITE_OFF, size=size)
