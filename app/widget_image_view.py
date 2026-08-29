"""The micrograph viewport: a pannable, zoomable image, and the plain scaled
image used for the result figures.

ImageView carries the zoom rules the rest of the app relies on — see _scale for
why a transform can go degenerate and why recovering from that is not optional.
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets


def pil_to_qpix(im):
    im = im.convert("RGB")
    qim = QtGui.QImage(im.tobytes("raw", "RGB"), im.width, im.height,
                       im.width * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qim.copy())


class ImageView(QtWidgets.QGraphicsView):
    """Pannable/zoomable image (trackpad pinch + ⌘-wheel; drag to pan; ⌘0 fits).

    The zoom is bounded relative to the fitted scale — see MIN_ZOOM/MAX_ZOOM.
    """

    clicked = QtCore.Signal(float, float)   # image coords of a plain (no-drag) click
    right_clicked = QtCore.Signal(float, float)   # ...of a plain right-click

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._fit = True
        self._press = None
        self.setRenderHints(QtGui.QPainter.SmoothPixmapTransform | QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        # zooming in still pans (drag / trackpad), but without the scrollbars
        # appearing along the edges of the micrograph
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QtGui.QColor(0, 0, 0, 0))
        self.setStyleSheet("QGraphicsView{background:transparent;border:none;}")
        self.viewport().setAutoFillBackground(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def set_image(self, pil):
        pix = pil_to_qpix(pil)
        if self._item is None:
            self._item = self._scene.addPixmap(pix)
        else:
            self._item.setPixmap(pix)
        self._scene.setSceneRect(QtCore.QRectF(pix.rect()))
        if self._fit:
            self.fit()

    def clear_image(self):
        if self._item is not None:
            self._scene.removeItem(self._item)
            self._item = None

    # How far from the fitted view zooming may go. Bounds matter for more than
    # taste: QGraphicsView zooms by MULTIPLYING its transform, so a single bad
    # factor is permanent — 0 makes the matrix singular and inf/NaN poisons it,
    # and every later scale() keeps it there. That is the "zoom stops working
    # until I restart the app" bug (reported 2026-08-02): a trackpad pinch
    # delivers a NativeGesture value of -1.0 often enough (and wheel deltas can
    # be large enough) to reach it, and nothing here could ever recover, because
    # fit() only runs on resize while _fit is True — which zooming has cleared.
    MIN_ZOOM, MAX_ZOOM = 0.5, 60.0        # x the fitted scale

    def fit(self):
        if self._item is not None:
            self.resetTransform()
            self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)
            self._fit_scale = self._scale() or 1.0
            self._fit = True

    def _scale(self):
        """Current zoom, or None if the transform has gone degenerate."""
        m = self.transform().m11()
        return float(m) if math.isfinite(m) and m > 1e-12 else None

    def _zoom(self, f):
        if not math.isfinite(f) or f <= 0:
            return                          # a gesture we can't act on
        cur = self._scale()
        if cur is None:                     # already poisoned -> heal, don't stack
            self.fit()
            return
        base = getattr(self, "_fit_scale", 0.0) or cur
        f = min(max(f, 0.2), 5.0)           # no single event may leap the range
        target = min(max(cur * f, base * self.MIN_ZOOM), base * self.MAX_ZOOM)
        if abs(target - cur) < 1e-9:
            return                          # already at the stop
        self.scale(target / cur, target / cur)
        self._fit = False

    def wheelEvent(self, e):
        if e.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
            self._zoom(1.0015 ** e.angleDelta().y()); e.accept()
        else:
            super().wheelEvent(e)

    def event(self, e):
        if e.type() == QtCore.QEvent.NativeGesture and \
                e.gestureType() == QtCore.Qt.ZoomNativeGesture:
            self._zoom(1.0 + e.value()); return True
        return super().event(e)

    def mouseDoubleClickEvent(self, e):
        # Deliberately does NOTHING (user rule, 2026-07-30). Double-click used to
        # reset the zoom, but labelling means clicking the same particle twice in
        # a row all the time, and the view kept jumping back to fit.
        e.accept()

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        press, self._press = self._press, None
        super().mouseReleaseEvent(e)
        # a click is a press+release without a drag in between (pan untouched)
        if press is not None and \
                (e.position().toPoint() - press).manhattanLength() < 5:
            p = self.mapToScene(e.position().toPoint())
            if e.button() == QtCore.Qt.RightButton:
                self.right_clicked.emit(p.x(), p.y())
            else:
                self.clicked.emit(p.x(), p.y())

    def resizeEvent(self, e):
        if self._fit:
            self.fit()
        super().resizeEvent(e)


class ScaledImage(QtWidgets.QLabel):
    """A QLabel that keeps a pixmap and scales it to fit (aspect-preserving)."""

    def __init__(self, placeholder="", on_resize=None, radius=0):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("color:#9aa4ae;")
        self._placeholder = placeholder
        self.setText(placeholder)
        # A QLabel holding a pixmap reports that pixmap as its minimum size, so a
        # figure rendered even slightly too large PINS the label and the layout
        # can never shrink it again — the panel then keeps a squat chart with a
        # gap under it for good. Ignoring the hint hands the size back to the
        # layout, which is the only thing that should decide it here.
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                           QtWidgets.QSizePolicy.Ignored)
        self._pix = None
        self._on_resize = on_resize
        self._radius = radius     # display-px corner radius (matches the card)

    def set_image(self, pil):
        self._pix = pil_to_qpix(pil)
        self._rescale()

    def clear_img(self):
        self._pix = None
        self.setText(self._placeholder)

    def resizeEvent(self, e):
        self._rescale()
        if self._on_resize is not None:
            self._on_resize()
        super().resizeEvent(e)

    def _rescale(self):
        if self._pix is None:
            return
        # Render at physical (Retina) resolution so downscaled figures stay crisp.
        dpr = self.devicePixelRatioF()
        target = QtCore.QSize(max(1, int(self.width() * dpr)),
                              max(1, int(self.height() * dpr)))
        scaled = self._pix.scaled(target, QtCore.Qt.KeepAspectRatio,
                                  QtCore.Qt.SmoothTransformation)
        if self._radius > 0:
            rounded = QtGui.QPixmap(scaled.size())
            rounded.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(rounded)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            clip = QtGui.QPainterPath()
            r = self._radius * dpr
            clip.addRoundedRect(QtCore.QRectF(rounded.rect()), r, r)
            p.setClipPath(clip)
            p.drawPixmap(0, 0, scaled)
            p.end()
            scaled = rounded
        scaled.setDevicePixelRatio(dpr)
        self.setPixmap(scaled)
