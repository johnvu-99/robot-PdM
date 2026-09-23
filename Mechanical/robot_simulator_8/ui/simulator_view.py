"""
The 3D viewport widget.

It owns no simulation state. Frames are pulled from the mailbox by the main
window's poll timer, and mouse input is turned into camera commands that are
posted to the simulation thread.
"""

from PyQt5.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter
from PyQt5.QtWidgets import QSizePolicy, QWidget

import config

BACKGROUND = QColor(18, 20, 24)
PLACEHOLDER_TEXT = QColor(130, 140, 152)
BANNER_BACKGROUND = QColor(190, 40, 45, 210)
BANNER_TEXT = QColor(255, 255, 255)
HINT_TEXT = QColor(150, 160, 172)


class SimulatorView(QWidget):

    command = pyqtSignal(str, object)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._image = None
        self._buffer = None          # keeps the bytes alive behind the QImage
        self._serial = 0
        self._banner = ""
        self._status = "Starting"

        self._drag_button = None
        self._drag_x = 0
        self._drag_y = 0

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(config.RESIZE_DEBOUNCE_MS)
        self._resize_timer.timeout.connect(self._publish_viewport)

    # -- frame input --------------------------------------------------------

    def update_from_mailbox(self, mailbox):
        """Called by the UI poll timer. Repaints only when a frame is new."""
        item = mailbox.take(self._serial)
        if item is None:
            return False
        serial, data, width, height = item
        self._serial = serial
        self._buffer = data
        self._image = QImage(data, width, height, width * 4, QImage.Format_RGBA8888)
        self.update()
        return True

    def set_banner(self, text):
        if text == self._banner:
            return
        self._banner = text
        self.update()

    def set_status(self, text):
        self._status = text
        if self._image is None:
            self.update()

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)

        if self._image is None:
            painter.setPen(PLACEHOLDER_TEXT)
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, self._status)
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(self._fitted_rect(), self._image)

        if self._banner:
            self._draw_banner(painter)
        self._draw_hint(painter)

    def _fitted_rect(self):
        """Letterbox the frame into the widget without distorting it."""
        area_w = self.width()
        area_h = self.height()
        image_w = self._image.width()
        image_h = self._image.height()
        if image_w <= 0 or image_h <= 0:
            return self.rect()

        scale = min(float(area_w) / image_w, float(area_h) / image_h)
        width = int(image_w * scale)
        height = int(image_h * scale)
        return QRect((area_w - width) // 2, (area_h - height) // 2, width, height)

    def _draw_banner(self, painter):
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        painter.setFont(font)
        rect = QRect(0, 0, self.width(), 34)
        painter.fillRect(rect, BANNER_BACKGROUND)
        painter.setPen(BANNER_TEXT)
        painter.drawText(rect, Qt.AlignCenter, self._banner)

    def _draw_hint(self, painter):
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(HINT_TEXT)
        rect = QRect(10, self.height() - 24, self.width() - 20, 18)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter,
                         "Drag to orbit, middle or right drag to pan, "
                         "wheel to zoom, keys 1-5 for views")

    # -- sizing -------------------------------------------------------------

    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)
        # Debounced: dragging a window edge must not reallocate render buffers
        # on every mouse move.
        self._resize_timer.start()

    def showEvent(self, event):
        QWidget.showEvent(self, event)
        self._resize_timer.start()

    def _publish_viewport(self):
        ratio = self.devicePixelRatioF()
        width = int(self.width() * ratio)
        height = int(self.height() * ratio)
        if width <= 0 or height <= 0:
            return
        self.command.emit("set_viewport", {"width": width, "height": height})

    # -- input --------------------------------------------------------------

    def mousePressEvent(self, event):
        self._drag_button = event.button()
        self._drag_x = event.x()
        self._drag_y = event.y()
        self.setFocus(Qt.MouseFocusReason)

    def mouseReleaseEvent(self, event):
        self._drag_button = None

    def mouseMoveEvent(self, event):
        if self._drag_button is None:
            return
        dx = event.x() - self._drag_x
        dy = event.y() - self._drag_y
        self._drag_x = event.x()
        self._drag_y = event.y()
        if dx == 0 and dy == 0:
            return
        if self._drag_button == Qt.LeftButton:
            self.command.emit("camera_orbit", {"dx": float(dx), "dy": float(dy)})
        else:
            self.command.emit("camera_pan", {"dx": float(dx), "dy": float(dy)})

    def wheelEvent(self, event):
        notches = event.angleDelta().y() / 120.0
        if notches:
            self.command.emit("camera_zoom", {"notches": float(notches)})

    def mouseDoubleClickEvent(self, event):
        self.command.emit("camera_preset", {"name": "Home"})

    def keyPressEvent(self, event):
        keys = {
            Qt.Key_1: "Home",
            Qt.Key_2: "Front",
            Qt.Key_3: "Side",
            Qt.Key_4: "Top",
            Qt.Key_5: "Isometric",
        }
        name = keys.get(event.key())
        if name is None:
            QWidget.keyPressEvent(self, event)
            return
        self.command.emit("camera_preset", {"name": name})
