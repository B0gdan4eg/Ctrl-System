"""
Виджет для отображения изображений с зумом колесом и панорамированием
"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from config.settings import MIN_IMAGE_VIEWER_SIZE


class ImageViewer(QGraphicsView):
    """QGraphicsView с зумом колесом мыши и панорамированием через drag."""

    MIN_SCALE = 0.05
    MAX_SCALE = 40.0
    ZOOM_IN_FACTOR = 1.25
    ZOOM_OUT_FACTOR = 1.0 / 1.25

    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title
        self.setMinimumSize(MIN_IMAGE_VIEWER_SIZE, MIN_IMAGE_VIEWER_SIZE)
        self.setStyleSheet(
            "border: 2px solid #555; background-color: #2b2b2b;"
        )

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self._placeholder = QGraphicsSimpleTextItem(f"{title}\n\nNo image")
        self._placeholder.setBrush(QColor("white"))
        f = QFont()
        f.setPointSize(14)
        self._placeholder.setFont(f)
        self._scene.addItem(self._placeholder)

        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._has_image = False
        self._user_zoomed = False

    def display_image(self, image: np.ndarray) -> None:
        if image is None:
            return

        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.dtype == np.uint16:
            display_image = (rgb_image / 1023.0 * 255.0).astype(np.uint8)
        else:
            display_image = rgb_image

        # Контигуальный буфер — Qt не делает копию, нужно держать ссылку.
        self._buffer = np.ascontiguousarray(display_image)
        h, w, ch = self._buffer.shape
        qt_image = QImage(self._buffer.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self._placeholder.setVisible(False)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        # Первый показ — fit-to-view. После пользовательского зума сохраняем масштаб.
        if not self._has_image or not self._user_zoomed:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._has_image = True

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._placeholder.setVisible(True)
        self._has_image = False
        self._user_zoomed = False
        self.resetTransform()

    def reset_zoom(self) -> None:
        self._user_zoomed = False
        if self._has_image:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self._has_image:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = self.ZOOM_IN_FACTOR if delta > 0 else self.ZOOM_OUT_FACTOR
        new_scale = self.transform().m11() * factor
        if new_scale < self.MIN_SCALE or new_scale > self.MAX_SCALE:
            return

        self.scale(factor, factor)
        self._user_zoomed = True
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and not self._user_zoomed:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        self.reset_zoom()
        event.accept()
