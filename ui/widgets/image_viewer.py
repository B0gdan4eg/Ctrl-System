"""
Виджет для отображения изображений
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from config.settings import MIN_IMAGE_VIEWER_SIZE


class ImageViewer(QLabel):
    """Виджет для отображения изображений с автомасштабированием"""
    
    def __init__(self, title: str = ""):
        """
        Args:
            title: Заголовок виджета
        """
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(MIN_IMAGE_VIEWER_SIZE, MIN_IMAGE_VIEWER_SIZE)
        self.setStyleSheet(
            "border: 2px solid #555; "
            "background-color: #2b2b2b; "
            "color: white; "
            "font-size: 14px;"
        )
        self.setText(f"{title}\n\nNo image")
        
    def display_image(self, image: np.ndarray):
        """
        Отображает OpenCV изображение (BGR формат)
        
        Args:
            image: OpenCV изображение (BGR или grayscale)
        """
        if image is None:
            return
        
        # Конвертация BGR -> RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Нормализация для отображения
        if image.dtype == np.uint16:
            display_image = (rgb_image / 1023.0 * 255.0).astype(np.uint8)
        else:
            display_image = rgb_image
        
        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(
            display_image.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
    
    def clear_image(self):
        """Очищает отображаемое изображение"""
        self.clear()
        self.setText(f"{self.title}\n\nNo image")