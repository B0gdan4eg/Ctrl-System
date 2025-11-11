#!/usr/bin/env python3
"""
Скрипт для тестирования камеры

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import sys
import cv2
from PySide6.QtWidgets import QApplication
from ui.widgets.camera_widget import CameraDialog


def main():
    """Запуск тестового окна камеры"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dialog = CameraDialog()
    dialog.setMinimumSize(800, 700)
    dialog.show()
    
    # Обработчик снимка
    def on_image_captured(image, filename):
        print(f"✅ Изображение захвачено: {filename}")
        print(f"   Размер: {image.shape}")
    
    dialog.image_captured.connect(on_image_captured)
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()