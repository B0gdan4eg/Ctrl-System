#!/usr/bin/env python3
"""
Точка входа в приложение детекции дефектов
"""

import sys
import os

# Инициализируем COM в STA-режиме ДО Qt и torch,
# иначе OleInitialize() падает с 0x80010106 ("режим потока уже установлен"),
# а нативный QFileDialog ("Загрузить изображение") виснет при открытии.
# COINIT_APARTMENTTHREADED == 0x2 (значение 0x0 = COINIT_MULTITHREADED — это и был баг).
if sys.platform == "win32":
    try:
        import ctypes
        COINIT_APARTMENTTHREADED = 0x2
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        # 0x80010106 = RPC_E_CHANGED_MODE — COM уже инициализирован в другом режиме.
        # Это не критично, главное — мы попытались установить STA первыми.
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui import DefectDetectionApp
from utils import app_logger, resource_path


def get_resource_path(relative_path):
    """Получить абсолютный путь к ресурсу (работает для dev и PyInstaller)"""
    return resource_path(relative_path)


def main():
    """Главная функция запуска приложения"""
    app_logger.info("="*50)
    app_logger.info("Запуск приложения детекции дефектов")
    app_logger.info("="*50)

    try:
        # Создание приложения Qt
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        # Установка иконки приложения
        icon_path = get_resource_path('icon.ico')
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(str(icon_path)))
            app_logger.info(f"Иконка приложения установлена: {icon_path}")
        else:
            app_logger.warning(f"Файл иконки не найден: {icon_path}")
        
        # Создание и отображение главного окна
        window = DefectDetectionApp()
        window.show()
        
        app_logger.info("Приложение успешно запущено")
        
        # Запуск event loop
        exit_code = app.exec()
        
        app_logger.info(f"Приложение завершено с кодом: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        app_logger.error(f"Критическая ошибка при запуске: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()