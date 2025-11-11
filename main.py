#!/usr/bin/env python3
"""
Точка входа в приложение детекции дефектов
"""

import sys
from PySide6.QtWidgets import QApplication
from ui import DefectDetectionApp
from utils import app_logger


def main():
    """Главная функция запуска приложения"""
    app_logger.info("="*50)
    app_logger.info("Запуск приложения детекции дефектов")
    app_logger.info("="*50)
    
    try:
        # Создание приложения Qt
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
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