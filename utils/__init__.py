"""
Утилиты
"""
import os
import sys
from pathlib import Path

from .logger import setup_logger, app_logger


def resource_path(relative_path: str) -> Path:
    """Абсолютный путь к ресурсу для dev И PyInstaller-сборки.

    В сборке PyInstaller (--onedir) данные из spec лежат в `sys._MEIPASS`
    (= `dist/Ctrl-System/_internal/`). build_exe.py при этом переносит
    best_model.pth и icon.ico в корень dist рядом с .exe — поэтому если
    файла в _MEIPASS нет, ищем в папке исполняемого файла.
    В dev режиме — относительно cwd проекта.
    """
    if hasattr(sys, '_MEIPASS'):
        candidate = Path(sys._MEIPASS) / relative_path
        if candidate.exists():
            return candidate
        return Path(sys.executable).resolve().parent / relative_path
    return Path(os.path.abspath('.')) / relative_path


__all__ = ['setup_logger', 'app_logger', 'resource_path']