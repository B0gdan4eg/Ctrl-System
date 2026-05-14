"""
Утилита для логирования
"""

import logging
import re
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "defect_detection", log_dir: str = "logs") -> logging.Logger:
    """
    Настраивает логгер для приложения

    Args:
        name: Имя логгера
        log_dir: Директория для логов

    Returns:
        Настроенный логгер
    """
    # Санация имени для использования в пути файла
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

    # Создаем директорию для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Файловый handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Консольный handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Добавляем handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Глобальный логгер для приложения
app_logger = setup_logger()