"""
Модуль для обработки изображений (нормализация, предобработка)
"""

import cv2
import numpy as np
from typing import Tuple

from config.settings import (
    BILATERAL_FILTER_D,
    BILATERAL_FILTER_SIGMA_COLOR,
    BILATERAL_FILTER_SIGMA_SPACE
)


def apply_histogram_normalization(
    image: np.ndarray,
    std_range: float = 2.5
) -> np.ndarray:
    """
    Применяет нормализацию Mean ± Std Dev
    
    Args:
        image: Входное изображение
        std_range: Диапазон стандартных отклонений
        
    Returns:
        Нормализованное изображение
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Определяем максимальное значение в зависимости от типа данных
    if gray.dtype == np.uint8:
        gray = (gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)
        max_val = 1023
    elif gray.dtype == np.uint16:
        max_val = 1023
    else:
        gray = gray.astype(np.uint16)
        max_val = 1023
    
    # Статистика
    mean = np.mean(gray)
    std = np.std(gray)
    
    input_min = int(max(0, mean - std_range * std))
    input_max = int(min(max_val, mean + std_range * std))
    
    if input_max <= input_min:
        input_max = input_min + 1
    
    # Применение levels adjustment
    gray_float = gray.astype(np.float32)
    normalized = (gray_float - input_min) / (input_max - input_min)
    normalized = np.clip(normalized, 0, 1) * max_val
    result = normalized.astype(np.uint16)
    
    # Если входное было цветное, применяем ко всем каналам
    if len(image.shape) == 3:
        result_color = np.zeros_like(image, dtype=np.uint16)
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            if image.dtype == np.uint8:
                channel = channel * (max_val / 255.0)
            normalized_channel = (channel - input_min) / (input_max - input_min)
            normalized_channel = np.clip(normalized_channel, 0, 1) * max_val
            result_color[:, :, c] = normalized_channel.astype(np.uint16)
        return result_color
    
    return result


def preprocess_for_detection(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Предобработка изображения для детекции дефектов
    
    Args:
        image: Входное изображение
        target_size: Целевой размер (width, height)
        
    Returns:
        Обработанное изображение
    """
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()
    
    # Конвертация в uint8 если нужно (bilateralFilter требует uint8 или float32)
    if img.dtype == np.uint16:
        img = (img / 1023.0 * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Шумоподавление
    img_denoised = cv2.bilateralFilter(
        img,
        BILATERAL_FILTER_D,
        BILATERAL_FILTER_SIGMA_COLOR,
        BILATERAL_FILTER_SIGMA_SPACE
    )
    
    # Resize
    img_resized = cv2.resize(img_denoised, target_size, interpolation=cv2.INTER_AREA)
    
    return img_resized


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Конвертирует изображение в grayscale с поддержкой разных форматов
    
    Args:
        image: Входное изображение
        
    Returns:
        Grayscale изображение
    """
    if len(image.shape) == 2:
        return image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Нормализация для 12-bit изображений
    if image.dtype == np.uint16:
        gray = (gray / 4).astype(np.uint16)
    
    return gray


def prepare_for_display(image: np.ndarray) -> np.ndarray:
    """
    Подготавливает изображение для отображения (конвертация в uint8 RGB)
    
    Args:
        image: Входное изображение
        
    Returns:
        RGB изображение uint8 для отображения
    """
    if len(image.shape) == 2:
        # Grayscale -> RGB
        if image.dtype == np.uint16:
            img_uint8 = (image / 1023.0 * 255.0).astype(np.uint8)
        else:
            img_uint8 = image
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        # Уже RGB
        if image.dtype == np.uint16:
            return (image / 1023.0 * 255.0).astype(np.uint8)
        else:
            return image