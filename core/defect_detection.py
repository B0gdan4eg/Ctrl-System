"""
Модуль для детекции дефектов с использованием нейронной сети
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from config.settings import COLOR_DEFECT, COLOR_ZONE_BORDER, DEFECT_OVERLAY_ALPHA


def predict_defects(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Предсказание дефектов с использованием U-Net модели
    
    Args:
        model: Обученная модель
        image: Входное изображение (должно быть предобработано)
        device: Устройство для вычислений (CPU/GPU)
        threshold: Порог для бинаризации предсказания
        
    Returns:
        Бинарная маска дефектов (0 или 255)
    """
    model.eval()
    
    # Убедимся что изображение RGB и uint8
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    if image_rgb.dtype == np.uint16:
        image_rgb = (image_rgb / 1023.0 * 255.0).astype(np.uint8)
    
    # Нормализация для модели
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
    
    # Постобработка
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    
    return mask


def visualize_defects(
    image: np.ndarray,
    defect_mask: np.ndarray,
    contours: list = None,
    alpha: float = DEFECT_OVERLAY_ALPHA
) -> np.ndarray:
    """
    Визуализирует дефекты на изображении
    
    Args:
        image: Исходное изображение (RGB)
        defect_mask: Маска дефектов
        contours: Опционально - контуры зон для отрисовки
        alpha: Прозрачность наложения дефектов
        
    Returns:
        Изображение с визуализированными дефектами
    """
    result = image.copy()
    
    # Красное наложение для дефектов
    mask_colored = np.zeros_like(result)
    mask_colored[defect_mask > 0] = COLOR_DEFECT
    
    result = cv2.addWeighted(result, 1.0, mask_colored, alpha, 0)
    
    # Рисуем контуры зон если предоставлены
    if contours is not None:
        for idx, cnt in enumerate(contours, 1):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), COLOR_ZONE_BORDER, 2)
            cv2.putText(
                result,
                f"Zone {idx}",
                (x+5, y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_ZONE_BORDER,
                2
            )
    
    return result


def calculate_defect_statistics(
    defect_mask: np.ndarray,
    zone_area: int = None
) -> dict:
    """
    Вычисляет статистику дефектов
    
    Args:
        defect_mask: Маска дефектов
        zone_area: Опционально - площадь зоны для расчета процента
        
    Returns:
        Словарь со статистикой
    """
    defect_pixels = np.sum(defect_mask > 0)
    
    stats = {
        'defect_pixels': int(defect_pixels),
        'has_defects': defect_pixels > 0
    }
    
    if zone_area is not None:
        stats['defect_percentage'] = (defect_pixels / zone_area * 100) if zone_area > 0 else 0
    
    return stats


def create_defect_heatmap(
    defect_mask: np.ndarray,
    original_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Создает тепловую карту дефектов
    
    Args:
        defect_mask: Маска дефектов
        original_shape: Исходный размер изображения (height, width)
        
    Returns:
        Тепловая карта дефектов
    """
    # Resize к исходному размеру
    mask_resized = cv2.resize(
        defect_mask,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Применяем colormap для визуализации
    heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    
    return heatmap