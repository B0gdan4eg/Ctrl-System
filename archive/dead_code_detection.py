"""
Архив: неиспользуемый код из core/defect_detection.py
Перемещено: 2026-05-14
Причина: функции нигде не вызываются в production коде
"""
import cv2
import numpy as np
from typing import Tuple

from config.settings import COLOR_DEFECT, COLOR_ZONE_BORDER, DEFECT_OVERLAY_ALPHA


# === Из core/defect_detection.py ===

def visualize_defects(
    image: np.ndarray,
    defect_mask: np.ndarray,
    contours: list = None,
    alpha: float = DEFECT_OVERLAY_ALPHA,
    pixel_size_mm: float = None
) -> np.ndarray:
    """
    Визуализирует дефекты на изображении

    Args:
        image: Исходное изображение (RGB)
        defect_mask: Маска дефектов
        contours: Опционально - контуры зон для отрисовки
        alpha: Прозрачность наложения дефектов
        pixel_size_mm: Размер одного пикселя в мм (для вывода размеров зон)

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

            # Надпись "Zone X"
            cv2.putText(
                result,
                f"Zone {idx}",
                (x+5, y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_ZONE_BORDER,
                2
            )

            # Размер зоны в мм — снизу под bbox, жирным
            if pixel_size_mm is not None and pixel_size_mm > 0:
                width_mm = w * pixel_size_mm
                height_mm = h * pixel_size_mm
                size_text = f"{width_mm:.3f} x {height_mm:.3f} mm"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 3
                (tw, th), _ = cv2.getTextSize(size_text, font, font_scale, thickness)

                img_h, img_w = result.shape[:2]
                gap = 12
                tx = x + (w - tw) // 2
                tx = max(5, min(tx, img_w - tw - 5))
                ty = y + h + gap + th
                # Если снизу не помещается — поднимем над bbox
                if ty > img_h - 5:
                    ty = max(th + 5, y - gap)

                cv2.putText(
                    result, size_text, (tx, ty),
                    font, font_scale, COLOR_ZONE_BORDER, thickness, cv2.LINE_AA,
                )

    return result


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
