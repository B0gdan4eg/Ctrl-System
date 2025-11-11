"""
Модуль для поиска и выделения зон на изображении
"""

import cv2
import numpy as np
from typing import List, Tuple

from config.settings import CLIPL_LIMIT_VALUES, THRESHOLD_VALUES


def find_top_zones(
    img_gray_original: np.ndarray,
    top_n: int = 5,
    min_area_ratio: float = 0.3,
    max_area_ratio: float = 0.95
) -> List[np.ndarray]:
    """
    Находит топ-N зон на изображении
    
    Args:
        img_gray_original: Исходное grayscale изображение
        top_n: Количество зон для поиска
        min_area_ratio: Минимальное отношение площади зоны к максимальной
        max_area_ratio: Максимальное отношение площади зоны к изображению
        
    Returns:
        Список контуров найденных зон
    """
    img_for_search = img_gray_original.copy()
    img_area = img_gray_original.shape[0] * img_gray_original.shape[1]

    # Нормализация
    in_min, in_max = np.percentile(img_for_search, 1), np.percentile(img_for_search, 99)
    img_for_search = (img_for_search - in_min) / (in_max - in_min) * 255
    img_for_search = np.clip(img_for_search, 0, 255).astype(np.uint8)

    best_contours = []
    current_target = top_n
    
    while current_target >= 1 and len(best_contours) < current_target:
        for clipLimit in CLIPL_LIMIT_VALUES:
            for threshold_val in THRESHOLD_VALUES:
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_for_search)
                
                _, mask = cv2.threshold(img_clahe, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    continue
                
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                good_contours = []
                for cnt in contours[:current_target * 3]:
                    area = cv2.contourArea(cnt)
                    area_ratio_to_image = area / img_area
                    
                    if area_ratio_to_image > max_area_ratio:
                        continue
                    
                    if len(good_contours) > 0:
                        max_good_area = cv2.contourArea(good_contours[0])
                        if area < max_good_area * min_area_ratio:
                            continue
                    
                    good_contours.append(cnt)
                
                if len(good_contours) >= current_target:
                    best_contours = good_contours[:current_target]
                    break
                
                if len(good_contours) > len(best_contours):
                    best_contours = good_contours[:current_target]
            
            if len(best_contours) >= current_target:
                break
        
        if len(best_contours) >= current_target:
            all_good = True
            if len(best_contours) > 0:
                max_area = cv2.contourArea(best_contours[0])
                for cnt in best_contours:
                    area = cv2.contourArea(cnt)
                    if area < max_area * min_area_ratio:
                        all_good = False
                        break
            
            if all_good:
                break
        
        current_target -= 1
    
    # Fallback: если ничего не найдено
    if len(best_contours) == 0:
        best_contours = _fallback_zone_search(img_for_search, top_n, max_area_ratio, img_area)
    
    # Сортировка по горизонтальной позиции
    best_contours = sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])
    return best_contours


def _fallback_zone_search(
    img_for_search: np.ndarray,
    top_n: int,
    max_area_ratio: float,
    img_area: int
) -> List[np.ndarray]:
    """
    Резервный метод поиска зон с базовыми параметрами
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_for_search)
    _, mask = cv2.threshold(img_clahe, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered = [c for c in contours if cv2.contourArea(c) / img_area <= max_area_ratio]
    best_contours = sorted(filtered, key=cv2.contourArea, reverse=True)[:top_n]
    
    return best_contours


def extract_zone_roi(
    image: np.ndarray,
    contour: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Извлекает ROI (Region of Interest) из изображения по контуру
    
    Args:
        image: Исходное изображение
        contour: Контур зоны
        
    Returns:
        Tuple из (ROI изображение, bbox координаты (x, y, w, h))
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w].copy()
    return roi, (x, y, w, h)