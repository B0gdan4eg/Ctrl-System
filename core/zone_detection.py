"""
Модуль для поиска и выделения зон на изображении
Интегрирует улучшенные методы детекции с фильтрацией артефактов
"""

import cv2
import numpy as np
from typing import List, Tuple

from config.settings import CLIPL_LIMIT_VALUES, THRESHOLD_VALUES

# Кэш CLAHE объектов для избежания повторного создания
_CLAHE_CACHE: dict = {}


def _get_clahe(clip_limit: float, tile_grid_size: tuple = (8, 8)) -> 'cv2.CLAHE':
    key = (clip_limit, tile_grid_size)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return _CLAHE_CACHE[key]


def apply_image_correction(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Применяет коррекцию яркости и контраста к изображению

    Args:
        image: Входное изображение (grayscale)
        brightness: Значение яркости (-100 до +100)
        contrast: Множитель контраста (0.1 до 5.0)

    Returns:
        Скорректированное изображение
    """
    img = image.astype(np.float32)
    img = contrast * img + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def pixels_to_mm(
    pixels: int,
    pixel_size_mm: float
) -> float:
    """
    Конвертирует пиксели в миллиметры

    Args:
        pixels: Количество пикселей
        pixel_size_mm: Размер одного пикселя в мм

    Returns:
        Размер в миллиметрах
    """
    return pixels * pixel_size_mm


def get_zone_size_mm(
    contour: np.ndarray,
    pixel_size_mm: float
) -> Tuple[float, float]:
    """
    Получает размер зоны в миллиметрах

    Args:
        contour: Контур зоны
        pixel_size_mm: Размер одного пикселя в мм

    Returns:
        Tuple из (ширина в мм, высота в мм)
    """
    x, y, w, h = cv2.boundingRect(contour)
    width_mm = pixels_to_mm(w, pixel_size_mm)
    height_mm = pixels_to_mm(h, pixel_size_mm)
    return width_mm, height_mm


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
    _denom = float(in_max - in_min)
    if _denom < 1e-6:
        img_for_search = np.zeros_like(img_for_search, dtype=np.uint8)
    else:
        img_for_search = np.clip((img_for_search.astype(np.float32) - in_min) / _denom * 255, 0, 255).astype(np.uint8)

    best_contours = []
    current_target = top_n
    
    while current_target >= 1 and len(best_contours) < current_target:
        for clipLimit in CLIPL_LIMIT_VALUES:
            for threshold_val in THRESHOLD_VALUES:
                img_clahe = _get_clahe(clipLimit, (8, 8)).apply(img_for_search)
                
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
            break

        current_target -= 1
    
    # Fallback: если ничего не найдено
    if len(best_contours) == 0:
        best_contours = _fallback_zone_search(img_for_search, top_n, max_area_ratio, img_area)
    
    # Сортировка по горизонтальной позиции
    best_contours = sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])
    return best_contours


def find_top_zones_improved(
    img_gray_original: np.ndarray,
    top_n: int = 5,
    min_area_ratio: float = 0.3,
    max_area_ratio: float = 0.95,
    min_absolute_area: int = 5000,
    aspect_ratio_filter: bool = True,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0,
    brightness: int = 0,
    contrast: float = 1.0
) -> List[np.ndarray]:
    """
    Улучшенный поиск зон с расширенной фильтрацией артефактов

    Args:
        img_gray_original: Исходное grayscale изображение
        top_n: Количество зон для поиска
        min_area_ratio: Минимальное отношение площади зоны к максимальной (0.0-1.0)
        max_area_ratio: Максимальное отношение площади зоны к изображению (0.0-1.0)
        min_absolute_area: Минимальная площадь в пикселях (отсекает мелкий мусор)
        aspect_ratio_filter: Использовать ли фильтр по пропорциям
        min_aspect_ratio: Минимальное соотношение сторон
        max_aspect_ratio: Максимальное соотношение сторон
        brightness: Коррекция яркости (-100 до +100)
        contrast: Коррекция контраста (0.1 до 5.0)

    Returns:
        Список контуров найденных зон
    """
    # Применяем коррекцию изображения
    img_for_search = apply_image_correction(img_gray_original.copy(), brightness, contrast)
    img_area = img_for_search.shape[0] * img_for_search.shape[1]

    # Нормализация
    in_min, in_max = np.percentile(img_for_search, 1), np.percentile(img_for_search, 99)
    _denom = float(in_max - in_min)
    if _denom < 1e-6:
        img_for_search = np.zeros_like(img_for_search, dtype=np.uint8)
    else:
        img_for_search = np.clip((img_for_search.astype(np.float32) - in_min) / _denom * 255, 0, 255).astype(np.uint8)

    best_contours = []
    best_score = 0

    # Расширенный перебор параметров для лучшего результата
    clip_limits = [2.0, 3.0, 4.0, 5.0, 7.0]
    threshold_vals = [15, 20, 25, 30, 40, 50]

    for clipLimit in clip_limits:
        for threshold_val in threshold_vals:
            img_clahe = _get_clahe(clipLimit, (8, 8)).apply(img_for_search)

            _, mask = cv2.threshold(img_clahe, threshold_val, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Фильтрация контуров
            good_contours = []
            for cnt in contours:
                # 1. Фильтр по площади
                area = cv2.contourArea(cnt)
                area_ratio_to_image = area / img_area

                if area_ratio_to_image > max_area_ratio:
                    continue

                # 2. Фильтр по абсолютной площади (отсекает мелкий мусор)
                if area < min_absolute_area:
                    continue

                # 3. Фильтр по пропорциям (отсекает слишком узкие/вытянутые объекты)
                if aspect_ratio_filter:
                    x, y, w, h = cv2.boundingRect(cnt)

                    if w == 0 or h == 0:
                        continue

                    # Соотношение сторон (всегда >= 1)
                    aspect_ratio = max(w, h) / min(w, h)

                    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                        continue

                # 4. Фильтр по компактности (отсекает слишком рваные контуры)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.05:  # Слишком вытянутая/изломанная форма
                        continue

                # 5. Фильтр по заполненности bbox (отсекает Г-образные и странные формы)
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_area = w * h
                extent = area / bbox_area if bbox_area > 0 else 0
                if extent < 0.3:  # Зона должна заполнять хотя бы 30% своего bbox
                    continue

                good_contours.append(cnt)

            if len(good_contours) == 0:
                continue

            # Сортируем по площади
            good_contours = sorted(good_contours, key=cv2.contourArea, reverse=True)

            # Убираем аномально большие контуры (слипшиеся зоны).
            # Ожидаем top_n зон — ни одна не должна занимать больше 3/top_n
            # площади изображения. Это защищает min_area_ratio от "отравления"
            # единственным гигантским blob'ом.
            max_reasonable = min(max_area_ratio, 3.0 / top_n)
            good_contours = [c for c in good_contours
                             if cv2.contourArea(c) / img_area <= max_reasonable]

            # Фильтр по относительному размеру
            if len(good_contours) > 0:
                max_area = cv2.contourArea(good_contours[0])
                filtered = [c for c in good_contours
                           if cv2.contourArea(c) >= max_area * min_area_ratio]

                # Оценка качества результата
                if len(filtered) >= top_n:
                    # Считаем score: чем ближе к top_n и равномернее зоны, тем лучше
                    count_score = abs(len(filtered) - top_n)

                    # Проверяем равномерность размеров
                    areas = [cv2.contourArea(c) for c in filtered[:top_n]]
                    if len(areas) > 1:
                        area_std = np.std(areas) / np.mean(areas)
                        uniformity_score = 1.0 - min(area_std, 1.0)
                    else:
                        uniformity_score = 1.0

                    total_score = (1.0 / (1 + count_score)) * 0.6 + uniformity_score * 0.4

                    if total_score > best_score:
                        best_score = total_score
                        best_contours = filtered[:top_n]

    # Fallback: если ничего не найдено
    if len(best_contours) == 0:
        best_contours = _fallback_zone_search(
            img_for_search, top_n, max_area_ratio, img_area, min_absolute_area
        )

    # Сортировка по горизонтальной позиции (слева направо)
    best_contours = sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])
    return best_contours


def _fallback_zone_search(
    img_for_search: np.ndarray,
    top_n: int,
    max_area_ratio: float,
    img_area: int,
    min_absolute_area: int = 0
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

    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_absolute_area:
            continue
        if area / img_area > max_area_ratio:
            continue
        filtered.append(c)

    return sorted(filtered, key=cv2.contourArea, reverse=True)[:top_n]


def extract_zone_roi(
    image: np.ndarray,
    contour: np.ndarray,
    padding: int = 0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Извлекает ROI (Region of Interest) из изображения по контуру

    Args:
        image: Исходное изображение
        contour: Контур зоны
        padding: Отступ вокруг ROI в пикселях (опционально)

    Returns:
        Tuple из (ROI изображение, bbox координаты (x, y, w, h))
    """
    x, y, w, h = cv2.boundingRect(contour)

    if padding > 0:
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

    roi = image[y:y+h, x:x+w].copy()
    return roi, (x, y, w, h)


def analyze_zones(contours: List[np.ndarray], img_shape: Tuple[int, int]) -> dict:
    """
    Анализирует найденные зоны и возвращает статистику

    Args:
        contours: Список контуров зон
        img_shape: Размер изображения (height, width)

    Returns:
        Словарь со статистикой зон
    """
    if len(contours) == 0:
        return {'count': 0}

    areas = [cv2.contourArea(c) for c in contours]

    analysis = {
        'count': len(contours),
        'min_area': min(areas),
        'max_area': max(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'total_coverage': sum(areas) / (img_shape[0] * img_shape[1]) * 100
    }

    # Соотношения сторон
    aspect_ratios = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) > 0:
            aspect_ratios.append(max(w, h) / min(w, h))

    if aspect_ratios:
        analysis['mean_aspect_ratio'] = np.mean(aspect_ratios)
        analysis['max_aspect_ratio'] = max(aspect_ratios)

    return analysis
