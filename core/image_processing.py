"""
Модуль для обработки изображений (нормализация, предобработка)
"""

import os

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
    
    # Если входное было цветное, применяем те же границы ко всем каналам
    if len(image.shape) == 3:
        result_color = np.zeros_like(image, dtype=np.uint16)
        scale = max_val / 255.0 if image.dtype == np.uint8 else 1.0
        for c in range(3):
            channel = image[:, :, c].astype(np.float32) * scale
            normalized_channel = (channel - input_min) / (input_max - input_min)
            result_color[:, :, c] = (np.clip(normalized_channel, 0, 1) * max_val).astype(np.uint16)
        return result_color
    
    return result


def resize_with_letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, dict]:
    """
    Масштабирование с сохранением пропорций (letterbox, чёрный padding).
    """
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    if len(image.shape) == 3:
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    pad_info = {
        'scale': scale,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'new_w': new_w,
        'new_h': new_h,
        'orig_w': src_w,
        'orig_h': src_h,
    }

    return canvas, pad_info


def restore_mask_letterbox(
    mask: np.ndarray,
    pad_info: dict
) -> np.ndarray:
    """
    Восстанавливает маску из letterbox-пространства в исходный размер.
    """
    pad_x = pad_info['pad_x']
    pad_y = pad_info['pad_y']
    new_w = pad_info['new_w']
    new_h = pad_info['new_h']
    orig_w = pad_info['orig_w']
    orig_h = pad_info['orig_h']

    mask_cropped = mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    mask_original = cv2.resize(
        mask_cropped,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_original


def _compute_tile_starts(length: int, tile: int, min_overlap: int) -> list:
    """Равномерные стартовые координаты тайлов с гарантированным перекрытием."""
    if length <= tile:
        return [0]
    stride_max = max(1, tile - min_overlap)
    n = max(2, int(np.ceil((length - tile) / stride_max)) + 1)
    if n == 1:
        return [0]
    return [int(round(i * (length - tile) / (n - 1))) for i in range(n)]


def _make_tile_window(tile_h: int, tile_w: int) -> np.ndarray:
    """Косинусное окно (приближение Hann), 1.0 в центре, ~0 на краях."""
    y = np.cos(np.pi * (np.arange(tile_h) - (tile_h - 1) / 2.0) / (tile_h - 1)) ** 2
    x = np.cos(np.pi * (np.arange(tile_w) - (tile_w - 1) / 2.0) / (tile_w - 1)) ** 2
    w = np.outer(y, x).astype(np.float32)
    return w + 0.01  # минимальный вес, чтобы избежать деления на ноль


def prepare_zone_tiles(
    roi: np.ndarray,
    tile_size: int = 1024,
    min_overlap: int = 128,
) -> dict:
    """Готовит зону к tiled inference. Если зона помещается в один tile_size×tile_size —
    возвращает letterbox-режим (как старый pipeline). Иначе нарезает тайлы с перекрытием.

    Returns dict:
      - mode: 'letterbox' | 'tiled'
      - tiles: list[np.ndarray] — 8-bit grayscale tiles of shape (tile_size, tile_size)
      - roi_shape: (H, W) исходной ROI (после bilateral)
      - pad_info: для 'letterbox' режима (для restore_mask_letterbox)
      - tile_origins: для 'tiled' — список (x, y, valid_w, valid_h) в координатах roi
      - roi_gray: full-resolution grayscale guide для refine
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if gray.dtype == np.uint16:
        gray = (gray / 1023.0 * 255.0).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    gray = cv2.bilateralFilter(
        gray,
        BILATERAL_FILTER_D,
        BILATERAL_FILTER_SIGMA_COLOR,
        BILATERAL_FILTER_SIGMA_SPACE,
    )

    H, W = gray.shape
    if H <= tile_size and W <= tile_size:
        tiled, pad_info = resize_with_letterbox(gray, (tile_size, tile_size))
        return {
            'mode': 'letterbox',
            'tiles': [tiled],
            'roi_shape': (H, W),
            'pad_info': pad_info,
            'tile_origins': None,
            'roi_gray': gray,
        }

    ys = _compute_tile_starts(H, tile_size, min_overlap)
    xs = _compute_tile_starts(W, tile_size, min_overlap)
    tiles = []
    origins = []
    for y in ys:
        for x in xs:
            yh = min(tile_size, H - y)
            xw = min(tile_size, W - x)
            tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
            tile[:yh, :xw] = gray[y:y + yh, x:x + xw]
            tiles.append(tile)
            origins.append((x, y, xw, yh))

    return {
        'mode': 'tiled',
        'tiles': tiles,
        'roi_shape': (H, W),
        'pad_info': None,
        'tile_origins': origins,
        'roi_gray': gray,
    }


def stitch_tile_probs(
    probs_tiles: list,
    tile_origins: list,
    roi_shape: Tuple[int, int],
    tile_size: int = 1024,
) -> np.ndarray:
    """Сшивает probs-карты от тайлов в единую карту размера ROI через weighted blending."""
    H, W = roi_shape
    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    window = _make_tile_window(tile_size, tile_size)

    for probs, (x, y, vw, vh) in zip(probs_tiles, tile_origins):
        sub_w = window[:vh, :vw]
        accum[y:y + vh, x:x + vw] += probs[:vh, :vw].astype(np.float32) * sub_w
        weight[y:y + vh, x:x + vw] += sub_w

    return accum / np.maximum(weight, 1e-6)


def preprocess_for_detection(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, dict]:
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
    
    img_resized, pad_info = resize_with_letterbox(img_denoised, target_size)
    return img_resized, pad_info


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Конвертирует изображение в grayscale с поддержкой разных форматов
    
    Args:
        image: Входное изображение
        
    Returns:
        Grayscale изображение
    """
    if len(image.shape) == 2:
        if image.dtype == np.uint16:
            return (image / 4).astype(np.uint16)  # 12-bit (0-4095) → 10-bit (0-1023)
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
        # BGR -> RGB для отображения
        if image.dtype == np.uint16:
            _img_u8 = (image / 1023.0 * 255.0).astype(np.uint8)
        else:
            _img_u8 = image
        return cv2.cvtColor(_img_u8, cv2.COLOR_BGR2RGB)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    Читает изображение по пути с любыми Unicode-символами (кириллица, № и т.д.)

    cv2.imread на Windows использует ANSI-кодировку и возвращает None
    для путей с не-ASCII символами, поэтому читаем через np.fromfile + imdecode.

    Args:
        path: Путь к файлу (str или Path)
        flags: Флаги cv2 (IMREAD_COLOR, IMREAD_UNCHANGED, IMREAD_GRAYSCALE...)

    Returns:
        Изображение numpy или None, если файл не прочитан
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, IOError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image, params=None):
    """
    Сохраняет изображение по пути с любыми Unicode-символами (кириллица, № и т.д.)

    cv2.imwrite на Windows использует ANSI-кодировку и молча не создаёт файл
    для путей с не-ASCII символами, поэтому пишем через imencode + np.tofile.

    Args:
        path: Путь к файлу (str или Path)
        image: Изображение numpy
        params: Дополнительные параметры cv2.imencode (например [cv2.IMWRITE_JPEG_QUALITY, 95])

    Returns:
        True, если файл сохранён
    """
    ext = os.path.splitext(str(path))[1] or '.png'
    try:
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except (OSError, IOError, cv2.error):
        return False