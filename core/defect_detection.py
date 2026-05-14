"""
Модуль для детекции дефектов с использованием нейронной сети
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List

from config.settings import COLOR_DEFECT, COLOR_ZONE_BORDER, DEFECT_OVERLAY_ALPHA
from .image_processing import prepare_for_display

# Кэш structuring elements для избежания повторного создания в цикле
_SE_CACHE: dict = {}


def _get_se(ring: int) -> np.ndarray:
    if ring not in _SE_CACHE:
        size = 2 * ring + 1
        _SE_CACHE[ring] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return _SE_CACHE[ring]


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image_rgb = prepare_for_display(image)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    return image_rgb


def predict_defects_batch_probs(
    model: nn.Module,
    images: List[np.ndarray],
    device: torch.device,
    tta: bool = False,
    chunk_size: int = 4,
) -> List[np.ndarray]:
    """Батчевый инференс. Возвращает список (H, W) float32 probability maps в [0, 1].

    Если tta=True — каждый тайл прогоняется дважды (оригинал + 180°), probs усредняются.

    chunk_size — макс размер батча для одного forward. Большие батчи дробятся
    последовательно, чтобы избежать OOM (CPU и VRAM).
    """
    if not images:
        return []

    model.eval()

    rgb_list = [_to_rgb_uint8(img) for img in images]
    arr = np.stack(rgb_list, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device, memory_format=torch.channels_last, non_blocking=True)

    if tta:
        rotated = torch.flip(tensor, dims=[2, 3])
        combined = torch.cat([tensor, rotated], dim=0)
    else:
        combined = tensor

    total = combined.size(0)
    pred_chunks = []
    with torch.no_grad():
        for i in range(0, total, chunk_size):
            chunk_pred = model(combined[i:i + chunk_size])
            pred_chunks.append(torch.sigmoid(chunk_pred))
    prediction = torch.cat(pred_chunks, dim=0)

    if tta:
        N = tensor.size(0)
        preds_orig = prediction[:N]
        preds_rot_back = torch.flip(prediction[N:], dims=[2, 3])
        prediction = (preds_orig + preds_rot_back) * 0.5

    probs = prediction.squeeze(1).cpu().numpy()
    return [probs[i] for i in range(probs.shape[0])]


def predict_defects_batch(
    model: nn.Module,
    images: List[np.ndarray],
    device: torch.device,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """Батчевый инференс с простым thresholding (legacy, без edge-aware refinement)."""
    probs = predict_defects_batch_probs(model, images, device)
    return [(p > threshold).astype(np.uint8) * 255 for p in probs]


def _guided_filter_gray(guide_u8: np.ndarray, src_f32: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Простой guided filter (He, Sun, Tang 2010), реализация через boxFilter."""
    I = guide_u8.astype(np.float32) / 255.0
    p = src_f32.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    return mean_a * I + mean_b


def refine_defect_mask(
    probs: np.ndarray,
    guide_rgb: np.ndarray,
    threshold_low: float = 0.01,
    threshold_high: float = 0.7,
    morph_kernel: int = 3,
    guided_radius: int = 4,
    guided_eps: float = 0.005,
    edge_threshold: float = 0.5,
    small_component_max: int = 80,
    max_ring_width: int = None,
) -> np.ndarray:
    """Двухпроходный refine для чётких границ.

    Стратегия:
      - Большие дефекты (соединённые компоненты с уверенным ядром p > thr_high):
        сглаживаем probs guided-фильтром с малым eps (резкое уважение к перепадам
        яркости guide), потом порог `edge_threshold`. Граница «защёлкивается»
        на реальные перепады яркости.
      - Мелкие дефекты (мелкие компоненты p > thr_low без сильного ядра):
        оставляем как есть после морфологии — guided filter их размыл бы.
    """
    if not probs.any():
        return np.zeros(probs.shape, dtype=np.uint8)

    thr_lo = float(threshold_low)
    thr_hi = max(thr_lo, float(threshold_high))
    thr_edge = max(thr_lo, float(edge_threshold))

    weak = probs > thr_lo
    if not weak.any():
        return np.zeros(probs.shape, dtype=np.uint8)

    # Guide: grayscale зоны
    if guide_rgb.ndim == 3:
        guide_u8 = cv2.cvtColor(guide_rgb, cv2.COLOR_BGR2GRAY)
    else:
        guide_u8 = guide_rgb
    if guide_u8.dtype != np.uint8:
        guide_u8 = guide_u8.astype(np.uint8)

    # Edge-aware probs (агрессивный eps — резкая чувствительность к границам guide)
    probs_f32 = probs.astype(np.float32, copy=False)
    filtered = _guided_filter_gray(guide_u8, probs_f32, radius=guided_radius, eps=guided_eps)
    filtered = np.clip(filtered, 0.0, 1.0)

    # Бинарная маска по edge-aware вероятностям
    large_mask = (filtered > thr_edge).astype(np.uint8) * 255

    # Мелкие дефекты: компоненты в weak-маске без strong-ядра, маленького размера.
    # Их guided filter может «съесть» — добавляем напрямую.
    small_mask = np.zeros_like(large_mask)
    weak_u8 = weak.astype(np.uint8)
    num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(weak_u8, connectivity=8)
    strong = probs > thr_hi
    for lbl in range(1, num_w):
        area = stats_w[lbl, cv2.CC_STAT_AREA]
        if area > small_component_max:
            continue
        if strong[labels_w == lbl].any():
            # Мелкий, но с уверенным ядром — пусть large_mask его подхватит
            continue
        small_mask[labels_w == lbl] = 255

    combined = cv2.bitwise_or(large_mask, small_mask)

    # Лёгкая морфология
    if morph_kernel and morph_kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    # Intensity-based shrinking: пиксели маски, чья яркость близка к локальному
    # фону, не похожи на дефект → выкидываем. Это сильнее всего урезает «пухлые»
    # маски, где модель уверенно расширила probs за реальный край.
    combined = _intensity_shrink(combined, guide_u8, k_sigma=2.5, ring_width=25, max_ring_width=max_ring_width)

    return combined


def _intensity_shrink(
    mask_u8: np.ndarray,
    guide_u8: np.ndarray,
    k_sigma: float = 2.5,
    ring_width: int = 25,
    min_diff: float = 12.0,
    bg_std_max: float = 22.0,
    max_ring_width: int = None,
) -> np.ndarray:
    """Урезает маску по яркостной близости к локальному фону.

    Адаптивная ширина кольца — для крупных компонент кольцо растёт вместе с
    габаритом, чтобы гарантированно выйти за края дефекта. Если фон шумный
    (высокий bg_std) или сэмпл мал — пропускаем shrink (доверяем edge-aware).

    `max_ring_width` (None по умолчанию — поведение прода без cap): верхняя
    граница на ring. Используется в eval, где fused probs из catch-all tiled
    могут давать гигантские компоненты, и dilate большим kernel зависает.
    """
    if not mask_u8.any():
        return mask_u8

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    H, W = mask_u8.shape
    guide_f = guide_u8.astype(np.float32)
    out = np.zeros_like(mask_u8)

    for lbl in range(1, num):
        x, y, w, h, area = (
            stats[lbl, cv2.CC_STAT_LEFT],
            stats[lbl, cv2.CC_STAT_TOP],
            stats[lbl, cv2.CC_STAT_WIDTH],
            stats[lbl, cv2.CC_STAT_HEIGHT],
            stats[lbl, cv2.CC_STAT_AREA],
        )

        # Адаптивная ширина кольца: для большой компоненты кольцо должно выйти за её край
        max_dim = max(w, h)
        adaptive_ring = max(ring_width, int(max_dim * 0.6) + 8)
        if max_ring_width is not None:
            adaptive_ring = min(adaptive_ring, max_ring_width)

        # Большая компонента → мягче shrink (probs модели вероятно правее, чем фон)
        eff_k = k_sigma if area < 1500 else 1.5

        pad = adaptive_ring + 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        local_labels = labels[y0:y1, x0:x1]
        local_mask = (local_labels == lbl).astype(np.uint8)
        if local_mask.sum() == 0:
            continue
        local_guide = guide_f[y0:y1, x0:x1]

        dilate_k = _get_se(adaptive_ring)
        dilated = cv2.dilate(local_mask, dilate_k)
        other = ((local_labels != 0) & (local_labels != lbl)).astype(np.uint8)
        ring = (dilated == 1) & (local_mask == 0) & (other == 0)

        if ring.sum() < 40:
            out[y0:y1, x0:x1][local_mask == 1] = 255
            continue

        bg_pixels = local_guide[ring]
        bg_mean = float(bg_pixels.mean())
        bg_std = float(bg_pixels.std())

        # Safety check: если фон сильно зашумлён / неоднороден — кольцо плохое
        # (часть кольца попала на другой дефект, край зоны, или внутрь нашей компоненты)
        if bg_std > bg_std_max:
            out[y0:y1, x0:x1][local_mask == 1] = 255
            continue

        threshold = max(eff_k * bg_std, min_diff)
        diff = np.abs(local_guide - bg_mean)
        anomaly = diff > threshold
        kept = (local_mask == 1) & anomaly

        # Если съели слишком много (>80% компонента) — это подозрительно, доверяем модели
        if kept.sum() < 0.2 * local_mask.sum():
            out[y0:y1, x0:x1][local_mask == 1] = 255
            continue

        out[y0:y1, x0:x1][kept] = 255

    return out


# ГОСТ 11141-84 «Детали оптические. Классы чистоты поверхностей».
# Лимиты в мм для классов I-IXa (Таблица 2 ГОСТ):
#   «точка» — компактный дефект (отношение сторон bbox ≤ 3:1)
#   «царапина» — вытянутый дефект (отношение сторон > 3:1)
_GOST_POINT_LIMITS_MM = [
    (0.020, "I"),
    (0.050, "II"),
    (0.100, "III"),
    (0.300, "IV"),
    (0.500, "V"),
    (0.700, "VI"),
    (1.000, "VII"),
    (1.500, "VIIIa"),
    (3.000, "IX"),
]
_GOST_SCRATCH_LIMITS_MM = [
    (0.004, "I"),
    (0.006, "II"),
    (0.010, "III"),
    (0.020, "IV"),
    (0.040, "V"),
    (0.060, "VI"),
    (0.100, "VII"),
    (0.200, "VIII"),
    (0.300, "VIIIa"),
    (0.400, "IXa"),
]
_GOST_CLASS_ORDER = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "VIIIa", "IX", "IXa", "хуже IXa"]


def _gost_lookup(size_mm: float, limits) -> str:
    for limit, cls in limits:
        if size_mm <= limit:
            return cls
    return "хуже IXa"


def classify_defects_gost(mask_u8: np.ndarray, pixel_size_mm: float) -> dict:
    """Классификация дефектов по ГОСТ 11141-84.

    Возвращает:
      {
        'defects':  [{'kind': 'point'|'scratch', 'size_mm': ..., 'bbox': (x,y,w,h), 'class': 'III', 'area_px': ...}, ...],
        'worst_class': 'IV' | 'хуже IXa' | 'I' (если дефектов нет),
        'count_points': N,
        'count_scratches': N,
      }
    """
    result = {
        'defects': [],
        'worst_class': 'I',
        'count_points': 0,
        'count_scratches': 0,
    }
    if mask_u8 is None or not mask_u8.any() or pixel_size_mm <= 0:
        return result

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return result

    worst_idx = 0  # начнём с 'I' — лучшего класса
    for lbl in range(1, num):
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        if aspect > 3.0:
            kind = 'scratch'
            size_mm = long_side * pixel_size_mm
            cls = _gost_lookup(size_mm, _GOST_SCRATCH_LIMITS_MM)
            result['count_scratches'] += 1
        else:
            kind = 'point'
            size_mm = long_side * pixel_size_mm  # «размер большей оси» по ГОСТ
            cls = _gost_lookup(size_mm, _GOST_POINT_LIMITS_MM)
            result['count_points'] += 1

        result['defects'].append({
            'kind': kind,
            'size_mm': size_mm,
            'bbox': (x, y, w, h),
            'area_px': area,
            'class': cls,
        })
        if cls in _GOST_CLASS_ORDER:
            idx = _GOST_CLASS_ORDER.index(cls)
            if idx > worst_idx:
                worst_idx = idx
        else:
            worst_idx = len(_GOST_CLASS_ORDER) - 1

    result['worst_class'] = _GOST_CLASS_ORDER[worst_idx]
    return result


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
    if model.training:
        model.eval()

    # Убедимся что изображение RGB и uint8
    image_rgb = prepare_for_display(image)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
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


def apply_defects_overlay(
    image: np.ndarray,
    defect_mask: np.ndarray,
    alpha: float = DEFECT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Накладывает красную полупрозрачную заливку дефектов на изображение."""
    if defect_mask is None:
        return image.copy()
    result = image.copy()
    overlay = np.zeros_like(result)
    overlay[defect_mask > 0] = COLOR_DEFECT
    return cv2.addWeighted(result, 1.0, overlay, alpha, 0)


def draw_zones(
    image: np.ndarray,
    contours: list,
    pixel_size_mm: float = None,
) -> np.ndarray:
    """Рисует bbox зон, надписи Zone N и размер в мм. Без красной заливки дефектов."""
    result = image.copy()
    if not contours:
        return result

    for idx, cnt in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), COLOR_ZONE_BORDER, 2)
        cv2.putText(
            result, f"Zone {idx}", (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ZONE_BORDER, 2,
        )
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
            if ty > img_h - 5:
                ty = max(th + 5, y - gap)
            cv2.putText(
                result, size_text, (tx, ty),
                font, font_scale, COLOR_ZONE_BORDER, thickness, cv2.LINE_AA,
            )
    return result


# Moved to archive/dead_code_detection.py


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


# Moved to archive/dead_code_detection.py