"""
Воркер для многопоточной обработки изображений
"""

import threading
import traceback
import cv2
import numpy as np
import torch
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from core import (
    convert_to_grayscale,
    find_top_zones,
    apply_histogram_normalization,
    preprocess_for_detection,
    restore_mask_letterbox,
    resize_with_letterbox,
    predict_defects,
    predict_defects_batch_probs,
    refine_defect_mask,
    classify_defects_gost,
    apply_defects_overlay,
    draw_zones,
    prepare_for_display,
    prepare_zone_tiles,
    stitch_tile_probs,
)
from core.zone_segmentation import ZoneSegmenter
from config.settings import ZONE_MODEL_PATH, USE_AI_ZONE_DETECTION, PIXEL_SIZE_MM


# Singleton-сегментатор: один экземпляр на процесс, модель грузится лениво
# при первом вызове и переиспользуется во всех последующих обработках.
_zone_segmenter: ZoneSegmenter | None = None
_zone_segmenter_lock = threading.Lock()


def _get_zone_segmenter() -> ZoneSegmenter:
    global _zone_segmenter
    if _zone_segmenter is None:
        with _zone_segmenter_lock:
            if _zone_segmenter is None:
                from utils import resource_path
                _zone_segmenter = ZoneSegmenter(model_path=resource_path(ZONE_MODEL_PATH))
    return _zone_segmenter


def _restore_probs_letterbox(probs: np.ndarray, pad_info: dict) -> np.ndarray:
    """Восстанавливает float probs из letterbox-пространства в исходный размер ROI."""
    pad_x = pad_info['pad_x']
    pad_y = pad_info['pad_y']
    new_w = pad_info['new_w']
    new_h = pad_info['new_h']
    orig_w = pad_info['orig_w']
    orig_h = pad_info['orig_h']
    cropped = probs[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


class ProcessingWorker(QThread):
    """Рабочий поток для обработки изображения"""
    
    progress = Signal(str)  # Сообщения о прогрессе
    zone_processed = Signal(int, object, object)  # (номер зоны, изображение, маска)
    # (clean_image без overlay, defects_mask, statistics, contours) — overlay рендерится в UI по чекбоксу
    finished = Signal(object, object, object, object)
    error = Signal(str)
    
    def __init__(self, image_path: Path, model, device, params: dict):
        """
        Args:
            image_path: Путь к изображению
            model: Модель для детекции
            device: Устройство (CPU/GPU)
            params: Параметры обработки
        """
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.device = device
        self.params = params
        
    def run(self):
        """Основной метод обработки"""
        try:
            self.progress.emit("📂 Загрузка изображения...")
            
            # Загрузка
            img = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                self.error.emit(f"Ошибка загрузки: {self.image_path}")
                return
            
            # Конвертация в grayscale
            img_gray = convert_to_grayscale(img)
            original_shape = img_gray.shape
            
            # ШАГ 1: Поиск зон — сначала нейросеть, фолбэк на классику
            contours = []
            used_ai = False
            if USE_AI_ZONE_DETECTION:
                segmenter = _get_zone_segmenter()
                if segmenter.is_available:
                    self.progress.emit("🤖 Поиск зон через U-Net...")
                    contours = segmenter.predict_contours(
                        img_gray,
                        top_n=self.params['num_zones'],
                        min_area_ratio=self.params['min_area_ratio'],
                        max_area_ratio=self.params['max_area_ratio'],
                    )
                    used_ai = bool(contours)
                    if not contours:
                        self.progress.emit("⚠️ Нейросеть не нашла зон, переключаюсь на классику")

            if not contours:
                self.progress.emit("🔍 Поиск зон классическим алгоритмом...")
                contours = find_top_zones(
                    img_gray,
                    top_n=self.params['num_zones'],
                    min_area_ratio=self.params['min_area_ratio'],
                    max_area_ratio=self.params['max_area_ratio']
                )

            if not contours:
                self.error.emit("Зоны не найдены на изображении")
                return

            method = "U-Net" if used_ai else "классика"
            self.progress.emit(f"✓ Найдено зон: {len(contours)} ({method})")
            
            # Создание выходного изображения
            result_image = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
            all_defects_mask = np.zeros(original_shape, dtype=np.uint8)
            
            stats = {
                'zones_found': len(contours),
                'zones_with_defects': 0,
                'total_defect_pixels': 0,
                'zone_details': [],
                'gost': None,
            }
            
            # Фаза A: подготовка всех зон (CPU-only, без forward)
            prepared = []
            for idx, cnt in enumerate(contours, 1):
                p = self._prepare_zone(cnt, idx, len(contours), img_gray)
                prepared.append(p)

            # Фаза B: два батч-forward'а — все sliding tiles и все letterbox tiles
            # со всех зон и всех контрастных вариантов
            valid = [p for p in prepared if p['variants']]
            masks_resized: list[np.ndarray | None] = [None] * len(prepared)
            if valid:
                small_tiles: list = []
                large_tiles: list = []
                for p in valid:
                    p['_small_slices'] = []
                    p['_large_slices'] = []
                    for v in p['variants']:
                        ss = len(small_tiles)
                        small_tiles.extend(v['tile_info']['tiles'])
                        p['_small_slices'].append((ss, len(small_tiles)))
                        ls = len(large_tiles)
                        large_tiles.extend(v['tile_info_large']['tiles'])
                        p['_large_slices'].append((ls, len(large_tiles)))

                small_size = int(self.params['model_size'])
                dev_str = str(self.device)
                n_variants = len(valid[0]['variants']) if valid else 0
                self.progress.emit(
                    f"🤖 Inference на {dev_str}: {len(valid)} зон × {n_variants} контр., "
                    f"{len(small_tiles)} sliding + {len(large_tiles)} letterbox {small_size}×{small_size}..."
                )
                small_probs_list = predict_defects_batch_probs(
                    self.model, small_tiles, self.device, tta=False, chunk_size=8
                )
                large_probs_list = predict_defects_batch_probs(
                    self.model, large_tiles, self.device, tta=False, chunk_size=8
                )

                thr_lo = float(self.params['detection_threshold'])
                thr_hi = max(thr_lo, 0.7)

                for p in valid:
                    probs_per_variant = []
                    guide = None
                    for vi, v in enumerate(p['variants']):
                        ti = v['tile_info']
                        ti_l = v['tile_info_large']
                        H, W = ti['roi_shape']
                        ss, se = p['_small_slices'][vi]
                        ls, _le = p['_large_slices'][vi]

                        if ti['mode'] == 'letterbox':
                            ps_full = _restore_probs_letterbox(small_probs_list[ss], ti['pad_info'])
                        else:
                            ps_full = stitch_tile_probs(
                                small_probs_list[ss:se], ti['tile_origins'], (H, W), tile_size=small_size
                            )
                        pl_full = _restore_probs_letterbox(large_probs_list[ls], ti_l['pad_info'])

                        probs_per_variant.append(ps_full)
                        probs_per_variant.append(pl_full)
                        guide = ti['roi_gray']  # последний (наиболее контрастный)

                    fused = np.maximum.reduce(probs_per_variant)
                    full_mask = refine_defect_mask(
                        fused, guide, threshold_low=thr_lo, threshold_high=thr_hi,
                    )
                    masks_resized[p['index']] = full_mask

            # Фаза C: финализация — раскладываем маски, обновляем result_image и stats
            for i, p in enumerate(prepared):
                zone_stats = self._finalize_zone(p, masks_resized[i], result_image, all_defects_mask)
                if zone_stats['has_defects']:
                    stats['zones_with_defects'] += 1
                    stats['total_defect_pixels'] += zone_stats['defect_pixels']
                stats['zone_details'].append(zone_stats)

            # Классификация по ГОСТ 11141-84 (на агрегированной маске)
            stats['gost'] = classify_defects_gost(all_defects_mask, pixel_size_mm=PIXEL_SIZE_MM)
            
            # Чистая визуализация — только зоны + подписи (без красной заливки).
            # Overlay рендерится в UI по чекбоксу.
            clean_image = draw_zones(result_image, contours, pixel_size_mm=PIXEL_SIZE_MM)

            self.progress.emit("✅ Обработка завершена!")
            self.finished.emit(clean_image, all_defects_mask, stats, contours)
            
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).exception("ProcessingWorker error")
            self.error.emit(f"Ошибка обработки: {str(e)}")
    
    # Контрасты (std_range) для multi-prep fusion. Каждое значение даёт отдельный
    # препроцессинг → отдельные тайлы → probs объединяются через max.
    HIST_CONTRASTS = (2.0, 3.0)

    def _prepare_zone(
        self,
        contour: np.ndarray,
        zone_idx: int,
        total_zones: int,
        img_gray: np.ndarray,
    ) -> dict:
        """Готовит зону с несколькими контрастами histogram-нормализации."""
        x, y, w, h = cv2.boundingRect(contour)
        self.progress.emit(f"🔧 Подготовка зоны {zone_idx}/{total_zones} ({w}x{h})...")

        roi_orig = img_gray[y:y+h, x:x+w].copy()
        info = {
            'index': zone_idx - 1,
            'zone_id': zone_idx,
            'bbox': (x, y, w, h),
            'variants': [],
            'roi_display': None,
            'too_small': roi_orig.size < 100,
        }

        if info['too_small']:
            return info

        small_size = int(self.params['model_size'])

        # Если apply_hist_norm выключен — один проход без контрастной обработки
        contrasts = list(self.HIST_CONTRASTS) if self.params['apply_hist_norm'] else [None]

        last_roi_c = roi_orig  # инициализация на случай пустого contrasts
        for c in contrasts:
            if c is not None:
                roi_c = apply_histogram_normalization(roi_orig, std_range=c)
            else:
                roi_c = roi_orig.copy()
            last_roi_c = roi_c  # сохраняем последний

            tile_info = prepare_zone_tiles(roi_c, tile_size=small_size, min_overlap=small_size // 4)
            roi_gray = tile_info['roi_gray']
            large_canvas, large_pad = resize_with_letterbox(roi_gray, (small_size, small_size))
            tile_info_large = {
                'mode': 'letterbox',
                'tiles': [large_canvas],
                'roi_shape': tile_info['roi_shape'],
                'pad_info': large_pad,
                'tile_origins': None,
                'roi_gray': roi_gray,
            }
            info['variants'].append({
                'contrast': c,
                'tile_info': tile_info,
                'tile_info_large': tile_info_large,
            })

        # roi_display — берём наиболее контрастный вариант (последний), без повторного вызова
        info['roi_display'] = prepare_for_display(last_roi_c)
        return info

    def _finalize_zone(
        self,
        info: dict,
        defect_mask_resized: np.ndarray | None,
        result_image: np.ndarray,
        all_defects_mask: np.ndarray,
    ) -> dict:
        """Размещает готовую маску в общем кадре и считает статистику."""
        x, y, w, h = info['bbox']
        zone_idx = info['zone_id']

        if info['too_small'] or defect_mask_resized is None:
            return {
                'zone_id': zone_idx,
                'bbox': (x, y, w, h),
                'has_defects': False,
                'defect_pixels': 0,
            }

        result_image[y:y+h, x:x+w] = info['roi_display']
        all_defects_mask[y:y+h, x:x+w] = defect_mask_resized

        defect_pixels = int(np.sum(defect_mask_resized > 0))
        self.zone_processed.emit(zone_idx, info['roi_display'], defect_mask_resized)

        return {
            'zone_id': zone_idx,
            'bbox': (x, y, w, h),
            'has_defects': defect_pixels > 0,
            'defect_pixels': defect_pixels,
        }