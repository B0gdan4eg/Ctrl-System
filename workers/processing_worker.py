"""
Воркер для многопоточной обработки изображений
"""

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
    predict_defects,
    visualize_defects,
    prepare_for_display
)


class ProcessingWorker(QThread):
    """Рабочий поток для обработки изображения"""
    
    progress = Signal(str)  # Сообщения о прогрессе
    zone_processed = Signal(int, object, object)  # (номер зоны, изображение, маска)
    finished = Signal(object, object)  # (результат, статистика)
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
            
            # ШАГ 1: Поиск зон
            self.progress.emit("🔍 Поиск зон на изображении...")
            contours = find_top_zones(
                img_gray, 
                top_n=self.params['num_zones'],
                min_area_ratio=self.params['min_area_ratio'],
                max_area_ratio=self.params['max_area_ratio']
            )
            
            if not contours:
                self.error.emit("Зоны не найдены на изображении")
                return
            
            self.progress.emit(f"✓ Найдено зон: {len(contours)}")
            
            # Создание выходного изображения
            result_image = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
            all_defects_mask = np.zeros(original_shape, dtype=np.uint8)
            
            stats = {
                'zones_found': len(contours),
                'zones_with_defects': 0,
                'total_defect_pixels': 0,
                'zone_details': []
            }
            
            # Обработка каждой зоны
            for idx, cnt in enumerate(contours, 1):
                zone_stats = self._process_zone(
                    cnt, idx, len(contours),
                    img_gray, result_image, all_defects_mask
                )
                
                if zone_stats['has_defects']:
                    stats['zones_with_defects'] += 1
                    stats['total_defect_pixels'] += zone_stats['defect_pixels']
                
                stats['zone_details'].append(zone_stats)
            
            # Визуализация дефектов
            result_with_defects = visualize_defects(
                result_image,
                all_defects_mask,
                contours
            )
            
            self.progress.emit("✅ Обработка завершена!")
            self.finished.emit(result_with_defects, stats)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Ошибка: {str(e)}\n{traceback.format_exc()}")
    
    def _process_zone(
        self,
        contour: np.ndarray,
        zone_idx: int,
        total_zones: int,
        img_gray: np.ndarray,
        result_image: np.ndarray,
        all_defects_mask: np.ndarray
    ) -> dict:
        """
        Обрабатывает одну зону
        
        Returns:
            Статистика по зоне
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        self.progress.emit(f"🔧 Обработка зоны {zone_idx}/{total_zones} ({w}x{h})...")
        
        # Вырезаем зону
        roi = img_gray[y:y+h, x:x+w].copy()
        
        if roi.size < 100:
            return {
                'zone_id': zone_idx,
                'bbox': (x, y, w, h),
                'has_defects': False,
                'defect_pixels': 0
            }
        
        # ШАГ 2: Гистограммная нормализация
        if self.params['apply_hist_norm']:
            roi = apply_histogram_normalization(
                roi, 
                std_range=self.params['std_range']
            )
        
        # ШАГ 3: Предобработка
        roi_processed = preprocess_for_detection(
            roi, 
            target_size=(self.params['model_size'], self.params['model_size'])
        )
        
        # ШАГ 4: Детекция дефектов
        defect_mask = predict_defects(
            self.model, 
            roi_processed, 
            self.device,
            threshold=self.params['detection_threshold']
        )
        
        # ШАГ 5: Resize маски обратно к размеру зоны
        defect_mask_resized = cv2.resize(
            defect_mask, 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Помещаем обратно в исходное изображение
        roi_display = prepare_for_display(roi)
        result_image[y:y+h, x:x+w] = roi_display
        
        # Накладываем маску дефектов
        all_defects_mask[y:y+h, x:x+w] = defect_mask_resized
        
        # Статистика
        defect_pixels = np.sum(defect_mask_resized > 0)
        
        # Сигнал о завершении обработки зоны
        self.zone_processed.emit(zone_idx, roi_display, defect_mask_resized)
        
        return {
            'zone_id': zone_idx,
            'bbox': (x, y, w, h),
            'has_defects': defect_pixels > 0,
            'defect_pixels': int(defect_pixels)
        }
        