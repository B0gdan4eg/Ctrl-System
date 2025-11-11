"""
Воркер для обработки изображений с камеры в реальном времени

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.camera import CameraCapture
from core import (
    convert_to_grayscale,
    find_top_zones,
    apply_histogram_normalization,
    preprocess_for_detection,
    predict_defects
)


class RealtimeCameraWorker(QThread):
    """Воркер для обработки кадров с камеры в реальном времени"""
    
    processed_frame = Signal(object, object)  # (кадр с визуализацией, статистика)
    progress = Signal(str)  # Сообщения о прогрессе
    error = Signal(str)  # Ошибки
    fps_update = Signal(float)  # Обновление FPS
    
    def __init__(self, camera_id, model, device, params):
        """
        Args:
            camera_id: ID камеры
            model: Модель для детекции
            device: Устройство (CPU/GPU)
            params: Параметры обработки
        """
        super().__init__()
        self.camera_id = camera_id
        self.model = model
        self.device = device
        self.params = params
        self.is_running = False
        self.camera_thread = None
        self.frame_count = 0
        self.skip_frames = params.get('skip_frames', 5)  # Обрабатывать каждый N-й кадр
        
    def run(self):
        """Основной цикл обработки"""
        try:
            self.progress.emit("📷 Запуск камеры...")
            
            # Запускаем камеру
            self.camera_thread = CameraCapture(self.camera_id)
            self.camera_thread.frame_captured.connect(self.process_frame)
            self.camera_thread.error.connect(self.on_camera_error)
            self.camera_thread.start()
            
            self.is_running = True
            self.progress.emit("✅ Камера запущена. Обработка кадров...")
            
            # Ждем завершения
            self.camera_thread.wait()
            
        except Exception as e:
            import traceback
            self.error.emit(f"Ошибка: {str(e)}\n{traceback.format_exc()}")
    
    def process_frame(self, frame):
        """
        Обрабатывает один кадр
        
        Args:
            frame: Кадр с камеры
        """
        if not self.is_running:
            return
        
        self.frame_count += 1
        
        # Пропускаем кадры для повышения производительности
        if self.frame_count % self.skip_frames != 0:
            # Просто показываем исходный кадр
            self.processed_frame.emit(frame, None)
            return
        
        try:
            import time
            start_time = time.time()
            
            # Конвертация в grayscale
            img_gray = convert_to_grayscale(frame)
            
            # Поиск зон
            contours = find_top_zones(
                img_gray,
                top_n=self.params['num_zones'],
                min_area_ratio=self.params['min_area_ratio'],
                max_area_ratio=self.params['max_area_ratio']
            )
            
            if not contours:
                self.processed_frame.emit(frame, {'zones_found': 0})
                return
            
            # Создаем выходное изображение
            result = frame.copy()
            all_defects = 0
            zones_with_defects = 0
            
            # Обрабатываем зоны (только первую для скорости)
            for idx, cnt in enumerate(contours[:1], 1):  # Только первую зону
                x, y, w, h = cv2.boundingRect(cnt)
                
                if w < 50 or h < 50:
                    continue
                
                roi = img_gray[y:y+h, x:x+w].copy()
                
                # Нормализация
                if self.params.get('apply_hist_norm', True):
                    roi = apply_histogram_normalization(
                        roi,
                        std_range=self.params['std_range']
                    )
                
                # Предобработка
                roi_processed = preprocess_for_detection(
                    roi,
                    target_size=(self.params['model_size'], self.params['model_size'])
                )
                
                # Детекция
                defect_mask = predict_defects(
                    self.model,
                    roi_processed,
                    self.device,
                    threshold=self.params['detection_threshold']
                )
                
                # Resize маски
                defect_mask_resized = cv2.resize(
                    defect_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Визуализация дефектов
                if np.sum(defect_mask_resized) > 0:
                    zones_with_defects += 1
                    all_defects += np.sum(defect_mask_resized > 0)
                    
                    # Красное наложение
                    mask_colored = np.zeros_like(result[y:y+h, x:x+w])
                    mask_colored[defect_mask_resized > 0] = [0, 0, 255]
                    result[y:y+h, x:x+w] = cv2.addWeighted(
                        result[y:y+h, x:x+w], 0.7,
                        mask_colored, 0.3,
                        0
                    )
                
                # Рисуем контур зоны
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    result,
                    f"Zone {idx}",
                    (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Статистика
            stats = {
                'zones_found': len(contours),
                'zones_with_defects': zones_with_defects,
                'total_defects': all_defects,
                'frame_number': self.frame_count
            }
            
            # FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_update.emit(fps)
            
            # Отправляем результат
            self.processed_frame.emit(result, stats)
            
        except Exception as e:
            self.error.emit(f"Ошибка обработки кадра: {str(e)}")
    
    def on_camera_error(self, error_msg):
        """Обработчик ошибок камеры"""
        self.error.emit(f"Ошибка камеры: {error_msg}")
        self.stop()
    
    def stop(self):
        """Остановка обработки"""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        self.progress.emit("⏹️ Обработка остановлена")


class ContinuousCameraWorker(QThread):
    """Воркер для непрерывного мониторинга с камеры"""
    
    defect_detected = Signal(object, dict)  # (кадр, информация о дефекте)
    stats_update = Signal(dict)  # Обновление статистики
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, camera_id, model, device, params):
        super().__init__()
        self.camera_id = camera_id
        self.model = model
        self.device = device
        self.params = params
        self.is_running = False
        self.camera_thread = None
        
        # Статистика
        self.total_frames = 0
        self.frames_with_defects = 0
        self.total_defects = 0
        
    def run(self):
        """Основной цикл мониторинга"""
        try:
            self.progress.emit("🔍 Запуск непрерывного мониторинга...")
            
            self.camera_thread = CameraCapture(self.camera_id)
            self.camera_thread.frame_captured.connect(self.monitor_frame)
            self.camera_thread.error.connect(self.on_camera_error)
            self.camera_thread.start()
            
            self.is_running = True
            self.camera_thread.wait()
            
        except Exception as e:
            import traceback
            self.error.emit(f"Ошибка: {str(e)}\n{traceback.format_exc()}")
    
    def monitor_frame(self, frame):
        """Мониторинг кадра на наличие дефектов"""
        if not self.is_running:
            return
        
        self.total_frames += 1
        
        # Обрабатываем каждый 10-й кадр
        if self.total_frames % 10 != 0:
            return
        
        try:
            # Быстрая проверка на дефекты
            img_gray = convert_to_grayscale(frame)
            contours = find_top_zones(img_gray, top_n=1)
            
            if not contours:
                return
            
            x, y, w, h = cv2.boundingRect(contours[0])
            roi = img_gray[y:y+h, x:x+w]
            
            if roi.size < 100:
                return
            
            # Нормализация и предобработка
            roi = apply_histogram_normalization(roi)
            roi_processed = preprocess_for_detection(roi, target_size=(256, 256))
            
            # Детекция
            defect_mask = predict_defects(
                self.model,
                roi_processed,
                self.device,
                threshold=self.params['detection_threshold']
            )
            
            defect_pixels = np.sum(defect_mask > 0)
            
            # Если найдены дефекты
            if defect_pixels > 100:  # Порог минимального размера дефекта
                self.frames_with_defects += 1
                self.total_defects += defect_pixels
                
                defect_info = {
                    'frame_number': self.total_frames,
                    'defect_pixels': defect_pixels,
                    'zone': (x, y, w, h),
                    'severity': 'high' if defect_pixels > 1000 else 'medium'
                }
                
                self.defect_detected.emit(frame, defect_info)
            
            # Обновление статистики
            stats = {
                'total_frames': self.total_frames,
                'frames_with_defects': self.frames_with_defects,
                'total_defects': self.total_defects,
                'defect_rate': (self.frames_with_defects / self.total_frames * 100)
                                if self.total_frames > 0 else 0
            }
            
            self.stats_update.emit(stats)
            
        except Exception as e:
            self.error.emit(f"Ошибка мониторинга: {str(e)}")
    
    def on_camera_error(self, error_msg):
        """Обработчик ошибок камеры"""
        self.error.emit(f"Ошибка камеры: {error_msg}")
        self.stop()
    
    def stop(self):
        """Остановка мониторинга"""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        self.progress.emit("⏹️ Мониторинг остановлен")