"""
Модуль для работы с камерой

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from PySide6.QtCore import QThread, Signal, QTimer


class CameraCapture(QThread):
    """Поток для захвата изображений с камеры"""
    
    frame_captured = Signal(object)  # Сигнал с новым кадром
    error = Signal(str)  # Сигнал ошибки
    
    def __init__(self, camera_id: int = 0, backend: int = None):
        """
        Args:
            camera_id: ID камеры (0 для первой камеры)
            backend: Backend для OpenCV (None для автоматического выбора)
        """
        super().__init__()
        self.camera_id = camera_id
        self.backend = backend
        self.capture = None
        self.is_running = False
        self.fps = 30
        
    def _try_open_camera(self):
        """Пытается открыть камеру с разными backends"""
        import platform
        
        # Список backends для проверки (приоритет для Windows)
        if platform.system() == "Windows":
            backends = [
                ('CAP_DSHOW', cv2.CAP_DSHOW),      # DirectShow (Windows) - ПРИОРИТЕТ
                ('CAP_MSMF', cv2.CAP_MSMF),        # Media Foundation (Windows)
                ('CAP_ANY', cv2.CAP_ANY),          # Auto
            ]
        elif platform.system() == "Linux":
            backends = [
                ('CAP_V4L2', cv2.CAP_V4L2),        # Video4Linux (Linux)
                ('CAP_ANY', cv2.CAP_ANY),
            ]
        elif platform.system() == "Darwin":
            backends = [
                ('CAP_AVFOUNDATION', cv2.CAP_AVFOUNDATION),  # AVFoundation (macOS)
                ('CAP_ANY', cv2.CAP_ANY),
            ]
        else:
            backends = [('CAP_ANY', cv2.CAP_ANY)]
        
        if self.backend is not None:
            # Если backend указан явно
            self.capture = cv2.VideoCapture(self.camera_id, self.backend)
            if self.capture.isOpened():
                return True
            return False
        
        # Пробуем разные backends
        for name, backend in backends:
            try:
                self.capture = cv2.VideoCapture(self.camera_id, backend)
                if self.capture.isOpened():
                    # Проверяем что можем читать кадры
                    ret, frame = self.capture.read()
                    if ret and frame is not None:
                        print(f"✅ Камера открыта через {name}")
                        return True
                    self.capture.release()
            except:
                pass
        
        return False
        
    def run(self):
        """Основной цикл захвата кадров"""
        try:
            if not self._try_open_camera():
                self.error.emit(f"Не удалось открыть камеру {self.camera_id}")
                return
            
            # Настройка камеры
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            
            while self.is_running:
                ret, frame = self.capture.read()
                
                if ret:
                    self.frame_captured.emit(frame)
                else:
                    self.error.emit("Ошибка чтения кадра с камеры")
                    break
                
                # Контроль FPS
                self.msleep(int(1000 / self.fps))
                
        except Exception as e:
            self.error.emit(f"Ошибка камеры: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Остановка захвата"""
        self.is_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
    
    def take_snapshot(self) -> Optional[np.ndarray]:
        """
        Делает снимок с камеры
        
        Returns:
            Кадр или None в случае ошибки
        """
        if self.capture is not None and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None
    
    def set_resolution(self, width: int, height: int):
        """Устанавливает разрешение камеры"""
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get_resolution(self) -> Tuple[int, int]:
        """Возвращает текущее разрешение камеры"""
        if self.capture is not None:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)


class CameraManager:
    """Менеджер для управления несколькими камерами"""
    
    @staticmethod
    def get_available_cameras(max_cameras: int = 10) -> List[int]:
        """
        Находит доступные камеры в системе
        
        Args:
            max_cameras: Максимальное количество камер для проверки
            
        Returns:
            Список ID доступных камер
        """
        import platform
        
        available = []
        
        # Список backends для проверки (по платформе)
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif platform.system() == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        elif platform.system() == "Darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        
        for i in range(max_cameras):
            # Пробуем разные backends
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        # Проверяем что можем читать кадры
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if i not in available:
                                available.append(i)
                            cap.release()
                            break
                        cap.release()
                except:
                    pass
        
        return available
    
    @staticmethod
    def get_camera_info(camera_id: int) -> dict:
        """
        Получает информацию о камере
        
        Args:
            camera_id: ID камеры
            
        Returns:
            Словарь с информацией о камере
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return {"available": False}
        
        info = {
            "available": True,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "backend": cap.getBackendName()
        }
        
        cap.release()
        return info
    
    @staticmethod
    def test_camera(camera_id: int, duration: int = 3) -> bool:
        """
        Тестирует камеру
        
        Args:
            camera_id: ID камеры
            duration: Длительность теста в секундах
            
        Returns:
            True если камера работает
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return False
        
        success = False
        frames_captured = 0
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, _ = cap.read()
            if ret:
                frames_captured += 1
        
        success = frames_captured > 0
        cap.release()
        
        return success


class CameraSnapshot:
    """Класс для работы со снимками с камеры"""
    
    @staticmethod
    def capture_single_frame(camera_id: int = 0) -> Optional[np.ndarray]:
        """
        Захватывает один кадр с камеры
        
        Args:
            camera_id: ID камеры
            
        Returns:
            Кадр или None
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return None
        
        # Пропускаем несколько первых кадров для стабилизации
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    @staticmethod
    def capture_multiple_frames(
        camera_id: int = 0,
        num_frames: int = 5,
        interval_ms: int = 100
    ) -> List[np.ndarray]:
        """
        Захватывает несколько кадров
        
        Args:
            camera_id: ID камеры
            num_frames: Количество кадров
            interval_ms: Интервал между кадрами в мс
            
        Returns:
            Список кадров
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return []
        
        frames = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            
            if i < num_frames - 1:
                cv2.waitKey(interval_ms)
        
        cap.release()
        return frames
    
    @staticmethod
    def capture_best_frame(
        camera_id: int = 0,
        num_samples: int = 10
    ) -> Optional[np.ndarray]:
        """
        Захватывает несколько кадров и возвращает наименее размытый
        
        Args:
            camera_id: ID камеры
            num_samples: Количество кадров для анализа
            
        Returns:
            Лучший кадр или None
        """
        frames = CameraSnapshot.capture_multiple_frames(
            camera_id, num_samples, 50
        )
        
        if not frames:
            return None
        
        # Оценка резкости через Лапласиан
        best_frame = None
        best_score = -1
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if score > best_score:
                best_score = score
                best_frame = frame
        
        return best_frame


class CameraCalibration:
    """Класс для калибровки камеры"""
    
    @staticmethod
    def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
        """
        Регулирует яркость
        
        Args:
            frame: Исходный кадр
            factor: Коэффициент яркости (1.0 = без изменений)
            
        Returns:
            Кадр с измененной яркостью
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
        """
        Регулирует контраст
        
        Args:
            frame: Исходный кадр
            factor: Коэффициент контраста (1.0 = без изменений)
            
        Returns:
            Кадр с измененным контрастом
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l = np.clip((l - 128) * factor + 128, 0, 255).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def auto_white_balance(frame: np.ndarray) -> np.ndarray:
        """
        Автоматическая балансировка белого
        
        Args:
            frame: Исходный кадр
            
        Returns:
            Кадр с балансировкой белого
        """
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """
        Улучшает качество кадра
        
        Args:
            frame: Исходный кадр
            
        Returns:
            Улучшенный кадр
        """
        # Шумоподавление
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Повышение резкости
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened