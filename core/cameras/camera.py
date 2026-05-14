"""
Модуль для работы с камерой

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import numpy as np
from typing import Optional, List
from PySide6.QtCore import QThread, Signal


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
        import time
        
        if platform.system() == "Windows":
            backends = [
                ('CAP_DSHOW', cv2.CAP_DSHOW),
                ('CAP_MSMF', cv2.CAP_MSMF),
            ]
        elif platform.system() == "Linux":
            backends = [
                ('CAP_V4L2', cv2.CAP_V4L2),
                ('CAP_ANY', cv2.CAP_ANY),
            ]
        elif platform.system() == "Darwin":
            backends = [
                ('CAP_AVFOUNDATION', cv2.CAP_AVFOUNDATION),
                ('CAP_ANY', cv2.CAP_ANY),
            ]
        else:
            backends = [('CAP_ANY', cv2.CAP_ANY)]
        
        if self.backend is not None:
            for attempt in range(3):
                try:
                    self.capture = cv2.VideoCapture(self.camera_id, self.backend)
                    if self.capture.isOpened():
                        time.sleep(1.0)  # Даем камере время инициализироваться
                        
                        # Проверяем несколько кадров
                        for _ in range(5):
                            ret, frame = self.capture.read()
                            if ret and frame is not None and np.std(frame) > 10:
                                return True
                            time.sleep(0.1)
                    
                    self.capture.release()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"❌ Попытка {attempt+1}: {e}")
            return False
        
        # Пробуем разные backends
        for name, backend in backends:
            try:
                print(f"🔄 Пробую {name}...")
                
                if self.capture is not None:
                    self.capture.release()
                    time.sleep(0.5)
                
                self.capture = cv2.VideoCapture(self.camera_id, backend)
                
                if self.capture.isOpened():
                    # КРИТИЧНО: даем камере время инициализироваться
                    time.sleep(1.2)
                    
                    # Пробуем прочитать несколько кадров и проверяем валидность
                    valid_count = 0
                    for attempt in range(10):
                        ret, frame = self.capture.read()
                        if ret and frame is not None:
                            std_val = np.std(frame)
                            mean_val = np.mean(frame)
                            
                            # Проверяем что это не шум
                            if std_val > 10 and 10 < mean_val < 245:
                                valid_count += 1
                                if valid_count >= 3:
                                    print(f"✅ Камера открыта через {name}")
                                    return True
                        time.sleep(0.1)
                    
                    self.capture.release()
            except Exception as e:
                print(f"❌ {name}: {e}")
            
            time.sleep(0.3)
        
        return False
        
    def run(self):
        """Основной цикл захвата кадров"""
        try:
            # Диагностика
            print(f"🔍 Попытка открыть камеру {self.camera_id}")
            available = CameraManager.get_available_cameras()
            print(f"📋 Доступные камеры: {available}")
            
            if not self._try_open_camera():
                self.error.emit(f"Не удалось открыть камеру {self.camera_id}")
                return
            
            # Настройка камеры
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Попытка установить формат (может помочь с некоторыми камерами)
            try:
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            except:
                pass
            
            # КРИТИЧНО: Инициализация и очистка буфера
            print("⏳ Инициализация камеры...")
            import time
            time.sleep(1.5)
            
            print("🧹 Очистка буфера камеры...")
            valid_frames = 0
            max_attempts = 50
            
            for attempt in range(max_attempts):
                ret, frame = self.capture.read()
                
                if ret and frame is not None:
                    # Проверяем что кадр не "шум"
                    mean_val = np.mean(frame)
                    std_val = np.std(frame)
                    
                    # Нормальный кадр имеет std > 10 и mean не близко к 0 или 255
                    if std_val > 10 and 10 < mean_val < 245:
                        valid_frames += 1
                        print(f"✓ Валидный кадр {valid_frames}/3 (std={std_val:.1f}, mean={mean_val:.1f})")
                        if valid_frames >= 3:
                            print("✅ Камера готова к работе")
                            break
                    else:
                        if valid_frames > 0:
                            print(f"✗ Невалидный кадр (std={std_val:.1f}, mean={mean_val:.1f})")
                        valid_frames = 0
                
                time.sleep(0.1)
            
            if valid_frames < 3:
                self.error.emit("Камера не выдает валидное изображение после инициализации")
                return
            
            self.is_running = True
            consecutive_errors = 0
            
            while self.is_running:
                ret, frame = self.capture.read()
                
                if ret and frame is not None:
                    # Дополнительная проверка валидности в основном цикле
                    std_val = np.std(frame)
                    if std_val > 5:  # Это не шум
                        self.frame_captured.emit(frame)
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        if consecutive_errors > 10:
                            self.error.emit("Камера начала выдавать шум")
                            break
                else:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
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
            # Читаем несколько кадров для гарантии свежести
            for _ in range(3):
                ret, frame = self.capture.read()
            
            if ret and frame is not None and np.std(frame) > 10:
                return frame
        return None
    
    # Moved to archive/dead_code_camera.py


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
        import time
        
        available = []
        
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif platform.system() == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        elif platform.system() == "Darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        
        for i in range(max_cameras):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        time.sleep(0.5)  # Даем время на инициализацию
                        
                        # Проверяем что можем читать валидные кадры
                        ret, frame = cap.read()
                        if ret and frame is not None and np.std(frame) > 10:
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
        import time
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return {"available": False}
        
        time.sleep(0.5)
        
        info = {
            "available": True,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "backend": cap.getBackendName()
        }
        
        cap.release()
        return info
    
    # Moved to archive/dead_code_camera.py


# Moved to archive/dead_code_camera.py