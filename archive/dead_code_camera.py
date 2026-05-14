"""
Архив: неиспользуемый код из core/cameras/camera.py
Перемещено: 2026-05-14
Причина: классы и методы нигде не вызываются
"""
import cv2
import numpy as np
from typing import Optional, List


# === Из core/cameras/camera.py ===

# --- Класс CameraCalibration ---

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


# --- Класс CameraSnapshot ---

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
        import time

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            return None

        time.sleep(1.0)  # Инициализация

        # Пропускаем несколько первых кадров для стабилизации
        for _ in range(10):
            cap.read()
            time.sleep(0.05)

        # Берем лучший из нескольких кадров
        best_frame = None
        best_std = 0

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                std_val = np.std(frame)
                if std_val > best_std:
                    best_std = std_val
                    best_frame = frame
            time.sleep(0.05)

        cap.release()

        return best_frame if best_std > 10 else None

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
        import time

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            return []

        time.sleep(1.0)  # Инициализация

        # Прогрев
        for _ in range(10):
            cap.read()

        frames = []

        for i in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None and np.std(frame) > 10:
                frames.append(frame)

            if i < num_frames - 1:
                time.sleep(interval_ms / 1000.0)

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


# --- Метод CameraManager.test_camera ---

# Оригинальный класс: CameraManager (core/cameras/camera.py)
# Метод нигде не вызывается в production коде

def test_camera(camera_id: int, duration: int = 3) -> bool:
    """
    Тестирует камеру

    Args:
        camera_id: ID камеры
        duration: Длительность теста в секундах

    Returns:
        True если камера работает
    """
    import time

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        return False

    time.sleep(1.0)  # Инициализация

    success = False
    frames_captured = 0
    valid_frames = 0

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frames_captured += 1
            if frame is not None and np.std(frame) > 10:
                valid_frames += 1

    success = valid_frames > 0
    cap.release()

    print(f"Тест камеры {camera_id}: {frames_captured} кадров, {valid_frames} валидных")
    return success


# --- Методы CameraCapture.set_resolution / get_resolution ---

# Оригинальный класс: CameraCapture (core/cameras/camera.py)
# Методы нигде не вызываются в production коде

def set_resolution(capture, width: int, height: int):
    """Устанавливает разрешение камеры"""
    if capture is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def get_resolution(capture):
    """Возвращает текущее разрешение камеры"""
    if capture is not None:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    return (0, 0)
