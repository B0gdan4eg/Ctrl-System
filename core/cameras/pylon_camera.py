"""
Модуль для работы с GigE Vision камерами через Basler Pylon SDK
Поддерживает камеры разных производителей (JAI, Basler, и др.)

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import numpy as np
from typing import Optional, List, Dict
from PySide6.QtCore import QThread, Signal

# Попытка импорта pypylon
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False
    print("WARNING: pypylon не установлен. GigE Vision камеры недоступны.")
    print("Установите: pip install pypylon")


class PylonCameraCapture(QThread):
    """Поток для захвата изображений с GigE Vision камеры через Pylon"""

    frame_captured = Signal(object)  # numpy array BGR
    error = Signal(str)
    camera_info = Signal(dict)

    def __init__(self, device_index: int = 0):
        super().__init__()

        if not PYLON_AVAILABLE:
            raise RuntimeError("pypylon не установлен")

        self.device_index = device_index
        self.camera = None
        self.is_running = False
        self.converter = None

    def run(self):
        """Основной цикл захвата"""
        try:
            # Получаем фабрику
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if not devices:
                self.error.emit("GigE камеры не найдены")
                return

            if self.device_index >= len(devices):
                self.error.emit(f"Камера {self.device_index} не найдена. Доступно: {len(devices)}")
                return

            # Информация о камере
            dev_info = devices[self.device_index]
            info = {
                'vendor': dev_info.GetVendorName(),
                'model': dev_info.GetModelName(),
                'serial_number': dev_info.GetSerialNumber(),
            }
            self.camera_info.emit(info)
            print(f"Pylon: Подключение к {info['vendor']} {info['model']}")

            # Создаём камеру
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(dev_info))
            self.camera.Open()

            # Настройка конвертера в BGR
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Запуск захвата
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_running = True

            print("Pylon: Захват запущен")

            while self.is_running and self.camera.IsGrabbing():
                try:
                    # Получаем кадр (таймаут 5 сек)
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    if grabResult.GrabSucceeded():
                        # Конвертируем в BGR
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray().copy()

                        self.frame_captured.emit(frame)
                    else:
                        print(f"Pylon: Ошибка захвата: {grabResult.ErrorCode}")

                    grabResult.Release()

                except pylon.TimeoutException:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"Pylon: Ошибка: {e}")

        except Exception as e:
            import traceback
            self.error.emit(f"Pylon ошибка: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.stop()

    def stop(self):
        """Остановка захвата"""
        self.is_running = False

        if self.camera is not None:
            try:
                self.camera.StopGrabbing()
                self.camera.Close()
            except:
                pass
            self.camera = None

        print("Pylon: Остановлен")

    def set_exposure(self, exposure_us: float):
        """Установка экспозиции в микросекундах"""
        if self.camera is not None and self.camera.IsOpen():
            try:
                self.camera.ExposureTime.SetValue(exposure_us)
            except:
                try:
                    self.camera.ExposureTimeAbs.SetValue(exposure_us)
                except Exception as e:
                    print(f"Pylon: Не удалось установить экспозицию: {e}")

    def set_gain(self, gain_db: float):
        """Установка усиления в dB"""
        if self.camera is not None and self.camera.IsOpen():
            try:
                self.camera.Gain.SetValue(gain_db)
            except:
                try:
                    self.camera.GainRaw.SetValue(int(gain_db))
                except Exception as e:
                    print(f"Pylon: Не удалось установить gain: {e}")


class PylonCameraManager:
    """Менеджер для работы с камерами через Pylon"""

    @staticmethod
    def is_available() -> bool:
        """Проверка доступности Pylon"""
        return PYLON_AVAILABLE

    @staticmethod
    def get_available_cameras() -> List[Dict]:
        """Получение списка камер"""
        if not PYLON_AVAILABLE:
            return []

        cameras = []
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            for i, dev in enumerate(devices):
                cameras.append({
                    'index': i,
                    'vendor': dev.GetVendorName(),
                    'model': dev.GetModelName(),
                    'serial_number': dev.GetSerialNumber(),
                    'type': 'GigE Vision (Pylon)',
                })
        except Exception as e:
            print(f"Pylon: Ошибка получения списка камер: {e}")

        return cameras

    @staticmethod
    def test_camera(device_index: int = 0) -> bool:
        """Тест камеры"""
        if not PYLON_AVAILABLE:
            return False

        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if not devices or device_index >= len(devices):
                return False

            camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))
            camera.Open()

            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            success = grabResult.GrabSucceeded()
            grabResult.Release()

            camera.StopGrabbing()
            camera.Close()

            return success

        except Exception as e:
            print(f"Pylon: Тест провален: {e}")
            return False


def capture_single_frame(device_index: int = 0) -> Optional[np.ndarray]:
    """Захват одного кадра"""
    if not PYLON_AVAILABLE:
        return None

    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if not devices or device_index >= len(devices):
            return None

        camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))
        camera.Open()

        # Конвертер
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        frame = None
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            frame = image.GetArray().copy()

        grabResult.Release()
        camera.StopGrabbing()
        camera.Close()

        return frame

    except Exception as e:
        print(f"Pylon: Ошибка захвата: {e}")
        return None


# Тест
if __name__ == "__main__":
    print("=== Pylon Camera Test ===")
    print(f"Pylon доступен: {PYLON_AVAILABLE}")

    if PYLON_AVAILABLE:
        cameras = PylonCameraManager.get_available_cameras()
        print(f"\nНайдено камер: {len(cameras)}")

        for cam in cameras:
            print(f"  [{cam['index']}] {cam['vendor']} {cam['model']} (S/N: {cam['serial_number']})")
