"""
Модуль для работы с GigE Vision камерами (JAI, Basler, и др.)
Использует библиотеку Harvester для доступа к GenICam устройствам

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.

Требования:
- pip install harvesters
- GenTL producer (.cti файл) от производителя камеры:
  - JAI SDK (для JAI камер)
  - Pylon (для Basler)
  - Matrix Vision (mvGenTL)
  - Stemmer Imaging
"""

import os
import sys
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from PySide6.QtCore import QThread, Signal

# Попытка импорта Harvester
try:
    from harvesters.core import Harvester
    HARVESTER_AVAILABLE = True
except ImportError:
    HARVESTER_AVAILABLE = False
    print("WARNING: Harvester не установлен. GigE Vision камеры недоступны.")
    print("Установите: pip install harvesters")


def find_cti_files() -> List[str]:
    """
    Ищет GenTL producer (.cti) файлы в стандартных местах

    Returns:
        Список путей к найденным .cti файлам
    """
    cti_files = []

    # Стандартные пути для поиска GenTL producers
    search_paths = []

    # 1. Сначала ищем в папке с приложением (для PyInstaller сборки)
    if getattr(sys, 'frozen', False):
        # Запущено из PyInstaller сборки
        app_dir = os.path.dirname(sys.executable)
    else:
        # Запущено из исходников
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Проверяем CTI в папке приложения
    for cti_name in ['ic4-gentl-gev.cti', 'mvGenTLProducer.cti', 'TIS_GenTL.cti']:
        cti_path = os.path.join(app_dir, cti_name)
        if os.path.exists(cti_path):
            cti_files.append(cti_path)
            print(f"GigE: Найден CTI в папке приложения: {cti_path}")

    if sys.platform == 'win32':
        # Windows пути
        program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
        program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')

        search_paths = [
            # The Imaging Source (бесплатный)
            os.path.join(program_files, 'The Imaging Source Europe GmbH', 'IC4 GenTL Driver for GigEVision Devices', 'bin'),
            # JAI SDK
            os.path.join(program_files, 'JAI', 'SDK'),
            os.path.join(program_files_x86, 'JAI', 'SDK'),
            # eBUS SDK (Pleora)
            os.path.join(program_files, 'Pleora Technologies Inc'),
            os.path.join(program_files_x86, 'Pleora Technologies Inc'),
            # Matrix Vision
            os.path.join(program_files, 'MATRIX VISION', 'mvIMPACT Acquire'),
            os.path.join(program_files_x86, 'MATRIX VISION', 'mvIMPACT Acquire'),
            # Basler Pylon
            os.path.join(program_files, 'Basler', 'pylon'),
            os.path.join(program_files_x86, 'Basler', 'pylon'),
            # Stemmer Imaging
            os.path.join(program_files, 'STEMMER IMAGING', 'Common Vision Blox'),
            os.path.join(program_files_x86, 'STEMMER IMAGING', 'Common Vision Blox'),
            # GenTL standard path
            os.environ.get('GENICAM_GENTL64_PATH', ''),
            os.environ.get('GENICAM_GENTL32_PATH', ''),
        ]
    else:
        # Linux пути
        search_paths = [
            '/opt/JAI',
            '/opt/pleora',
            '/opt/mvIMPACT_Acquire',
            '/opt/pylon',
            os.environ.get('GENICAM_GENTL64_PATH', ''),
            os.environ.get('GENICAM_GENTL32_PATH', ''),
        ]

    # Поиск .cti файлов
    for base_path in search_paths:
        if not base_path or not os.path.exists(base_path):
            continue

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.cti'):
                    cti_path = os.path.join(root, file)
                    if cti_path not in cti_files:
                        cti_files.append(cti_path)

    return cti_files


class GigECameraCapture(QThread):
    """Поток для захвата изображений с GigE Vision камеры"""

    frame_captured = Signal(object)  # Сигнал с новым кадром (numpy array)
    error = Signal(str)  # Сигнал ошибки
    camera_info = Signal(dict)  # Информация о камере

    def __init__(self, device_index: int = 0, cti_file: str = None):
        """
        Args:
            device_index: Индекс устройства (0 для первой камеры)
            cti_file: Путь к GenTL producer (.cti файл)
        """
        super().__init__()

        if not HARVESTER_AVAILABLE:
            raise RuntimeError("Harvester не установлен")

        self.device_index = device_index
        self.cti_file = cti_file
        self.harvester = None
        self.ia = None  # Image Acquirer
        self.is_running = False
        self.fps = 30

    def _init_harvester(self) -> bool:
        """Инициализация Harvester и подключение GenTL producer"""
        try:
            self.harvester = Harvester()

            # Если CTI файл не указан, ищем автоматически
            if not self.cti_file:
                cti_files = find_cti_files()
                if not cti_files:
                    self.error.emit("GenTL producer (.cti) не найден. Установите JAI SDK или другой GenTL provider.")
                    return False
                self.cti_file = cti_files[0]
                print(f"Используется GenTL producer: {self.cti_file}")

            # Загружаем CTI файл
            self.harvester.add_file(self.cti_file)
            self.harvester.update()

            return True

        except Exception as e:
            self.error.emit(f"Ошибка инициализации Harvester: {str(e)}")
            return False

    def run(self):
        """Основной цикл захвата кадров"""
        try:
            print(f"GigE Camera: Запуск камеры {self.device_index}")

            # Инициализация
            if not self._init_harvester():
                return

            # Проверяем наличие устройств
            device_count = len(self.harvester.device_info_list)
            print(f"GigE Camera: Найдено устройств: {device_count}")

            if device_count == 0:
                self.error.emit("GigE Vision камеры не найдены. Проверьте подключение и сетевые настройки.")
                return

            if self.device_index >= device_count:
                self.error.emit(f"Устройство {self.device_index} не найдено. Доступно: {device_count}")
                return

            # Выводим информацию о найденных камерах
            for i, dev_info in enumerate(self.harvester.device_info_list):
                print(f"  [{i}] {dev_info.vendor} - {dev_info.model} (S/N: {dev_info.serial_number})")

            # Создаем Image Acquirer
            self.ia = self.harvester.create(self.device_index)

            # Получаем информацию о камере
            dev_info = self.harvester.device_info_list[self.device_index]
            info = {
                'vendor': dev_info.vendor,
                'model': dev_info.model,
                'serial_number': dev_info.serial_number,
                'display_name': dev_info.display_name,
            }
            self.camera_info.emit(info)

            # Настройка параметров (если доступны)
            try:
                node_map = self.ia.remote_device.node_map

                # Попытка установить режим непрерывной съемки
                if hasattr(node_map, 'AcquisitionMode'):
                    node_map.AcquisitionMode.value = 'Continuous'

                # Получаем разрешение
                width = node_map.Width.value if hasattr(node_map, 'Width') else 'N/A'
                height = node_map.Height.value if hasattr(node_map, 'Height') else 'N/A'
                print(f"GigE Camera: Разрешение {width}x{height}")

            except Exception as e:
                print(f"GigE Camera: Не удалось настроить параметры: {e}")

            # Запускаем захват
            self.ia.start()
            self.is_running = True
            print("GigE Camera: Захват запущен")

            # Основной цикл
            while self.is_running:
                try:
                    # Получаем буфер с кадром (с таймаутом)
                    with self.ia.fetch(timeout=2.0) as buffer:
                        # Получаем компонент изображения
                        component = buffer.payload.components[0]

                        # Конвертируем в numpy array
                        frame = self._convert_to_numpy(component)

                        if frame is not None:
                            self.frame_captured.emit(frame)

                except TimeoutError:
                    # Таймаут - это нормально, продолжаем
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"GigE Camera: Ошибка чтения кадра: {e}")

                # Контроль FPS
                self.msleep(int(1000 / self.fps))

        except Exception as e:
            import traceback
            self.error.emit(f"GigE Camera ошибка: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.stop()

    def _convert_to_numpy(self, component) -> Optional[np.ndarray]:
        """
        Конвертирует компонент изображения в numpy array

        Args:
            component: Компонент изображения из Harvester

        Returns:
            numpy array в формате BGR (для совместимости с OpenCV)
        """
        try:
            # Получаем данные
            width = component.width
            height = component.height
            pixel_format = component.data_format

            # Создаем numpy array из данных
            if hasattr(component, 'data'):
                data = component.data
            else:
                data = np.array(component)

            # Определяем формат и конвертируем
            if 'Mono' in pixel_format or 'mono' in pixel_format.lower():
                # Grayscale
                if 'Mono8' in pixel_format:
                    frame = data.reshape((height, width))
                elif 'Mono10' in pixel_format or 'Mono12' in pixel_format or 'Mono16' in pixel_format:
                    frame = data.reshape((height, width))
                    # Нормализуем до 8 бит
                    frame = (frame / frame.max() * 255).astype(np.uint8)
                else:
                    frame = data.reshape((height, width))

                # Конвертируем в BGR для совместимости
                frame = np.stack([frame, frame, frame], axis=-1)

            elif 'RGB' in pixel_format:
                # RGB формат
                frame = data.reshape((height, width, 3))
                # Конвертируем RGB -> BGR
                frame = frame[:, :, ::-1].copy()

            elif 'BGR' in pixel_format:
                # BGR формат (уже готов)
                frame = data.reshape((height, width, 3))

            elif 'BayerRG' in pixel_format or 'BayerGB' in pixel_format or 'BayerGR' in pixel_format or 'BayerBG' in pixel_format:
                # Bayer формат - нужна демозаика
                import cv2
                raw = data.reshape((height, width))

                # Определяем паттерн Bayer
                if 'BayerRG' in pixel_format:
                    pattern = cv2.COLOR_BayerRG2BGR
                elif 'BayerGB' in pixel_format:
                    pattern = cv2.COLOR_BayerGB2BGR
                elif 'BayerGR' in pixel_format:
                    pattern = cv2.COLOR_BayerGR2BGR
                elif 'BayerBG' in pixel_format:
                    pattern = cv2.COLOR_BayerBG2BGR
                else:
                    pattern = cv2.COLOR_BayerBG2BGR

                frame = cv2.cvtColor(raw, pattern)

            else:
                # Неизвестный формат - пробуем как grayscale
                print(f"GigE Camera: Неизвестный формат {pixel_format}, пробуем как Mono8")
                frame = data.reshape((height, width))
                frame = np.stack([frame, frame, frame], axis=-1)

            return frame.astype(np.uint8)

        except Exception as e:
            print(f"GigE Camera: Ошибка конвертации кадра: {e}")
            return None

    def stop(self):
        """Остановка захвата"""
        self.is_running = False

        if self.ia is not None:
            try:
                self.ia.stop()
                self.ia.destroy()
            except:
                pass
            self.ia = None

        if self.harvester is not None:
            try:
                self.harvester.reset()
            except:
                pass
            self.harvester = None

        print("GigE Camera: Остановлен")

    def set_exposure(self, exposure_us: float):
        """
        Устанавливает время экспозиции

        Args:
            exposure_us: Время экспозиции в микросекундах
        """
        if self.ia is not None:
            try:
                node_map = self.ia.remote_device.node_map
                if hasattr(node_map, 'ExposureTime'):
                    node_map.ExposureTime.value = exposure_us
                elif hasattr(node_map, 'ExposureTimeAbs'):
                    node_map.ExposureTimeAbs.value = exposure_us
            except Exception as e:
                print(f"GigE Camera: Не удалось установить экспозицию: {e}")

    def set_gain(self, gain_db: float):
        """
        Устанавливает усиление

        Args:
            gain_db: Усиление в dB
        """
        if self.ia is not None:
            try:
                node_map = self.ia.remote_device.node_map
                if hasattr(node_map, 'Gain'):
                    node_map.Gain.value = gain_db
                elif hasattr(node_map, 'GainRaw'):
                    node_map.GainRaw.value = int(gain_db)
            except Exception as e:
                print(f"GigE Camera: Не удалось установить gain: {e}")


class GigECameraManager:
    """Менеджер для работы с GigE Vision камерами"""

    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность GigE Vision функциональности"""
        return HARVESTER_AVAILABLE

    @staticmethod
    def get_available_cameras(cti_file: str = None) -> List[Dict]:
        """
        Получает список доступных GigE Vision камер

        Args:
            cti_file: Путь к GenTL producer (опционально)

        Returns:
            Список словарей с информацией о камерах
        """
        if not HARVESTER_AVAILABLE:
            return []

        cameras = []
        h = None

        try:
            h = Harvester()

            # Определяем CTI файл
            if not cti_file:
                cti_files = find_cti_files()
                if not cti_files:
                    print("GigE: GenTL producer не найден")
                    return []
                cti_file = cti_files[0]

            h.add_file(cti_file)
            h.update()

            for i, dev_info in enumerate(h.device_info_list):
                cameras.append({
                    'index': i,
                    'vendor': dev_info.vendor,
                    'model': dev_info.model,
                    'serial_number': dev_info.serial_number,
                    'display_name': dev_info.display_name,
                    'type': 'GigE Vision',
                    'cti_file': cti_file,
                })

        except Exception as e:
            print(f"GigE: Ошибка получения списка камер: {e}")
        finally:
            if h is not None:
                try:
                    h.reset()
                except:
                    pass

        return cameras

    @staticmethod
    def get_cti_files() -> List[str]:
        """Возвращает список найденных GenTL producers"""
        return find_cti_files()

    @staticmethod
    def test_camera(device_index: int = 0, cti_file: str = None) -> bool:
        """
        Тестирует подключение к камере

        Args:
            device_index: Индекс устройства
            cti_file: Путь к GenTL producer

        Returns:
            True если камера работает
        """
        if not HARVESTER_AVAILABLE:
            return False

        h = None
        ia = None

        try:
            h = Harvester()

            if not cti_file:
                cti_files = find_cti_files()
                if not cti_files:
                    return False
                cti_file = cti_files[0]

            h.add_file(cti_file)
            h.update()

            if len(h.device_info_list) == 0:
                return False

            if device_index >= len(h.device_info_list):
                return False

            # Пробуем захватить кадр
            ia = h.create(device_index)
            ia.start()

            with ia.fetch(timeout=5.0) as buffer:
                component = buffer.payload.components[0]
                if component.width > 0 and component.height > 0:
                    return True

            return False

        except Exception as e:
            print(f"GigE: Тест камеры провален: {e}")
            return False
        finally:
            if ia is not None:
                try:
                    ia.stop()
                    ia.destroy()
                except:
                    pass
            if h is not None:
                try:
                    h.reset()
                except:
                    pass


# Пример использования
if __name__ == "__main__":
    print("=== GigE Vision Camera Test ===")
    print(f"Harvester доступен: {HARVESTER_AVAILABLE}")

    if HARVESTER_AVAILABLE:
        print("\nПоиск GenTL producers...")
        cti_files = find_cti_files()
        for cti in cti_files:
            print(f"  - {cti}")

        print("\nПоиск камер...")
        cameras = GigECameraManager.get_available_cameras()
        for cam in cameras:
            print(f"  [{cam['index']}] {cam['vendor']} {cam['model']} (S/N: {cam['serial_number']})")

        if cameras:
            print("\nТест первой камеры...")
            if GigECameraManager.test_camera(0):
                print("  Камера работает!")
            else:
                print("  Камера не отвечает")
