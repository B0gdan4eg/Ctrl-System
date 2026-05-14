"""
Модуль для работы с JAI GigE Vision камерами через Pleora eBUS SDK
Поддерживает камеры JAI RM-4200GE и другие GigE Vision камеры

Реализует работу с камерой аналогично camera_viewer.py:
- Использует CopyToBitmap для извлечения изображения
- Быстрая конвертация через ctypes.memmove
- Формат пикселей: Mono8 -> RGB24

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import sys
import os
import numpy as np
import ctypes
import traceback
import time
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from PySide6.QtCore import QThread, Signal

# Импорт логгера приложения
try:
    from utils import app_logger
    _USE_APP_LOGGER = True
except ImportError:
    _USE_APP_LOGGER = False
    import logging
    app_logger = logging.getLogger("ebus_camera")

# Путь к eBUS SDK по умолчанию
DEFAULT_EBUS_SDK_PATH = r"C:\Program Files\Common Files\Pleora\eBUS SDK"

# Флаг доступности eBUS SDK
EBUS_AVAILABLE = False
_ebus_initialized = False

# Глобальные ссылки на .NET типы (заполняются после инициализации)
PvSystem = None
PvDevice = None
PvStream = None
PvPipeline = None
UInt32 = None
Bitmap = None
PixelFormat = None
ImageLockMode = None
SystemDrawing = None


def log_ebus(message: str, level: str = "INFO"):
    """
    Логирование eBUS сообщений через app_logger

    Args:
        message: Текст сообщения
        level: Уровень логирования (INFO, WARN, ERROR, DEBUG)
    """
    log_message = f"[eBUS] {message}"

    if level == "DEBUG":
        app_logger.debug(log_message)
    elif level == "WARN":
        app_logger.warning(log_message)
    elif level == "ERROR":
        app_logger.error(log_message)
    else:
        app_logger.info(log_message)


def init_ebus_sdk(sdk_path: str = None) -> bool:
    """
    Инициализация eBUS SDK

    Args:
        sdk_path: Путь к папке eBUS SDK (опционально)

    Returns:
        True если инициализация успешна
    """
    global EBUS_AVAILABLE, _ebus_initialized
    global PvSystem, PvDevice, PvStream, PvPipeline
    global UInt32, Bitmap, PixelFormat, ImageLockMode, SystemDrawing

    if _ebus_initialized:
        log_ebus(f"SDK уже инициализирован, EBUS_AVAILABLE={EBUS_AVAILABLE}", "DEBUG")
        return EBUS_AVAILABLE

    log_ebus("=== Начало инициализации eBUS SDK ===")

    if sdk_path is None:
        # Пробуем импортировать из настроек
        try:
            from config.settings import EBUS_SDK_PATH
            sdk_path = EBUS_SDK_PATH
            log_ebus(f"Путь SDK из config.settings: {sdk_path}", "DEBUG")
        except ImportError:
            sdk_path = DEFAULT_EBUS_SDK_PATH
            log_ebus(f"Используется путь по умолчанию: {sdk_path}", "DEBUG")

    log_ebus(f"Путь к SDK: {sdk_path}")

    # Проверяем существование папки SDK
    if not os.path.exists(sdk_path):
        log_ebus(f"Папка SDK не найдена: {sdk_path}", "ERROR")
        _ebus_initialized = True
        return False

    log_ebus(f"Папка SDK существует: {sdk_path}", "DEBUG")

    dll_path = os.path.join(sdk_path, "PvDotNet.dll")
    if not os.path.exists(dll_path):
        log_ebus(f"PvDotNet.dll не найден: {dll_path}", "ERROR")
        _ebus_initialized = True
        return False

    log_ebus(f"PvDotNet.dll найден: {dll_path}", "DEBUG")

    # Показываем содержимое папки SDK
    try:
        files = os.listdir(sdk_path)
        dll_files = [f for f in files if f.endswith('.dll')]
        log_ebus(f"DLL файлы в SDK: {dll_files}", "DEBUG")
    except Exception as e:
        log_ebus(f"Не удалось прочитать содержимое SDK: {e}", "WARN")

    try:
        # Добавляем путь в sys.path
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
            log_ebus(f"Добавлен путь в sys.path: {sdk_path}", "DEBUG")

        # Импортируем pythonnet
        log_ebus("Импорт pythonnet (clr)...", "DEBUG")
        import clr
        log_ebus(f"pythonnet импортирован успешно", "DEBUG")

        # Загружаем сборки
        log_ebus(f"Загрузка сборки: {dll_path}", "DEBUG")
        clr.AddReference(dll_path)
        log_ebus("PvDotNet.dll загружен", "DEBUG")

        log_ebus("Загрузка сборки: System.Drawing", "DEBUG")
        clr.AddReference("System.Drawing")
        log_ebus("System.Drawing загружен", "DEBUG")

        # Импортируем типы
        log_ebus("Импорт типов из PvDotNet...", "DEBUG")
        from PvDotNet import PvSystem as _PvSystem
        from PvDotNet import PvDevice as _PvDevice
        from PvDotNet import PvStream as _PvStream
        from PvDotNet import PvPipeline as _PvPipeline
        log_ebus("Типы PvDotNet импортированы", "DEBUG")

        log_ebus("Импорт типов из System...", "DEBUG")
        from System import UInt32 as _UInt32
        from System.Drawing import Bitmap as _Bitmap
        from System.Drawing.Imaging import PixelFormat as _PixelFormat
        from System.Drawing.Imaging import ImageLockMode as _ImageLockMode
        import System.Drawing as _SystemDrawing
        log_ebus("Типы System импортированы", "DEBUG")

        # Сохраняем ссылки глобально
        PvSystem = _PvSystem
        PvDevice = _PvDevice
        PvStream = _PvStream
        PvPipeline = _PvPipeline
        UInt32 = _UInt32
        Bitmap = _Bitmap
        PixelFormat = _PixelFormat
        ImageLockMode = _ImageLockMode
        SystemDrawing = _SystemDrawing

        EBUS_AVAILABLE = True
        _ebus_initialized = True
        log_ebus(f"=== eBUS SDK инициализирован УСПЕШНО ===")
        return True

    except ImportError as e:
        log_ebus(f"Ошибка импорта: {e}", "ERROR")
        log_ebus(f"Traceback: {traceback.format_exc()}", "DEBUG")
        log_ebus("Установите: pip install pythonnet", "ERROR")
        _ebus_initialized = True
        return False
    except Exception as e:
        log_ebus(f"Ошибка инициализации: {e}", "ERROR")
        log_ebus(f"Traceback: {traceback.format_exc()}", "DEBUG")
        _ebus_initialized = True
        return False


def bitmap_to_numpy(bitmap, buffer: np.ndarray = None) -> np.ndarray:
    """
    Конвертация System.Drawing.Bitmap в numpy array

    Использует LockBits + ctypes.memmove для быстрой конвертации

    Args:
        bitmap: System.Drawing.Bitmap объект
        buffer: Опциональный буфер для переиспользования

    Returns:
        numpy array формата HxWx3 (BGR)
    """
    width = bitmap.Width
    height = bitmap.Height

    # Блокируем биты для чтения
    rect = SystemDrawing.Rectangle(0, 0, width, height)
    bitmap_data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, bitmap.PixelFormat)

    try:
        # Получаем указатель как целое
        ptr = bitmap_data.Scan0.ToInt64()
        stride = bitmap_data.Stride
        bytes_per_pixel = 3  # Format24bppRgb

        # Общее количество байт
        total_bytes = stride * height

        # Создаем или переиспользуем буфер
        if buffer is None or buffer.size != total_bytes:
            buffer = np.empty(total_bytes, dtype=np.uint8)

        # Прямое копирование памяти через ctypes
        ctypes.memmove(buffer.ctypes.data, ptr, total_bytes)

        # Изменяем форму с учетом stride (может быть padding)
        img = buffer.reshape((height, stride))

        # Убираем padding если stride != width * bytes_per_pixel
        if stride != width * bytes_per_pixel:
            img = img[:, :width * bytes_per_pixel].copy()

        # Изменяем форму на HxWx3
        img = img.reshape((height, width, bytes_per_pixel))

        return img

    finally:
        bitmap.UnlockBits(bitmap_data)


class EbusCameraCapture(QThread):
    """
    Поток для захвата изображений с JAI GigE камеры через eBUS SDK

    Реализует захват кадров аналогично camera_viewer.py:
    - Использует PvPipeline для буферизации
    - CopyToBitmap для конвертации
    - ctypes.memmove для быстрого копирования в numpy

    Signals:
        frame_captured: Испускается при захвате кадра (numpy array BGR)
        error: Испускается при ошибке (str)
        camera_info: Испускается при подключении (dict с информацией о камере)
        stats_updated: Испускается при обновлении статистики (dict)
    """

    frame_captured = Signal(object)  # numpy array BGR
    error = Signal(str)
    camera_info = Signal(dict)
    stats_updated = Signal(dict)  # Статистика: fps, frame_count, errors, etc.

    def __init__(self, connection_id: str = None, buffer_count: int = 16):
        """
        Инициализация захвата с GigE камеры

        Args:
            connection_id: ID подключения камеры (опционально, берется первая найденная)
            buffer_count: Количество буферов в пайплайне (по умолчанию 16)
        """
        super().__init__()

        log_ebus(f"Инициализация EbusCameraCapture")
        log_ebus(f"  connection_id: {connection_id}")
        log_ebus(f"  buffer_count: {buffer_count}")

        if not init_ebus_sdk():
            log_ebus("КРИТИЧЕСКАЯ ОШИБКА: eBUS SDK не инициализирован", "ERROR")
            raise RuntimeError("eBUS SDK не инициализирован. Проверьте установку Pleora eBUS SDK.")

        self.connection_id = connection_id
        self.buffer_count = buffer_count

        self.is_running = False
        self.device = None
        self.stream = None
        self.pipeline = None
        self.bitmap = None

        # Параметры камеры
        self._exposure = None
        self._gain = None

        # Статистика
        self._frame_count = 0
        self._error_count = 0
        self._fps = 0.0
        self._fps_time = 0.0
        self._fps_frame_count = 0
        self._start_time = None
        self._last_frame_time = None

        # Размеры изображения
        self._width = 0
        self._height = 0

    def run(self):
        """
        Основной цикл захвата кадров с камеры

        Последовательность:
        1. Поиск камеры через PvSystem.Find()
        2. Подключение через PvDevice.CreateAndConnect()
        3. Открытие потока через PvStream.CreateAndOpen()
        4. Настройка PvPipeline с буферами
        5. Установка PixelFormat=Mono8
        6. Запуск AcquisitionStart
        7. Цикл: RetrieveNextBuffer -> CopyToBitmap -> bitmap_to_numpy -> emit
        """
        log_ebus("=" * 60)
        log_ebus("ЗАПУСК ЗАХВАТА С GigE КАМЕРЫ")
        log_ebus("=" * 60)
        self._start_time = time.time()

        try:
            # === ШАГ 1: Поиск камер ===
            log_ebus("[ШАГ 1/7] Поиск GigE Vision камер...")
            system = PvSystem()
            log_ebus("PvSystem создан, выполняется Find()...")

            find_start = time.time()
            system.Find()
            find_time = time.time() - find_start
            log_ebus(f"Find() завершен за {find_time:.2f} сек")

            device_count = system.DeviceCount
            log_ebus(f"Найдено устройств: {device_count}")

            if device_count == 0:
                error_msg = (
                    "GigE камеры не найдены!\n\n"
                    "Проверьте:\n"
                    "1. Камера подключена к сети и включена\n"
                    "2. IP адрес камеры в той же подсети (169.254.x.x)\n"
                    "3. Windows Firewall разрешает GigE Vision\n"
                    "4. VPN/proxy отключены\n"
                    "5. Сетевой адаптер поддерживает Jumbo Frames"
                )
                log_ebus(error_msg, "ERROR")
                self.error.emit(error_msg)
                return

            # === ШАГ 2: Информация о найденных камерах ===
            log_ebus("[ШАГ 2/7] Информация о найденных устройствах:")
            log_ebus("-" * 50)
            for i in range(device_count):
                try:
                    dev_info = system.GetDeviceInfo(i)
                    log_ebus(f"[Камера {i}]")
                    log_ebus(f"  Модель:      {dev_info.ModelName}")
                    log_ebus(f"  ConnectionID: {dev_info.ConnectionID}")
                    if hasattr(dev_info, 'Vendor'):
                        log_ebus(f"  Производитель: {dev_info.Vendor}")
                    if hasattr(dev_info, 'SerialNumber'):
                        log_ebus(f"  Серийный номер: {dev_info.SerialNumber}")
                    if hasattr(dev_info, 'IPAddress'):
                        log_ebus(f"  IP адрес:    {dev_info.IPAddress}")
                    if hasattr(dev_info, 'MACAddress'):
                        log_ebus(f"  MAC адрес:   {dev_info.MACAddress}")
                except Exception as e:
                    log_ebus(f"[Камера {i}] Ошибка: {e}", "WARN")
            log_ebus("-" * 50)

            # === ШАГ 3: Выбор камеры ===
            log_ebus("[ШАГ 3/7] Выбор камеры для подключения...")
            device_info = None

            if self.connection_id:
                log_ebus(f"Поиск камеры с ID: {self.connection_id}")
                for i in range(device_count):
                    info = system.GetDeviceInfo(i)
                    if str(info.ConnectionID) == str(self.connection_id):
                        device_info = info
                        log_ebus(f"Камера найдена: индекс {i}")
                        break

                if device_info is None:
                    error_msg = f"Камера с ID '{self.connection_id}' не найдена"
                    log_ebus(error_msg, "ERROR")
                    self.error.emit(error_msg)
                    return
            else:
                log_ebus("ID не указан, используется первая найденная камера")
                device_info = system.GetDeviceInfo(0)

            # Отправляем информацию о камере
            camera_info = {
                'vendor': str(device_info.Vendor) if hasattr(device_info, 'Vendor') else 'Unknown',
                'model': str(device_info.ModelName),
                'serial_number': str(device_info.SerialNumber) if hasattr(device_info, 'SerialNumber') else 'Unknown',
                'connection_id': str(device_info.ConnectionID),
                'ip_address': str(device_info.IPAddress) if hasattr(device_info, 'IPAddress') else 'Unknown',
                'mac_address': str(device_info.MACAddress) if hasattr(device_info, 'MACAddress') else 'Unknown',
            }
            self.camera_info.emit(camera_info)
            log_ebus(f"Выбрана камера: {camera_info['model']} ({camera_info['connection_id']})")

            # === ШАГ 4: Подключение к камере ===
            log_ebus("[ШАГ 4/7] Подключение к камере...")
            connect_start = time.time()

            log_ebus("  Создание PvDevice...")
            self.device = PvDevice.CreateAndConnect(device_info.ConnectionID)
            log_ebus(f"  PvDevice создан: {self.device}")

            log_ebus("  Создание PvStream...")
            self.stream = PvStream.CreateAndOpen(device_info.ConnectionID)
            log_ebus(f"  PvStream создан")
            log_ebus(f"    LocalIPAddress: {self.stream.LocalIPAddress}")
            log_ebus(f"    LocalPort: {self.stream.LocalPort}")

            # SetStreamDestination для камер которые этого требуют
            try:
                self.device.SetStreamDestination(self.stream.LocalIPAddress, self.stream.LocalPort)
                log_ebus("  SetStreamDestination: OK")
            except Exception as e:
                log_ebus(f"  SetStreamDestination: {e} (может быть нормально)", "DEBUG")

            connect_time = time.time() - connect_start
            log_ebus(f"Подключение завершено за {connect_time:.2f} сек")

            # === ШАГ 5: Настройка Pipeline ===
            log_ebus("[ШАГ 5/7] Настройка буферного пайплайна...")
            self.pipeline = PvPipeline(self.stream)

            payload_size = self.device.PayloadSize
            log_ebus(f"  PayloadSize: {payload_size} байт ({payload_size / 1024 / 1024:.2f} МБ)")

            self.pipeline.BufferSize = payload_size
            self.pipeline.BufferCount = self.buffer_count

            total_buffer_mb = (payload_size * self.buffer_count) / 1024 / 1024
            log_ebus(f"  BufferCount: {self.buffer_count}")
            log_ebus(f"  Всего памяти буферов: {total_buffer_mb:.2f} МБ")

            log_ebus("  Запуск pipeline...")
            self.pipeline.Start()
            log_ebus("  Pipeline запущен")

            # === ШАГ 6: Настройка формата пикселей ===
            log_ebus("[ШАГ 6/7] Настройка параметров камеры...")

            # PixelFormat
            try:
                pf = self.device.Parameters.Get("PixelFormat")
                old_format = pf.ValueString
                pf.ValueString = "Mono8"
                log_ebus(f"  PixelFormat: {old_format} -> Mono8")
            except Exception as e:
                log_ebus(f"  PixelFormat: не удалось установить ({e})", "WARN")

            # Обновляем буфер после смены формата
            new_payload = self.device.PayloadSize
            if new_payload != payload_size:
                log_ebus(f"  PayloadSize изменился: {payload_size} -> {new_payload}")
                self.pipeline.BufferSize = new_payload

            # Размеры изображения
            self._width = int(self.device.Parameters.Get("Width").Value)
            self._height = int(self.device.Parameters.Get("Height").Value)
            log_ebus(f"  Разрешение: {self._width}x{self._height}")
            log_ebus(f"  Всего пикселей: {self._width * self._height:,}")

            # Создаем bitmap для конвертации (24bpp RGB как требует CopyToBitmap)
            log_ebus(f"  Создание Bitmap для конвертации...")
            self.bitmap = Bitmap(self._width, self._height, PixelFormat.Format24bppRgb)
            bitmap_size_mb = (self._width * self._height * 3) / 1024 / 1024
            log_ebus(f"  Bitmap создан: {self._width}x{self._height}x3 ({bitmap_size_mb:.2f} МБ)")

            # Применяем пользовательские параметры
            if self._exposure is not None:
                self._set_exposure_internal(self._exposure)
                log_ebus(f"  Экспозиция: {self._exposure} мкс")
            if self._gain is not None:
                self._set_gain_internal(self._gain)
                log_ebus(f"  Усиление: {self._gain} dB")

            # Читаем текущие параметры
            try:
                exp_val = self.device.Parameters.Get("ExposureTime").Value
                log_ebus(f"  Текущая экспозиция: {exp_val} мкс")
            except:
                pass
            try:
                gain_val = self.device.Parameters.Get("Gain").Value
                log_ebus(f"  Текущее усиление: {gain_val}")
            except:
                pass

            # === ШАГ 7: Запуск захвата ===
            log_ebus("[ШАГ 7/7] Запуск захвата изображений...")
            self.device.StreamEnable()
            log_ebus("  StreamEnable: OK")

            acq_start = self.device.Parameters.Get("AcquisitionStart")
            acq_start.Execute()
            log_ebus("  AcquisitionStart: OK")

            self.is_running = True
            log_ebus("=" * 60)
            log_ebus("ЗАХВАТ ЗАПУЩЕН УСПЕШНО")
            log_ebus("=" * 60)

            # === ОСНОВНОЙ ЦИКЛ ЗАХВАТА ===
            numpy_buffer = None
            self._fps_time = time.time()
            self._fps_frame_count = 0
            stats_update_interval = 30  # Обновлять статистику каждые N кадров

            while self.is_running:
                try:
                    # Получаем следующий буфер (таймаут 100мс)
                    ret = self.pipeline.RetrieveNextBuffer(None, UInt32(100))

                    # Обработка возвращаемого значения (может быть tuple или result)
                    if isinstance(ret, tuple):
                        result, buffer = ret
                    else:
                        result = ret
                        buffer = None

                    # Проверка результата
                    if result is None or not result.IsOK:
                        self._error_count += 1
                        if self._error_count <= 3:
                            log_ebus(f"RetrieveNextBuffer: timeout/error (#{self._error_count})", "DEBUG")
                        continue

                    if buffer is None or not buffer.OperationResult.IsOK:
                        if buffer:
                            self.pipeline.ReleaseBuffer(buffer)
                        self._error_count += 1
                        if self._error_count <= 3:
                            log_ebus(f"Buffer OperationResult не OK (#{self._error_count})", "DEBUG")
                        continue

                    image = buffer.Image
                    if image is None:
                        self.pipeline.ReleaseBuffer(buffer)
                        self._error_count += 1
                        if self._error_count <= 3:
                            log_ebus(f"Image is None (#{self._error_count})", "DEBUG")
                        continue

                    try:
                        # Копируем в bitmap (SDK конвертирует Mono8 -> RGB24)
                        image.CopyToBitmap(self.bitmap)

                        # Конвертируем в numpy array
                        frame = bitmap_to_numpy(self.bitmap, numpy_buffer)

                        self._frame_count += 1
                        self._fps_frame_count += 1
                        self._last_frame_time = time.time()

                        # Логирование первых кадров
                        if self._frame_count == 1:
                            log_ebus(f"Первый кадр получен! Shape: {frame.shape}, dtype: {frame.dtype}")
                        elif self._frame_count <= 5:
                            log_ebus(f"Кадр #{self._frame_count} получен")

                        # Расчет FPS
                        current_time = time.time()
                        elapsed = current_time - self._fps_time
                        if elapsed >= 1.0:  # Обновляем FPS каждую секунду
                            self._fps = self._fps_frame_count / elapsed
                            self._fps_time = current_time
                            self._fps_frame_count = 0

                        # Логирование статистики
                        if self._frame_count % 100 == 0:
                            total_time = current_time - self._start_time
                            avg_fps = self._frame_count / total_time if total_time > 0 else 0
                            log_ebus(
                                f"Статистика: кадров={self._frame_count}, "
                                f"FPS={self._fps:.1f}, "
                                f"средний FPS={avg_fps:.1f}, "
                                f"ошибок={self._error_count}"
                            )

                        # Отправляем статистику
                        if self._frame_count % stats_update_interval == 0:
                            stats = {
                                'frame_count': self._frame_count,
                                'fps': self._fps,
                                'error_count': self._error_count,
                                'width': self._width,
                                'height': self._height,
                                'uptime': current_time - self._start_time,
                            }
                            self.stats_updated.emit(stats)

                        # Отправляем кадр
                        self.frame_captured.emit(frame)
                        self._error_count = 0  # Сброс счетчика ошибок при успехе

                    except Exception as e:
                        self._error_count += 1
                        if self._frame_count < 10 or self._error_count <= 5:
                            log_ebus(f"Ошибка обработки кадра: {e}", "ERROR")
                            log_ebus(traceback.format_exc(), "DEBUG")

                    self.pipeline.ReleaseBuffer(buffer)

                except Exception as e:
                    if self.is_running:
                        self._error_count += 1
                        if self._error_count <= 10:
                            log_ebus(f"Ошибка в цикле захвата: {e}", "ERROR")

        except Exception as e:
            error_msg = f"Критическая ошибка eBUS: {str(e)}"
            log_ebus(error_msg, "ERROR")
            log_ebus(traceback.format_exc(), "ERROR")
            self.error.emit(f"{error_msg}\n\nПодробности:\n{traceback.format_exc()}")
        finally:
            # Финальная статистика
            if self._start_time:
                total_time = time.time() - self._start_time
                avg_fps = self._frame_count / total_time if total_time > 0 else 0
                log_ebus("=" * 60)
                log_ebus("ЗАВЕРШЕНИЕ ЗАХВАТА")
                log_ebus(f"  Время работы: {total_time:.1f} сек")
                log_ebus(f"  Всего кадров: {self._frame_count}")
                log_ebus(f"  Средний FPS: {avg_fps:.1f}")
                log_ebus(f"  Всего ошибок: {self._error_count}")
                log_ebus("=" * 60)
            self._cleanup()

    def _cleanup(self):
        """
        Очистка ресурсов камеры

        Порядок очистки:
        1. AcquisitionStop - остановка захвата
        2. StreamDisable - отключение потока
        3. Pipeline.Stop - остановка буферизации
        4. Stream.Close - закрытие потока
        5. Device.Disconnect - отключение от камеры
        6. Bitmap.Dispose - освобождение памяти
        """
        log_ebus("Очистка ресурсов камеры...")

        # 1. Останавливаем захват
        if self.device is not None:
            try:
                log_ebus("  [1/6] AcquisitionStop...")
                self.device.Parameters.Get("AcquisitionStop").Execute()
                log_ebus("  [1/6] AcquisitionStop: OK")
            except Exception as e:
                log_ebus(f"  [1/6] AcquisitionStop: {e}", "DEBUG")

            try:
                log_ebus("  [2/6] StreamDisable...")
                self.device.StreamDisable()
                log_ebus("  [2/6] StreamDisable: OK")
            except Exception as e:
                log_ebus(f"  [2/6] StreamDisable: {e}", "DEBUG")

        # 2. Останавливаем пайплайн
        if self.pipeline is not None:
            try:
                log_ebus("  [3/6] Pipeline.Stop...")
                self.pipeline.Stop()
                log_ebus("  [3/6] Pipeline.Stop: OK")
            except Exception as e:
                log_ebus(f"  [3/6] Pipeline.Stop: {e}", "DEBUG")
            self.pipeline = None

        # 3. Закрываем stream
        if self.stream is not None:
            try:
                log_ebus("  [4/6] Stream.Close...")
                self.stream.Close()
                log_ebus("  [4/6] Stream.Close: OK")
            except Exception as e:
                log_ebus(f"  [4/6] Stream.Close: {e}", "DEBUG")
            self.stream = None

        # 4. Отключаемся от камеры
        if self.device is not None:
            try:
                log_ebus("  [5/6] Device.Disconnect...")
                self.device.Disconnect()
                log_ebus("  [5/6] Device.Disconnect: OK")
            except Exception as e:
                log_ebus(f"  [5/6] Device.Disconnect: {e}", "DEBUG")
            self.device = None

        # 5. Освобождаем bitmap
        if self.bitmap is not None:
            try:
                log_ebus("  [6/6] Bitmap.Dispose...")
                self.bitmap.Dispose()
                log_ebus("  [6/6] Bitmap.Dispose: OK")
            except Exception as e:
                log_ebus(f"  [6/6] Bitmap.Dispose: {e}", "DEBUG")
            self.bitmap = None

        log_ebus("Очистка завершена")

    def stop(self):
        """Остановка захвата"""
        log_ebus("Запрос на остановку захвата...")
        self.is_running = False

    @property
    def fps(self) -> float:
        """Текущий FPS"""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Количество захваченных кадров"""
        return self._frame_count

    @property
    def error_count(self) -> int:
        """Количество ошибок"""
        return self._error_count

    @property
    def resolution(self) -> Tuple[int, int]:
        """Разрешение камеры (width, height)"""
        return (self._width, self._height)

    def get_statistics(self) -> Dict:
        """
        Получение текущей статистики захвата

        Returns:
            Словарь с статистикой: fps, frame_count, error_count, resolution, uptime
        """
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time

        return {
            'fps': self._fps,
            'frame_count': self._frame_count,
            'error_count': self._error_count,
            'width': self._width,
            'height': self._height,
            'uptime': uptime,
            'is_running': self.is_running,
        }

    def _set_exposure_internal(self, value: float):
        """Внутренний метод установки экспозиции"""
        if self.device is not None:
            try:
                exp = self.device.Parameters.Get("ExposureTime")
                exp.Value = value
                log_ebus(f"ExposureTime установлен: {value}")
            except:
                try:
                    exp = self.device.Parameters.Get("ExposureTimeAbs")
                    exp.Value = value
                    log_ebus(f"ExposureTimeAbs установлен: {value}")
                except Exception as e:
                    log_ebus(f"Не удалось установить экспозицию: {e}", "WARN")

    def _set_gain_internal(self, value: float):
        """Внутренний метод установки усиления"""
        if self.device is not None:
            try:
                gain = self.device.Parameters.Get("Gain")
                gain.Value = value
                log_ebus(f"Gain установлен: {value}")
            except:
                try:
                    gain = self.device.Parameters.Get("GainRaw")
                    gain.Value = int(value)
                    log_ebus(f"GainRaw установлен: {int(value)}")
                except Exception as e:
                    log_ebus(f"Не удалось установить gain: {e}", "WARN")

    def set_exposure(self, value: float):
        """
        Установка экспозиции в микросекундах

        Args:
            value: Значение экспозиции в мкс
        """
        self._exposure = value
        if self.is_running:
            self._set_exposure_internal(value)

    def set_gain(self, value: float):
        """
        Установка усиления в dB

        Args:
            value: Значение усиления в dB
        """
        self._gain = value
        if self.is_running:
            self._set_gain_internal(value)


class EbusCameraManager:
    """
    Менеджер для работы с GigE Vision камерами через eBUS SDK

    Предоставляет статические методы для:
    - Проверки доступности SDK
    - Поиска камер в сети
    - Тестирования подключения
    - Захвата одиночного кадра
    """

    @staticmethod
    def is_available() -> bool:
        """
        Проверка доступности eBUS SDK

        Returns:
            True если SDK инициализирован успешно
        """
        return init_ebus_sdk()

    @staticmethod
    def get_available_cameras() -> List[Dict]:
        """
        Получение списка доступных GigE камер

        Выполняет broadcast поиск камер в сети и возвращает
        информацию о каждой найденной камере.

        Returns:
            Список словарей с информацией о камерах:
            - index: Индекс камеры
            - vendor: Производитель
            - model: Модель
            - serial_number: Серийный номер
            - connection_id: ID подключения (используется для Connect)
            - ip_address: IP адрес камеры
            - mac_address: MAC адрес камеры
            - type: Тип камеры ('GigE Vision (eBUS)')
        """
        log_ebus("Поиск GigE Vision камер...")

        if not init_ebus_sdk():
            log_ebus("eBUS SDK не инициализирован", "WARN")
            return []

        cameras = []
        try:
            system = PvSystem()

            search_start = time.time()
            system.Find()
            search_time = time.time() - search_start

            device_count = system.DeviceCount
            log_ebus(f"Поиск завершен за {search_time:.2f} сек, найдено: {device_count}")

            for i in range(device_count):
                try:
                    dev = system.GetDeviceInfo(i)
                    cam_info = {
                        'index': i,
                        'vendor': str(dev.Vendor) if hasattr(dev, 'Vendor') else 'Unknown',
                        'model': str(dev.ModelName),
                        'serial_number': str(dev.SerialNumber) if hasattr(dev, 'SerialNumber') else 'Unknown',
                        'connection_id': str(dev.ConnectionID),
                        'type': 'GigE Vision (eBUS)',
                    }

                    # IP и MAC адреса
                    if hasattr(dev, 'IPAddress'):
                        cam_info['ip_address'] = str(dev.IPAddress)
                    if hasattr(dev, 'MACAddress'):
                        cam_info['mac_address'] = str(dev.MACAddress)

                    log_ebus(f"  [{i}] {cam_info['model']} @ {cam_info.get('ip_address', 'N/A')}")
                    cameras.append(cam_info)
                except Exception as e:
                    log_ebus(f"  [{i}] Ошибка: {e}", "WARN")

        except Exception as e:
            log_ebus(f"Ошибка поиска камер: {e}", "ERROR")

        return cameras

    @staticmethod
    def test_camera(connection_id: str = None) -> bool:
        """
        Тест подключения к камере

        Args:
            connection_id: ID подключения (опционально)

        Returns:
            True если тест успешен
        """
        log_ebus(f"=== EbusCameraManager.test_camera(connection_id={connection_id}) ===")

        if not init_ebus_sdk():
            log_ebus("SDK не инициализирован", "ERROR")
            return False

        device = None
        stream = None
        pipeline = None

        try:
            system = PvSystem()
            log_ebus("Поиск камер...")
            system.Find()

            if system.DeviceCount == 0:
                log_ebus("Камеры не найдены", "ERROR")
                return False

            log_ebus(f"Найдено камер: {system.DeviceCount}")

            # Получаем device info
            if connection_id:
                device_info = None
                for i in range(system.DeviceCount):
                    info = system.GetDeviceInfo(i)
                    if str(info.ConnectionID) == str(connection_id):
                        device_info = info
                        break
                if device_info is None:
                    log_ebus(f"Камера {connection_id} не найдена", "ERROR")
                    return False
            else:
                device_info = system.GetDeviceInfo(0)

            log_ebus(f"Тестирование камеры: {device_info.ModelName} ({device_info.ConnectionID})")

            # Подключаемся
            log_ebus("Подключение...")
            device = PvDevice.CreateAndConnect(device_info.ConnectionID)
            stream = PvStream.CreateAndOpen(device_info.ConnectionID)

            try:
                device.SetStreamDestination(stream.LocalIPAddress, stream.LocalPort)
            except:
                pass

            # Создаем пайплайн
            log_ebus("Создание pipeline...")
            pipeline = PvPipeline(stream)
            pipeline.BufferSize = device.PayloadSize
            pipeline.BufferCount = 4
            pipeline.Start()

            # Запускаем захват
            log_ebus("Запуск захвата...")
            device.StreamEnable()
            device.Parameters.Get("AcquisitionStart").Execute()

            # Пробуем получить кадр
            log_ebus("Ожидание кадра (5 сек)...")
            ret = pipeline.RetrieveNextBuffer(None, UInt32(5000))

            if isinstance(ret, tuple):
                result, buffer = ret
            else:
                result = ret
                buffer = None

            success = result is not None and result.IsOK
            log_ebus(f"Результат: {'УСПЕХ' if success else 'НЕУДАЧА'}")

            if buffer:
                pipeline.ReleaseBuffer(buffer)

            # Останавливаем
            device.Parameters.Get("AcquisitionStop").Execute()
            device.StreamDisable()

            return success

        except Exception as e:
            log_ebus(f"Тест провален: {e}", "ERROR")
            log_ebus(traceback.format_exc(), "DEBUG")
            return False
        finally:
            if pipeline:
                try:
                    pipeline.Stop()
                except:
                    pass
            if stream:
                try:
                    stream.Close()
                except:
                    pass
            if device:
                try:
                    device.Disconnect()
                except:
                    pass

    @staticmethod
    def capture_single_frame(connection_id: str = None) -> Optional[np.ndarray]:
        """
        Захват одного кадра

        Args:
            connection_id: ID подключения (опционально)

        Returns:
            numpy array (BGR) или None при ошибке
        """
        log_ebus(f"=== EbusCameraManager.capture_single_frame(connection_id={connection_id}) ===")

        if not init_ebus_sdk():
            return None

        device = None
        stream = None
        pipeline = None
        bitmap = None

        try:
            system = PvSystem()
            system.Find()

            if system.DeviceCount == 0:
                log_ebus("Камеры не найдены", "ERROR")
                return None

            # Получаем device info
            if connection_id:
                device_info = None
                for i in range(system.DeviceCount):
                    info = system.GetDeviceInfo(i)
                    if str(info.ConnectionID) == str(connection_id):
                        device_info = info
                        break
                if device_info is None:
                    log_ebus(f"Камера {connection_id} не найдена", "ERROR")
                    return None
            else:
                device_info = system.GetDeviceInfo(0)

            log_ebus(f"Захват с камеры: {device_info.ModelName}")

            # Подключаемся
            device = PvDevice.CreateAndConnect(device_info.ConnectionID)
            stream = PvStream.CreateAndOpen(device_info.ConnectionID)

            try:
                device.SetStreamDestination(stream.LocalIPAddress, stream.LocalPort)
            except:
                pass

            # Устанавливаем Mono8
            try:
                pf = device.Parameters.Get("PixelFormat")
                pf.ValueString = "Mono8"
            except:
                pass

            # Создаем пайплайн
            pipeline = PvPipeline(stream)
            pipeline.BufferSize = device.PayloadSize
            pipeline.BufferCount = 4
            pipeline.Start()

            # Получаем размеры
            width = int(device.Parameters.Get("Width").Value)
            height = int(device.Parameters.Get("Height").Value)
            log_ebus(f"Разрешение: {width}x{height}")

            # Создаем bitmap
            bitmap = Bitmap(width, height, PixelFormat.Format24bppRgb)

            # Запускаем захват
            device.StreamEnable()
            device.Parameters.Get("AcquisitionStart").Execute()

            # Получаем кадр
            frame = None
            ret = pipeline.RetrieveNextBuffer(None, UInt32(5000))

            if isinstance(ret, tuple):
                result, buffer = ret
            else:
                result = ret
                buffer = None

            if result is not None and result.IsOK and buffer and buffer.OperationResult.IsOK:
                image = buffer.Image
                if image:
                    image.CopyToBitmap(bitmap)
                    frame = bitmap_to_numpy(bitmap)
                    log_ebus(f"Кадр захвачен: {frame.shape}")
                pipeline.ReleaseBuffer(buffer)
            else:
                log_ebus("Не удалось получить кадр", "ERROR")

            # Останавливаем
            device.Parameters.Get("AcquisitionStop").Execute()
            device.StreamDisable()

            return frame

        except Exception as e:
            log_ebus(f"Ошибка захвата: {e}", "ERROR")
            log_ebus(traceback.format_exc(), "DEBUG")
            return None
        finally:
            if bitmap:
                try:
                    bitmap.Dispose()
                except:
                    pass
            if pipeline:
                try:
                    pipeline.Stop()
                except:
                    pass
            if stream:
                try:
                    stream.Close()
                except:
                    pass
            if device:
                try:
                    device.Disconnect()
                except:
                    pass


# Автоматическая инициализация SDK при импорте модуля
# Это нужно чтобы EBUS_AVAILABLE был установлен правильно
# при импорте: from core.ebus_camera import EBUS_AVAILABLE
try:
    init_ebus_sdk()
except Exception as e:
    log_ebus(f"Ошибка автоинициализации SDK: {e}", "DEBUG")


# Тест модуля
if __name__ == "__main__":
    print("=" * 60)
    print("eBUS Camera Module Test")
    print("=" * 60)

    print(f"\nИнициализация SDK...")
    result = init_ebus_sdk()
    print(f"Результат: {result}")
    print(f"EBUS_AVAILABLE: {EBUS_AVAILABLE}")

    if EBUS_AVAILABLE:
        print("\n" + "=" * 60)
        print("Поиск камер...")
        print("=" * 60)

        cameras = EbusCameraManager.get_available_cameras()
        print(f"\nНайдено камер: {len(cameras)}")

        for cam in cameras:
            print(f"\n  [{cam['index']}] {cam['model']}")
            print(f"      Connection ID: {cam['connection_id']}")
            print(f"      Vendor: {cam['vendor']}")
            print(f"      Serial: {cam['serial_number']}")
            if 'ip_address' in cam:
                print(f"      IP: {cam['ip_address']}")

        if cameras:
            print("\n" + "=" * 60)
            print("Тест подключения к первой камере...")
            print("=" * 60)

            if EbusCameraManager.test_camera():
                print("\n*** ТЕСТ ПРОЙДЕН ***")
            else:
                print("\n*** ТЕСТ ПРОВАЛЕН ***")
    else:
        print("\neBUS SDK недоступен")
