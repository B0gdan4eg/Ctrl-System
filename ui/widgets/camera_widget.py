"""
Виджет камеры для пользовательского интерфейса

Поддерживает камеры:
- USB камеры через OpenCV
- GigE Vision камеры через Pleora eBUS SDK (JAI)
- GigE Vision камеры через Basler Pylon

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QGroupBox, QMainWindow
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QKeySequence, QShortcut

from core.cameras.camera import CameraCapture, CameraManager

# Импорт логгера
try:
    from utils import app_logger
except ImportError:
    import logging
    app_logger = logging.getLogger("camera_widget")

# Импорт GigE камер (опционально)
try:
    from core.cameras.pylon_camera import PylonCameraCapture, PylonCameraManager, PYLON_AVAILABLE
except ImportError:
    PYLON_AVAILABLE = False
    PylonCameraCapture = None
    PylonCameraManager = None

try:
    from core.cameras.ebus_camera import EbusCameraCapture, EbusCameraManager, EBUS_AVAILABLE
except ImportError:
    EBUS_AVAILABLE = False
    EbusCameraCapture = None
    EbusCameraManager = None

app_logger.info(f"CameraWidget: EBUS_AVAILABLE={EBUS_AVAILABLE}, PYLON_AVAILABLE={PYLON_AVAILABLE}")


class CameraFullscreenWindow(QMainWindow):
    """
    Полноэкранное окно стриминга камеры.
    Пробел — сделать снимок, Escape — закрыть.
    """

    snapshot_requested = Signal()
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Камера — стриминг")
        self.setStyleSheet("background-color: black;")

        # Центральный виджет с видео
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, stretch=1)

        # Подсказка снизу
        hint = QLabel("Пробел — снимок  |  Esc — закрыть")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(11)
        hint.setFont(font)
        hint.setStyleSheet("color: rgba(255,255,255,160); background: transparent; padding: 6px;")
        hint.setFixedHeight(32)
        layout.addWidget(hint)

        # Горячие клавиши
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.snapshot_requested.emit)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close)

    def update_frame(self, qt_pixmap: QPixmap):
        """Обновляет кадр в окне (вызывается из CameraWidget)."""
        if qt_pixmap.isNull():
            return
        scaled = qt_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class CameraWidget(QWidget):
    """
    Виджет для работы с камерой

    Поддерживает:
    - USB камеры через OpenCV
    - GigE камеры через eBUS SDK (JAI RM-4200GE и др.)
    - GigE камеры через Pylon SDK (Basler)

    Signals:
        snapshot_taken: Испускается при снимке (numpy array BGR)
    """

    snapshot_taken = Signal(object)  # Сигнал со снимком

    def __init__(self):
        super().__init__()
        app_logger.info("Инициализация CameraWidget")

        self.camera_thread = None
        self.current_frame = None
        self.is_preview_active = False
        self.fullscreen_window = None
        self._frame_buffer = None

        # Статистика GigE камеры
        self._camera_fps = 0.0
        self._camera_frame_count = 0
        self._camera_info = {}

        self.init_ui()
        app_logger.info("CameraWidget инициализирован")
        
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)

        # === Выбор камеры (одна строка) ===
        camera_group = QGroupBox("📷 Камера")
        camera_layout = QHBoxLayout(camera_group)

        camera_layout.addWidget(QLabel("Тип:"))
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.setFixedWidth(160)
        if EBUS_AVAILABLE:
            self.camera_type_combo.addItem("GigE (eBUS/JAI)", "ebus")
        if PYLON_AVAILABLE:
            self.camera_type_combo.addItem("GigE (Pylon/Basler)", "pylon")
        self.camera_type_combo.addItem("USB (OpenCV)", "usb")
        self.camera_type_combo.currentIndexChanged.connect(self.on_camera_type_changed)
        camera_layout.addWidget(self.camera_type_combo)

        camera_layout.addWidget(QLabel("Камера:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedWidth(280)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_layout.addWidget(self.camera_combo)

        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.setFixedWidth(110)
        self.refresh_btn.clicked.connect(self.detect_cameras)
        camera_layout.addWidget(self.refresh_btn)

        camera_layout.addStretch()
        layout.addWidget(camera_group)
        
        # === Превью ===
        preview_group = QGroupBox("🎥 Превью")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet(
            "border: 2px solid #555; "
            "background-color: #2b2b2b; "
            "color: white;"
        )
        self.preview_label.setText("Камера не активна")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group, stretch=1)

        # === Кнопки управления ===
        buttons_layout = QHBoxLayout()

        self.preview_btn = QPushButton("▶️ Запустить превью")
        self.preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn.setMinimumHeight(40)
        buttons_layout.addWidget(self.preview_btn)

        self.snapshot_btn = QPushButton("📸 Сделать снимок")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.snapshot_btn.setMinimumHeight(40)
        self.snapshot_btn.setEnabled(False)
        buttons_layout.addWidget(self.snapshot_btn)

        layout.addLayout(buttons_layout)

        # === Информация ===
        self.info_label = QLabel("📡 Нажмите «Обновить» для поиска камер")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def on_camera_type_changed(self, index):
        """Обработчик смены типа камеры"""
        camera_type = self.camera_type_combo.currentData()
        app_logger.info(f"Смена типа камеры: {camera_type}")

        if self.is_preview_active:
            self.stop_preview()
        self.camera_combo.clear()
        self.info_label.setText("📡 Нажмите «Обновить» для поиска камер")

    def detect_cameras(self):
        """Обнаружение доступных камер выбранного типа"""
        camera_type = self.camera_type_combo.currentData()
        self.camera_combo.clear()

        app_logger.info(f"Поиск камер типа: {camera_type}")

        if camera_type == "usb":
            # USB камеры через OpenCV
            cameras = CameraManager.get_available_cameras()
            app_logger.info(f"Найдено USB камер: {len(cameras) if cameras else 0}")

            if cameras:
                for cam_id in cameras:
                    info = CameraManager.get_camera_info(cam_id)
                    label = f"USB #{cam_id} ({info['width']}x{info['height']})"
                    self.camera_combo.addItem(label, {'type': 'usb', 'id': cam_id})
                self.info_label.setText(f"✅ Найдено USB камер: {len(cameras)}")
            else:
                self.info_label.setText("❌ USB камеры не найдены")

        elif camera_type == "ebus" and EBUS_AVAILABLE:
            # GigE камеры через eBUS SDK (JAI и др.)
            app_logger.info("Поиск GigE камер через eBUS SDK...")
            cameras = EbusCameraManager.get_available_cameras()
            app_logger.info(f"Найдено eBUS камер: {len(cameras)}")

            if cameras:
                for cam in cameras:
                    ip_addr = cam.get('ip_address', '')
                    if ip_addr:
                        label = f"{cam['model']} @ {ip_addr}"
                    else:
                        label = f"{cam['model']} ({cam['connection_id']})"
                    self.camera_combo.addItem(label, {
                        'type': 'ebus',
                        'connection_id': cam['connection_id'],
                        'model': cam['model'],
                        'ip_address': ip_addr,
                    })
                    app_logger.info(f"  Добавлена камера: {label}")
                self.info_label.setText(f"✅ Найдено GigE камер (eBUS): {len(cameras)}")
            else:
                self.info_label.setText("❌ GigE камеры (eBUS) не найдены")
                app_logger.warning("GigE камеры не найдены через eBUS SDK")

        elif camera_type == "pylon" and PYLON_AVAILABLE:
            # GigE камеры через Pylon
            app_logger.info("Поиск GigE камер через Pylon SDK...")
            cameras = PylonCameraManager.get_available_cameras()
            app_logger.info(f"Найдено Pylon камер: {len(cameras)}")

            if cameras:
                for cam in cameras:
                    label = f"{cam['vendor']} {cam['model']} (S/N: {cam['serial_number']})"
                    self.camera_combo.addItem(label, {'type': 'pylon', 'index': cam['index']})
                self.info_label.setText(f"✅ Найдено Pylon камер: {len(cameras)}")
            else:
                self.info_label.setText("❌ Pylon камеры не найдены")
        else:
            self.info_label.setText("❌ Выбранный тип камеры недоступен")
            app_logger.warning(f"Тип камеры недоступен: {camera_type}")
    
    def on_camera_changed(self, index):
        """Обработчик смены камеры"""
        if self.is_preview_active:
            self.stop_preview()
    
    def toggle_preview(self):
        """Переключение превью"""
        if self.is_preview_active:
            self.stop_preview()
        else:
            self.start_preview()
    
    def start_preview(self):
        """Запуск превью камеры"""
        camera_data = self.camera_combo.currentData()

        if camera_data is None:
            self.info_label.setText("❌ Камера не выбрана")
            app_logger.warning("Попытка запуска превью без выбранной камеры")
            return

        camera_type = camera_data.get('type')
        self.info_label.setText(f"🔄 Запуск камеры ({camera_type})...")
        app_logger.info(f"Запуск превью: тип={camera_type}, данные={camera_data}")

        try:
            # Создаем поток камеры в зависимости от типа
            if camera_type == 'usb':
                camera_id = camera_data.get('id')
                app_logger.info(f"Создание USB камеры: id={camera_id}")
                self.camera_thread = CameraCapture(camera_id)

            elif camera_type == 'ebus' and EBUS_AVAILABLE:
                connection_id = camera_data.get('connection_id')
                app_logger.info(f"Создание eBUS камеры: connection_id={connection_id}")
                self.camera_thread = EbusCameraCapture(connection_id=connection_id)

                # Подключаем дополнительные сигналы для eBUS камеры
                self.camera_thread.camera_info.connect(self.on_camera_info)
                self.camera_thread.stats_updated.connect(self.on_stats_updated)

            elif camera_type == 'pylon' and PYLON_AVAILABLE:
                device_index = camera_data.get('index')
                app_logger.info(f"Создание Pylon камеры: index={device_index}")
                self.camera_thread = PylonCameraCapture(device_index=device_index)
            else:
                error_msg = f"Неизвестный тип камеры: {camera_type}"
                self.info_label.setText(f"❌ {error_msg}")
                app_logger.error(error_msg)
                return

            # Подключаем общие сигналы
            self.camera_thread.frame_captured.connect(self.on_frame_captured)
            self.camera_thread.error.connect(self.on_camera_error)

            # Запускаем поток
            app_logger.info("Запуск потока камеры...")
            self.camera_thread.start()

            self.is_preview_active = True
            self.preview_btn.setText("⏸️ Остановить превью")
            self.snapshot_btn.setEnabled(True)
            self.camera_combo.setEnabled(False)
            self.camera_type_combo.setEnabled(False)

            # Открываем полноэкранное окно стриминга
            self._open_fullscreen()

            # Проверяем статус через секунду
            QTimer.singleShot(1000, self.check_preview_status)

        except Exception as e:
            error_msg = f"Ошибка запуска камеры: {e}"
            self.info_label.setText(f"❌ {error_msg}")
            app_logger.error(error_msg, exc_info=True)
    
    def stop_preview(self):
        """Остановка превью камеры"""
        app_logger.info("Остановка превью камеры...")

        if self.camera_thread:
            app_logger.info("Остановка потока камеры...")
            self.camera_thread.stop()
            self.camera_thread.wait(5000)  # Ждем до 5 секунд

            if self.camera_thread.isRunning():
                app_logger.warning("Поток камеры не остановился за 5 сек, принудительное завершение")
                self.camera_thread.terminate()

            self.camera_thread = None
            app_logger.info("Поток камеры остановлен")

        self.is_preview_active = False
        self.preview_btn.setText("▶️ Запустить превью")
        self.snapshot_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.camera_type_combo.setEnabled(True)
        self.preview_label.setText("Камера не активна")

        # Показываем финальную статистику для GigE камер
        if self._camera_frame_count > 0:
            self.info_label.setText(
                f"⏹️ Камера остановлена | Всего кадров: {self._camera_frame_count}"
            )
        else:
            self.info_label.setText("⏹️ Камера остановлена")

        # Сбрасываем статистику
        self._camera_fps = 0.0
        self._camera_frame_count = 0
        self._camera_info = {}

        # Закрываем полноэкранное окно если открыто
        if self.fullscreen_window is not None:
            try:
                self.fullscreen_window.closed.disconnect(self._on_fullscreen_closed)
            except (RuntimeError, TypeError):
                pass
            self.fullscreen_window.close()
            self.fullscreen_window = None

    def _open_fullscreen(self):
        """Открывает полноэкранное окно стриминга."""
        if self.fullscreen_window is not None:
            self.fullscreen_window.show()
            return

        self.fullscreen_window = CameraFullscreenWindow()
        self.fullscreen_window.snapshot_requested.connect(self.take_snapshot)
        self.fullscreen_window.closed.connect(self._on_fullscreen_closed)
        self.fullscreen_window.showFullScreen()

    def _on_fullscreen_closed(self):
        """Пользователь закрыл полноэкранное окно — останавливаем превью."""
        self.fullscreen_window = None
        if self.is_preview_active:
            self.stop_preview()

    def on_frame_captured(self, frame):
        """Обработка полученного кадра"""
        if frame is None:
            return
            
        self.current_frame = frame.copy()

        # Конвертируем BGR -> RGB для Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Создаем QImage (буфер хранится как атрибут чтобы избежать use-after-free)
        self._frame_buffer = np.ascontiguousarray(rgb_image)
        qt_image = QImage(
            self._frame_buffer.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Создаем pixmap
        pixmap = QPixmap.fromImage(qt_image)

        # Обновляем полноэкранное окно (масштабирование внутри него)
        if self.fullscreen_window is not None:
            self.fullscreen_window.update_frame(pixmap)

        # Масштабируем для превью в виджете
        label_size = self.preview_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
    
    def take_snapshot(self):
        """Делает снимок"""
        if self.current_frame is not None:
            self.snapshot_taken.emit(self.current_frame.copy())
            self.info_label.setText(
                f"📸 Снимок сделан ({self.current_frame.shape[1]}x{self.current_frame.shape[0]})"
            )
    
    def on_camera_error(self, error_msg):
        """Обработчик ошибок камеры"""
        app_logger.error(f"Ошибка камеры: {error_msg}")
        self.info_label.setText(f"❌ Ошибка: {error_msg[:100]}")
        self.stop_preview()

    def on_camera_info(self, info: dict):
        """
        Обработчик информации о подключенной GigE камере

        Args:
            info: Словарь с информацией о камере
        """
        self._camera_info = info
        app_logger.info(f"Информация о камере: {info}")

        # Формируем строку для отображения
        model = info.get('model', 'Unknown')
        vendor = info.get('vendor', '')
        ip = info.get('ip_address', '')

        status_text = f"📷 {vendor} {model}"
        if ip:
            status_text += f" @ {ip}"

        self.info_label.setText(status_text)

    def on_stats_updated(self, stats: dict):
        """
        Обработчик обновления статистики GigE камеры

        Args:
            stats: Словарь со статистикой: fps, frame_count, error_count, etc.
        """
        self._camera_fps = stats.get('fps', 0.0)
        self._camera_frame_count = stats.get('frame_count', 0)

        # Обновляем информацию
        fps = stats.get('fps', 0.0)
        frames = stats.get('frame_count', 0)
        width = stats.get('width', 0)
        height = stats.get('height', 0)
        errors = stats.get('error_count', 0)

        model = self._camera_info.get('model', 'GigE камера')

        status_parts = [f"📷 {model}"]
        status_parts.append(f"{width}x{height}")
        status_parts.append(f"FPS: {fps:.1f}")
        status_parts.append(f"Кадров: {frames}")
        if errors > 0:
            status_parts.append(f"⚠️ Ошибок: {errors}")

        self.info_label.setText(" | ".join(status_parts))
    
    def check_preview_status(self):
        """Проверка статуса превью"""
        if self.is_preview_active:
            if self.current_frame is not None:
                self.info_label.setText(f"✅ Камера активна ({self.current_frame.shape[1]}x{self.current_frame.shape[0]})")
            else:
                self.info_label.setText("⚠️ Камера запущена, но кадры не поступают")
    
    # Moved to archive/dead_code_ui.py

    def closeEvent(self, event):
        """Обработчик закрытия виджета"""
        if self.is_preview_active:
            self.stop_preview()
        super().closeEvent(event)


