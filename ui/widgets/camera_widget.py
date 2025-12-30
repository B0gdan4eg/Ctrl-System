"""
Виджет камеры для пользовательского интерфейса

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QGroupBox, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap

from core.camera import CameraCapture, CameraManager, CameraCalibration


class CameraWidget(QWidget):
    """Виджет для работы с камерой"""
    
    snapshot_taken = Signal(object)  # Сигнал со снимком
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.current_frame = None
        self.is_preview_active = False
        
        # Настройки
        self.brightness = 1.0
        self.contrast = 1.0
        self.auto_wb = False
        self.enhance = False
        
        self.init_ui()
        self.detect_cameras()
        
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # === Выбор камеры ===
        camera_group = QGroupBox("📷 Камера")
        camera_layout = QHBoxLayout(camera_group)
        
        camera_layout.addWidget(QLabel("Камера:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_layout.addWidget(self.camera_combo)
        
        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.clicked.connect(self.detect_cameras)
        camera_layout.addWidget(self.refresh_btn)
        
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
        
        layout.addWidget(preview_group)
        
        # === Управление ===
        controls_group = QGroupBox("🎛️ Управление")
        controls_layout = QVBoxLayout(controls_group)
        
        # Кнопки
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
        
        controls_layout.addLayout(buttons_layout)
        
        # Настройки изображения
        settings_layout = QVBoxLayout()
        
        # Яркость
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("☀️ Яркость:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(50, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_value = QLabel("1.0")
        brightness_layout.addWidget(self.brightness_value)
        settings_layout.addLayout(brightness_layout)
        
        # Контраст
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("🔆 Контраст:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_value = QLabel("1.0")
        contrast_layout.addWidget(self.contrast_value)
        settings_layout.addLayout(contrast_layout)
        
        # Чекбоксы
        checkboxes_layout = QHBoxLayout()
        
        self.wb_checkbox = QCheckBox("⚖️ Баланс белого")
        self.wb_checkbox.stateChanged.connect(self.on_wb_changed)
        checkboxes_layout.addWidget(self.wb_checkbox)
        
        self.enhance_checkbox = QCheckBox("✨ Улучшение")
        self.enhance_checkbox.stateChanged.connect(self.on_enhance_changed)
        checkboxes_layout.addWidget(self.enhance_checkbox)
        
        settings_layout.addLayout(checkboxes_layout)
        
        controls_layout.addLayout(settings_layout)
        
        layout.addWidget(controls_group)
        
        # === Информация ===
        self.info_label = QLabel("Камеры не найдены")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def detect_cameras(self):
        """Обнаружение доступных камер"""
        cameras = CameraManager.get_available_cameras()
        
        self.camera_combo.clear()
        
        if cameras:
            for cam_id in cameras:
                info = CameraManager.get_camera_info(cam_id)
                label = f"Камера {cam_id} ({info['width']}x{info['height']})"
                self.camera_combo.addItem(label, cam_id)
            
            self.info_label.setText(f"✅ Найдено камер: {len(cameras)}")
        else:
            self.info_label.setText("❌ Камеры не найдены")
    
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
        """Запуск превью"""
        camera_id = self.camera_combo.currentData()
        
        if camera_id is None:
            self.info_label.setText("❌ Камера не выбрана")
            return
        
        self.info_label.setText(f"🔄 Запуск камеры {camera_id}...")
        
        # Создаем поток камеры
        self.camera_thread = CameraCapture(camera_id)
        self.camera_thread.frame_captured.connect(self.on_frame_captured)
        self.camera_thread.error.connect(self.on_camera_error)
        self.camera_thread.start()
        
        self.is_preview_active = True
        self.preview_btn.setText("⏸️ Остановить превью")
        self.snapshot_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        
        # Ждем немного и проверяем
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, self.check_preview_status)
    
    def stop_preview(self):
        """Остановка превью"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread = None
        
        self.is_preview_active = False
        self.preview_btn.setText("▶️ Запустить превью")
        self.snapshot_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.preview_label.setText("Камера не активна")
        self.info_label.setText("⏹️ Камера остановлена")
    
    def on_frame_captured(self, frame):
        """Обработка полученного кадра"""
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        
        # Применяем настройки
        processed = self.apply_settings(frame)
        
        # Конвертируем BGR -> RGB для Qt
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Создаем QImage
        qt_image = QImage(
            rgb_image.data.tobytes(),
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Создаем pixmap и масштабируем
        pixmap = QPixmap.fromImage(qt_image)
        
        # Получаем размер label с учетом соотношения сторон
        label_size = self.preview_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_label.setPixmap(scaled_pixmap)
    
    def apply_settings(self, frame):
        """Применяет текущие настройки к кадру"""
        result = frame.copy()
        
        # Яркость
        if self.brightness != 1.0:
            result = CameraCalibration.adjust_brightness(result, self.brightness)
        
        # Контраст
        if self.contrast != 1.0:
            result = CameraCalibration.adjust_contrast(result, self.contrast)
        
        # Баланс белого
        if self.auto_wb:
            result = CameraCalibration.auto_white_balance(result)
        
        # Улучшение
        if self.enhance:
            result = CameraCalibration.enhance_frame(result)
        
        return result
    
    def on_brightness_changed(self, value):
        """Обработчик изменения яркости"""
        self.brightness = value / 100.0
        self.brightness_value.setText(f"{self.brightness:.1f}")
    
    def on_contrast_changed(self, value):
        """Обработчик изменения контраста"""
        self.contrast = value / 100.0
        self.contrast_value.setText(f"{self.contrast:.1f}")
    
    def on_wb_changed(self, state):
        """Обработчик баланса белого"""
        self.auto_wb = state == Qt.CheckState.Checked.value
    
    def on_enhance_changed(self, state):
        """Обработчик улучшения"""
        self.enhance = state == Qt.CheckState.Checked.value
    
    def take_snapshot(self):
        """Делает снимок"""
        if self.current_frame is not None:
            # Применяем настройки
            snapshot = self.apply_settings(self.current_frame)
            
            # Отправляем сигнал
            self.snapshot_taken.emit(snapshot)
            
            self.info_label.setText(
                f"📸 Снимок сделан ({snapshot.shape[1]}x{snapshot.shape[0]})"
            )
    
    def on_camera_error(self, error_msg):
        """Обработчик ошибок камеры"""
        self.info_label.setText(f"❌ Ошибка: {error_msg}")
        self.stop_preview()
    
    def check_preview_status(self):
        """Проверка статуса превью"""
        if self.is_preview_active:
            if self.current_frame is not None:
                self.info_label.setText(f"✅ Камера активна ({self.current_frame.shape[1]}x{self.current_frame.shape[0]})")
            else:
                self.info_label.setText("⚠️ Камера запущена, но кадры не поступают")
    
    def get_current_frame(self):
        """Возвращает текущий кадр с настройками"""
        if self.current_frame is not None:
            return self.apply_settings(self.current_frame)
        return None
    
    def closeEvent(self, event):
        """Обработчик закрытия виджета"""
        if self.is_preview_active:
            self.stop_preview()
        super().closeEvent(event)


class CameraDialog(QWidget):
    """Диалог камеры с расширенными возможностями"""
    
    image_captured = Signal(object, str)  # Сигнал (изображение, имя файла)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Камера")
        self.init_ui()
        
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # Виджет камеры
        self.camera_widget = CameraWidget()
        self.camera_widget.snapshot_taken.connect(self.on_snapshot_taken)
        layout.addWidget(self.camera_widget)
        
        # Кнопки действий
        actions_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Сохранить снимок")
        self.save_btn.clicked.connect(self.save_snapshot)
        self.save_btn.setEnabled(False)
        actions_layout.addWidget(self.save_btn)
        
        self.use_btn = QPushButton("✅ Использовать для обработки")
        self.use_btn.clicked.connect(self.use_snapshot)
        self.use_btn.setEnabled(False)
        actions_layout.addWidget(self.use_btn)
        
        layout.addLayout(actions_layout)
        
        self.last_snapshot = None
    
    def on_snapshot_taken(self, snapshot):
        """Обработчик снимка"""
        self.last_snapshot = snapshot
        self.save_btn.setEnabled(True)
        self.use_btn.setEnabled(True)
    
    def save_snapshot(self):
        """Сохранение снимка"""
        if self.last_snapshot is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_snapshot_{timestamp}.png"
            
            cv2.imwrite(filename, self.last_snapshot)
            
            self.camera_widget.info_label.setText(f"💾 Сохранено: {filename}")
    
    def use_snapshot(self):
        """Использование снимка для обработки"""
        if self.last_snapshot is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_snapshot_{timestamp}.png"
            
            self.image_captured.emit(self.last_snapshot, filename)
            self.camera_widget.info_label.setText("✅ Снимок передан для обработки")