#!/usr/bin/env python3
"""
Полный конвейер обработки изображений с детекцией дефектов
Интегрирует: вырезание зон → обработка → детекция → восстановление
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, 
    QProgressBar, QSplitter, QTextEdit, QTabWidget, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================================
# U-NET MODEL (из TRAIN.py)
# ============================================================================

class DoubleConv(nn.Module):
    """Блок: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net архитектура для сегментации"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip_connection = skip_connections[idx//2]
            
            # Обработка несоответствия размеров
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True
                )
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return self.final_conv(x)


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def find_top_zones(img_gray_original, top_n=5, min_area_ratio=0.3, max_area_ratio=0.95):
    """
    Находит топ-N зон на изображении (из cuting.py)
    """
    img_for_search = img_gray_original.copy()
    img_area = img_gray_original.shape[0] * img_gray_original.shape[1]

    # Нормализация
    in_min, in_max = np.percentile(img_for_search, 1), np.percentile(img_for_search, 99)
    img_for_search = (img_for_search - in_min) / (in_max - in_min) * 255
    img_for_search = np.clip(img_for_search, 0, 255).astype(np.uint8)

    clipLimit_values = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    threshold_values = [30, 25, 20, 15, 10]
    
    best_contours = []
    current_target = top_n
    
    while current_target >= 1 and len(best_contours) < current_target:
        for clipLimit in clipLimit_values:
            for threshold_val in threshold_values:
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_for_search)
                
                _, mask = cv2.threshold(img_clahe, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    continue
                
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                good_contours = []
                for cnt in contours[:current_target * 3]:
                    area = cv2.contourArea(cnt)
                    area_ratio_to_image = area / img_area
                    
                    if area_ratio_to_image > max_area_ratio:
                        continue
                    
                    if len(good_contours) > 0:
                        max_good_area = cv2.contourArea(good_contours[0])
                        if area < max_good_area * min_area_ratio:
                            continue
                    
                    good_contours.append(cnt)
                
                if len(good_contours) >= current_target:
                    best_contours = good_contours[:current_target]
                    break
                
                if len(good_contours) > len(best_contours):
                    best_contours = good_contours[:current_target]
            
            if len(best_contours) >= current_target:
                break
        
        if len(best_contours) >= current_target:
            all_good = True
            if len(best_contours) > 0:
                max_area = cv2.contourArea(best_contours[0])
                for cnt in best_contours:
                    area = cv2.contourArea(cnt)
                    if area < max_area * min_area_ratio:
                        all_good = False
                        break
            
            if all_good:
                break
        
        current_target -= 1
    
    if len(best_contours) == 0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_for_search)
        _, mask = cv2.threshold(img_clahe, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered = [c for c in contours if cv2.contourArea(c) / img_area <= max_area_ratio]
        best_contours = sorted(filtered, key=cv2.contourArea, reverse=True)[:top_n]
    
    best_contours = sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])
    return best_contours


def apply_histogram_normalization(image, std_range=2.5):
    """
    Применяет нормализацию Mean ± Std Dev (из hist.py)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Определяем максимальное значение в зависимости от типа данных
    if gray.dtype == np.uint8:
        gray = (gray.astype(np.float32) * (1023.0 / 255.0)).astype(np.uint16)
        max_val = 1023
    elif gray.dtype == np.uint16:
        # Уже в правильном формате, определяем диапазон
        max_val = 1023
    else:
        # На случай других типов
        gray = gray.astype(np.uint16)
        max_val = 1023
    
    # Статистика
    mean = np.mean(gray)
    std = np.std(gray)
    
    input_min = int(max(0, mean - std_range * std))
    input_max = int(min(max_val, mean + std_range * std))
    
    if input_max <= input_min:
        input_max = input_min + 1
    
    # Применение levels adjustment
    gray_float = gray.astype(np.float32)
    normalized = (gray_float - input_min) / (input_max - input_min)
    normalized = np.clip(normalized, 0, 1) * max_val
    result = normalized.astype(np.uint16)
    
    # Если входное было цветное, применяем ко всем каналам
    if len(image.shape) == 3:
        result_color = np.zeros_like(image, dtype=np.uint16)
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            if image.dtype == np.uint8:
                channel = channel * (max_val / 255.0)
            normalized_channel = (channel - input_min) / (input_max - input_min)
            normalized_channel = np.clip(normalized_channel, 0, 1) * max_val
            result_color[:, :, c] = normalized_channel.astype(np.uint16)
        return result_color
    
    return result


def preprocess_for_detection(image, target_size=(512, 512)):
    """
    Предобработка для детекции (из prep.py)
    """
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()
    
    # Конвертация в uint8 если нужно (bilateralFilter требует uint8 или float32)
    if img.dtype == np.uint16:
        img = (img / 1023.0 * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Шумоподавление
    img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Resize
    img_resized = cv2.resize(img_denoised, target_size, interpolation=cv2.INTER_AREA)
    
    return img_resized


def predict_defects(model, image, device, threshold=0.5):
    """
    Предсказание дефектов с использованием U-Net
    """
    model.eval()
    
    # Убедимся что изображение RGB и uint8
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    if image_rgb.dtype == np.uint16:
        image_rgb = (image_rgb / 1023.0 * 255.0).astype(np.uint8)
    
    # Нормализация для модели
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
    
    # Постобработка
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    
    return mask


# ============================================================================
# PROCESSING WORKER (для многопоточности)
# ============================================================================

class ProcessingWorker(QThread):
    """Рабочий поток для обработки изображения"""
    
    progress = Signal(str)  # Сообщения о прогрессе
    zone_processed = Signal(int, object, object)  # (номер зоны, изображение, маска)
    finished = Signal(object, object)  # (результат, статистика)
    error = Signal(str)
    
    def __init__(self, image_path, model, device, params):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.device = device
        self.params = params
        
    def run(self):
        try:
            self.progress.emit("📂 Загрузка изображения...")
            
            # Загрузка
            img = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                self.error.emit(f"Ошибка загрузки: {self.image_path}")
                return
            
            # Конвертация в grayscale
            if img.dtype == np.uint16:
                img = (img / 4).astype(np.uint16)
            
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img.copy()
            
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
                'processing_time': 0
            }
            
            # Обработка каждой зоны
            for idx, cnt in enumerate(contours, 1):
                x, y, w, h = cv2.boundingRect(cnt)
                
                self.progress.emit(f"🔧 Обработка зоны {idx}/{len(contours)} ({w}x{h})...")
                
                # Вырезаем зону
                roi = img_gray[y:y+h, x:x+w].copy()
                
                if roi.size < 100:
                    continue
                
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
                # Конвертируем ROI в RGB для визуализации
                if roi.dtype == np.uint16:
                    roi_display = (roi / 1023.0 * 255.0).astype(np.uint8)
                else:
                    roi_display = roi
                
                roi_rgb = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2RGB) if len(roi_display.shape) == 2 else roi_display
                result_image[y:y+h, x:x+w] = roi_rgb
                
                # Накладываем маску дефектов
                all_defects_mask[y:y+h, x:x+w] = defect_mask_resized
                
                # Статистика
                defect_pixels = np.sum(defect_mask_resized > 0)
                if defect_pixels > 0:
                    stats['zones_with_defects'] += 1
                    stats['total_defect_pixels'] += defect_pixels
                
                # Сигнал о завершении обработки зоны
                self.zone_processed.emit(idx, roi_rgb, defect_mask_resized)
            
            # Визуализация дефектов
            result_with_defects = result_image.copy()
            
            # Красное наложение для дефектов
            mask_colored = np.zeros_like(result_image)
            mask_colored[all_defects_mask > 0] = [0, 0, 255]  # Красный
            
            result_with_defects = cv2.addWeighted(
                result_with_defects, 1.0,
                mask_colored, 0.5,
                0
            )
            
            # Рисуем контуры зон
            for idx, cnt in enumerate(contours, 1):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_with_defects, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    result_with_defects, f"Zone {idx}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            
            self.progress.emit("✅ Обработка завершена!")
            self.finished.emit(result_with_defects, stats)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Ошибка: {str(e)}\n{traceback.format_exc()}")


# ============================================================================
# IMAGE VIEWER
# ============================================================================

class ImageViewer(QLabel):
    """Виджет для отображения изображений"""
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet(
            "border: 2px solid #555; "
            "background-color: #2b2b2b; "
            "color: white; "
            "font-size: 14px;"
        )
        self.setText(f"{title}\n\nNo image")
        
    def display_image(self, image):
        """Отображает OpenCV изображение (BGR формат)"""
        if image is None:
            return
        
        # Конвертация BGR -> RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Нормализация для отображения
        if image.dtype == np.uint16:
            display_image = (rgb_image / 1023.0 * 255.0).astype(np.uint8)
        else:
            display_image = rgb_image
        
        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


# ============================================================================
# MAIN WINDOW
# ============================================================================

class DefectDetectionApp(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.model = None
        self.device = None
        self.result_image = None
        self.stats = None
        
        self.init_ui()
        self.apply_style()
        self.load_model()
        
    def init_ui(self):
        """Инициализация UI"""
        self.setWindowTitle("Система детекции дефектов - Полный конвейер")
        self.setGeometry(100, 50, 1600, 1000)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Заголовок
        # header = QLabel("🔬 СИСТЕМА ДЕТЕКЦИИ ДЕФЕКТОВ")
        # header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # header_font = QFont()
        # header_font.setPointSize(18)
        # header_font.setBold(True)
        # header.setFont(header_font)
        # header.setStyleSheet("color: #4CAF50; padding: 10px;")
        # main_layout.addWidget(header)
        
        # Splitter для разделения панелей
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # === ЛЕВАЯ ПАНЕЛЬ - Управление ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Загрузка изображения
        file_group = QGroupBox("📂 Файл")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Загрузить изображение")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumHeight(40)
        file_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        left_layout.addWidget(file_group)
        
        # Параметры обработки
        params_group = QGroupBox("⚙️ Параметры обработки")
        params_layout = QVBoxLayout(params_group)
        
        # Количество зон
        zones_layout = QHBoxLayout()
        zones_layout.addWidget(QLabel("Количество зон:"))
        self.zones_spin = QSpinBox()
        self.zones_spin.setRange(1, 10)
        self.zones_spin.setValue(5)
        zones_layout.addWidget(self.zones_spin)
        params_layout.addLayout(zones_layout)
        
        # Std range для гистограммы
        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Std Range (σ):"))
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(1.0, 5.0)
        self.std_spin.setValue(2.5)
        self.std_spin.setSingleStep(0.5)
        std_layout.addWidget(self.std_spin)
        params_layout.addLayout(std_layout)
        
        # Порог детекции
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог детекции:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(self.threshold_spin)
        params_layout.addLayout(threshold_layout)
        
        # Размер для модели
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер для модели:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["256", "512", "1024"])
        self.size_combo.setCurrentText("1024")
        size_layout.addWidget(self.size_combo)
        params_layout.addLayout(size_layout)
        
        # Применять гистограммную нормализацию
        self.hist_norm_check = QCheckBox("Применять нормализацию гистограммы")
        self.hist_norm_check.setChecked(True)
        params_layout.addWidget(self.hist_norm_check)
        
        left_layout.addWidget(params_group)
        
        # Кнопка запуска
        self.process_btn = QPushButton("ЗАПУСТИТЬ ОБРАБОТКУ")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 16px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        left_layout.addWidget(self.process_btn)
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Лог обработки
        log_group = QGroupBox("📋 Лог обработки")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        left_layout.addWidget(log_group)
        
        # Статистика
        stats_group = QGroupBox("📊 Статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Нет данных")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        left_layout.addWidget(stats_group)
        
        # Сохранение результата
        save_group = QGroupBox("💾 Сохранение")
        save_layout = QVBoxLayout(save_group)
        
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        left_layout.addWidget(save_group)
        
        left_layout.addStretch()
        
        # === ПРАВАЯ ПАНЕЛЬ - Визуализация ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tabs для разных видов
        self.tabs = QTabWidget()
        
        # Tab 1: Результат
        self.result_viewer = ImageViewer("Результат с дефектами")
        self.tabs.addTab(self.result_viewer, "📸 Результат")
        
        # Tab 2: Оригинал
        self.original_viewer = ImageViewer("Исходное изображение")
        self.tabs.addTab(self.original_viewer, "🖼️ Оригинал")
        
        right_layout.addWidget(self.tabs)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        # Статус бар
        self.status_label = QLabel("Готов к работе")
        self.statusBar().addWidget(self.status_label)
        
    def apply_style(self):
        """Применение темной темы"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
    
    def load_model(self):
        """Загрузка модели U-Net"""
        self.log("🔧 Инициализация модели U-Net...")
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = UNet(in_channels=3, out_channels=1)
            self.model = self.model.to(self.device)
            
            # Попытка загрузить веса если есть
            model_path = Path('best_model.pth')
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.log(f"✅ Модель загружена из {model_path}")
            else:
                self.log("⚠️ Веса модели не найдены, используется неинициализированная модель")
                self.log("   Поместите файл 'best_model.pth' в текущую директорию")
            
            self.model.eval()
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            self.log(f"✅ Модель готова (устройство: {device_name})")
            
        except Exception as e:
            self.log(f"❌ Ошибка загрузки модели: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{str(e)}")
    
    def log(self, message):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def load_image(self):
        """Загрузка изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.file_label.setText(f"📄 {self.current_image_path.name}")
            self.log(f"📂 Загружен файл: {self.current_image_path.name}")
            
            # Отображаем оригинал
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.original_viewer.display_image(img)
            
            self.process_btn.setEnabled(True)
    
    def start_processing(self):
        """Запуск обработки"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
        
        if self.model is None:
            QMessageBox.warning(self, "Ошибка", "Модель не загружена")
            return
        
        # Собираем параметры
        params = {
            'num_zones': self.zones_spin.value(),
            'min_area_ratio': 0.3,
            'max_area_ratio': 0.95,
            'std_range': self.std_spin.value(),
            'detection_threshold': self.threshold_spin.value(),
            'model_size': int(self.size_combo.currentText()),
            'apply_hist_norm': self.hist_norm_check.isChecked()
        }
        
        self.log("="*50)
        self.log("ЗАПУСК ОБРАБОТКИ")
        self.log(f"   Зон для поиска: {params['num_zones']}")
        self.log(f"   Std Range: ±{params['std_range']}σ")
        self.log(f"   Порог детекции: {params['detection_threshold']}")
        self.log(f"   Размер модели: {params['model_size']}x{params['model_size']}")
        self.log("="*50)
        
        # Блокируем кнопки
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Создаем и запускаем worker
        self.worker = ProcessingWorker(
            self.current_image_path,
            self.model,
            self.device,
            params
        )
        
        self.worker.progress.connect(self.log)
        self.worker.zone_processed.connect(self.on_zone_processed)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        
        self.worker.start()
    
    def on_zone_processed(self, zone_num, image, mask):
        """Обработка завершена для одной зоны"""
        defect_pixels = np.sum(mask > 0)
        if defect_pixels > 0:
            self.log(f"   Zone {zone_num}: найдено {defect_pixels} пикселей дефектов")
    
    def on_processing_finished(self, result_image, stats):
        """Обработка завершена"""
        self.result_image = result_image
        self.stats = stats
        
        # Отображаем результат
        self.result_viewer.display_image(result_image)
        
        # Обновляем статистику
        stats_text = f"""
        ✅ Обработка завершена!
        
        📊 Статистика:
        • Найдено зон: {stats['zones_found']}
        • Зон с дефектами: {stats['zones_with_defects']}
        • Пикселей дефектов: {stats['total_defect_pixels']}
        """
        
        self.stats_label.setText(stats_text)
        self.log("="*50)
        self.log("✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        self.log(f"   Найдено зон: {stats['zones_found']}")
        self.log(f"   Зон с дефектами: {stats['zones_with_defects']}")
        self.log("="*50)
        
        # Разблокируем кнопки
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText(
            f"✅ Готово | Зон: {stats['zones_found']} | С дефектами: {stats['zones_with_defects']}"
        )
    
    def on_processing_error(self, error_msg):
        """Ошибка при обработке"""
        self.log(f"❌ ОШИБКА: {error_msg}")
        
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        # Разблокируем кнопки
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText("❌ Ошибка обработки")
    
    def save_result(self):
        """Сохранение результата"""
        if self.result_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff);;All Files (*)"
        )
        
        if file_path:
            success = cv2.imwrite(file_path, self.result_image)
            if success:
                self.log(f"💾 Результат сохранен: {file_path}")
                QMessageBox.information(self, "Успех", f"Результат сохранен:\n{file_path}")
            else:
                self.log(f"❌ Ошибка сохранения: {file_path}")
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DefectDetectionApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()