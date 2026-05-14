"""
Главное окно приложения
"""

import cv2
import tempfile
import torch
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QSplitter, QTabWidget
)
from PySide6.QtCore import Qt

from config.settings import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    MODEL_PATH, MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS, MODEL_FEATURES
)
from models import UNet
from workers import ProcessingWorker
from core import apply_defects_overlay
from .widgets import ImageViewer, ControlPanel, StatsPanel
from .widgets.camera_widget import CameraWidget
from .styles.dark_theme import get_dark_theme


class DefectDetectionApp(QMainWindow):
    """Главное окно приложения детекции дефектов"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.model = None
        self.device = None
        self.model_loaded = False
        self.result_image = None
        self.clean_image = None
        self.defects_mask = None
        self.contours = None
        self.stats = None
        self.worker = None

        self.init_ui()
        self.apply_style()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Система детекции дефектов - Полный конвейер")
        self.setGeometry(100, 50, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Splitter для разделения панелей
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # === ЛЕВАЯ ПАНЕЛЬ - Управление ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Панель управления
        self.control_panel = ControlPanel()
        self.control_panel.load_image_clicked.connect(self.load_image)
        self.control_panel.start_processing_clicked.connect(self.start_processing)
        self.control_panel.save_result_clicked.connect(self.save_result)
        self.control_panel.show_defects_changed.connect(self._on_show_defects_toggled)
        left_layout.addWidget(self.control_panel)
        
        # Панель статистики и логов
        self.stats_panel = StatsPanel()
        left_layout.addWidget(self.stats_panel)
        
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
        
        # Tab 3: Камера
        self.camera_widget = CameraWidget()
        self.camera_widget.snapshot_taken.connect(self.on_camera_snapshot)
        self.tabs.addTab(self.camera_widget, "📷 Камера")
        
        right_layout.addWidget(self.tabs)
        
        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
        
    def apply_style(self):
        """Применение темной темы"""
        self.setStyleSheet(get_dark_theme())
    
    def load_model(self):
        """Загрузка модели U-Net"""
        self.stats_panel.log("🔧 Инициализация модели U-Net...")
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = UNet(
                in_channels=MODEL_IN_CHANNELS,
                out_channels=MODEL_OUT_CHANNELS,
                features=MODEL_FEATURES
            )
            self.model = self.model.to(self.device)

            # Попытка загрузить веса (resource_path = поддержка PyInstaller _MEIPASS)
            from utils import resource_path
            model_path = resource_path(MODEL_PATH)
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.stats_panel.log(f"✅ Модель загружена из {model_path}")
            else:
                self.stats_panel.log("⚠️ Веса модели не найдены, используется неинициализированная модель")
                self.stats_panel.log(f"   Поместите файл '{MODEL_PATH}' в текущую директорию")

            self.model.eval()
            self._apply_device_optimizations()
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            self.stats_panel.log(f"✅ Модель готова (устройство: {device_name})")

            # Модель успешно загружена
            self.model_loaded = True

        except Exception as e:
            self.stats_panel.log(f"❌ Ошибка загрузки модели: {str(e)}")
            self.model = None
            self.model_loaded = False
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось загрузить модель:\n{str(e)}"
            )

    def _apply_device_optimizations(self):
        """Подбирает оптимизации под текущее устройство (CUDA / CPU стенд)."""
        if self.device.type == 'cuda':
            # channels_last: bit-identical, ускоряет conv на NHWC-friendly GPU
            self.model = self.model.to(memory_format=torch.channels_last)
            return

        # CPU стенд — выжимаем максимум на чистом torch (без torch.compile —
        # на Windows он требует MSVC build tools).
        import os
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(max(2, n_threads // 2))
        except RuntimeError:
            # set_num_interop_threads нельзя вызывать после первого parallel-вызова
            pass
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        # channels_last на CPU тоже ускоряет conv'ы с oneDNN/MKL-DNN
        try:
            self.model = self.model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        self.stats_panel.log(
            f"⚙️ CPU: {n_threads} threads, MKL-DNN + channels_last"
        )

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
            self.control_panel.set_file_name(self.current_image_path.name)
            self.stats_panel.log(f"📂 Загружен файл: {self.current_image_path.name}")
            
            # Отображаем оригинал
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.original_viewer.display_image(img)
                self.control_panel.set_processing_enabled(True)
            else:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    "Не удалось загрузить изображение"
                )
    
    def start_processing(self, params: dict):
        """
        Запуск обработки изображения

        Args:
            params: Параметры обработки
        """
        if self.current_image_path is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return

        # Lazy loading модели - загружаем только при первой обработке
        if not self.model_loaded:
            self.stats_panel.log("⏳ Первый запуск - загрузка модели...")
            self.load_model()
            if self.model is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить модель")
                return
        
        self.stats_panel.log("="*50)
        self.stats_panel.log("ЗАПУСК ОБРАБОТКИ")
        self.stats_panel.log(f"   Зон для поиска: {params['num_zones']}")
        self.stats_panel.log(f"   Std Range: ±{params['std_range']}σ")
        self.stats_panel.log(f"   Порог детекции: {params['detection_threshold']}")
        self.stats_panel.log(f"   Размер модели: {params['model_size']}x{params['model_size']}")
        self.stats_panel.log("="*50)
        
        # Блокируем интерфейс
        self.control_panel.set_controls_enabled(False)
        self.control_panel.show_progress(True)
        
        # Остановить старый воркер если запущен
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(3000)
            self.worker.deleteLater()
            self.worker = None

        # Создаем и запускаем worker
        self.worker = ProcessingWorker(
            self.current_image_path,
            self.model,
            self.device,
            params
        )
        
        self.worker.progress.connect(self.stats_panel.log)
        self.worker.zone_processed.connect(self.on_zone_processed)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        
        self.worker.start()
    
    def on_zone_processed(self, zone_num: int, image, mask):
        """
        Обработчик завершения обработки зоны
        
        Args:
            zone_num: Номер зоны
            image: Изображение зоны
            mask: Маска дефектов
        """
        defect_pixels = mask.sum() // 255
        if defect_pixels > 0:
            self.stats_panel.log(
                f"   Zone {zone_num}: найдено {defect_pixels} пикселей дефектов"
            )
    
    def on_processing_finished(self, clean_image, defects_mask, stats: dict, contours):
        """
        Обработчик завершения обработки. Хранит «чистое» изображение и маску
        отдельно, чтобы рисовать overlay по чекбоксу без повторного прогона модели.
        """
        self.clean_image = clean_image
        self.defects_mask = defects_mask
        self.stats = stats
        self.contours = contours

        self._redraw_result()

        # Обновляем статистику
        self.stats_panel.update_statistics(stats)
        
        self.stats_panel.log("="*50)
        self.stats_panel.log("✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        self.stats_panel.log(f"   Найдено зон: {stats['zones_found']}")
        self.stats_panel.log(f"   Зон с дефектами: {stats['zones_with_defects']}")
        self.stats_panel.log("="*50)
        
        # Разблокируем интерфейс
        self.control_panel.set_controls_enabled(True)
        self.control_panel.set_processing_enabled(True)
        self.control_panel.set_save_enabled(True)
        self.control_panel.show_progress(False)
        
        self.statusBar().showMessage(
            f"✅ Готово | Зон: {stats['zones_found']} | С дефектами: {stats['zones_with_defects']}"
        )
    
    def _redraw_result(self):
        """Перерисовывает result_viewer: clean + опциональный defects overlay."""
        if self.clean_image is None:
            return
        if self.control_panel.is_show_defects() and self.defects_mask is not None:
            self.result_image = apply_defects_overlay(self.clean_image, self.defects_mask)
        else:
            self.result_image = self.clean_image
        self.result_viewer.display_image(self.result_image)

    def _on_show_defects_toggled(self, _checked: bool):
        self._redraw_result()

    def on_processing_error(self, error_msg: str):
        """
        Обработчик ошибки при обработке
        
        Args:
            error_msg: Сообщение об ошибке
        """
        self.stats_panel.log(f"❌ ОШИБКА: {error_msg}")
        
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        # Разблокируем интерфейс
        self.control_panel.set_controls_enabled(True)
        self.control_panel.set_processing_enabled(True)
        self.control_panel.show_progress(False)
        
        self.statusBar().showMessage("❌ Ошибка обработки")
    
    def save_result(self):
        """Сохранение результата"""
        if self.result_image is None:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Нет результата для сохранения"
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff);;All Files (*)"
        )
        
        if file_path:
            from pathlib import Path as _Path
            _ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
            _save_path = _Path(file_path)
            if _save_path.suffix.lower() not in _ALLOWED_EXT:
                _save_path = _save_path.with_suffix('.png')
                file_path = str(_save_path)
            success = cv2.imwrite(file_path, self.result_image)
            if success:
                self.stats_panel.log(f"💾 Результат сохранен: {file_path}")
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Результат сохранен:\n{file_path}"
                )
            else:
                self.stats_panel.log(f"❌ Ошибка сохранения: {file_path}")
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    "Не удалось сохранить файл"
                )
    
    def on_camera_snapshot(self, snapshot):
        """
        Обработчик снимка с камеры

        Args:
            snapshot: Изображение с камеры
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import os as _os
        _app_temp_dir = Path(_os.path.dirname(_os.path.abspath(__file__))).parent / "temp"
        _app_temp_dir.mkdir(exist_ok=True)
        temp_path = _app_temp_dir / f"camera_snapshot_{timestamp}.png"

        cv2.imwrite(str(temp_path), snapshot)
        
        # Устанавливаем как текущее изображение
        self.current_image_path = temp_path
        self.control_panel.set_file_name(f"📷 {temp_path.name}")
        self.stats_panel.log(f"📷 Получен снимок с камеры: {temp_path.name}")
        
        # Отображаем в оригинале
        self.original_viewer.display_image(snapshot)
        
        # Включаем обработку
        self.control_panel.set_processing_enabled(True)
        
        # Переключаемся на вкладку "Оригинал"
        self.tabs.setCurrentIndex(1)