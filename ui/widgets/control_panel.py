"""
Панель управления обработкой
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QProgressBar
)
from PySide6.QtCore import Signal

from config.settings import (
    DEFAULT_NUM_ZONES,
    DEFAULT_STD_RANGE,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_MODEL_SIZE
)


class ControlPanel(QWidget):
    """Панель управления параметрами обработки"""
    
    load_image_clicked = Signal()
    start_processing_clicked = Signal(dict)  # Отправляет параметры
    save_result_clicked = Signal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # === Загрузка файла ===
        file_group = QGroupBox("📂 Файл")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Загрузить изображение")
        self.load_btn.clicked.connect(self.load_image_clicked.emit)
        self.load_btn.setMinimumHeight(40)
        file_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        layout.addWidget(file_group)
        
        # === Параметры обработки ===
        params_group = QGroupBox("⚙️ Параметры обработки")
        params_layout = QVBoxLayout(params_group)
        
        # Количество зон
        zones_layout = QHBoxLayout()
        zones_layout.addWidget(QLabel("Количество зон:"))
        self.zones_spin = QSpinBox()
        self.zones_spin.setRange(1, 10)
        self.zones_spin.setValue(DEFAULT_NUM_ZONES)
        zones_layout.addWidget(self.zones_spin)
        params_layout.addLayout(zones_layout)
        
        # Std range
        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Std Range (σ):"))
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(1.0, 5.0)
        self.std_spin.setValue(DEFAULT_STD_RANGE)
        self.std_spin.setSingleStep(0.5)
        std_layout.addWidget(self.std_spin)
        params_layout.addLayout(std_layout)
        
        # Порог детекции
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог детекции:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setValue(DEFAULT_DETECTION_THRESHOLD)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(self.threshold_spin)
        params_layout.addLayout(threshold_layout)
        
        # Размер для модели
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Размер для модели:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["256", "512", "1024"])
        self.size_combo.setCurrentText(str(DEFAULT_MODEL_SIZE))
        size_layout.addWidget(self.size_combo)
        params_layout.addLayout(size_layout)
        
        # Применять нормализацию
        self.hist_norm_check = QCheckBox("Применять нормализацию гистограммы")
        self.hist_norm_check.setChecked(True)
        params_layout.addWidget(self.hist_norm_check)
        
        layout.addWidget(params_group)
        
        # === Кнопка запуска ===
        self.process_btn = QPushButton("ЗАПУСТИТЬ ОБРАБОТКУ")
        self.process_btn.clicked.connect(self._on_process_clicked)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 16px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        layout.addWidget(self.process_btn)
        
        # === Прогресс бар ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # === Кнопка сохранения ===
        save_group = QGroupBox("💾 Сохранение")
        save_layout = QVBoxLayout(save_group)
        
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.clicked.connect(self.save_result_clicked.emit)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        layout.addWidget(save_group)
        
        layout.addStretch()
    
    def _on_process_clicked(self):
        """Обработчик нажатия кнопки обработки"""
        params = self.get_parameters()
        self.start_processing_clicked.emit(params)
    
    def get_parameters(self) -> dict:
        """
        Получает текущие параметры обработки
        
        Returns:
            Словарь с параметрами
        """
        return {
            'num_zones': self.zones_spin.value(),
            'min_area_ratio': 0.3,
            'max_area_ratio': 0.95,
            'std_range': self.std_spin.value(),
            'detection_threshold': self.threshold_spin.value(),
            'model_size': int(self.size_combo.currentText()),
            'apply_hist_norm': self.hist_norm_check.isChecked()
        }
    
    def set_file_name(self, filename: str):
        """Устанавливает имя загруженного файла"""
        self.file_label.setText(f"📄 {filename}")
    
    def set_processing_enabled(self, enabled: bool):
        """Включает/выключает возможность обработки"""
        self.process_btn.setEnabled(enabled)
    
    def set_save_enabled(self, enabled: bool):
        """Включает/выключает возможность сохранения"""
        self.save_btn.setEnabled(enabled)
    
    def set_controls_enabled(self, enabled: bool):
        """Включает/выключает все контролы"""
        self.load_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.zones_spin.setEnabled(enabled)
        self.std_spin.setEnabled(enabled)
        self.threshold_spin.setEnabled(enabled)
        self.size_combo.setEnabled(enabled)
        self.hist_norm_check.setEnabled(enabled)
    
    def show_progress(self, show: bool = True, indeterminate: bool = True):
        """
        Показывает/скрывает прогресс бар
        
        Args:
            show: Показать или скрыть
            indeterminate: Неопределенный прогресс (анимация)
        """
        self.progress_bar.setVisible(show)
        if show and indeterminate:
            self.progress_bar.setRange(0, 0)
        elif show:
            self.progress_bar.setRange(0, 100)