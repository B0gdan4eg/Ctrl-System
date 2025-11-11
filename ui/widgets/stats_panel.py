"""
Панель отображения статистики и логов
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox,
    QLabel, QTextEdit
)
from datetime import datetime


class StatsPanel(QWidget):
    """Панель статистики и логирования"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Инициализация UI"""
        layout = QVBoxLayout(self)
        
        # === Лог обработки ===
        log_group = QGroupBox("📋 Лог обработки")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # === Статистика ===
        stats_group = QGroupBox("📊 Статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Нет данных")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def log(self, message: str, timestamp: bool = True):
        """
        Добавляет сообщение в лог
        
        Args:
            message: Сообщение для логирования
            timestamp: Добавлять ли временную метку
        """
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{ts}] {message}"
        else:
            formatted_message = message
        
        self.log_text.append(formatted_message)
    
    def clear_log(self):
        """Очищает лог"""
        self.log_text.clear()
    
    def update_statistics(self, stats: dict):
        """
        Обновляет отображение статистики
        
        Args:
            stats: Словарь со статистикой
        """
        if not stats:
            self.stats_label.setText("Нет данных")
            return
        
        stats_text = f"""
✅ Обработка завершена!

📊 Статистика:
• Найдено зон: {stats.get('zones_found', 0)}
• Зон с дефектами: {stats.get('zones_with_defects', 0)}
• Пикселей дефектов: {stats.get('total_defect_pixels', 0)}
        """
        
        # Добавляем детали по зонам если есть
        zone_details = stats.get('zone_details', [])
        if zone_details:
            stats_text += "\n\n📍 Детали по зонам:\n"
            for zone in zone_details:
                status = "✓" if zone['has_defects'] else "○"
                stats_text += f"{status} Зона {zone['zone_id']}: {zone['defect_pixels']} пикселей\n"
        
        self.stats_label.setText(stats_text.strip())
    
    def clear_statistics(self):
        """Очищает статистику"""
        self.stats_label.setText("Нет данных")