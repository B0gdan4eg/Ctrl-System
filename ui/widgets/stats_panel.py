"""
Панель отображения статистики, отчёта о дефектах и логов
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox,
    QLabel, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from datetime import datetime

from config.settings import PIXEL_SIZE_MM

PIXEL_AREA_MM2 = PIXEL_SIZE_MM ** 2


class StatsPanel(QWidget):
    """Панель статистики, отчёта и логирования"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # === Вердикт ===
        verdict_group = QGroupBox("🏁 Результат проверки")
        verdict_layout = QVBoxLayout(verdict_group)

        self.verdict_label = QLabel("—")
        self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_label.setMinimumHeight(56)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.verdict_label.setFont(font)
        self.verdict_label.setStyleSheet(
            "color: #aaaaaa; background: transparent; border-radius: 4px;"
        )
        verdict_layout.addWidget(self.verdict_label)
        layout.addWidget(verdict_group)

        # === Отчёт о дефектах ===
        report_group = QGroupBox("📊 Отчёт")
        report_layout = QVBoxLayout(report_group)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(160)
        self.report_text.setPlaceholderText("Здесь появится отчёт после обработки")
        report_layout.addWidget(self.report_text)
        layout.addWidget(report_group)

        # === Лог обработки ===
        log_group = QGroupBox("📋 Лог")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Лог
    # ------------------------------------------------------------------

    def log(self, message: str, timestamp: bool = True):
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.append(f"[{ts}] {message}")
        else:
            self.log_text.append(message)

    # Moved to archive/dead_code_ui.py

    # ------------------------------------------------------------------
    # Отчёт и вердикт
    # ------------------------------------------------------------------

    def update_statistics(self, stats: dict):
        if not stats:
            self._set_verdict_unknown()
            self.report_text.clear()
            return

        zones_found = stats.get('zones_found', 0)
        zones_with_defects = stats.get('zones_with_defects', 0)
        total_defect_pixels = stats.get('total_defect_pixels', 0)
        zone_details = stats.get('zone_details', [])

        # Вердикт по ГОСТ 11141-84: годными считаются классы I-IV, всё хуже — брак.
        gost_class = (stats.get('gost') or {}).get('worst_class') or 'I'
        is_pass = gost_class in ('I', 'II', 'III', 'IV')

        verdict = "ГОДЕН" if is_pass else "БРАК"
        color = "#27ae60" if is_pass else "#c0392b"
        self.verdict_label.setText(f"{verdict}\nГОСТ: класс {gost_class}")
        self.verdict_label.setStyleSheet(
            f"color: white; background-color: {color}; border-radius: 4px;"
        )

        # Площадь дефектов в мм²
        pixel_area = PIXEL_AREA_MM2
        total_defect_mm2 = total_defect_pixels * pixel_area

        lines = []
        lines.append(f"Зон найдено:       {zones_found}")
        lines.append(f"Зон с дефектами:   {zones_with_defects}")
        lines.append(f"Площадь дефектов:  {total_defect_mm2:.4f} мм²")

        gost = stats.get('gost') or {}
        lines.append("")
        lines.append("По ГОСТ 11141-84:")
        lines.append(f"  Класс чистоты:   {gost.get('worst_class', 'I')}  (допустимы I-IV)")
        lines.append(f"  Точек:           {gost.get('count_points', 0)}")
        lines.append(f"  Царапин:         {gost.get('count_scratches', 0)}")
        biggest = max(gost.get('defects', []), key=lambda d: d['size_mm'], default=None)
        if biggest is not None:
            kind_ru = 'точка' if biggest['kind'] == 'point' else 'царапина'
            lines.append(f"  Самый крупный:   {kind_ru} {biggest['size_mm']:.3f} мм (кл. {biggest['class']})")

        lines.append("")

        if zone_details:
            lines.append("Детали по зонам:")
            for z in zone_details:
                x, y, w, h = z['bbox']
                w_mm = w * PIXEL_SIZE_MM
                h_mm = h * PIXEL_SIZE_MM
                d_px = z['defect_pixels']
                d_mm2 = d_px * pixel_area
                status = "ДЕФЕКТ" if z['has_defects'] else "норма"
                lines.append(
                    f"  Зона {z['zone_id']}: {w_mm:.1f}×{h_mm:.1f} мм  "
                    f"│ дефект {d_mm2:.4f} мм²  │ {status}"
                )

        self.report_text.setPlainText("\n".join(lines))

    def _set_verdict_unknown(self):
        self.verdict_label.setText("—")
        self.verdict_label.setStyleSheet(
            "color: #aaaaaa; background: transparent; border-radius: 4px;"
        )

    # Moved to archive/dead_code_ui.py
