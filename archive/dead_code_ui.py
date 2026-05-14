"""
Архив: неиспользуемые UI методы
Перемещено: 2026-05-14
Причина: методы нигде не вызываются за пределами своих определений
"""


# === Из ui/widgets/stats_panel.py — класс StatsPanel ===

def clear_log(self):
    """Очищает лог обработки"""
    self.log_text.clear()


def clear_statistics(self):
    """Сбрасывает вердикт и отчёт о дефектах"""
    self._set_verdict_unknown()
    self.report_text.clear()


# === Из ui/widgets/camera_widget.py — класс CameraWidget ===

def get_current_frame(self):
    """Возвращает текущий кадр"""
    return self.current_frame.copy() if self.current_frame is not None else None
