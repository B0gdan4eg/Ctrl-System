"""
Воркер для энергетического расчёта (поверхность негодности) по папке образца.

Оборачивает scripts/run_pipeline.py (калибровка -> детекция зон -> лучшая
полоса на зону -> поверхность негодности по каждой зоне -> вердикт) в
QThread, чтобы не блокировать UI на время расчёта (может занимать секунды
на деталь с несколькими зонами и спектральными полосами).
"""

import os
import sys
import traceback

from PySide6.QtCore import QThread, Signal

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


class EnergyPipelineWorker(QThread):
    """Запускает run_pipeline.run() на папке образца в фоновом потоке"""

    progress = Signal(str)
    finished_ok = Signal(dict, str)  # (res, out_dir)
    error = Signal(str)

    def __init__(self, specimen_dir: str, k: float = 4.0):
        super().__init__()
        self.specimen_dir = specimen_dir
        self.k = k

    def run(self):
        try:
            if _SCRIPTS_DIR not in sys.path:
                sys.path.insert(0, _SCRIPTS_DIR)
            import run_pipeline  # noqa: E402 (динамический sys.path выше)

            self.progress.emit(f"⚙️ Калибровка и детекция зон: {self.specimen_dir}")
            res = run_pipeline.run(self.specimen_dir, k=self.k)

            name = os.path.basename(self.specimen_dir.rstrip("/\\"))
            out_dir = os.path.join(_SCRIPTS_DIR, "..", "results", "v2_pipeline", name)
            out_dir = os.path.normpath(out_dir)
            self.progress.emit("💾 Сохранение результатов...")
            run_pipeline.save(res, out_dir, name)

            self.finished_ok.emit(res, out_dir)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
