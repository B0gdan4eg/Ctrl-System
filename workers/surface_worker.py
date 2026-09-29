"""
Воркер построения карт сигнала для вкладки «3D-карта» (PyVista).

Расчёт (scripts/surface_maps.py) крутится в QThread: калибровка, мультиспектральная
детекция зон, композит/фон/p% и позонный поиск дефектов могут занимать секунды.
Результат — dict с numpy-картами — уходит в GUI одним сигналом.
"""

import os
import sys
import traceback

from PySide6.QtCore import QThread, Signal

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


class SurfaceMapsWorker(QThread):
    """Считает surface_maps.build_signal_maps() на папке образца в фоновом потоке."""

    progress = Signal(str)
    finished_ok = Signal(object)  # dict с numpy-картами (см. surface_maps.build_signal_maps)
    error = Signal(str)

    def __init__(self, specimen_dir: str, k: float = 4.0, p_min: float = 20.0):
        super().__init__()
        self.specimen_dir = specimen_dir
        self.k = k
        self.p_min = p_min

    def run(self):
        try:
            if _SCRIPTS_DIR not in sys.path:
                sys.path.insert(0, _SCRIPTS_DIR)
            import surface_maps  # noqa: E402 (динамический sys.path выше)

            self.progress.emit(f"⚙️ Калибровка + мультиспектральная детекция зон: {self.specimen_dir}")
            maps = surface_maps.build_signal_maps(self.specimen_dir, k=self.k, p_min=self.p_min)

            rows = maps["rows"]
            n_win = sum(1 for r in rows if r["in_window"])
            n_dark = sum(r["dark_px"] for r in rows)
            n_white = sum(r["white_px"] for r in rows)
            self.progress.emit(f"   зон: {len(rows)}, в окне 2000±250: {n_win}, "
                               f"дефекты: тёмн={n_dark}px бел={n_white}px")
            self.finished_ok.emit(maps)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")
