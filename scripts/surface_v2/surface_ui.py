#!/usr/bin/env python3
"""
v2 UI: выбираешь папку детали -> программа сама калибрует, ищет зоны/полосы и
строит интерактивный 3D (крути/зумь/двигай мышью прямо в окне).

Стиль взят из UI оригинального Ctrl-System (тёмная тема, левая панель управления +
лог, справа просмотр). 3D рисуется через Plotly и встраивается в окно как
QWebEngineView, поэтому вертится мышью без выхода в браузер.

Два режима:
  • «Яркость + плотность (по зонам)» — поверхность негодности по каждой зоне
    (лучшая полоса к 2000) + слой плотности дефектов + подвижная секущая по Z.
  • «Весь фильтр (без зон)» — обзорный 3D всего фильтра (попиксельно лучшая
    ненасыщенная полоса), без шага детекции зон.

Запуск:  python scripts/surface_ui.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

# COM в STA ДО Qt — иначе нативный QFileDialog виснет на Windows (см. main.py оригинала).
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    except Exception:
        pass

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl  # noqa: E402
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402

# Тёмная тема оригинала — грузим файл напрямую, чтобы не тянуть torch из ui/__init__.
_theme_spec = importlib.util.spec_from_file_location(
    "dark_theme", os.path.join(_ROOT, "ui", "styles", "dark_theme.py"))
_theme = importlib.util.module_from_spec(_theme_spec)
_theme_spec.loader.exec_module(_theme)
get_dark_theme = _theme.get_dark_theme

# Пайплайн v2
from surface_3d_interactive import build_maps, make_html  # noqa: E402
from whole_filter_3d import save_3d_html  # noqa: E402
from calibrate_multispectral import (  # noqa: E402
    discover_frames, load_calibrated,
)
from whole_filter_3d import best_band_per_pixel  # noqa: E402

DEFAULT_BASE = r"D:\Ctrl-System v2\WTS 05.08.2026"
OUT_DIR = os.path.join(_ROOT, "results", "v2_filter3d")

MODE_ZONES = "Рельеф + поверхность негодности ДИ (по зонам)"
MODE_WHOLE = "Весь фильтр (без зон)"


class BuildWorker(QThread):
    """Фоновый расчёт + генерация HTML (не блокирует окно)."""
    progress = Signal(str)
    done = Signal(str)        # путь к html
    error = Signal(str)

    def __init__(self, spec: str, mode: str, step: int, flat_per_zone: bool = False):
        super().__init__()
        self.spec = spec
        self.mode = mode
        self.step = step
        self.flat_per_zone = flat_per_zone

    def run(self):
        try:
            name = os.path.basename(self.spec.rstrip("/\\"))
            out = os.path.join(OUT_DIR, name)
            os.makedirs(out, exist_ok=True)

            self.progress.emit(f"📂 Деталь: {name}")
            fr = discover_frames(self.spec)
            self.progress.emit(
                f"   кадры: dark={bool(fr['dark'])} flatГ={bool(fr['flat_halogen'])} "
                f"flatД={bool(fr['flat_diode'])} полос={len(fr['bands'])}")
            if not fr["bands"]:
                raise RuntimeError("в папке не найдено спектральных полос — это папка детали?")

            if self.mode == MODE_ZONES:
                self.progress.emit("⚙️ Калибровка + детекция зон + поверхность негодности по ДИ…")
                relief, bg, labels, density = build_maps(self.spec)
                import numpy as _np
                zmask = bg > 0
                if zmask.any():
                    self.progress.emit(f"   фон B по зонам ≈ {float(_np.median(bg[zmask])):.0f} DN, "
                                       f"ДИ-порог 0.8·B ≈ {0.8 * float(_np.median(bg[zmask])):.0f} DN")
                    self.progress.emit(f"   плотность дефектов max={float(density.max()):.2f} "
                                       f"(бугры = кластеры)")
                mode_txt = "плоская по зоне" if self.flat_per_zone else "по фону B(x,y)"
                self.progress.emit(f"🌐 Рендер 3D (яркость+плотность + ДИ-поверхность: {mode_txt})…")
                path = make_html(relief, bg, labels, density, out, name, step=self.step,
                                 flat_per_zone=self.flat_per_zone)
            else:
                self.progress.emit("⚙️ Калибровка + попиксельно лучшая полоса…")
                cal, raw = load_calibrated(fr)
                comp, satmap = best_band_per_pixel(cal, raw)
                self.progress.emit(f"   композит {comp.shape}, насыщ.пикс={int(satmap.sum())}")
                self.progress.emit("🌐 Рендер интерактивного 3D (весь фильтр)…")
                path = save_3d_html(comp, out, name, step=max(4, self.step // 2))

            self.progress.emit(f"✅ Готово: {os.path.basename(path)}")
            self.done.emit(path)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class SurfaceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spec = None
        self.worker = None
        self._init_ui()
        self.setStyleSheet(get_dark_theme())

    def _init_ui(self):
        self.setWindowTitle("Ctrl-System v2 — поверхность негодности (3D)")
        self.setGeometry(80, 50, 1500, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ===== ЛЕВАЯ ПАНЕЛЬ =====
        left = QWidget()
        ll = QVBoxLayout(left)

        gb = QGroupBox("Деталь")
        gl = QVBoxLayout(gb)
        self.btn_pick = QPushButton("📁 Выбрать папку детали…")
        self.btn_pick.clicked.connect(self.pick_folder)
        gl.addWidget(self.btn_pick)
        self.lbl_path = QLabel("папка не выбрана")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("color:#9ad;")
        gl.addWidget(self.lbl_path)
        ll.addWidget(gb)

        gb2 = QGroupBox("Параметры 3D")
        g2 = QVBoxLayout(gb2)
        g2.addWidget(QLabel("Тип поверхности:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([MODE_ZONES, MODE_WHOLE])
        g2.addWidget(self.cmb_mode)
        row = QHBoxLayout()
        row.addWidget(QLabel("Детализация (шаг, меньше=детальнее):"))
        self.spn_step = QSpinBox()
        self.spn_step.setRange(2, 30)
        self.spn_step.setValue(10)
        row.addWidget(self.spn_step)
        g2.addLayout(row)
        self.chk_flat = QCheckBox("Плоская ДИ-поверхность (ступени по зонам)")
        self.chk_flat.setToolTip("Вкл: 0.8·медиана(B) на зону (ровные плато).\n"
                                 "Выкл: 0.8·B(x,y) — следует за фоном зоны (гладко изогнута).")
        g2.addWidget(self.chk_flat)
        ll.addWidget(gb2)

        self.btn_build = QPushButton("▶ Построить 3D")
        self.btn_build.setEnabled(False)
        self.btn_build.clicked.connect(self.build)
        ll.addWidget(self.btn_build)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # бегунок неопределённости
        self.progress.setVisible(False)
        ll.addWidget(self.progress)

        gb3 = QGroupBox("Лог")
        g3 = QVBoxLayout(gb3)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        g3.addWidget(self.log)
        ll.addWidget(gb3, stretch=1)

        # ===== ПРАВАЯ ПАНЕЛЬ: 3D =====
        right = QWidget()
        rl = QVBoxLayout(right)
        self.web = QWebEngineView()
        self.web.setHtml(
            "<body style='background:#1e1e1e;color:#888;font-family:sans-serif'>"
            "<div style='display:flex;height:90vh;align-items:center;justify-content:center'>"
            "Выбери папку детали и нажми «Построить 3D»</div></body>")
        rl.addWidget(self.web)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.statusBar().showMessage("Готов к работе")

    # ----- действия -----
    def _log(self, msg: str):
        self.log.append(msg)

    def pick_folder(self):
        start = DEFAULT_BASE if os.path.isdir(DEFAULT_BASE) else ""
        d = QFileDialog.getExistingDirectory(self, "Выберите папку детали", start)
        if d:
            self.spec = d
            self.lbl_path.setText(d)
            self.btn_build.setEnabled(True)
            self._log(f"📂 Выбрано: {d}")

    def build(self):
        if not self.spec:
            return
        self.btn_build.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage("Обработка…")
        self.log.clear()

        self.worker = BuildWorker(self.spec, self.cmb_mode.currentText(), self.spn_step.value(),
                                  flat_per_zone=self.chk_flat.isChecked())
        self.worker.progress.connect(self._log)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_done(self, html_path: str):
        self.web.setUrl(QUrl.fromLocalFile(html_path))
        self.progress.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_pick.setEnabled(True)
        self.statusBar().showMessage(f"✅ Готово: {os.path.basename(html_path)}")

    def on_error(self, msg: str):
        self.progress.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_pick.setEnabled(True)
        self.statusBar().showMessage("❌ Ошибка")
        self._log("❌ " + msg)
        QMessageBox.critical(self, "Ошибка обработки", msg.split("\n")[0])


def main():
    app = QApplication(sys.argv)
    win = SurfaceUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
