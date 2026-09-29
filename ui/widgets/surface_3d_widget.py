"""
Вкладка «3D-карта» главного окна: выбор папки детали -> калибровка, детекция
зон/полос, сплошная карта сигнала всего фильтра, интерактивный 3D на PyVista
(нативный VTK-рендер внутри вкладки — без браузера и plotly).

Высота рельефа — два режима:
  • «Обзор: яркость (DN)» — сырая карта сигнала (попиксельно лучшая ненасыщенная
    полоса); полосы-плато, фон между ними — тёмная долина;
  • «Дефекты: отклонение p%» — знаковое отклонение от фона B(x,y): фон ~0
    (плоское плато), тёмные дефекты — ямы, белые — пики. Видны сразу.

Коридор ДИ (поверхности B·(1±p/100) в DN-режиме, плоскости ±p в p%-режиме)
двигается слайдером строгости; дефекты (позонно в родной полосе, см.
scripts/surface_maps.py) отмечаются точками: красные — тёмные, жёлтые — белые.

Геометрия: pv.ImageData + warp_by_scalar — равномерная сетка, скаляры идут в
C-порядке (столбец меняется быстрее всего), т.е. ровно как индексы numpy
[row, col]; никакой перестановки осей, как было бы с StructuredGrid.
"""

import os

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog, QMessageBox, QSlider,
)
from PySide6.QtCore import Qt, QTimer

import pyvista as pv
from pyvistaqt import QtInteractor

from config.settings import SPECIMEN_BASE_DIR
from workers import SurfaceMapsWorker

MODE_DN = "Обзор: яркость (DN)"
MODE_PCT = "Дефекты: отклонение p%"

_Z_SCALE_DN = 0.4    # сжатие Z в DN-режиме (DN до ~4400 при XY 2048)
_Z_SCALE_PCT = 6.0   # растяжение Z в p%-режиме (отклонения до ±100%)
_PCT_CLIM = 60.0     # симметричный диапазон цвета в p%-режиме (±60%)
_MARKER_LIMIT = 20000  # макс. точек-маркеров на класс дефекта (прореживание)


class Surface3DTab(QWidget):
    """Вкладка 3D-карты сигнала (поверхность негодности) по папке детали."""

    def __init__(self, log=None, parent=None):
        super().__init__(parent)
        self.spec = None
        self.worker = None
        self.maps = None            # dict с numpy-картами (SurfaceMapsWorker.finished_ok)
        self._corridor_actors = []  # актёры коридора — пересаживаются слайдером
        self._corr_B = None         # уменьшенное B для коридора (DN-режим)
        self._dn_mode = True        # режим на момент последнего _render()
        self._log = log or (lambda msg: None)
        self._slider_timer = QTimer(self)  # слайдер spams события — рендерим с паузой
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(60)
        self._slider_timer.timeout.connect(self._update_corridor)
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # --- строка 1: папка + запуск ---
        row1 = QHBoxLayout()
        self.btn_pick = QPushButton("📁 Выбрать папку детали…")
        self.btn_pick.clicked.connect(self.pick_folder)
        row1.addWidget(self.btn_pick)

        self.lbl_path = QLabel("папка не выбрана")
        self.lbl_path.setStyleSheet("color:#9ad;")
        row1.addWidget(self.lbl_path, stretch=1)

        self.btn_build = QPushButton("▶ Построить 3D")
        self.btn_build.setEnabled(False)
        self.btn_build.clicked.connect(self.build)
        row1.addWidget(self.btn_build)
        root.addLayout(row1)

        # --- строка 2: параметры ---
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Высота рельефа:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([MODE_DN, MODE_PCT])
        self.cmb_mode.currentTextChanged.connect(self._render)
        row2.addWidget(self.cmb_mode)

        row2.addWidget(QLabel("Шаг сетки:"))
        self.spn_step = QSpinBox()
        self.spn_step.setRange(2, 30)
        self.spn_step.setValue(8)
        self.spn_step.setToolTip("Меньше — детальнее (шаг по пикселям кадра)")
        self.spn_step.valueChanged.connect(self._render)
        row2.addWidget(self.spn_step)

        self.chk_corridor = QCheckBox("Коридор ДИ")
        self.chk_corridor.setChecked(True)
        self.chk_corridor.toggled.connect(self._render)
        row2.addWidget(self.chk_corridor)

        row2.addWidget(QLabel("ДИ-строгость p:"))
        self.sld_p = QSlider(Qt.Orientation.Horizontal)
        self.sld_p.setRange(5, 60)
        self.sld_p.setValue(20)
        self.sld_p.setFixedWidth(160)
        self.sld_p.valueChanged.connect(self._on_p_changed)
        row2.addWidget(self.sld_p)
        self.lbl_p = QLabel("20 %")
        row2.addWidget(self.lbl_p)

        self.chk_markers = QCheckBox("Маркеры дефектов")
        self.chk_markers.setChecked(True)
        self.chk_markers.toggled.connect(self._render)
        row2.addWidget(self.chk_markers)
        row2.addStretch(1)
        root.addLayout(row2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # бегунок неопределённости
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # --- 3D-просмотр (нативный VTK) ---
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_text("Выбери папку детали и нажми «Построить 3D»",
                              font_size=12, color="#888888")
        root.addWidget(self.plotter, stretch=1)

    # ----- действия -----
    def pick_folder(self):
        start = SPECIMEN_BASE_DIR if SPECIMEN_BASE_DIR and os.path.isdir(SPECIMEN_BASE_DIR) else ""
        d = QFileDialog.getExistingDirectory(self, "Выберите папку детали", start)
        if d:
            self.spec = d
            self.lbl_path.setText(d)
            self.btn_build.setEnabled(True)
            self._log(f"📂 Выбрано: {d}")

    def build(self):
        if not self.spec:
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Построение уже выполняется",
                                    "Дождитесь завершения текущего построения.")
            return

        self._set_busy(True)
        self._log("=" * 50)
        self._log(f"🌐 ПОСТРОЕНИЕ 3D-КАРТЫ: {self.spec}")

        self.worker = SurfaceMapsWorker(self.spec)
        self.worker.progress.connect(self._log)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_done(self, maps: dict):
        self.maps = maps
        self._set_busy(False)
        for r in maps["rows"]:
            win = "" if r["in_window"] else " (вне окна)"
            self._log(f"   зона {r['zone']} x={r['x']}: полоса {r['band']} "
                      f"медиана={r['median']}{win} | тёмн={r['dark_px']}px бел={r['white_px']}px")
        self._log("✅ Карты готовы")
        self._log("=" * 50)
        self._render()

    def on_error(self, msg: str):
        self._set_busy(False)
        self._log("❌ " + msg)
        QMessageBox.critical(self, "Ошибка построения 3D", msg.split("\n")[0])

    def cleanup(self):
        """Освобождение VTK-рендера (вызвать при закрытии главного окна)."""
        try:
            self.plotter.close()
        except Exception:
            pass

    # ----- рендеринг -----
    def _on_p_changed(self, value: int):
        self.lbl_p.setText(f"{value} %")
        self._slider_timer.start()

    def _set_busy(self, busy: bool):
        self.btn_build.setEnabled(not busy and self.spec is not None)
        self.btn_pick.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _height_field(self, arr_sub, dn_mode: bool):
        """Карта высот текущего режима в координатах кадра (пиксели)."""
        return arr_sub  # массивы composite/B/dev уже в пиксельных координатах

    def _render(self):
        """Полная перестройка сцены под текущий режим/шаг/чекбоксы."""
        if self.maps is None:
            return
        step = self.spn_step.value()
        comp = self.maps["composite"][::step, ::step]
        dev = self.maps["dev"][::step, ::step]
        self._dn_mode = self.cmb_mode.currentText() == MODE_DN

        z = comp if self._dn_mode else dev
        ny, nx = z.shape

        p = self.plotter
        p.clear()

        # основная поверхность: плоская сетка + warp по высоте
        grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(step, step, 1),
                            origin=(0.0, 0.0, 0.0))
        grid.point_data["value"] = z.ravel(order="C")
        surf = grid.warp_by_scalar("value", factor=1.0, normal=(0, 0, 1)).extract_surface(
            algorithm="dataset_surface")
        if self._dn_mode:
            p.add_mesh(surf, scalars="value", cmap="gray", smooth_shading=True,
                       show_scalar_bar=False)
        else:
            p.add_mesh(surf, scalars="value", cmap="RdBu_r",
                       clim=[-_PCT_CLIM, _PCT_CLIM], smooth_shading=False,
                       lighting=False, show_scalar_bar=False)

        self._corridor_actors = []
        self._corr_B = self.maps["B"][::step, ::step]
        if self.chk_corridor.isChecked():
            self._add_corridor(step, nx, ny)

        if self.chk_markers.isChecked():
            self._add_markers(step, self._dn_mode)

        zscale = _Z_SCALE_DN if self._dn_mode else _Z_SCALE_PCT
        p.set_scale(1, 1, zscale)
        p.reset_camera()
        mode_name = "DN" if self._dn_mode else "p%"
        p.add_text(f"режим {mode_name}  |  шаг {step}",
                   font_size=10, color="#888888", position="upper_left")

    def _add_corridor(self, step: int, nx: int, ny: int):
        """Границы ДИ: в DN-режиме поверхности B·(1∓p/100), в p%-режиме плоскости ±p."""
        pv_ = self.sld_p.value() / 100.0
        p = self.plotter
        if self._dn_mode:
            zs = (self._corr_B * (1.0 - pv_), self._corr_B * (1.0 + pv_))
        else:
            zs = (np.full((ny, nx), -self.sld_p.value(), np.float32),
                  np.full((ny, nx), self.sld_p.value(), np.float32))
        for z, color in zip(zs, ("#ff5555", "#55ffff")):
            grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(step, step, 1),
                                origin=(0.0, 0.0, 0.0))
            grid.point_data["z"] = z.ravel(order="C")
            mesh = grid.warp_by_scalar("z", factor=1.0, normal=(0, 0, 1)).extract_surface(
                algorithm="dataset_surface")
            actor = p.add_mesh(mesh, color=color, opacity=0.30, show_scalar_bar=False)
            self._corridor_actors.append(actor)

    def _update_corridor(self):
        """Слайдер p: убрать старые границы ДИ и нарисовать на новой высоте
        (основной рельеф и маркеры не трогаем)."""
        if self.maps is None or not self.chk_corridor.isChecked():
            return
        for actor in self._corridor_actors:
            self.plotter.remove_actor(actor, reset_camera=False)
        step = self.spn_step.value()
        ny, nx = self._corr_B.shape
        self._add_corridor(step, nx, ny)
        self.plotter.render()

    def _add_markers(self, step: int, dn_mode: bool):
        """Точки дефектов на рельефе: красные — тёмные, жёлтые — белые."""
        zfield = self.maps["composite"] if dn_mode else self.maps["dev"]
        zmin, zmax = float(zfield.min()), float(zfield.max())
        lift = (zmax - zmin) * 0.01
        for mask, color in ((self.maps["dark"], "#ff2020"), (self.maps["white"], "#ffe000")):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            if len(xs) > _MARKER_LIMIT:  # прореживание для скорости рендера
                idx = np.linspace(0, len(xs) - 1, _MARKER_LIMIT).astype(int)
                ys, xs = ys[idx], xs[idx]
            zs = zfield[ys, xs] + lift
            pts = pv.PolyData(np.column_stack([xs, ys, zs]))
            self.plotter.add_points(pts, color=color, point_size=4,
                                    render_points_as_spheres=True)
