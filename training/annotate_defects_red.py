"""
Mini-app для разметки дефектов на 2048×2048 кропах.

Возможности:
  • Список изображений слева со статусами:  ⏳ pending / ✅ annotated / 🚫 skipped
  • Auto-mode (A): клик на дефект → flood-fill авто-выделение границы
  • Manual-mode (M): рисуешь полигон руками, Enter закрывает
  • Tolerance slider — порог для flood-fill (3–60)
  • DEL — пропустить картинку
  • Z   — отменить последний полигон
  • R   — очистить все полигоны на текущем кадре
  • S   — сохранить и перейти к следующей
  • Esc — отменить незакрытый ручной полигон
  • Backspace — назад к предыдущей картинке
  • Колесо мыши — zoom

Прогресс и метки сохраняются автоматически:
  D:/train/labels_2048.json    — VIA-совместимый JSON (читается msak_from_json.py)
  D:/train/_progress.json      — статус каждого файла, восстанавливается между запусками

Запуск:
  pip install pyside6 opencv-python numpy
  python D:/train/annotate_defects.py
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QPolygonF,
    QKeySequence, QShortcut, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QPushButton, QLabel, QSlider, QRadioButton, QMessageBox
)


# ─── Пути ─────────────────────────────────────────────────────────────────────
IMAGES_DIR    = Path("D:/train/base_red_zones_n5_2048")
LABELS_PATH   = Path("D:/train/labels_red_2048.json")
PROGRESS_PATH = Path("D:/train/_progress_red.json")

# Цвета полигонов
COLOR_AUTO   = QColor( 80, 220, 120, 90)   # зелёный полупрозрачный — авто
COLOR_MANUAL = QColor(255, 200,   0, 90)   # жёлтый полупрозрачный — ручной
COLOR_SAVED  = QColor( 80, 180, 255, 90)   # голубой — восстановленный из JSON


# ─── Канвас ───────────────────────────────────────────────────────────────────
class AnnotationCanvas(QGraphicsView):
    polygon_added = Signal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        self.gray: np.ndarray | None = None
        self.pixmap_item = None
        self.polygon_items: list = []   # [(QGraphicsPolygonItem, [(x,y),...], kind)]
        self.manual_pts: list = []
        self.manual_preview = None

        self.mode = "auto"              # "auto" / "manual"
        self.tolerance = 15

    # ── загрузка / очистка ──
    def load_image(self, path: Path):
        self._scene.clear()
        self.polygon_items.clear()
        self.manual_pts.clear()
        self.manual_preview = None
        self.pixmap_item = None
        self.gray = None

        bgr = cv2.imread(str(path))
        if bgr is None:
            return False
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.pixmap_item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(0, 0, w, h)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        return True

    def restore_polygons(self, polygons: list):
        for pts in polygons:
            self._add_polygon(pts, "saved")

    # ── зум колесом ──
    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 1 / 1.25
        self.scale(f, f)

    # ── клик ──
    def mousePressEvent(self, e):
        if self.gray is None:
            return super().mousePressEvent(e)

        if e.button() == Qt.LeftButton:
            sp = self.mapToScene(e.pos())
            x, y = int(sp.x()), int(sp.y())
            h, w = self.gray.shape
            if 0 <= x < w and 0 <= y < h:
                if self.mode == "auto":
                    self._auto_detect(x, y)
                else:
                    self.manual_pts.append((x, y))
                    self._refresh_manual_preview()
            return

        if e.button() == Qt.RightButton and self.mode == "manual":
            self.finish_manual()
            return

        super().mousePressEvent(e)

    # ── auto-detect через flood-fill ──
    def _auto_detect(self, x: int, y: int):
        h, w = self.gray.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(
            self.gray.copy(), flood_mask, (x, y), 255,
            loDiff=self.tolerance, upDiff=self.tolerance,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8)
        )
        mask = flood_mask[1:-1, 1:-1]

        # морфология — закрыть мелкие дырки
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        # тот, что содержит точку клика, иначе крупнейший
        chosen = None
        for c in contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                if chosen is None or cv2.contourArea(c) > cv2.contourArea(chosen):
                    chosen = c
        if chosen is None:
            chosen = max(contours, key=cv2.contourArea)

        if cv2.contourArea(chosen) < 10:
            return

        eps = max(0.5, 0.0015 * cv2.arcLength(chosen, True))
        approx = cv2.approxPolyDP(chosen, eps, True)
        if len(approx) < 3:
            return

        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        self._add_polygon(pts, "auto")
        self.polygon_added.emit()

    # ── manual ──
    def _refresh_manual_preview(self):
        if self.manual_preview is not None:
            self._scene.removeItem(self.manual_preview)
            self.manual_preview = None
        if not self.manual_pts:
            return
        poly = QPolygonF([QPointF(x, y) for x, y in self.manual_pts])
        self.manual_preview = self._scene.addPolygon(
            poly,
            QPen(QColor(255, 200, 0, 240), 2),
            QBrush(QColor(255, 200, 0, 60))
        )

    def finish_manual(self):
        if len(self.manual_pts) >= 3:
            self._add_polygon(list(self.manual_pts), "manual")
            self.polygon_added.emit()
        self.manual_pts.clear()
        if self.manual_preview is not None:
            self._scene.removeItem(self.manual_preview)
            self.manual_preview = None

    def cancel_manual(self):
        self.manual_pts.clear()
        if self.manual_preview is not None:
            self._scene.removeItem(self.manual_preview)
            self.manual_preview = None

    # ── общие операции с полигонами ──
    def _add_polygon(self, pts: list, kind: str):
        fill = {"auto": COLOR_AUTO, "manual": COLOR_MANUAL, "saved": COLOR_SAVED}[kind]
        border = QColor(fill); border.setAlpha(230)
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        item = self._scene.addPolygon(poly, QPen(border, 2), QBrush(fill))
        self.polygon_items.append((item, pts, kind))

    def undo_last(self):
        if self.manual_pts:
            self.manual_pts.pop()
            self._refresh_manual_preview()
            return
        if self.polygon_items:
            item, _, _ = self.polygon_items.pop()
            self._scene.removeItem(item)

    def clear_polygons(self):
        for item, _, _ in self.polygon_items:
            self._scene.removeItem(item)
        self.polygon_items.clear()
        self.cancel_manual()

    def get_polygons(self) -> list:
        return [pts for _, pts, _ in self.polygon_items]


# ─── Главное окно ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Annotator")
        self.resize(1600, 1000)

        self.images = sorted(IMAGES_DIR.glob("*.png"))
        if not self.images:
            QMessageBox.critical(self, "Ошибка", f"Нет PNG в {IMAGES_DIR}")
            sys.exit(1)

        self.progress = self._load_json(PROGRESS_PATH, {})    # filename -> status
        self.labels   = self._load_json(LABELS_PATH, {})      # filename -> VIA entry

        self.current_idx = self._first_pending_idx()

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_list()
        self._load_current()

    # ── UI ──
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Левая панель
        left = QVBoxLayout()
        left.addWidget(QLabel("Изображения:"))
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(280)
        self.list_widget.itemClicked.connect(self._on_list_click)
        left.addWidget(self.list_widget, 1)
        self.stats_label = QLabel("")
        left.addWidget(self.stats_label)
        root.addLayout(left)

        # Центр: контролы + канвас + кнопки
        center = QVBoxLayout()

        ctrl = QHBoxLayout()
        self.mode_auto   = QRadioButton("Auto-click [A]")
        self.mode_manual = QRadioButton("Manual [M]")
        self.mode_auto.setChecked(True)
        self.mode_auto.toggled.connect(lambda v: v and self._set_mode("auto"))
        self.mode_manual.toggled.connect(lambda v: v and self._set_mode("manual"))
        ctrl.addWidget(self.mode_auto)
        ctrl.addWidget(self.mode_manual)
        ctrl.addSpacing(20)

        ctrl.addWidget(QLabel("Tolerance:"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(3, 60)
        self.tol_slider.setValue(15)
        self.tol_slider.setMaximumWidth(180)
        self.tol_label = QLabel("15")
        self.tol_slider.valueChanged.connect(self._on_tol_change)
        ctrl.addWidget(self.tol_slider)
        ctrl.addWidget(self.tol_label)
        ctrl.addStretch()
        center.addLayout(ctrl)

        self.canvas = AnnotationCanvas()
        center.addWidget(self.canvas, 1)

        btns = QHBoxLayout()
        self._btn(btns, "← Назад [Backspace]", self.prev_image)
        self._btn(btns, "🚫 Пропустить [Del]", self.skip_image)
        self._btn(btns, "Очистить [R]",        self.clear_current)
        self._btn(btns, "Отменить [Z]",        self.canvas.undo_last)
        self._btn(btns, "Закрыть полигон [Enter]", self.canvas.finish_manual)
        btns.addStretch()
        save_btn = QPushButton("✅ Сохранить и далее [S]")
        save_btn.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        save_btn.clicked.connect(self.save_and_next)
        btns.addWidget(save_btn)
        center.addLayout(btns)

        root.addLayout(center, 1)

    def _btn(self, layout, text, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        layout.addWidget(b)

    def _on_tol_change(self, v):
        self.canvas.tolerance = v
        self.tol_label.setText(str(v))

    def _setup_shortcuts(self):
        sc = lambda key, fn: QShortcut(QKeySequence(key), self, activated=fn)
        sc("A",          lambda: self.mode_auto.setChecked(True))
        sc("M",          lambda: self.mode_manual.setChecked(True))
        sc(Qt.Key_Delete,    self.skip_image)
        sc("Z",          self.canvas.undo_last)
        sc("R",          self.clear_current)
        sc(Qt.Key_Return,    self.canvas.finish_manual)
        sc(Qt.Key_Enter,     self.canvas.finish_manual)
        sc(Qt.Key_Escape,    self.canvas.cancel_manual)
        sc("S",          self.save_and_next)
        sc(Qt.Key_Backspace, self.prev_image)

    # ── persistence ──
    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default
        return default

    def _save_progress(self):
        PROGRESS_PATH.write_text(
            json.dumps(self.progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_labels(self):
        LABELS_PATH.write_text(
            json.dumps(self.labels, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _first_pending_idx(self):
        for i, p in enumerate(self.images):
            if self.progress.get(p.name) not in ("annotated", "skipped"):
                return i
        return 0

    # ── список ──
    def _refresh_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        icons = {"annotated": "✅ ", "skipped": "🚫 "}
        for p in self.images:
            icon = icons.get(self.progress.get(p.name, ""), "⏳ ")
            self.list_widget.addItem(QListWidgetItem(f"{icon}{p.name}"))
        if 0 <= self.current_idx < self.list_widget.count():
            self.list_widget.setCurrentRow(self.current_idx)
            self.list_widget.scrollToItem(self.list_widget.item(self.current_idx))
        self.list_widget.blockSignals(False)
        self._update_stats()

    def _update_stats(self):
        done = sum(1 for v in self.progress.values() if v == "annotated")
        skip = sum(1 for v in self.progress.values() if v == "skipped")
        total = len(self.images)
        self.stats_label.setText(
            f"✅ {done}    🚫 {skip}    ⏳ {total - done - skip}    /  {total}"
        )

    def _on_list_click(self, item):
        self.current_idx = self.list_widget.row(item)
        self._load_current()

    # ── режимы / навигация ──
    def _set_mode(self, mode):
        self.canvas.mode = mode
        self.canvas.cancel_manual()

    def _load_current(self):
        if not (0 <= self.current_idx < len(self.images)):
            return
        p = self.images[self.current_idx]
        self.canvas.load_image(p)

        # восстановить уже сохранённые полигоны
        if p.name in self.labels:
            polys = []
            regions = self.labels[p.name].get("regions", {})
            iterable = regions.values() if isinstance(regions, dict) else regions
            for r in iterable:
                sa = r.get("shape_attributes", {})
                if sa.get("name") == "polygon":
                    xs, ys = sa.get("all_points_x", []), sa.get("all_points_y", [])
                    if xs and ys:
                        polys.append(list(zip(xs, ys)))
            self.canvas.restore_polygons(polys)

        self.setWindowTitle(f"[{self.current_idx+1}/{len(self.images)}]  {p.name}")
        self.list_widget.setCurrentRow(self.current_idx)

    def next_image(self):
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self._load_current()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_current()

    # ── действия ──
    def clear_current(self):
        self.canvas.clear_polygons()

    def skip_image(self):
        p = self.images[self.current_idx]
        self.progress[p.name] = "skipped"
        self.labels.pop(p.name, None)
        self._save_progress(); self._save_labels()
        self._refresh_list()
        self.next_image()

    def save_and_next(self):
        self.canvas.finish_manual()
        p = self.images[self.current_idx]
        polys = self.canvas.get_polygons()

        regions = {}
        for i, pts in enumerate(polys):
            regions[str(i)] = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [int(x) for x, _ in pts],
                    "all_points_y": [int(y) for _, y in pts],
                },
                "region_attributes": {}
            }

        self.labels[p.name] = {
            "filename": p.name,
            "size": p.stat().st_size,
            "regions": regions,
            "file_attributes": {}
        }
        self.progress[p.name] = "annotated"

        self._save_progress(); self._save_labels()
        self._refresh_list()
        self.next_image()


# ─── entry point ──────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
