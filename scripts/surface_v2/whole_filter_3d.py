"""
v2: карта яркости и 3D ВСЕГО фильтра без детекции зон (по просьбе — убрать шаг 2).

Вместо Otsu→контуры→зоны: попиксельный выбор лучшей полосы — для каждого пикселя берём
самую яркую НЕнасыщенную калиброванную полосу. Фон (тёмный во всех полосах) остаётся тёмным,
зоны получают значение своей полосы. Зоны как объекты не нужны.

ВАЖНО: max-композит хорош для ОБЗОРНОЙ карты/3D, но для ДЕТЕКЦИИ дефектов опасен —
дефект, тёмный в «родной» полосе зоны, может быть не тёмным в другой, и max его спрячет.
Поэтому поиск дефектов оставляем по-зонно в родной полосе (run_pipeline.py), а это —
визуализация всего фильтра.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

import glob as _glob  # noqa: E402
from surface_of_unfitness import imwrite_u, _to_u8, imread_u16  # noqa: E402
from calibrate_multispectral import discover_frames, load_calibrated, SAT  # noqa: E402


def load_raw_glob(pattern: str) -> tuple[dict, dict]:
    """Загрузка сырых снимков по glob БЕЗ калибровки (нет тёмнового/flat).
    cal=raw (некалиброванные) — для попиксельного max и 3D."""
    files = sorted(_glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"нет файлов по шаблону: {pattern}")
    raw = {}
    for f in files:
        raw[os.path.splitext(os.path.basename(f))[0]] = imread_u16(f).astype(np.float32)
    print("[i] загружено сырых кадров:", list(raw))
    return dict(raw), raw  # cal=raw


def best_band_per_pixel(cal: dict, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Попиксельно: самая яркая НЕнасыщенная полоса. Возвращает (composite, sat_map)."""
    bands = list(cal)
    cals = np.stack([cal[b] for b in bands])   # (B,H,W)
    raws = np.stack([raw[b] for b in bands])
    sat = raws >= SAT
    masked = np.where(sat, -np.inf, cals)       # насыщенные не выбираем
    composite = np.max(masked, axis=0)
    allsat = sat.all(axis=0)
    composite = np.where(allsat | ~np.isfinite(composite), cals.max(axis=0), composite)
    return composite.astype(np.float32), allsat


def save_3d(comp: np.ndarray, out: str, name: str, step: int = 12,
            floor_pct: float = 40.0) -> None:
    """3D всего фильтра. Фон опускаем под общий пол, чтобы зоны-плато читались рельефом."""
    H, W = comp.shape
    ys = np.arange(0, H, step); xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)
    Z = comp[::step, ::step]
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True,
                    rstride=1, cstride=1)
    ax.set_title(f"{name}: 3D яркости всего фильтра (попиксельно лучшая ненасыщенная полоса)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("яркость DN")
    ax.view_init(elev=40, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_filter3d.png"), dpi=120)
    plt.close(fig)


def save_3d_html(comp: np.ndarray, out: str, name: str, step: int = 8) -> str:
    """Интерактивный 3D (Plotly) -> standalone HTML: вращение/зум/панорама мышью в браузере."""
    import plotly.graph_objects as go
    Z = comp[::step, ::step]
    x = np.arange(0, comp.shape[1], step)
    y = np.arange(0, comp.shape[0], step)
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale="Viridis",
                                     colorbar=dict(title="DN"))])
    fig.update_layout(
        title=f"{name}: 3D яркости всего фильтра — крути/зумь/двигай мышью",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="яркость DN",
                   aspectratio=dict(x=1, y=1, z=0.45)),
        width=1200, height=850, margin=dict(l=0, r=0, t=40, b=0))
    path = os.path.join(out, f"{name}_filter3d.html")
    fig.write_html(path, include_plotlyjs=True)  # inline -> работает офлайн
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen", default=None, help="папка детали (с калибровкой)")
    ap.add_argument("--images", default=None, help="glob сырых снимков (без калибровки)")
    ap.add_argument("--name", default=None, help="имя вывода для режима --images")
    ap.add_argument("--step", type=int, default=12, help="прореживание для 3D")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.images:
        cal, raw = load_raw_glob(args.images)
        name = args.name or "raw3d"
    elif args.specimen:
        frames = discover_frames(args.specimen)
        cal, raw = load_calibrated(frames)
        name = os.path.basename(args.specimen.rstrip("/\\"))
    else:
        ap.error("нужен --specimen или --images")
    comp, satmap = best_band_per_pixel(cal, raw)
    out = args.out or os.path.join("results", "v2_filter3d", name)
    os.makedirs(out, exist_ok=True)

    imwrite_u(os.path.join(out, f"{name}_composite.png"), _to_u8(comp))
    imwrite_u(os.path.join(out, f"{name}_composite16.png"),
              np.clip(comp, 0, 4095).astype(np.uint16))
    save_3d(comp, out, name, step=args.step)
    html = save_3d_html(comp, out, name, step=max(6, args.step // 2))
    print(f"[i] composite {comp.shape}  насыщ. пикселей: {int(satmap.sum())}")
    print(f"[ok] PNG + интерактивный HTML -> {out}")
    print(f"[html] открой в браузере: {html}")


if __name__ == "__main__":
    main()
