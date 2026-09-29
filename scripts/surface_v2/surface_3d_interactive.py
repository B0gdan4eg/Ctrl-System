"""
v2: интерактивный 3D — рельеф яркости + «поверхность негодности» по ДИ (критерий
протокол.docx), подвижная по строгости p%.

Критерий ДИ относителен: p% = (B − img)/B·100. Значит абсолютный порог в DN —
это B·(1 − p/100), и он ПРИВЯЗАН К ФОНУ КОНКРЕТНОЙ ЗОНЫ: у яркой зоны 0.8·B высоко,
у тусклой — низко. Поэтому ДИ-поверхность строим для КАЖДОЙ зоны на её уровне B.

Для детали:
  - рельеф яркости по зонам (лучшая полоса к 2000, калиброванная) -> relief_map
  - фон B(x,y) по зонам -> bg_map
  - ДИ-поверхность негодности = B·(1 − p/100); слайдер двигает p (5..60%),
    дефолт 20% (= точечный/значимый тёмный). Где рельеф проваливается ниже
    ДИ-поверхности — там брак. Рядом панель «карта p%» того, что ниже поверхности.

Plotly -> standalone HTML (крути/зумь/двигай мышью, слайдер двигает строгость ДИ).
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

from surface_of_unfitness import process, estimate_background, estimate_noise_model_pooled  # noqa: E402
from calibrate_multispectral import (  # noqa: E402
    discover_frames, load_calibrated, detect_zones, zone_mask, best_band,
)

ERODE = 25
P_MIN = 20.0   # ДИ: точечный/значимый тёмный дефект 50 > p ≥ 20


def build_maps(spec: str):
    """Возвращает (relief, bg, labels, density): рельеф яркости, фон B, карту меток зон
    (0=вне зон, 1..N) и слой плотности дефектов S(x,y) в координатах кадра.
    density — глубина-в-сигмах, размытая гауссом: близкие дефекты складываются в высокий
    бугор, далёкие — в отдельные низкие. labels нужны для «плоской ДИ-поверхности»."""
    fr = discover_frames(spec)
    cal, raw = load_calibrated(fr)
    proj = np.stack(list(cal.values())).max(0)
    zones = detect_zones(proj)
    H, W = proj.shape
    relief = np.zeros((H, W), np.float32)
    bg = np.zeros((H, W), np.float32)
    labels = np.zeros((H, W), np.int32)
    density = np.zeros((H, W), np.float32)

    zframes = []
    for i, c in enumerate(zones):
        zm = zone_mask(proj.shape, c, erode=ERODE)
        band = best_band(zm, raw)["best"]
        frame = cal[band]
        zframes.append((i, zm, frame))

    # одна шумовая модель на всю деталь, из пула тайлов всех зон разом (design §4;
    # см. estimate_noise_model_pooled — устойчивее, чем оценка read/gain по одной
    # узкой по яркости зоне).
    samples = [(frame, zm, estimate_background(frame, zone=zm)) for _, zm, frame in zframes]
    noise = estimate_noise_model_pooled(samples, dark_path=fr["dark"]) if samples else None

    for i, zm, frame in zframes:
        r = process(frame, k=4.0, zone=zm, dark_path=fr["dark"], p_min=P_MIN, noise=noise)
        relief[zm] = frame[zm]      # фактическая яркость зоны (рельеф)
        bg[zm] = r.B[zm]            # фон B зоны (к нему привязана ДИ-поверхность)
        labels[zm] = i + 1
        # r.density уже маскирована по dark_mask внутри process() (только реальные
        # дефекты — чистая зона даёт плотность 0, без ложного пьедестала).
        density[zm] = r.density[zm]
    return relief, bg, labels, density


def make_html(relief: np.ndarray, bg: np.ndarray, labels: np.ndarray, density: np.ndarray,
              out: str, name: str, step: int = 10, n_levels: int = 24,
              flat_per_zone: bool = False) -> str:
    """flat_per_zone=False: ДИ-поверхность = 0.8·B(x,y) (следует за фоном, гладко изогнута).
    flat_per_zone=True:  ДИ-поверхность = 0.8·медиана(B зоны) (5 ровных ступеней-плато).

    Террейн = рельеф_яркости + масштаб·плотность: близкие дефекты складываются в высокий
    бугор (кластер), далёкие — в отдельные низкие. Цвет поверхности — по плотности (кластеры
    «горят»). Масштаб авто: пик плотности ≈ 40% диапазона яркости зон."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    zone = bg > 0
    Rl = relief[::step, ::step].astype(np.float32)
    Bg = bg[::step, ::step].astype(np.float32)
    Lab = labels[::step, ::step]
    Dn = density[::step, ::step].astype(np.float32)
    zd = zone[::step, ::step]
    xs = np.arange(0, relief.shape[1], step)
    ys = np.arange(0, relief.shape[0], step)

    # ВЫСОТА террейна = ЧИСТЫЙ рельеф яркости (НЕ добавляем плотность в высоту: бугры от
    # тёмных кластеров росли бы вверх и фальшиво протыкали верхнюю поверхность белых).
    # Плотность кластеров показываем ЦВЕТОМ поверхности — геометрия коридора остаётся честной.
    dmax = float(Dn[zd].max()) if zd.any() else 0.0
    Rl_z = np.where(zd, Rl, np.nan)            # террейн (вне зон — дыра)
    Dn_z = np.where(zd, Dn, np.nan)            # цвет = плотность дефектов (кластеры «горят»)

    # опорный фон для ДИ-поверхности: пиксельный B(x,y) или плоская медиана по зоне
    if flat_per_zone:
        Ref = np.zeros_like(Bg)
        for l in np.unique(Lab):
            if l == 0:
                continue
            sel = Lab == l
            Ref[sel] = float(np.median(Bg[sel]))
        mode_txt = "плоская по зоне (0.8·медиана B)"
    else:
        Ref = Bg
        mode_txt = "по фону B(x,y)"

    # уровни строгости ДИ: p% от 5 до 60, дефолт = P_MIN (20%)
    p_levels = np.linspace(5.0, 60.0, n_levels)
    default_idx = int(np.argmin(np.abs(p_levels - P_MIN)))

    def dark_di(p):                             # НИЖНЯЯ поверхность негодности Ref·(1−p/100)
        return np.where(zd, Ref * (1.0 - p / 100.0), np.nan)

    def white_di(p):                            # ВЕРХНЯЯ поверхность годности Ref·(1+p/100)
        return np.where(zd, Ref * (1.0 + p / 100.0), np.nan)

    def defect_pmap(p):                         # знаковое отклонение там, где вышли за коридор
        di_lo = Ref * (1.0 - p / 100.0)
        di_hi = Ref * (1.0 + p / 100.0)
        dev = (Rl - Ref) / np.maximum(Ref, 1e-6) * 100.0   # <0 темнее (тёмн.деф.), >0 ярче (белый)
        out = zd & ((Rl < di_lo) | (Rl > di_hi))
        return np.where(out, dev, np.nan)

    p0 = p_levels[default_idx]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=(f"{name}: рельеф яркости (цвет=плотность); коридор годности 0.8·B…1.2·B",
                        "Брак вне коридора (син=тёмный p≤−, красн=белый p≥+)"))

    # 0: террейн = ЧИСТЫЙ рельеф яркости, цвет = плотность дефектов (кластеры «горят»)
    fig.add_trace(go.Surface(z=Rl_z, x=xs, y=ys, surfacecolor=Dn_z, colorscale="Hot",
                             colorbar=dict(title="плотн.", x=0.52), name="яркость (цвет=плотн.)"),
                  row=1, col=1)
    # 1: НИЖНЯЯ поверхность негодности 0.8·B — тёмный дефект ныряет ПОД неё (p ≤ −порог)
    fig.add_trace(go.Surface(z=dark_di(p0), x=xs, y=ys, showscale=False, opacity=0.45,
                             colorscale=[[0, "red"], [1, "red"]], name="негодность тёмн. 0.8·B"),
                  row=1, col=1)
    # 2: ВЕРХНЯЯ поверхность негодности 1.2·B — белый дефект пробивает её СВЕРХУ (p ≥ +порог)
    fig.add_trace(go.Surface(z=white_di(p0), x=xs, y=ys, showscale=False, opacity=0.35,
                             colorscale=[[0, "cyan"], [1, "cyan"]], name="негодность бел. 1.2·B"),
                  row=1, col=1)
    # 3: карта знакового отклонения вне коридора
    fig.add_trace(go.Heatmap(z=defect_pmap(p0), x=xs, y=ys, colorscale="RdBu", reversescale=True,
                             zmin=-60, zmax=60, zmid=0, colorbar=dict(title="p%", x=1.0)),
                  row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    steps = []
    for p in p_levels:
        steps.append(dict(method="restyle",
                          args=[{"z": [dark_di(p), white_di(p), defect_pmap(p)]}, [1, 2, 3]],
                          label=f"{p:.0f}"))
    sliders = [dict(active=default_idx,
                    currentvalue={"prefix": "ДИ-строгость: брак при |p| ≥ ", "suffix": "%"},
                    pad={"t": 30}, steps=steps)]

    fig.update_layout(
        title=f"{name}: коридор годности B·(1±p%); выход вниз/вверх = брак (тёмн/белый) — {mode_txt}",
        sliders=sliders, width=1500, height=850, margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="яркость DN",
                   aspectratio=dict(x=1, y=1, z=0.5)))

    os.makedirs(out, exist_ok=True)
    suffix = "flat" if flat_per_zone else "bg"
    path = os.path.join(out, f"{name}_surface_di_{suffix}.html")
    fig.write_html(path, include_plotlyjs=True)
    if zone.any():
        print(f"[i] зон.пикс={int(zone.sum())}  B медиана по зонам="
              f"{float(np.median(bg[zone])):.0f}  коридор 0.8…1.2·B  "
              f"плотн.max={dmax:.2f} (цвет поверхности)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--flat", action="store_true", help="плоская ДИ-поверхность по зоне")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = os.path.basename(args.specimen.rstrip("/\\"))
    out = args.out or os.path.join("results", "v2_filter3d", name)
    relief, bg, labels, density = build_maps(args.specimen)
    path = make_html(relief, bg, labels, density, out, name, step=args.step,
                     flat_per_zone=args.flat)
    print(f"[ok] -> {path}")


if __name__ == "__main__":
    main()
