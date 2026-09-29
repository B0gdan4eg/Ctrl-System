"""
v2 карта сигнала ВСЕГО фильтра для вкладки «3D-карта» (PyVista).

Ключевые отличия от старого пути (surface_3d_interactive.build_maps + plotly):
  • карта НЕПРЕРЫВНА по всему кадру: фон между зонами — реальная тёмная
    долина, а не NaN-дыра; 3D читается как одна поверхность;
  • B(x,y) — непрерывный фон по композиту: вне зон nearest-fill от зон (тёмный фон
    кадра не занижает B у краёв полос), внутри зон — сепарабельный профиль полосы
    estimate_background_strip (P(x)+Q(y)), который НЕ провисает над широкими
    пятнами — коридор ДИ ровный, а ядро крупного дефекта остаётся дефектом;
  • p% — ЗНАКОВОЕ отклонение от фона: фон ~0 (плоское плато), тёмные дефекты
    < 0 (ямы), белые > 0 (пики) — дефекты видны на рельефе сразу;
  • зоны ищутся мультиспектрально (detect_zones_multiband): каждая полоса
    в своей калиброванной полосе, тусклые не теряются;
  • дефекты для маркеров считаются ПОЗОННО в родной полосе (process из
    surface_of_unfitness): max-композит мог бы спрятать дефект, тёмный в
    «родной» полосе зоны (см. whole_filter_3d.py, ВАЖНО).

Вердикт по бюджетам пикселей остаётся в run_pipeline.py — эта карта про
визуализацию (design §v2.3: рельеф = плато фон, ямы = тёмные, пики = белые).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from calibrate_multispectral import (  # noqa: E402
    discover_frames, load_calibrated, detect_zones_multiband, zone_mask, best_band,
)
from whole_filter_3d import best_band_per_pixel  # noqa: E402
from surface_of_unfitness import (  # noqa: E402
    process, estimate_background, estimate_background_strip, estimate_noise_model_pooled,
)

ERODE = 25        # эрозия маски зоны (срезает яркостный спад у границы), как в run_pipeline
SAT_ZONE = 0.10   # доля насыщенных, выше которой зона не измерима (как в run_pipeline)


def build_signal_maps(spec: str, k: float = 4.0, p_min: float = 20.0) -> dict:
    """Считает все карты для 3D-вкладки. Возвращает dict с numpy-массивами
    полного разрешения (H×W) и таблицей зон для лога:
      composite — сигнал: попиксельно лучшая ненасыщенная калиброванная полоса (DN)
      B         — непрерывный фон (DN)
      dev       — знаковое отклонение от фона, %: <0 тёмное, >0 белое (вне зон 0)
      union     — объединённая маска зон
      dark/white— маски дефектов (позонно в родной полосе, для маркеров)
      rows      — список зон (band/median/in_window/sat/dark_px/white_px) для лога
    """
    frames = discover_frames(spec)
    if not frames["bands"]:
        raise RuntimeError("в папке не найдено спектральных полос — это папка детали?")
    cal, raw = load_calibrated(frames)
    composite, satmap = best_band_per_pixel(cal, raw)
    H, W = composite.shape

    zones = detect_zones_multiband(composite, cal, raw)
    union = np.zeros((H, W), bool)
    zinfos = []
    for i, c in enumerate(zones):
        zm = zone_mask((H, W), c, erode=ERODE)
        union |= zm
        sel = best_band(zm, raw)
        zinfos.append(dict(i=i, zm=zm, sel=sel,
                           x=int(cv2.boundingRect(c)[0])))

    # --- непрерывный фон по композиту. ВНУТРИ зон — сепарабельный профиль полосы
    # estimate_background_strip: не провисает над широкими пятнами, поэтому коридор
    # ДИ ровный, а ядро крупного дефекта не сливается с фоном (образец мод в/№6,
    # пятно ~200px). ВНЕ зон — nearest-fill от ЗОНОВЫХ значений B, а не от композита:
    # nearest-fill композита (как в estimate_background) тащит сигнал края зоны в
    # промежуток — пятно у края зоны прогибает коридор и в зазорах между полосами.
    union_any = bool(union.any())
    if not union_any:                        # зоны не найдены — старый путь целиком
        B = estimate_background(composite)
    else:
        B = np.zeros_like(composite, np.float32)
        for zi in zinfos:
            B[zi["zm"]] = estimate_background_strip(composite, zi["zm"])[zi["zm"]]
        try:
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~union, return_distances=False,
                                         return_indices=True)
            B = B[tuple(idx)]
        except Exception:                    # fallback без scipy — общий уровень
            fill = float(np.median(B[union]))
            B = np.where(union, B, fill)

    # --- одна шумовая модель на деталь: пул тайлов всех зон разом (design §4).
    samples = []
    for zi in zinfos:
        band = zi["sel"]["best"]
        if zi["sel"]["bands"][band]["sat_frac"] <= SAT_ZONE:
            zi["Bz"] = estimate_background_strip(cal[band], zi["zm"])
            samples.append((cal[band], zi["zm"], zi["Bz"]))
    noise = estimate_noise_model_pooled(samples, dark_path=frames["dark"]) if samples else None

    # --- дефекты позонно в родной полосе (для маркеров в 3D) + таблица в лог.
    # strip_bg=True: фон зоны — профиль полосы, широкое пятно не тянет его за собой.
    dark_any = np.zeros((H, W), bool)
    white_any = np.zeros((H, W), bool)
    rows = []
    for zi in zinfos:
        band = zi["sel"]["best"]
        med = zi["sel"]["bands"][band]["median"]
        sat_frac = zi["sel"]["bands"][band]["sat_frac"]
        nd = nw = 0
        if noise is not None and sat_frac <= SAT_ZONE:
            r = process(cal[band], k=k, zone=zi["zm"], dark_path=frames["dark"],
                        p_min=p_min, noise=noise, strip_bg=True)
            dark_any |= r.dark_mask
            white_any |= r.white_mask
            nd, nw = int(r.dark_mask.sum()), int(r.white_mask.sum())
        rows.append(dict(zone=zi["i"], x=zi["x"], band=band, median=round(med),
                         in_window=zi["sel"]["in_window"], sat=round(sat_frac, 3),
                         dark_px=nd, white_px=nw))

    # --- знаковое отклонение p%: фон ~0, тёмный дефект < 0 (яма), белый > 0 (пик).
    # Вне зон оставляем 0 (долина фона не должна выглядеть «дефектом −100%»).
    Bs = np.maximum(B, 1e-3)
    dev = np.where(union, (composite - Bs) / Bs * 100.0, 0.0).astype(np.float32)

    return {"composite": composite, "B": B, "dev": dev, "union": union,
            "dark": dark_any, "white": white_any, "rows": rows, "satmap": satmap}
