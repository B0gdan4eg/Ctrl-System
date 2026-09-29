"""
v2: калибровка кадров + выбор лучшей карты яркости на зону (design §4, §v2.1).

В папке детали ~8 кадров:
  Темновой           — bias + тёмновой ток + горячие пиксели (dark)
  Засветка/Галоген   — flat-field для галогенных полос
  Засветка/Диод      — flat-field для диодных полос
  5 спектральных полос (панхром/красный/зелёный/синий/ИК) — у каждой свои зоны ярки

Калибровка (убирает СИСТЕМАТИКУ — bias, тёмновой ток, горячие пиксели, виньетирование,
неравномерность, пыль; случайный шот-шум калибровкой НЕ убирается — design §4):
    C = (S − Dark) / (Flat − Dark) · median(Flat − Dark)
Flat выбирается под источник: галоген для панхром/красный/ИК, диод для зелёный/синий.

Лучшая полоса на зону: самая яркая БЕЗ насыщения (насыщение → потолок, дефект невидим,
кривой %). Каждая зона анализируется в своей лучшей калиброванной полосе.

Кириллические пути — через imread_u16/imwrite_u (cv2.imread их не читает на Windows).
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(__file__))
from surface_of_unfitness import imread_u16, imwrite_u  # noqa: E402

SAT = 4070      # порог насыщения (потолок сенсора ~4083)
TARGET = 2000.0  # целевая яркость зоны (12-бит): лучший SNR + линейность (design §3)
WINDOW = 250.0   # допуск окна достоверности 2000±250


# ----------------------------------------------------------- обнаружение кадров
def _find(spec: str, *keys: str, depth: int | None = None) -> str | None:
    """depth: если задан, принимаются только файлы на этой глубине от spec
    (число компонентов относительного пути). Защита от выбора папки-группы
    (например 'WTS…/мод б' вместо '…/мод б/№10Б'): без неё ключевое слово
    'панхром' матчится в пути ГЛУБЖЕ, и калибровка тихо смешала бы кадры
    разных деталей."""
    for f in glob.glob(os.path.join(spec, "**", "*.png"), recursive=True):
        low = f.lower()
        if all(k in low for k in keys):
            if depth is not None:
                rel = os.path.relpath(f, spec)
                if len(rel.split(os.sep)) != depth:
                    continue
            return f
    return None


def discover_frames(spec: str) -> dict:
    """Находит dark, два flat и 5 спектральных полос по названиям папок.

    Полосы и dark ищутся на глубине 2 (spec/<полоса>/файл.png, spec/Темновой/файл.png),
    flat — на глубине 3 (spec/Засветка/<источник>/файл.png)."""
    bands = {
        "панхром": ("панхром", "halogen"),
        "красный": ("красн", "halogen"),
        "ИК":      ("ик (",   "halogen"),
        "зелёный": ("зел",    "diode"),
        "синий":   ("син",    "diode"),
    }
    found = {"dark": _find(spec, "темнов", depth=2),
             "flat_halogen": _find(spec, "засветк", "галог", depth=3),
             "flat_diode": _find(spec, "засветк", "диод", depth=3),
             "bands": {}}
    for name, (kw, src) in bands.items():
        p = _find(spec, kw, depth=2)
        if p:
            found["bands"][name] = {"path": p, "source": src}
    return found


# ----------------------------------------------------------------- калибровка
def calibrate(S: np.ndarray, D: np.ndarray, F: np.ndarray) -> np.ndarray:
    """C = (S−D)/(F−D)·median(F−D). Возвращает float32 в DN-масштабе."""
    fd = np.maximum(F - D, 1.0)
    sd = S - D
    return np.clip(sd / fd * float(np.median(fd)), 0, None).astype(np.float32)


def load_calibrated(frames: dict) -> tuple[dict, dict]:
    """Возвращает (cal_bands, raw_bands) — калиброванные и сырые float32-кадры."""
    dark = imread_u16(frames["dark"]).astype(np.float32)
    flats = {}
    if frames["flat_halogen"]:
        flats["halogen"] = imread_u16(frames["flat_halogen"]).astype(np.float32)
    if frames["flat_diode"]:
        flats["diode"] = imread_u16(frames["flat_diode"]).astype(np.float32)

    cal, raw = {}, {}
    for name, meta in frames["bands"].items():
        S = imread_u16(meta["path"]).astype(np.float32)
        raw[name] = S
        F = flats.get(meta["source"])
        cal[name] = calibrate(S, dark, F) if F is not None else (S - dark)
    return cal, raw


# --------------------------------------------------------------- детекция зон
def _segments(on: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Границы непрерывных True-сегментов в 1D-маске (≥ min_len)."""
    d = np.diff(on.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def detect_zones(proj: np.ndarray, min_w: int = 40, thr_frac: float = 0.12,
                 smooth: int = 31) -> list[np.ndarray]:
    """Зоны = вертикальные полосы при фикс. X. Сначала профиль яркости по столбцам
    ловит ВСЕ полосы (включая тусклые — порог как доля от максимума, а не глобальный Otsu,
    который задирается яркими зонами и режет тусклые). Затем в каждом X-боксе локальный
    Otsu даёт точную форму. Контуры слева направо в координатах полного кадра.
    """
    H, W = proj.shape
    col = cv2.blur(np.median(proj, axis=0).reshape(1, -1), (1, smooth)).ravel()
    xsegs = _segments(col > col.max() * thr_frac, min_w)

    contours = []
    for x0, x1 in xsegs:
        row = cv2.blur(np.median(proj[:, x0:x1], axis=1).reshape(1, -1), (1, smooth)).ravel()
        ysegs = _segments(row > row.max() * thr_frac, min_w)
        if not ysegs:
            continue
        # крупнейший сегмент, а не объединение всех: если в этом X-окне окажется два
        # несвязанных ярких участка по Y (шум/чужая зона), union их bounding box'ов
        # склеил бы разные объекты в один — берём один, самый протяжённый.
        y0, y1 = max(ysegs, key=lambda seg: seg[1] - seg[0])
        box = proj[y0:y1, x0:x1]
        lo, hi = np.percentile(box, [1, 99])
        u8 = np.clip((box - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        _t, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        big = max(cs, key=cv2.contourArea) + np.array([[x0, y0]])  # сдвиг в полные координаты
        contours.append(big)
    return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


def zone_mask(shape, contour, erode: int = 9) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    if erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    return m.astype(bool)


def detect_zones_multiband(composite: np.ndarray, cal: dict, raw: dict,
                           min_w: int = 40, thr_frac: float = 0.25,
                           smooth: int = 31) -> list[np.ndarray]:
    """Мультиспектральная детекция зон — БЕЗ дыр, в отличие от detect_zones(proj).

    detect_zones ищет полосы по профилю столбцов КОМПОЗИТА с порогом 12% от максимума:
    панхром (~4000 DN) задаёт максимум, и тусклые полосы (зелёный ~450 DN) под порог
    не проходят — зоны теряются, 3D получается с дырками (найдено 4 из 6 на №16А).

    Здесь профиль столбцов считается ПОКАЖДЕНО в своей калиброванной полосе и
    нормируется на свой 99-й перцентиль: каждая полоса видна ~1.0 в своей полосе,
    независимо от абсолютной яркости. Видимость = max по полосам; порог thr_frac
    отсекает фон. Форма зоны — локальный Otsu в X-боксе по композиту (попиксельно
    лучшая ненасыщенная полоса, см. best_band_per_pixel). API совместим с
    detect_zones: список контуров слева направо в координатах полного кадра.
    """
    H, W = composite.shape
    vis = np.zeros(W, np.float32)
    for S in cal.values():
        prof = cv2.blur(np.median(S, axis=0).reshape(1, -1), (1, smooth)).ravel()
        p99 = float(np.percentile(prof, 99))
        if p99 > 0:
            vis = np.maximum(vis, prof / p99)
    xsegs = _segments(vis > thr_frac, min_w)

    contours = []
    for x0, x1 in xsegs:
        row = cv2.blur(np.median(composite[:, x0:x1], axis=1).reshape(1, -1), (1, smooth)).ravel()
        ysegs = _segments(row > row.max() * thr_frac, min_w)
        if not ysegs:
            continue
        # крупнейший Y-сегмент: union bbox'ов нескольких несвязанных участков
        # склеил бы разные объекты в один (см. detect_zones)
        y0, y1 = max(ysegs, key=lambda seg: seg[1] - seg[0])
        box = composite[y0:y1, x0:x1]
        lo, hi = np.percentile(box, [1, 99])
        u8 = np.clip((box - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        _t, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        big = max(cs, key=cv2.contourArea) + np.array([[x0, y0]])
        contours.append(big)
    return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


# -------------------------------------------------- выбор лучшей полосы на зону
def best_band(zmask: np.ndarray, raw: dict, sat_frac_max: float = 0.02) -> dict:
    """Лучшая полоса = медиана ближайшая к окну достоверности 2000±250 (design §3),
    среди НЕнасыщенных. Не максимум: и ниже окна (шум), и выше (нелинейность/насыщение) плохо.

    Если все насыщены — берём с минимальной долей насыщенных. in_window=False, если даже
    лучшая полоса вне 2000±250 (измерение менее достоверно — фикс. экспозиция данных).

    sat_frac_max=2% здесь — это порог ВЫБОРА полосы (предпочесть почти-идеальную полосу
    другой). Это НЕ то же самое, что порог "зона вообще неизмерима" (SAT_ZONE=10% в
    run_pipeline.py) — тот гейт финальный и терпимее: зона бракуется как "насыщ" только
    если даже её лучшая полоса пересвечена больше чем на 10%, а не 2%.
    """
    info = {}
    for name, S in raw.items():
        v = S[zmask]
        info[name] = {"median": float(np.median(v)),
                      "sat_frac": float((v >= SAT).mean())}
    ok = {n: d for n, d in info.items() if d["sat_frac"] <= sat_frac_max}
    pool = ok if ok else info
    best = min(pool, key=lambda n: abs(pool[n]["median"] - TARGET))
    in_window = abs(info[best]["median"] - TARGET) <= WINDOW
    return {"best": best, "in_window": in_window, "bands": info}


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="v2 калибровка + лучшая карта на зону")
    ap.add_argument("--specimen", required=True, help="папка детали (…/№N)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = discover_frames(args.specimen)
    name = os.path.basename(args.specimen.rstrip("/\\"))
    out = args.out or os.path.join("results", "v2_calib", name)
    os.makedirs(out, exist_ok=True)

    print(f"[i] dark: {bool(frames['dark'])}  flat_halogen: {bool(frames['flat_halogen'])}"
          f"  flat_diode: {bool(frames['flat_diode'])}  bands: {list(frames['bands'])}")

    cal, raw = load_calibrated(frames)
    proj = np.stack(list(cal.values())).max(0)
    zones = detect_zones(proj)
    print(f"[i] зон найдено: {len(zones)}")

    band_names = list(raw)
    print("\nzone   x    w  | " + " ".join("%9s" % b for b in band_names) + " | лучшая")
    composite = np.zeros(proj.shape, np.float32)
    report = []
    for i, c in enumerate(zones):
        x, y, w, h = cv2.boundingRect(c)
        zm = zone_mask(proj.shape, c)
        sel = best_band(zm, raw)
        meds = sel["bands"]
        row = "z%-2d %4d %4d | " % (i, x, w)
        for b in band_names:
            sat = "!" if meds[b]["sat_frac"] > 0.02 else " "
            star = "*" if b == sel["best"] else " "
            row += "%8.0f%s" % (meds[b]["median"], (star if star == "*" else sat))
        win = "в окне 2000±250" if sel["in_window"] else "ВНЕ окна (%.0f)" % meds[sel["best"]]["median"]
        row += " | %-8s %s" % (sel["best"], win)
        print(row)
        composite[zm] = cal[sel["best"]][zm]  # зона из своей лучшей калиброванной полосы
        report.append({"zone": i, "x": x, "w": w, "best": sel["best"],
                       "median_cal": float(np.median(cal[sel["best"]][zm]))})

    # сохраняем композит (8-бит для просмотра + 16-бит для анализа)
    lo, hi = np.percentile(composite[composite > 0], [1, 99]) if (composite > 0).any() else (0, 1)
    u8 = np.clip((composite - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    imwrite_u(os.path.join(out, f"{name}_composite.png"), u8)
    imwrite_u(os.path.join(out, f"{name}_composite16.png"),
              np.clip(composite, 0, 4095).astype(np.uint16))
    print(f"\n[ok] композит и таблица -> {out}")


if __name__ == "__main__":
    main()
