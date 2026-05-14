"""
Лабораторный скрипт для разработки и сравнения алгоритмов поиска зон.

Запуск:
    python addition/zone_detection_lab.py --image path/to/image.tif --zones 5

Отображает side-by-side сравнение текущего алгоритма и экспериментального.
"""

import argparse
import time
import cv2
import numpy as np
import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.zone_detection import find_top_zones, find_top_zones_improved


# ===========================================================================
#  ЭКСПЕРИМЕНТАЛЬНЫЙ АЛГОРИТМ — сюда вносим правки
# ===========================================================================

def _detect_contrast_level(img: np.ndarray) -> str:
    """Определяет уровень контраста изображения."""
    p5, p95 = np.percentile(img, 5), np.percentile(img, 95)
    dynamic_range = p95 - p5
    mean_val = np.mean(img)

    if dynamic_range < 20:
        return "very_low"
    if dynamic_range < 50:
        return "low"
    if mean_val < 30 or mean_val > 225:
        return "extreme"
    if dynamic_range > 200:
        return "high"
    return "normal"


def _normalize_robust(img: np.ndarray) -> np.ndarray:
    """
    Робастная нормализация: обрезает выбросы (1-99 percentile),
    затем применяет гамма-коррекцию если изображение тёмное/светлое.
    """
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p99 <= p1:
        return img.copy()

    out = (img.astype(np.float32) - p1) / (p99 - p1)
    out = np.clip(out, 0, 1)

    # Гамма-коррекция: если среднее в нормализованном < 0.35 — осветляем,
    # если > 0.65 — затемняем, чтобы контраст зон был более выражен
    mean_norm = np.mean(out)
    if mean_norm < 0.35:
        gamma = 0.6  # осветление
    elif mean_norm > 0.65:
        gamma = 1.6  # затемнение
    else:
        gamma = 1.0

    if gamma != 1.0:
        out = np.power(out, gamma)

    return (out * 255).astype(np.uint8)


def _nms_contours(contours: list, iou_threshold: float = 0.3) -> list:
    """
    Non-Maximum Suppression для контуров по bbox IoU.
    Убирает перекрывающиеся дубли, оставляет больший.
    """
    if len(contours) <= 1:
        return contours

    boxes = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)
    areas = [cv2.contourArea(c) for c in contours]
    order = sorted(range(len(contours)), key=lambda i: areas[i], reverse=True)

    keep = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        x1, y1, w1, h1 = boxes[i]

        for j in order:
            if j in suppressed or j == i:
                continue
            x2, y2, w2, h2 = boxes[j]

            # Пересечение
            ix = max(x1, x2)
            iy = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)

            if ix2 <= ix or iy2 <= iy:
                continue  # нет пересечения

            inter = (ix2 - ix) * (iy2 - iy)
            union = w1 * h1 + w2 * h2 - inter
            iou = inter / union if union > 0 else 0

            if iou > iou_threshold:
                suppressed.add(j)

    return [contours[i] for i in keep]


def _score_candidates(candidates: list, top_n: int, img_w: int, img_h: int) -> float:
    """
    Оценивает набор кандидатов по трём критериям:
    1. count_score   — насколько близко кол-во зон к top_n
    2. uniformity    — равномерность площадей зон
    3. spread_score  — насколько зоны распределены по ширине изображения
                       (защита от рассыпания в один угол)
    """
    count = len(candidates)
    if count == 0:
        return 0.0

    areas = [cv2.contourArea(c) for c in candidates]
    count_score = count / top_n  # [0, 1], может быть > 1 если нашли больше

    uniformity = 1.0 - min(np.std(areas) / np.mean(areas), 1.0) if count > 1 else 1.0

    # Spread: берём центры X, смотрим насколько они покрывают ширину
    centers_x = []
    for c in candidates:
        x, y, w, h = cv2.boundingRect(c)
        centers_x.append(x + w / 2)
    if len(centers_x) > 1:
        span = (max(centers_x) - min(centers_x)) / img_w
        spread_score = min(span, 1.0)
    else:
        spread_score = 0.5

    return count_score * 0.5 + uniformity * 0.3 + spread_score * 0.2


def _find_zones_by_projection(
    img: np.ndarray,
    top_n: int,
    min_absolute_area: int,
    max_area_ratio: float,
    img_area: int,
    min_zone_width: int = 0,
) -> list:
    """
    Поиск зон через проекцию по столбцам.

    Ключевое свойство: оправа (металл, полностью непрозрачная) всегда
    ЗНАЧИТЕЛЬНО темнее любой зоны фильтра, даже самой тёмной.
    Поэтому порог задаётся как (min + fraction * range) — это надёжно
    отделяет оправу от зон вне зависимости от абсолютной яркости.

    Перебираем несколько значений fraction и берём то, при котором
    количество найденных сегментов ближе всего к top_n.
    """
    h, w = img.shape[:2]
    img_max = float(np.max(img)) if np.max(img) > 0 else 1.0

    # Среднее по каждому столбцу
    col_means = np.mean(img, axis=0).astype(np.float32)

    # Сглаживание — убираем шум, сохраняем границы зон
    sigma = max(5, w // 300)
    ksize = sigma * 6 + 1
    if ksize % 2 == 0:
        ksize += 1
    smoothed = cv2.GaussianBlur(col_means.reshape(1, -1), (ksize, 1), sigma)[0]

    prof_min = float(smoothed.min())
    prof_max = float(smoothed.max())
    if prof_max - prof_min < 1:
        return []

    def _segments_at_fraction(frac):
        """Возвращает список (x1, x2) для заданного fraction порога."""
        thr = prof_min + frac * (prof_max - prof_min)
        is_z = smoothed > thr
        segs = []
        in_z, sx = False, 0
        for i in range(w):
            if is_z[i] and not in_z:
                sx, in_z = i, True
            elif not is_z[i] and in_z:
                in_z = False
                if i - sx > 10:
                    segs.append((sx, i))
        if in_z and w - sx > 10:
            segs.append((sx, w))
        return segs

    # Ищем fraction при котором сегментов >= top_n
    # Начинаем с малых значений (чтобы захватить даже тёмные зоны)
    best_segments = []
    for frac in [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35]:
        segs = _segments_at_fraction(frac)
        if len(segs) >= top_n:
            best_segments = segs
            break
        if len(segs) > len(best_segments):
            best_segments = segs

    if not best_segments:
        return []

    # ── Gap-fill: локальный поиск в широких промежутках ──────────────────
    # Используется когда глобальный порог не нашёл нужное кол-во зон.
    # Позволяет обнаружить очень тёмные зоны (чёрный фильтр): их
    # col_means лишь на ~1-2% выше оправы, но ЛОКАЛЬНО внутри промежутка
    # контраст гораздо больше → фракция 0.4-0.7 от локального диапазона надёжно
    # отделяет зону от краёв оправы.
    if len(best_segments) < top_n and min_zone_width > 0:
        filled_sorted = sorted(best_segments)
        gap_list: list = []
        prev_end = 0
        for (sx1, sx2) in filled_sorted:
            if sx1 - prev_end >= min_zone_width * 2:
                gap_list.append((prev_end, sx1))
            prev_end = sx2
        if w - prev_end >= min_zone_width * 2:
            gap_list.append((prev_end, w))

        for (gx1, gx2) in gap_list:
            if len(best_segments) >= top_n:
                break
            local_prof = smoothed[gx1:gx2]
            loc_min = float(local_prof.min())
            loc_max = float(local_prof.max())
            loc_range = loc_max - loc_min
            if loc_range < 1.0:
                continue  # нет никакого контраста → не зона

            for loc_frac in [0.40, 0.50, 0.60, 0.70]:
                thr = loc_min + loc_frac * loc_range
                is_z = local_prof > thr
                in_z2, sx2 = False, 0
                new_segs: list = []
                for i in range(len(local_prof)):
                    if is_z[i] and not in_z2:
                        sx2, in_z2 = i, True
                    elif not is_z[i] and in_z2:
                        in_z2 = False
                        if i - sx2 >= min_zone_width:
                            new_segs.append((gx1 + sx2, gx1 + i))
                if in_z2 and len(local_prof) - sx2 >= min_zone_width:
                    new_segs.append((gx1 + sx2, gx2))

                if new_segs:
                    # Берём самый широкий сегмент из этого промежутка
                    best_seg = max(new_segs, key=lambda s: s[1] - s[0])
                    best_segments.append(best_seg)
                    break  # один сегмент на промежуток

    # Для каждого X-сегмента находим Y-границы через строчный профиль
    contours = []
    for (x1, x2) in best_segments:
        if min_zone_width > 0 and (x2 - x1) < min_zone_width:
            continue
        zone_col = img[:, x1:x2]
        row_means = np.mean(zone_col, axis=1)

        # Эталонная яркость — средняя треть по Y (тело зоны, не оправа)
        mid_y1, mid_y2 = h // 3, 2 * h // 3
        zone_brightness = float(np.mean(row_means[mid_y1:mid_y2]))

        # Пропускаем только абсолютно тёмные столбцы (= оправа)
        # Используем 1% от максимума изображения как порог "полного нуля"
        if zone_brightness < img_max * 0.01:
            continue

        # Y-порог: отрезаем края оправы
        # Берём 20% от яркости зоны, но не меньше 0.5% от максимума
        y_threshold = max(zone_brightness * 0.20, img_max * 0.005)
        y_mask = row_means > y_threshold
        ys = np.where(y_mask)[0]
        if len(ys) < 5:
            continue

        y1, y2 = int(ys[0]), int(ys[-1])
        zw, zh = x2 - x1, y2 - y1
        area = zw * zh

        if area < min_absolute_area or area > max_area_ratio * img_area:
            continue

        cnt = np.array(
            [[[x1, y1]], [[x2 - 1, y1]], [[x2 - 1, y2]], [[x1, y2]]],
            dtype=np.int32,
        )
        contours.append(cnt)

    # Берём top_n крупнейших, сортируем слева направо
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    contours = contours[:top_n]
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    return contours


def find_zones_experimental(
    img_gray_original: np.ndarray,
    top_n: int = 5,
    min_area_ratio: float = 0.3,
    max_area_ratio: float = 0.95,
    min_absolute_area: int = 5000,
    max_aspect_ratio: float = 20.0,
) -> list:
    """
    Экспериментальный алгоритм — два этапа:

    Этап 1 (проекция по столбцам):
        Не зависит от абсолютной яркости. Оправа всегда темнее любой зоны
        фильтра → адаптивный порог надёжно разделяет их.
        Перебирает fraction (2%…35%) пока не найдёт нужное кол-во сегментов.

    Этап 2 (CLAHE+threshold, fallback):
        Если проекция не дала top_n зон — пробуем CLAHE+threshold.
        Фиксы по сравнению с find_top_zones:
        - max_reasonable_zone_ratio: убирает слипшиеся blob'ы
        - min_absolute_area: отсекает мелкий мусор
        - max_aspect_ratio через minAreaRect
    """
    img_area = img_gray_original.shape[0] * img_gray_original.shape[1]

    # ── Этап 1: проекция ──────────────────────────────────────────────────
    # Адаптивная минимальная ширина зоны: ~1/35 ширины изображения
    min_zone_w = max(30, img_gray_original.shape[1] // 35)
    proj = _find_zones_by_projection(
        img_gray_original, top_n, min_absolute_area, max_area_ratio, img_area,
        min_zone_width=min_zone_w,
    )
    print(f"  [exp] projection: {len(proj)} зон")
    if len(proj) >= top_n:
        return proj

    # ── Этап 2: CLAHE+threshold ──────────────────────────────────────────
    print(f"  [exp] projection недостаточно, пробуем CLAHE+threshold...")

    in_min = np.percentile(img_gray_original, 1)
    in_max = np.percentile(img_gray_original, 99)
    img_norm = (img_gray_original.astype(np.float32) - in_min) / max(in_max - in_min, 1) * 255
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)

    # Разумный верхний лимит на размер одной зоны
    max_reasonable = min(max_area_ratio, 3.0 / top_n)

    from config.settings import CLIPL_LIMIT_VALUES, THRESHOLD_VALUES

    best_contours = list(proj)  # стартуем с тем что нашла проекция
    current_target = top_n

    while current_target >= 1 and len(best_contours) < current_target:
        for clipLimit in CLIPL_LIMIT_VALUES:
            for threshold_val in THRESHOLD_VALUES:
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_norm)

                _, mask = cv2.threshold(img_clahe, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                orig_max = float(img_gray_original.max()) if img_gray_original.max() > 0 else 255.0
                good = []
                for cnt in contours[:current_target * 3]:
                    area = cv2.contourArea(cnt)
                    if area / img_area > max_reasonable:
                        continue
                    if area < min_absolute_area:
                        continue
                    rect = cv2.minAreaRect(cnt)
                    rw, rh = rect[1]
                    if rw == 0 or rh == 0:
                        continue
                    if max(rw, rh) / min(rw, rh) > max_aspect_ratio:
                        continue
                    # Brightness filter: reject contours in completely dark areas
                    # (metal frame / background — not a filter zone)
                    cx_, cy_, cw_, ch_ = cv2.boundingRect(cnt)
                    roi_orig = img_gray_original[cy_:cy_+ch_, cx_:cx_+cw_]
                    if roi_orig.size > 0 and float(np.mean(roi_orig)) < orig_max * 0.03:
                        continue
                    if good:
                        if area < cv2.contourArea(good[0]) * min_area_ratio:
                            continue
                    good.append(cnt)

                # Не заменяем результат проекции на худший результат CLAHE;
                # только улучшаем если нашли больше зон
                if len(good) >= current_target:
                    best_contours = good[:current_target]
                    break
                if len(good) > len(best_contours):
                    best_contours = good[:current_target]

            if len(best_contours) >= current_target:
                break

        if len(best_contours) >= current_target:
            break
        current_target -= 1

    print(f"  [exp] итого: {len(best_contours)} зон")
    return sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])


# ===========================================================================
#  Утилиты визуализации
# ===========================================================================

COLORS = [
    (0, 255, 0),    # зелёный
    (0, 165, 255),  # оранжевый
    (255, 0, 255),  # пурпурный
    (0, 255, 255),  # голубой
    (255, 255, 0),  # жёлтый
]


def draw_zones(img_bgr: np.ndarray, contours: list, label: str, elapsed_ms: float,
               use_rotated_rect: bool = False) -> np.ndarray:
    out = img_bgr.copy()
    for i, cnt in enumerate(contours):
        color = COLORS[i % len(COLORS)]
        if use_rotated_rect:
            # Показываем повёрнутый прямоугольник (minAreaRect)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(out, [box], 0, color, 2)
            cx, cy = int(rect[0][0]), int(rect[0][1])
            angle = rect[2]
            cv2.putText(out, f"Z{i+1} {angle:.1f}°", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f"Z{i+1}", (x + 4, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Шапка
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(out, f"{label}  |  зон: {len(contours)}  |  {elapsed_ms:.0f} ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return out


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {path}")
    # Конвертируем в uint8 grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def resize_for_display(img: np.ndarray, max_w: int = 900, max_h: int = 700) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ===========================================================================
#  Печать метрик в консоль
# ===========================================================================

def print_metrics(label: str, contours: list, elapsed_ms: float):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Найдено зон : {len(contours)}")
    print(f"  Время       : {elapsed_ms:.1f} ms")
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        print(f"  Площади     : {[int(a) for a in areas]}")
        print(f"  Std/Mean    : {np.std(areas)/np.mean(areas):.3f}  (меньше = равномернее)")
        print("  Зоны (bbox):")
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"    Z{i+1}: x={x} y={y} w={w} h={h}  area={int(areas[i])}")


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Сравнение алгоритмов поиска зон")
    parser.add_argument("--image", required=True, help="Путь к изображению")
    parser.add_argument("--zones", type=int, default=5, help="Ожидаемое кол-во зон")
    parser.add_argument("--no-display", action="store_true", help="Только метрики, без окна")
    args = parser.parse_args()

    img_gray = load_image(args.image)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    top_n = args.zones

    # --- Текущий алгоритм (find_top_zones) ---
    t0 = time.perf_counter()
    contours_current = find_top_zones(img_gray, top_n=top_n)
    elapsed_current = (time.perf_counter() - t0) * 1000

    # --- Текущий improved (find_top_zones_improved) ---
    t0 = time.perf_counter()
    contours_improved = find_top_zones_improved(img_gray, top_n=top_n)
    elapsed_improved = (time.perf_counter() - t0) * 1000

    # --- Экспериментальный ---
    t0 = time.perf_counter()
    contours_exp = find_zones_experimental(img_gray, top_n=top_n)
    elapsed_exp = (time.perf_counter() - t0) * 1000

    print_metrics("find_top_zones (текущий)", contours_current, elapsed_current)
    print_metrics("find_top_zones_improved (текущий)", contours_improved, elapsed_improved)
    print_metrics("find_zones_experimental (новый)", contours_exp, elapsed_exp)

    if args.no_display:
        return

    # --- Визуализация ---
    vis_current  = draw_zones(img_bgr, contours_current,  "find_top_zones",         elapsed_current)
    vis_improved = draw_zones(img_bgr, contours_improved, "find_top_zones_improved", elapsed_improved)
    vis_exp      = draw_zones(img_bgr, contours_exp,      "experimental",            elapsed_exp,
                              use_rotated_rect=True)

    vis_current  = resize_for_display(vis_current)
    vis_improved = resize_for_display(vis_improved)
    vis_exp      = resize_for_display(vis_exp)

    # Выравниваем высоту перед конкатенацией
    h_max = max(vis_current.shape[0], vis_improved.shape[0], vis_exp.shape[0])
    def pad_h(img, h):
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        return img

    row = np.hstack([
        pad_h(vis_current, h_max),
        pad_h(vis_improved, h_max),
        pad_h(vis_exp, h_max),
    ])

    cv2.imshow("Zone Detection Lab  [Q — выход]", row)
    print("\nНажми Q или закрой окно для выхода")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(
                "Zone Detection Lab  [Q — выход]", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
