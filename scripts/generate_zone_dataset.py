"""
Генерация датасета для обучения U-Net сегментации зон фильтра.

Использует существующий классический алгоритм find_top_zones_improved
как weak supervision: прогоняет по корпусу изображений, генерирует бинарные
маски зон. Хорошие результаты идут в train/, спорные — в needs_review/
для ручной правки или отбраковки.

Запуск:
    python scripts/generate_zone_dataset.py --input-dir path/to/raw_images --output-dir path/to/dataset --top-n 5

Структура вывода:
    output_dir/
        images/         <-- исходные grayscale изображения (для обучения)
        masks/          <-- бинарные маски зон (255 = зона, 0 = фон)
        preview/        <-- визуализация для глазного контроля (image + mask overlay)
        needs_review/   <-- сюда уходят кейсы с подозрительной разметкой
        manifest.json   <-- статистика по каждому файлу
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.zone_detection import find_top_zones_improved


SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_grayscale(path: Path) -> np.ndarray | None:
    """Загружает grayscale-изображение и приводит к uint8 с percentile-стретчем
    (1-99%) — это вытягивает контраст для исходников в uint16/12-bit, у которых
    основная масса данных в нижней части диапазона."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img

    # Percentile-стретч для uint16/float — бьём по 1% и 99%, чтобы выбросы
    # не съели весь диапазон при конвертации в 8 бит.
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p99 <= p1:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    out = (img.astype(np.float32) - p1) / (p99 - p1) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def refine_zone_by_otsu(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Сужает контур зоны до фактически яркой части, отрезая попавшую
    в детекцию тёмную рамку фильтра.

    Внутри расширенного bbox контура делает Otsu-thresholding по реальной
    яркости пикселей (uint8 image). Otsu автоматически находит границу
    между bright filter zone и dark frame. Берётся самый крупный связный
    компонент яркой части — это и есть настоящая зона.

    Если что-то идёт не так — возвращает исходный контур (graceful fallback).
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 10:
        return contour

    # Расширяем bbox на 8% — Otsu увидит и тёмный фон тоже,
    # без этого может не сработать на узких полосах.
    pad = max(int(min(w, h) * 0.08), 6)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return contour

    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return contour
    # Пропускаем background (label 0), ищем самый большой яркий компонент
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    refined_local = (labels == largest).astype(np.uint8) * 255

    refined_contours, _ = cv2.findContours(refined_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not refined_contours:
        return contour
    refined = max(refined_contours, key=cv2.contourArea)

    # Sanity check: уточнённая зона не должна стать сильно меньше исходной —
    # иначе Otsu, похоже, разделил неправильно (например, на бликах).
    refined_area = cv2.contourArea(refined)
    original_area = cv2.contourArea(contour)
    if original_area > 0 and refined_area < original_area * 0.4:
        return contour

    # Переводим координаты из локального crop'а обратно в систему image
    refined = refined.copy()
    refined[:, :, 0] += x0
    refined[:, :, 1] += y0
    return refined


def contours_to_mask(contours, shape, smooth_kernel: int = 9) -> tuple[np.ndarray, list]:
    """Заливает контуры в бинарную маску и аккуратно сглаживает 'укусы'
    по краям без расширения за реальные границы зоны:
      1. closing небольшим ядром — затягивает рваные мелкие выемки
      2. opening тем же ядром — убирает торчащие шумовые отростки
      3. полигональная аппроксимация (approxPolyDP) — финальное сглаживание
         без расширения формы, как convex hull сделал бы

    Возвращает (mask, smoothed_contours) — второе для отрисовки в preview,
    чтобы зелёная заливка и контур совпадали.
    smooth_kernel = 0 отключает пост-обработку.
    """
    raw_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(raw_mask, contours, -1, 255, thickness=cv2.FILLED)

    if smooth_kernel <= 0:
        return raw_mask, list(contours)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed: list = []
    final_mask = np.zeros(shape, dtype=np.uint8)
    for cnt in found:
        if cv2.contourArea(cnt) < 100:
            continue
        # epsilon ~0.3% от длины периметра — лёгкое сглаживание шумных зубцов,
        # форма зоны сохраняется, в отличие от convex hull.
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        smoothed.append(approx)
        cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)

    return final_mask, smoothed


def make_preview(image: np.ndarray, mask: np.ndarray, contours) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)
    return blended


def evaluate_quality(contours, img_shape, top_n: int, min_zones: int) -> tuple[bool, str]:
    """Эвристика: считаем разметку 'хорошей' если найдено зон в диапазоне
    [min_zones, top_n] с разумным разбросом площадей. Иначе — на ручной просмотр."""
    if len(contours) == 0:
        return False, "no_zones_found"
    if not (min_zones <= len(contours) <= top_n):
        return False, f"zone_count_{len(contours)}_outside_[{min_zones},{top_n}]"

    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float64)
    if areas.min() <= 0:
        return False, "zero_area_contour"

    # Коэффициент вариации площадей: если зоны очень разные — подозрительно
    cv_ratio = areas.std() / areas.mean()
    if cv_ratio > 0.5:
        return False, f"high_area_variance_cv_{cv_ratio:.2f}"

    img_area = img_shape[0] * img_shape[1]
    coverage = areas.sum() / img_area
    if coverage < 0.05 or coverage > 0.95:
        return False, f"unreasonable_coverage_{coverage:.2f}"

    return True, "ok"


def process_one(
    image_path: Path,
    out_images: Path,
    out_masks: Path,
    out_preview: Path,
    out_review: Path,
    top_n: int,
    min_zones: int,
    smooth_kernel: int,
) -> dict:
    record = {
        'file': image_path.name,
        'status': None,
        'reason': None,
        'zones_found': 0,
    }

    image = load_grayscale(image_path)
    if image is None:
        record['status'] = 'load_failed'
        record['reason'] = 'cv2.imread returned None'
        return record

    contours = find_top_zones_improved(
        image,
        top_n=top_n,
        min_absolute_area=20000,
        max_aspect_ratio=20.0,
    )
    # Уточняем каждую зону Otsu-порогом — отрезаем тёмную рамку фильтра,
    # которую find_top_zones_improved частично захватывает.
    contours = [refine_zone_by_otsu(image, c) for c in contours]
    record['zones_found'] = len(contours)

    is_good, reason = evaluate_quality(contours, image.shape, top_n, min_zones)
    record['reason'] = reason

    mask, smoothed_contours = contours_to_mask(contours, image.shape, smooth_kernel=smooth_kernel)
    # В preview рисуем итоговые сглаженные контуры — чтобы зелёная заливка
    # и красная граница совпадали и не вводили в заблуждение.
    preview = make_preview(image, mask, smoothed_contours)

    base = image_path.stem
    if is_good:
        cv2.imwrite(str(out_images / f"{base}.png"), image)
        cv2.imwrite(str(out_masks / f"{base}.png"), mask)
        cv2.imwrite(str(out_preview / f"{base}.png"), preview)
        record['status'] = 'auto_labeled'
    else:
        review_dir = out_review / base
        review_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(review_dir / "image.png"), image)
        cv2.imwrite(str(review_dir / "mask_auto.png"), mask)
        cv2.imwrite(str(review_dir / "preview.png"), preview)
        record['status'] = 'needs_review'

    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Папка с исходными изображениями фильтров")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Куда складывать датасет")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Ожидаемое максимальное количество зон на изображении")
    parser.add_argument("--min-zones", type=int, default=4,
                        help="Минимум зон для приёмки (если < top_n — допускаем "
                             "снимки с отсутствующей крайней зоной)")
    parser.add_argument("--smooth-kernel", type=int, default=9,
                        help="Размер ядра для closing/opening краёв маски (0 = выкл)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Обработать только первые N файлов (0 = все)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input dir not found: {args.input_dir}")
        sys.exit(1)

    out_images = args.output_dir / "images"
    out_masks = args.output_dir / "masks"
    out_preview = args.output_dir / "preview"
    out_review = args.output_dir / "needs_review"
    for d in (out_images, out_masks, out_preview, out_review):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if args.limit > 0:
        files = files[:args.limit]

    print(f"Найдено изображений: {len(files)}")
    print(f"Ожидаемое число зон на снимок: {args.top_n}")
    print(f"Вывод: {args.output_dir.resolve()}")
    print("-" * 60)

    records = []
    counts = {'auto_labeled': 0, 'needs_review': 0, 'load_failed': 0}

    for i, fp in enumerate(files, 1):
        rec = process_one(fp, out_images, out_masks, out_preview, out_review,
                          args.top_n, args.min_zones, args.smooth_kernel)
        records.append(rec)
        counts[rec['status']] = counts.get(rec['status'], 0) + 1
        marker = {'auto_labeled': '+', 'needs_review': '?', 'load_failed': 'x'}.get(rec['status'], '.')
        print(f"  [{i:>4}/{len(files)}] {marker} {fp.name:<60} zones={rec['zones_found']} reason={rec['reason']}")

    manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'input_dir': str(args.input_dir.resolve()),
        'top_n_expected': args.top_n,
        'total': len(records),
        'counts': counts,
        'records': records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print("-" * 60)
    print(f"Готово. auto_labeled={counts.get('auto_labeled', 0)}  "
          f"needs_review={counts.get('needs_review', 0)}  "
          f"load_failed={counts.get('load_failed', 0)}")
    print(f"Манифест: {manifest_path}")
    print()
    print("Дальше:")
    print(f"  1. Просмотри {out_preview}/ глазами — выкини мусор в needs_review/")
    print(f"  2. По needs_review/ — поправь маску в редакторе или отбракуй")
    print(f"  3. Когда готово — обучи U-Net на images/ + masks/")


if __name__ == "__main__":
    main()
