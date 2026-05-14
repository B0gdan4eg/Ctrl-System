"""
Стенд для разработки и доводки алгоритма детекции зон.

Запуск:
    python addition/zone_detection_dev.py --image path/to/image.png
    python addition/zone_detection_dev.py --image path/to/image.png --save

Окна:
  1. Zone detection: OLD vs NEW  — сравнение алгоритмов поиска зон
  2. ROI trim (OLD zones)         — обрезка тёмных краёв на зонах find_top_zones
  3. ROI trim (NEW zones)         — обрезка тёмных краёв на зонах find_top_zones_improved
  4. Letterbox padding            — чёрный фон vs зеркальное отражение
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.zone_detection import find_top_zones, find_top_zones_improved


# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ ДЛЯ ПОДБОРА — меняй здесь
# ---------------------------------------------------------------------------

PARAMS = {
    # --- find_top_zones_improved ---
    'num_zones':         5,
    'min_area_ratio':    0.3,
    'max_area_ratio':    0.95,
    'min_absolute_area': 20000,
    'max_aspect_ratio':  20.0,
    'brightness':        0,
    'contrast':          1.0,

    # --- trim_roi_to_bright_area ---
    'trim_dark_percentile':  30.0,   # 0-100: чем выше, тем агрессивнее обрезка
    'trim_min_bright_ratio': 0.3,    # 0-1: мин. доля светлых пикселей в строке/столбце
    'trim_min_size_ratio':   0.5,    # 0-1: защита от чрезмерной обрезки

    # --- letterbox ---
    'use_reflect_padding': True,
}

# ---------------------------------------------------------------------------


def trim_roi_to_bright_area(roi, dark_percentile=30.0, min_bright_ratio=0.3,
                             min_size_ratio=0.5):
    """Обрезает тёмные края ROI (оправа/маска фильтра)."""
    h, w = roi.shape[:2]
    no_trim = (roi, (0, 0, w, h))

    if h < 10 or w < 10:
        return no_trim

    img_u8 = (roi / 1023.0 * 255.0).astype(np.uint8) if roi.dtype == np.uint16 else roi

    threshold = float(np.percentile(img_u8, dark_percentile))
    bright = (img_u8 > threshold).astype(np.float32)

    row_bright = bright.mean(axis=1)
    col_bright = bright.mean(axis=0)

    bright_rows = np.where(row_bright >= min_bright_ratio)[0]
    bright_cols = np.where(col_bright >= min_bright_ratio)[0]

    if len(bright_rows) == 0 or len(bright_cols) == 0:
        return no_trim

    y1, y2 = int(bright_rows[0]), int(bright_rows[-1]) + 1
    x1, x2 = int(bright_cols[0]), int(bright_cols[-1]) + 1
    crop_h, crop_w = y2 - y1, x2 - x1

    if crop_h < h * min_size_ratio or crop_w < w * min_size_ratio:
        return no_trim
    if x1 <= 1 and y1 <= 1 and (w - x2) <= 1 and (h - y2) <= 1:
        return no_trim

    return roi[y1:y2, x1:x2], (x1, y1, crop_w, crop_h)


def letterbox_resize(image, target_size=(512, 512), use_reflect=True):
    tw, th = target_size
    sh, sw = image.shape[:2]
    scale = min(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    t, b = pad_y, th - nh - pad_y
    l, r = pad_x, tw - nw - pad_x
    if t > 0 or b > 0 or l > 0 or r > 0:
        border = cv2.BORDER_REFLECT_101 if use_reflect else cv2.BORDER_CONSTANT
        canvas = cv2.copyMakeBorder(resized, t, b, l, r, border, value=0)
    else:
        canvas = resized
    return canvas, {'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y,
                    'new_w': nw, 'new_h': nh, 'orig_w': sw, 'orig_h': sh}


def to_u8(img):
    if img.dtype == np.uint16:
        return (img / 1023.0 * 255.0).astype(np.uint8)
    return img.copy()


def add_header(img, text, h=26):
    header = np.full((h, img.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(header, text, (6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([header, img])


def pad_to_height(img, target_h):
    if img.shape[0] < target_h:
        pad = np.full((target_h - img.shape[0], img.shape[1], 3), 25, dtype=np.uint8)
        return np.vstack([img, pad])
    return img


def draw_zones(image_bgr, contours, color, label_prefix=''):
    out = image_bgr.copy()
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        area_pct = cv2.contourArea(cnt) / (image_bgr.shape[0] * image_bgr.shape[1]) * 100
        label = f'{label_prefix}{i + 1} {w}x{h} {area_pct:.0f}%'
        cv2.putText(out, label, (x + 3, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def make_trim_panel(img_gray, contours, p, label=''):
    """Панель: для каждой зоны — orig ROI | trimmed ROI с подписями."""
    disp_h = 180
    panels = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img_gray[y:y + h, x:x + w].copy()

        trimmed, (tx, ty, tw, th) = trim_roi_to_bright_area(
            roi,
            dark_percentile=p['trim_dark_percentile'],
            min_bright_ratio=p['trim_min_bright_ratio'],
            min_size_ratio=p['trim_min_size_ratio'],
        )

        def thumb(img, dh):
            u8 = to_u8(img)
            sc = dh / max(u8.shape[0], 1)
            rw = max(1, int(u8.shape[1] * sc))
            return cv2.cvtColor(cv2.resize(u8, (rw, dh)), cv2.COLOR_GRAY2BGR)

        orig_t = thumb(roi, disp_h)
        trim_t = thumb(trimmed, disp_h)

        # Подписи
        cut_x = tx if tx > 1 else 0
        cut_y = ty if ty > 1 else 0
        cut_r = (w - tx - tw) if (w - tx - tw) > 1 else 0
        cut_b = (h - ty - th) if (h - ty - th) > 1 else 0

        cv2.putText(orig_t, f'Z{i+1} {w}x{h}', (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 200, 255), 1)
        cv2.putText(trim_t, f'Z{i+1} {tw}x{th}', (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 120), 1)
        cv2.putText(trim_t, f'-L{cut_x} -R{cut_r} -T{cut_y} -B{cut_b}', (3, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 120), 1)

        sep = np.full((disp_h, 3, 3), 70, dtype=np.uint8)
        panels.append(np.hstack([orig_t, sep, trim_t]))

    if not panels:
        blank = np.full((disp_h, 300, 3), 40, dtype=np.uint8)
        cv2.putText(blank, 'Zones not found', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 200), 1)
        return add_header(blank, label)

    # Приводим все к одной ширине
    max_w = max(p.shape[1] for p in panels)
    rows = []
    for pi in panels:
        if pi.shape[1] < max_w:
            pad = np.full((pi.shape[0], max_w - pi.shape[1], 3), 30, dtype=np.uint8)
            pi = np.hstack([pi, pad])
        rows.append(pi)
        rows.append(np.full((3, max_w, 3), 50, dtype=np.uint8))

    panel = np.vstack(rows[:-1])
    return add_header(panel, label)


def make_letterbox_panel(img_gray, contours):
    """Сравнивает letterbox с чёрным фоном и с отражением (первые 3 зоны)."""
    size = 180
    panels = []
    for i, cnt in enumerate(contours[:3]):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = to_u8(img_gray[y:y + h, x:x + w])

        black, _ = letterbox_resize(roi, (size, size), use_reflect=False)
        refl, _  = letterbox_resize(roi, (size, size), use_reflect=True)

        bk = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
        rf = cv2.cvtColor(refl,  cv2.COLOR_GRAY2BGR)

        cv2.putText(bk, f'Z{i+1} black', (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 100, 255), 1)
        cv2.putText(rf, f'Z{i+1} reflect', (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 120), 1)

        sep = np.full((size, 3, 3), 70, dtype=np.uint8)
        panels.append(np.hstack([bk, sep, rf]))

    if not panels:
        return add_header(np.full((180, 400, 3), 40, dtype=np.uint8),
                          'Letterbox: нет зон')

    max_w = max(p.shape[1] for p in panels)
    rows = []
    for pi in panels:
        if pi.shape[1] < max_w:
            pi = np.hstack([pi, np.full((pi.shape[0], max_w - pi.shape[1], 3), 30, dtype=np.uint8)])
        rows.append(pi)
        rows.append(np.full((3, max_w, 3), 50, dtype=np.uint8))

    return add_header(np.vstack(rows[:-1]),
                      'Letterbox: чёрный фон (слева) vs зеркальное отражение (справа)')


def print_zone_stats(label, contours, img_shape):
    img_area = img_shape[0] * img_shape[1]
    print(f'\n  [{label}] Найдено зон: {len(contours)}')
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = max(w, h) / max(min(w, h), 1)
        coverage = area / img_area * 100
        print(f'    Z{i+1}: pos=({x},{y}) size={w}x{h}  '
              f'area={int(area)}  aspect={aspect:.1f}  coverage={coverage:.1f}%')


def run(image_path, save_output=False, output_dir=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'Ошибка: не удалось загрузить {image_path}')
        sys.exit(1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    display_bgr = cv2.cvtColor(to_u8(img_gray), cv2.COLOR_GRAY2BGR)

    p = PARAMS
    print(f'\nИзображение: {image_path}  ({img_gray.shape[1]}x{img_gray.shape[0]}, {img_gray.dtype})')

    # --- Поиск зон ---
    contours_old = find_top_zones(
        img_gray,
        top_n=p['num_zones'],
        min_area_ratio=p['min_area_ratio'],
        max_area_ratio=p['max_area_ratio'],
    )
    print_zone_stats('find_top_zones (текущий)', contours_old, img_gray.shape)

    contours_new = find_top_zones_improved(
        img_gray,
        top_n=p['num_zones'],
        min_area_ratio=p['min_area_ratio'],
        max_area_ratio=p['max_area_ratio'],
        min_absolute_area=p['min_absolute_area'],
        max_aspect_ratio=p['max_aspect_ratio'],
        brightness=p['brightness'],
        contrast=p['contrast'],
    )
    print_zone_stats('find_top_zones_improved', contours_new, img_gray.shape)

    # --- Окно 1: сравнение алгоритмов ---
    vis_old = draw_zones(display_bgr, contours_old, (0, 200, 255), 'OLD-')
    vis_new = draw_zones(display_bgr, contours_new, (0, 255, 100), 'NEW-')

    max_h = 640
    scale = min(max_h / max(vis_old.shape[0], 1), 1.0)
    def rs(img):
        return cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                max(1, int(img.shape[0] * scale))))

    vis_old_s = add_header(rs(vis_old),
                           f'find_top_zones (OLD) — жёлтый: {len(contours_old)} зон')
    vis_new_s = add_header(rs(vis_new),
                           f'find_top_zones_improved (NEW) — зелёный: {len(contours_new)} зон')

    h_max = max(vis_old_s.shape[0], vis_new_s.shape[0])
    sep = np.full((h_max, 4, 3), 80, dtype=np.uint8)
    win1 = np.hstack([pad_to_height(vis_old_s, h_max), sep,
                      pad_to_height(vis_new_s, h_max)])

    # --- Окна 2 и 3: trim ---
    win2 = make_trim_panel(img_gray, contours_old, p,
                           label=f'ROI trim (OLD zones, {len(contours_old)} зон): orig | trimmed')
    win3 = make_trim_panel(img_gray, contours_new, p,
                           label=f'ROI trim (NEW zones, {len(contours_new)} зон): orig | trimmed')

    # --- Окно 4: letterbox ---
    # Показываем на зонах OLD (они корректные)
    lb_contours = contours_old if contours_old else contours_new
    win4 = make_letterbox_panel(img_gray, lb_contours)

    # Показываем
    cv2.imshow('Zone detection: OLD vs NEW', win1)
    cv2.imshow('ROI trim (OLD zones)', win2)
    cv2.imshow('ROI trim (NEW zones)', win3)
    cv2.imshow('Letterbox padding', win4)

    # Позиционируем окна чтобы не перекрывались
    cv2.moveWindow('Zone detection: OLD vs NEW', 0, 0)
    cv2.moveWindow('ROI trim (OLD zones)', 0, 700)
    cv2.moveWindow('ROI trim (NEW zones)', 700, 700)
    cv2.moveWindow('Letterbox padding', 1400, 700)

    if save_output:
        out_dir = Path(output_dir) if output_dir else Path(image_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(out_dir / f'{stem}_zones.png'), win1)
        cv2.imwrite(str(out_dir / f'{stem}_trim_old.png'), win2)
        cv2.imwrite(str(out_dir / f'{stem}_trim_new.png'), win3)
        cv2.imwrite(str(out_dir / f'{stem}_letterbox.png'), win4)
        print(f'\nСохранено в: {out_dir}')

    print('\nНажми любую клавишу для выхода...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Стенд разработки алгоритма детекции зон')
    parser.add_argument('--image', required=True, help='Путь к изображению')
    parser.add_argument('--save', action='store_true', help='Сохранить результаты')
    parser.add_argument('--output-dir', help='Папка для сохранения')
    args = parser.parse_args()
    run(args.image, save_output=args.save, output_dir=args.output_dir)
