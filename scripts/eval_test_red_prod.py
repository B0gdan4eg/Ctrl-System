"""Production-style evaluation на отложенном test split.

Повторяет pipeline `ProcessingWorker.run()` 1-в-1: фазы A (подготовка зон) →
B (два батч-forward'а: sliding + letterbox для всех зон сразу) → C
(refine + размещение маски).

Test split — это zone-кропы 2048×2048, которые уже выделены extract_red_zones.py.
ZoneSegmenter здесь не запускается: каждый image трактуется как одна "зона",
покрывающая весь кадр.

`max_ring_width=80` в refine — обязателен для 2048×2048: без cap _intensity_shrink
на catch-all компоненте делает dilate с эллипсом ~2400×2400 → 100+ сек/кадр.
В проде на маленькой зоне (230×1932) этого cap не нужно — компоненты компактные.
"""
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from core import (
    apply_histogram_normalization,
    predict_defects_batch_probs,
    refine_defect_mask,
    prepare_zone_tiles,
    stitch_tile_probs,
    resize_with_letterbox,
)
from config.settings import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_MODEL_SIZE,
)

TEST_DIR = Path(r'D:/train/prepared_red_2048/dataset/test')
MODEL_PATH = ROOT / 'best_model.pth'

# Те же параметры что в UI по умолчанию
THRESHOLD_LOW = float(DEFAULT_DETECTION_THRESHOLD)   # 0.01
THRESHOLD_HIGH = max(THRESHOLD_LOW, 0.7)
MODEL_SIZE = int(DEFAULT_MODEL_SIZE)                  # 256
# extract_red_zones.py уже применил apply_histogram_normalization(std_range=5)
# к этим zone-кропам. Повторная нормализация (std_range=2.0 и 3.0) поверх
# клипает в стенку → catch-all probs → refine не справляется. Поэтому здесь False.
APPLY_HIST_NORM = False

# Размер вырезаемых полос (в пикселях). Прод-зона ~230×1932 (аспект 1:8.4).
# Режем каждый 2048×2048 кадр на N_STRIPS вертикальных полос STRIP_WIDTH×2048,
# чтобы pipeline (особенно refine) работал на узкой ROI как в проде.
STRIP_WIDTH = 256
N_STRIPS = 2048 // STRIP_WIDTH  # = 8

# Те же контрасты что в ProcessingWorker.HIST_CONTRASTS
HIST_CONTRASTS = (2.0, 3.0)

# Защита от 100+ сек/кадр на 2048×2048 catch-all компонентах
MAX_RING_WIDTH = 80


def load_model(device):
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f'  model: epoch={ckpt.get("epoch")}, val_iou={ckpt.get("val_iou")}')
    return model


def _restore_probs_letterbox(probs, pad_info):
    pad_x = pad_info['pad_x']; pad_y = pad_info['pad_y']
    new_w = pad_info['new_w']; new_h = pad_info['new_h']
    orig_w = pad_info['orig_w']; orig_h = pad_info['orig_h']
    cropped = probs[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def prepare_zone(roi_orig: np.ndarray, small_size: int, apply_hist_norm: bool) -> dict:
    """Фаза A из ProcessingWorker._prepare_zone. ROI = весь image."""
    H, W = roi_orig.shape[:2]
    info = {
        'roi_shape': (H, W),
        'variants': [],
        'too_small': roi_orig.size < 100,
    }
    if info['too_small']:
        return info

    contrasts = list(HIST_CONTRASTS) if apply_hist_norm else [None]
    for c in contrasts:
        roi_c = apply_histogram_normalization(roi_orig, std_range=c) if c is not None else roi_orig.copy()

        tile_info = prepare_zone_tiles(roi_c, tile_size=small_size, min_overlap=small_size // 4)
        roi_gray = tile_info['roi_gray']
        large_canvas, large_pad = resize_with_letterbox(roi_gray, (small_size, small_size))
        tile_info_large = {
            'mode': 'letterbox',
            'tiles': [large_canvas],
            'roi_shape': tile_info['roi_shape'],
            'pad_info': large_pad,
            'tile_origins': None,
            'roi_gray': roi_gray,
        }
        info['variants'].append({
            'contrast': c,
            'tile_info': tile_info,
            'tile_info_large': tile_info_large,
        })
    return info


def _pipeline_on_strip(model, device, roi_gray: np.ndarray, small_size: int) -> np.ndarray:
    """Прогон pipeline на одной полосе (имитация прод-зоны). Возвращает бин. маску."""
    H, W = roi_gray.shape

    p = prepare_zone(roi_gray, small_size=small_size, apply_hist_norm=APPLY_HIST_NORM)
    if p['too_small'] or not p['variants']:
        return np.zeros((H, W), dtype=np.uint8)

    small_tiles = []
    large_tiles = []
    slices_small = []
    slices_large = []
    for v in p['variants']:
        ss = len(small_tiles)
        small_tiles.extend(v['tile_info']['tiles'])
        slices_small.append((ss, len(small_tiles)))
        ls = len(large_tiles)
        large_tiles.extend(v['tile_info_large']['tiles'])
        slices_large.append((ls, len(large_tiles)))

    small_probs_list = predict_defects_batch_probs(
        model, small_tiles, device, tta=False, chunk_size=64,
    )
    large_probs_list = predict_defects_batch_probs(
        model, large_tiles, device, tta=False, chunk_size=64,
    )

    probs_per_variant = []
    guide = None
    for vi, v in enumerate(p['variants']):
        ti = v['tile_info']
        ti_l = v['tile_info_large']
        ss, se = slices_small[vi]
        ls, _le = slices_large[vi]

        if ti['mode'] == 'letterbox':
            ps_full = _restore_probs_letterbox(small_probs_list[ss], ti['pad_info'])
        else:
            ps_full = stitch_tile_probs(
                small_probs_list[ss:se], ti['tile_origins'], (H, W), tile_size=small_size,
            )
        pl_full = _restore_probs_letterbox(large_probs_list[ls], ti_l['pad_info'])

        probs_per_variant.append(ps_full)
        probs_per_variant.append(pl_full)
        guide = ti['roi_gray']

    fused = np.maximum.reduce(probs_per_variant)
    return refine_defect_mask(
        fused, guide,
        threshold_low=THRESHOLD_LOW,
        threshold_high=THRESHOLD_HIGH,
        max_ring_width=MAX_RING_WIDTH,
    )


def process_image(model, device, img_bgr: np.ndarray, small_size: int) -> np.ndarray:
    """Полный pipeline на одном image: режем на N_STRIPS вертикальных полос
    STRIP_WIDTH×2048 (имитация узкой прод-зоны), прогоняем каждую отдельно,
    собираем pred mask обратно в (H, W)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    full_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(N_STRIPS):
        x0 = i * STRIP_WIDTH
        x1 = min(W, x0 + STRIP_WIDTH)
        strip = gray[:, x0:x1]
        strip_mask = _pipeline_on_strip(model, device, strip, small_size=small_size)
        full_mask[:, x0:x1] = strip_mask

    return full_mask


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model:  {MODEL_PATH}')
    print(f'Params: threshold_low={THRESHOLD_LOW}, model_size={MODEL_SIZE}, '
          f'hist_norm={APPLY_HIST_NORM}, contrasts={HIST_CONTRASTS}')
    model = load_model(device)

    images = sorted((TEST_DIR / 'images').glob('*.png'))
    print(f'Test images: {len(images)}')

    # Скалярные аккумуляторы
    tp = fp = fn = 0
    img_tp = img_fp = img_fn = img_tn = 0
    per_iou_list = []

    pbar = tqdm(images, desc='Eval', unit='img', ncols=100)
    for i, img_path in enumerate(pbar):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        mask_gt = cv2.imread(str(TEST_DIR / 'masks' / img_path.name), cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            continue
        gt = (mask_gt > 0).astype(np.uint8)

        t0 = time.perf_counter()
        pred_mask = process_image(model, device, img, small_size=MODEL_SIZE)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        pred = (pred_mask > 0).astype(np.uint8)

        # Диагностика: убедимся что размеры совпадают, посмотрим что внутри (первые 3 кадра)
        if i < 3:
            print(f'\n  [{img_path.name}] img.shape={img.shape} '
                  f'pred.shape={pred.shape} gt.shape={gt.shape} '
                  f'pred_px={int(pred.sum())} gt_px={int(gt.sum())}')
        assert pred.shape == gt.shape, f'shape mismatch: pred={pred.shape} gt={gt.shape}'

        tp_i = int(((pred == 1) & (gt == 1)).sum())
        fp_i = int(((pred == 1) & (gt == 0)).sum())
        fn_i = int(((pred == 0) & (gt == 1)).sum())
        tp += tp_i; fp += fp_i; fn += fn_i

        if (tp_i + fn_i) == 0 and fp_i == 0:
            piou = 1.0
        else:
            piou = tp_i / max(1, tp_i + fp_i + fn_i)
        per_iou_list.append(piou)

        has_pred = bool(pred.any()); has_gt = bool(gt.any())
        if has_pred and has_gt: img_tp += 1
        elif has_pred and not has_gt: img_fp += 1
        elif not has_pred and has_gt: img_fn += 1
        else: img_tn += 1

        pbar.set_postfix(t=f'{t_total:.1f}s', iou=f'{piou:.2f}')

        del img, mask_gt, gt, pred_mask, pred
        if (i + 1) % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print('\n=== Mode: production (full pipeline) ===')
    print(f'  threshold_low:  {THRESHOLD_LOW}')
    print(f'  threshold_high: {THRESHOLD_HIGH}')
    print(f'  pixel_iou:       {tp / max(1, tp + fp + fn):.4f}')
    print(f'  pixel_precision: {tp / max(1, tp + fp):.4f}')
    print(f'  pixel_recall:    {tp / max(1, tp + fn):.4f}')
    print(f'  pixel_dice:      {2 * tp / max(1, 2 * tp + fp + fn):.4f}')
    print(f'  per_image_iou_mean:   {float(np.mean(per_iou_list)):.4f}')
    print(f'  per_image_iou_median: {float(np.median(per_iou_list)):.4f}')
    print(f'  image_tp: {img_tp}  image_fp: {img_fp}  image_fn: {img_fn}  image_tn: {img_tn}')
    print(f'  image_precision: {img_tp / max(1, img_tp + img_fp):.4f}')
    print(f'  image_recall:    {img_tp / max(1, img_tp + img_fn):.4f}')


if __name__ == '__main__':
    main()
