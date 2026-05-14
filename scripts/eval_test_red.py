"""Evaluation на отложенном test split red-zone датасета.

Три режима, чтобы оценить вклад каждого этапа production-пайплайна:
  raw         — модель без post-processing: resize 2048→1024, forward, resize обратно
  tiled       — production tile inference: sliding 256 + letterbox 256, multi-contrast 2.0+3.0
                fusion через max, БЕЗ refine_defect_mask
  production  — то что в проде: tiled + refine_defect_mask (guided filter + hysteresis +
                intensity_shrink)

Метрики:
  Pixel-level (micro): IoU, Precision, Recall, Dice
  Per-image:           mean / median IoU
  Image-level:         TP / FP / FN / TN (есть дефект / нет)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # graceful fallback если tqdm нет
        return it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from core.image_processing import (
    prepare_zone_tiles, stitch_tile_probs,
    apply_histogram_normalization, resize_with_letterbox,
)
from core.defect_detection import predict_defects_batch_probs, refine_defect_mask

TEST_DIR = Path(r'D:/train/prepared_red_2048/dataset/test')
MODEL_PATH = ROOT / 'best_model.pth'

TILE_SIZE = 256
HIST_CONTRASTS = (2.0, 3.0)
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def load_model(device):
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f'  model: epoch={ckpt.get("epoch")}, val_iou={ckpt.get("val_iou")}')
    return model


def infer_raw(model, img_bgr, device):
    """resize 2048→1024 → forward → resize 1024→2048. Возвращает probs (H, W)."""
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_1024 = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_AREA)
    arr = img_1024.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    if device.type == 'cuda':
        t = t.to(device, memory_format=torch.channels_last)
    else:
        t = t.to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
    return cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)


def infer_tiled_fused(model, img_bgr, device, return_guide=False):
    """Production multi-contrast tiled inference, без refine. Возвращает fused probs (H, W).

    Использует ту же логику что в workers/processing_worker._prepare_zone и фазу B:
      для каждого контраста (2.0, 3.0): sliding 256 + letterbox 256 → 2 probs map'а
      fused = np.maximum.reduce(all probs map'ов)
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    all_probs_full = []
    last_guide = None
    for c in HIST_CONTRASTS:
        roi_c = apply_histogram_normalization(gray, std_range=c)
        ti = prepare_zone_tiles(roi_c, tile_size=TILE_SIZE, min_overlap=TILE_SIZE // 4)
        roi_gray = ti['roi_gray']
        last_guide = roi_gray

        # Sliding tiles
        small_probs = predict_defects_batch_probs(
            model, ti['tiles'], device, tta=False, chunk_size=8,
        )
        if ti['mode'] == 'letterbox':
            ps = restore_probs_letterbox(small_probs[0], ti['pad_info'])
        else:
            ps = stitch_tile_probs(small_probs, ti['tile_origins'], (H, W), tile_size=TILE_SIZE)
        all_probs_full.append(ps)

        # Letterbox 256×256 — вся зона в один тайл
        canvas, pad_info = resize_with_letterbox(roi_gray, (TILE_SIZE, TILE_SIZE))
        large_probs = predict_defects_batch_probs(
            model, [canvas], device, tta=False, chunk_size=1,
        )
        pl = restore_probs_letterbox(large_probs[0], pad_info)
        all_probs_full.append(pl)

    fused = np.maximum.reduce(all_probs_full)
    return (fused, last_guide) if return_guide else fused


def restore_probs_letterbox(probs, pad_info):
    pad_x = pad_info['pad_x']; pad_y = pad_info['pad_y']
    new_w = pad_info['new_w']; new_h = pad_info['new_h']
    orig_w = pad_info['orig_w']; orig_h = pad_info['orig_h']
    cropped = probs[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def evaluate_at_threshold(threshold, all_probs, all_gt):
    tp = fp = fn = 0
    per_iou = []
    img_tp = img_fp = img_fn = img_tn = 0
    for probs, gt in zip(all_probs, all_gt):
        pred = (probs > threshold).astype(np.uint8)
        tp_i = int(((pred == 1) & (gt == 1)).sum())
        fp_i = int(((pred == 1) & (gt == 0)).sum())
        fn_i = int(((pred == 0) & (gt == 1)).sum())
        tp += tp_i; fp += fp_i; fn += fn_i
        if (tp_i + fn_i) == 0 and fp_i == 0:
            per_iou.append(1.0)
        else:
            per_iou.append(tp_i / max(1, tp_i + fp_i + fn_i))
        has_pred = bool(pred.any()); has_gt = bool(gt.any())
        if has_pred and has_gt: img_tp += 1
        elif has_pred and not has_gt: img_fp += 1
        elif not has_pred and has_gt: img_fn += 1
        else: img_tn += 1
    return {
        'threshold': threshold,
        'pixel_iou':       tp / max(1, tp + fp + fn),
        'pixel_precision': tp / max(1, tp + fp),
        'pixel_recall':    tp / max(1, tp + fn),
        'pixel_dice':      2 * tp / max(1, 2 * tp + fp + fn),
        'per_image_iou_mean':   float(np.mean(per_iou)),
        'per_image_iou_median': float(np.median(per_iou)),
        'image_tp': img_tp, 'image_fp': img_fp, 'image_fn': img_fn, 'image_tn': img_tn,
        'image_precision': img_tp / max(1, img_tp + img_fp),
        'image_recall':    img_tp / max(1, img_tp + img_fn),
    }


def evaluate_production(all_probs, all_guides, all_gt, threshold_low=0.01, threshold_high=0.7):
    """Один проход evaluation для production: refine + бинарная маска. threshold_low —
    пользовательский порог (как в UI). Возвращает один результат (без sweep)."""
    tp = fp = fn = 0
    per_iou = []
    img_tp = img_fp = img_fn = img_tn = 0
    for probs, guide, gt in tqdm(list(zip(all_probs, all_guides, all_gt)),
                                 desc='Refine', unit='img', ncols=100):
        mask = refine_defect_mask(
            probs, guide,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )
        pred = (mask > 0).astype(np.uint8)
        tp_i = int(((pred == 1) & (gt == 1)).sum())
        fp_i = int(((pred == 1) & (gt == 0)).sum())
        fn_i = int(((pred == 0) & (gt == 1)).sum())
        tp += tp_i; fp += fp_i; fn += fn_i
        if (tp_i + fn_i) == 0 and fp_i == 0:
            per_iou.append(1.0)
        else:
            per_iou.append(tp_i / max(1, tp_i + fp_i + fn_i))
        has_pred = bool(pred.any()); has_gt = bool(gt.any())
        if has_pred and has_gt: img_tp += 1
        elif has_pred and not has_gt: img_fp += 1
        elif not has_pred and has_gt: img_fn += 1
        else: img_tn += 1
    return {
        'mode': 'production (refine)',
        'pixel_iou':       tp / max(1, tp + fp + fn),
        'pixel_precision': tp / max(1, tp + fp),
        'pixel_recall':    tp / max(1, tp + fn),
        'pixel_dice':      2 * tp / max(1, 2 * tp + fp + fn),
        'per_image_iou_mean':   float(np.mean(per_iou)),
        'per_image_iou_median': float(np.median(per_iou)),
        'image_tp': img_tp, 'image_fp': img_fp, 'image_fn': img_fn, 'image_tn': img_tn,
        'image_precision': img_tp / max(1, img_tp + img_fp),
        'image_recall':    img_tp / max(1, img_tp + img_fn),
    }


def print_sweep_header(mode):
    print(f'\n=== Mode: {mode} ===')
    print(f'{"thr":>6} {"pIoU":>7} {"pPrec":>7} {"pRec":>7} {"pDice":>7} '
          f'{"imgPrec":>8} {"imgRec":>8} {"TP/FP/FN/TN":>16}')


def print_row(r):
    print(
        f'{r["threshold"]:>6.2f} {r["pixel_iou"]:>7.4f} {r["pixel_precision"]:>7.4f} '
        f'{r["pixel_recall"]:>7.4f} {r["pixel_dice"]:>7.4f} '
        f'{r["image_precision"]:>8.4f} {r["image_recall"]:>8.4f} '
        f'{r["image_tp"]:>3}/{r["image_fp"]:>3}/{r["image_fn"]:>3}/{r["image_tn"]:>3}'
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model:  {MODEL_PATH}')
    model = load_model(device)

    images = sorted((TEST_DIR / 'images').glob('*.png'))
    print(f'Test images: {len(images)}')

    raw_probs = []
    tiled_probs = []
    guides = []
    gts = []

    pbar = tqdm(images, desc='Inference', unit='img', ncols=100)
    for img_path in pbar:
        img = cv2.imread(str(img_path))
        if img is None: continue
        mask = cv2.imread(str(TEST_DIR / 'masks' / img_path.name), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        gt = (mask > 0).astype(np.uint8)

        rp = infer_raw(model, img, device)
        raw_probs.append(rp)

        tp, guide = infer_tiled_fused(model, img, device, return_guide=True)
        tiled_probs.append(tp)
        guides.append(guide)
        gts.append(gt)

    # Threshold sweep для raw и tiled
    for mode, probs_list in [('raw', raw_probs), ('tiled (no refine)', tiled_probs)]:
        print_sweep_header(mode)
        best = None
        for thr in THRESHOLDS:
            r = evaluate_at_threshold(thr, probs_list, gts)
            print_row(r)
            score = 2 * r['image_precision'] * r['image_recall'] / max(1e-9, r['image_precision'] + r['image_recall'])
            if best is None or score > best[0]:
                best = (score, r)
        print(f'  best image-F1: thr={best[1]["threshold"]}, pIoU={best[1]["pixel_iou"]:.4f}')

    # Production режим: тот же threshold_low=0.01 что в UI
    print('\n=== Mode: production (refine + intensity_shrink) ===')
    prod = evaluate_production(tiled_probs, guides, gts, threshold_low=0.01, threshold_high=0.7)
    for k, v in prod.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
