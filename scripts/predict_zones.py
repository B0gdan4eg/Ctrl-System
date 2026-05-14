"""
Прогон обученной U-Net сегментации зон на тестовых снимках.

Запуск:
    python scripts/predict_zones.py \
        --model  D:/AI/zone_model/best_zone_unet.pth \
        --input  D:/AI/full_db \
        --output D:/AI/zone_predictions \
        --limit 10 --compare

С --compare добавляет side-by-side: [оригинал | классика | нейросетка | overlay].
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.unet import UNet

# Импортируем core/zone_detection.py напрямую файлом, минуя core/__init__.py,
# чтобы не тянуть PySide6 (которого нет в глобальном python — он только в venv).
import importlib.util as _il
_zd_path = Path(__file__).parent.parent / "core" / "zone_detection.py"
_spec = _il.spec_from_file_location("zone_detection_standalone", str(_zd_path))
_zd = _il.module_from_spec(_spec)
_spec.loader.exec_module(_zd)
find_top_zones_improved = _zd.find_top_zones_improved


SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_grayscale(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p99 <= p1:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = (img.astype(np.float32) - p1) / (p99 - p1) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def predict_mask(model, image: np.ndarray, image_size: int, device: torch.device,
                 threshold: float = 0.5) -> np.ndarray:
    """Предсказывает бинарную маску зон в исходном разрешении image."""
    h, w = image.shape
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    mask_small = (probs > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def classical_mask(image: np.ndarray) -> np.ndarray:
    contours = find_top_zones_improved(
        image, top_n=5, min_absolute_area=20000, max_aspect_ratio=20.0,
    )
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def overlay(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    layer = bgr.copy()
    layer[mask > 0] = color
    return cv2.addWeighted(bgr, 0.6, layer, 0.4, 0)


def make_label(text: str, w: int, h: int = 40) -> np.ndarray:
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def make_compare(image: np.ndarray, classical: np.ndarray, nn_mask: np.ndarray,
                 max_width: int = 1024) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cls_overlay = overlay(image, classical, (255, 0, 0))   # синий — классика
    nn_overlay = overlay(image, nn_mask, (0, 255, 0))      # зелёный — нейросетка

    diff_overlay = bgr.copy()
    only_classical = (classical > 0) & (nn_mask == 0)
    only_nn = (nn_mask > 0) & (classical == 0)
    both = (classical > 0) & (nn_mask > 0)
    diff_overlay[only_classical] = (255, 0, 0)
    diff_overlay[only_nn] = (0, 255, 0)
    diff_overlay[both] = (0, 200, 200)
    diff_overlay = cv2.addWeighted(bgr, 0.5, diff_overlay, 0.5, 0)

    panels = [bgr, cls_overlay, nn_overlay, diff_overlay]
    titles = ["Original", "Classical (blue)", "U-Net (green)", "Diff: blue=cls only / green=NN only / yellow=both"]

    h, w = image.shape
    target_w = max_width // len(panels)
    scale = target_w / w
    target_h = int(h * scale)

    rows = []
    for panel, title in zip(panels, titles):
        scaled = cv2.resize(panel, (target_w, target_h))
        label = make_label(title, target_w)
        rows.append(np.vstack([label, scaled]))

    return np.hstack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Путь к best_zone_unet.pth")
    parser.add_argument("--input", type=Path, required=True, help="Файл или папка")
    parser.add_argument("--output", type=Path, required=True, help="Куда сохранять результаты")
    parser.add_argument("--limit", type=int, default=10, help="Сколько файлов из папки обработать")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--compare", action="store_true",
                        help="Создавать side-by-side с классическим алгоритмом")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    features = ckpt.get('features', [32, 64, 128, 256])
    image_size = ckpt.get('image_size', 512)
    print(f"Model: features={features}, image_size={image_size}, "
          f"epoch={ckpt.get('epoch', '?')}, val_iou={ckpt.get('val_iou', '?'):.4f}")

    model = UNet(in_channels=1, out_channels=1, features=features).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(p for p in args.input.rglob("*")
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
        if args.limit > 0:
            files = files[:args.limit]

    print(f"Файлов на обработку: {len(files)}")

    for i, fp in enumerate(files, 1):
        image = load_grayscale(fp)
        if image is None:
            print(f"  [{i}/{len(files)}] x {fp.name}: load failed")
            continue

        nn_mask = predict_mask(model, image, image_size, device, args.threshold)
        cv2.imwrite(str(args.output / f"{fp.stem}_mask_nn.png"), nn_mask)

        if args.compare:
            cls_mask = classical_mask(image)
            cv2.imwrite(str(args.output / f"{fp.stem}_mask_cls.png"), cls_mask)
            compare_img = make_compare(image, cls_mask, nn_mask)
            cv2.imwrite(str(args.output / f"{fp.stem}_compare.png"), compare_img)

            # Метрики совпадения
            inter = ((cls_mask > 0) & (nn_mask > 0)).sum()
            union = ((cls_mask > 0) | (nn_mask > 0)).sum()
            iou = inter / union if union > 0 else 0
            print(f"  [{i}/{len(files)}] + {fp.name:<50} IoU(cls,NN)={iou:.3f}")
        else:
            print(f"  [{i}/{len(files)}] + {fp.name}")

    print(f"\nГотово. Результаты в {args.output}")


if __name__ == "__main__":
    main()
