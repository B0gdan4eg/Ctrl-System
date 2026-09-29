"""
Готовит датасет к обучению из частичной разметки.

Читает:
  D:/train/data_2048/         — кропы 2048×2048
  D:/train/labels_2048.json   — VIA-JSON разметка (от annotate_defects.py)
  D:/train/_progress.json     — статусы (skipped → исключаем)

Создаёт:
  D:/train/prepared_2048/dataset/
      train/images/ + train/masks/   (~85%)  [+ N аугментаций на файл]
      val/images/   + val/masks/     (~15%)  [без аугментаций]

Маски — uint8 PNG, 0/255 (255 = дефект). Negative-сэмплы — пустая маска.
Стратифицированный сплит на уровне ОРИГИНАЛОВ → нет data leak между
train и val (аугментации одного исходника не разъедутся по сплитам).

Аугментации (через aug_traintest.create_augmentation_pipeline):
  HFlip, VFlip, Rot90, ShiftScaleRotate, brightness/contrast, gamma,
  blur, noise, distortions, CLAHE.

Запуск:
  python D:/train/prep_train.py                          # без аугментаций
  python D:/train/prep_train.py --augment --num-augs 3   # +3 аугментации на train-файл (×4)
  python D:/train/prep_train.py --augment --num-augs 5 --val-fraction 0.15 --clean
"""

import sys
import io
import json
import shutil
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


IMAGES_DIR    = Path("D:/train/base_red_zones_n5_2048")
LABELS_PATH   = Path("D:/train/labels_red_2048.json")
PROGRESS_PATH = Path("D:/train/_progress_red.json")
OUT_ROOT      = Path("D:/train/prepared_red_2048/dataset")


def build_mask(regions, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    iterable = regions.values() if isinstance(regions, dict) else regions
    for r in iterable:
        sa = r.get("shape_attributes", {})
        name = sa.get("name")
        if name == "polygon":
            xs = sa.get("all_points_x", [])
            ys = sa.get("all_points_y", [])
            if len(xs) >= 3 and len(ys) == len(xs):
                pts = np.array(list(zip(xs, ys)), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        elif name == "rect":
            x = int(sa.get("x", 0)); y = int(sa.get("y", 0))
            ww = int(sa.get("width", 0)); hh = int(sa.get("height", 0))
            cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)
        elif name == "circle":
            cv2.circle(mask, (int(sa.get("cx", 0)), int(sa.get("cy", 0))),
                       int(sa.get("r", 0)), 255, -1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction",  type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--augment",   action="store_true",
                    help="Применить офлайн-аугментацию к train (Albumentations)")
    ap.add_argument("--num-augs",     type=int,   default=5,
                    help="Кол-во аугментированных копий на train-файл (только с --augment)")
    ap.add_argument("--clean", action="store_true",
                    help="Очистить prepared_red_2048/dataset перед записью")
    args = ap.parse_args()

    aug_pipeline = None
    if args.augment:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from aug_traintest import create_augmentation_pipeline  # type: ignore
            aug_pipeline = create_augmentation_pipeline()
            print(f"🎨 Аугментация включена: +{args.num_augs} копий на train-файл")
        except ImportError as e:
            print(f"❌ Не удалось импортировать aug_traintest ({e}).")
            print("   Установи albumentations:  pip install albumentations")
            return

    if not LABELS_PATH.exists():
        print(f"❌ Нет {LABELS_PATH}")
        return
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}

    skipped = {k for k, v in progress.items() if v == "skipped"}

    # Берём только размеченные (не skipped), для которых есть исходный кроп
    positives, negatives = [], []
    missing = 0
    for fname, entry in labels.items():
        if fname in skipped:
            continue
        img_path = IMAGES_DIR / fname
        if not img_path.exists():
            missing += 1
            continue
        regions = entry.get("regions", {})
        n_reg = len(regions) if isinstance(regions, (dict, list)) else 0
        (positives if n_reg > 0 else negatives).append((img_path, regions))

    print(f"Размечено всего: {len(labels)}")
    print(f"  с дефектами:  {len(positives)}")
    print(f"  без дефектов: {len(negatives)}")
    print(f"  пропущено (skipped): {len(skipped)}")
    if missing:
        print(f"  ⚠️  нет файла-кропа: {missing}")

    # Стратифицированный split (на уровне ОРИГИНАЛОВ, аугментации не разъедутся)
    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split3(lst):
        n = len(lst)
        n_val  = max(1, round(n * args.val_fraction))  if n else 0
        n_test = max(1, round(n * args.test_fraction)) if n else 0
        # train | val | test
        return lst[n_val + n_test:], lst[:n_val], lst[n_val:n_val + n_test]

    pos_train, pos_val, pos_test = split3(positives)
    neg_train, neg_val, neg_test = split3(negatives)
    train_set = pos_train + neg_train
    val_set   = pos_val   + neg_val
    test_set  = pos_test  + neg_test
    rng.shuffle(train_set); rng.shuffle(val_set); rng.shuffle(test_set)

    print(f"\nSplit (val={args.val_fraction}, test={args.test_fraction}):")
    print(f"  train: {len(train_set)}  ({len(pos_train)} pos + {len(neg_train)} neg)")
    print(f"  val:   {len(val_set)}    ({len(pos_val)} pos + {len(neg_val)} neg)")
    print(f"  test:  {len(test_set)}   ({len(pos_test)} pos + {len(neg_test)} neg)")

    # Папки
    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for sub in ("train/images", "train/masks",
                "val/images",   "val/masks",
                "test/images",  "test/masks"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    def write_split(split_name, items, do_aug: bool):
        img_dir  = OUT_ROOT / split_name / "images"
        mask_dir = OUT_ROOT / split_name / "masks"
        n_total = 0
        for img_path, regions in tqdm(items, desc=split_name):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = build_mask(regions, h, w)

            cv2.imwrite(str(img_dir  / img_path.name), img)
            cv2.imwrite(str(mask_dir / (img_path.stem + ".png")), mask)
            n_total += 1

            if do_aug and aug_pipeline is not None:
                for i in range(args.num_augs):
                    try:
                        out = aug_pipeline(image=img, mask=mask)
                    except Exception as e:
                        print(f"  ⚠️  aug {img_path.name}#{i+1}: {e}")
                        continue
                    aug_img  = out["image"]
                    aug_mask = out["mask"]
                    cv2.imwrite(
                        str(img_dir / f"{img_path.stem}_aug{i+1}{img_path.suffix}"),
                        aug_img
                    )
                    cv2.imwrite(
                        str(mask_dir / f"{img_path.stem}_aug{i+1}.png"),
                        aug_mask
                    )
                    n_total += 1
        return n_total

    n_train = write_split("train", train_set, do_aug=args.augment)
    n_val   = write_split("val",   val_set,   do_aug=False)
    n_test  = write_split("test",  test_set,  do_aug=False)
    print(f"\nЗаписано файлов:  train={n_train}, val={n_val}, test={n_test}")

    # Статистика по площади дефектов
    def defect_coverage(items):
        if not items:
            return 0.0
        total_pix, defect_pix = 0, 0
        for _, regions in items:
            m = build_mask(regions, 2048, 2048)
            total_pix  += m.size
            defect_pix += int((m > 0).sum())
        return defect_pix / total_pix * 100

    print(f"\n📊 Доля дефектных пикселей:")
    print(f"  train: {defect_coverage(train_set):.4f}%")
    print(f"  val:   {defect_coverage(val_set):.4f}%")
    print(f"  test:  {defect_coverage(test_set):.4f}%")

    print(f"\n✅ Готово: {OUT_ROOT}")
    print(f"\nЗапуск обучения:")
    print(f"  python D:/train/TRAIN_1024.py")
    print(f"  python D:/train/TRAIN_1024.py --epochs 60 --no-gradient-checkpoint  # если есть VRAM")


if __name__ == "__main__":
    main()
