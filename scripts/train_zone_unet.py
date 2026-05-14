"""
Обучение U-Net для сегментации рабочих зон фильтра.

Использует датасет, сгенерированный generate_zone_dataset.py:
    images/  — grayscale PNG
    masks/   — бинарные маски (255 = зона, 0 = фон)

Запуск:
    python scripts/train_zone_unet.py \
        --dataset D:/AI/zone_dataset_full \
        --output  D:/AI/zone_model \
        --epochs 50 --batch-size 4 --image-size 512

Результат:
    output/best_zone_unet.pth   — лучший чекпоинт по val IoU
    output/last_zone_unet.pth   — последний чекпоинт
    output/training_log.json    — метрики по эпохам
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.unet import UNet


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ZoneDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size: int, augment: bool):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment_pair(self, image: np.ndarray, mask: np.ndarray):
        # Случайные флипы
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        # Лёгкая яркость/контраст — учим робастности к разному освещению
        alpha = np.random.uniform(0.85, 1.15)  # contrast
        beta = np.random.uniform(-15, 15)       # brightness
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_t = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # [1,H,W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)  # [1,H,W]
        return image_t, mask_t


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = combined_loss(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_iou += iou_score(logits, masks) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = combined_loss(logits, masks)
        total_loss += loss.item() * images.size(0)
        total_iou += iou_score(logits, masks) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_iou / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Папка с подпапками images/ и masks/")
    parser.add_argument("--output", type=Path, required=True,
                        help="Куда сохранять чекпоинты и лог")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512,
                        help="Размер для resize (квадрат). Меньше = быстрее, "
                             "но точность по краям падает.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--features", type=int, nargs='+', default=[32, 64, 128, 256],
                        help="Размеры каналов U-Net. Меньше = легче и быстрее. "
                             "Дефолт [32,64,128,256] — компактная сеть для зон.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    images_dir = args.dataset / "images"
    masks_dir = args.dataset / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(f"ERROR: не найдены {images_dir} и/или {masks_dir}")
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.png"))
    mask_paths = [masks_dir / p.name for p in image_paths]
    pairs = [(i, m) for i, m in zip(image_paths, mask_paths) if m.exists()]
    if not pairs:
        print("ERROR: не найдено ни одной пары image/mask")
        sys.exit(1)

    image_paths, mask_paths = zip(*pairs)
    print(f"Найдено пар image/mask: {len(image_paths)}")

    full = ZoneDataset(list(image_paths), list(mask_paths), args.image_size, augment=False)
    val_size = max(1, int(len(full) * args.val_split))
    train_size = len(full) - val_size

    train_idx, val_idx = random_split(
        range(len(full)), [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_paths = [(image_paths[i], mask_paths[i]) for i in train_idx.indices]
    val_paths = [(image_paths[i], mask_paths[i]) for i in val_idx.indices]

    train_ds = ZoneDataset(*zip(*train_paths), image_size=args.image_size, augment=True)
    val_ds = ZoneDataset(*zip(*val_paths), image_size=args.image_size, augment=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", "GPU:", torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a')

    model = UNet(in_channels=1, out_channels=1, features=args.features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров U-Net: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = []
    best_iou = 0.0
    best_path = args.output / "best_zone_unet.pth"
    last_path = args.output / "last_zone_unet.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_iou = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch:>3}/{args.epochs} | "
              f"train loss {train_loss:.4f} iou {train_iou:.4f} | "
              f"val loss {val_loss:.4f} iou {val_iou:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_iou': train_iou,
            'val_loss': val_loss, 'val_iou': val_iou,
            'lr': lr_now, 'time_sec': elapsed,
        })

        ckpt = {
            'model_state_dict': model.state_dict(),
            'features': args.features,
            'image_size': args.image_size,
            'epoch': epoch,
            'val_iou': val_iou,
        }
        torch.save(ckpt, last_path)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ckpt, best_path)
            print(f"  -> saved best (IoU {best_iou:.4f})")

        (args.output / "training_log.json").write_text(
            json.dumps({'best_iou': best_iou, 'history': history},
                       ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    print(f"\nГотово. Best val IoU = {best_iou:.4f}")
    print(f"Best: {best_path}")
    print(f"Last: {last_path}")


if __name__ == "__main__":
    main()
