"""
Обучение U-Net для детекции дефектов — размер входа 1024×1024.

Изменения vs TRAIN.py:
  - IMAGE_SIZE = (1024, 1024)
  - batch_size = 2  (безопасно для 8 GB с AMP)
  - LR = 1e-4       (было 1e-6 — слишком маленький для обучения с нуля)
  - scheduler: CosineAnnealingLR вместо ReduceLROnPlateau
  - num_workers = 0  (Windows, без multiprocessing)
  - gradient_checkpointing опционально (--checkpoint флаг)
  - сохраняет лучший чекпоинт по val IoU (не Dice)

Запуск (глобальный Python 3.11 с GPU torch):
  python TRAIN_1024.py
  python TRAIN_1024.py --epochs 100 --batch-size 1  # если OOM
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
import warnings
warnings.filterwarnings('ignore')


# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

TRAIN_IMAGES_DIR = 'D:/train/prepared_2048/dataset/train/images'
TRAIN_MASKS_DIR  = 'D:/train/prepared_2048/dataset/train/masks'
VAL_IMAGES_DIR   = 'D:/train/prepared_2048/dataset/val/images'
VAL_MASKS_DIR    = 'D:/train/prepared_2048/dataset/val/masks'
OUTPUT_DIR       = 'D:/AI/defect_model_2048'

EPOCHS      = 80
BATCH_SIZE  = 2
LR          = 3e-4
IMAGE_SIZE  = (1024, 1024)   # 1024² → влезает в 8GB без checkpoint и быстро
FEATURES    = [64, 128, 256, 512]
NUM_WORKERS = 2

# Class imbalance: дефект ~0.2% пикселей → нужен pos_weight в BCE.
# Считаем автоматически из train-масок и клампим в этот диапазон.
POS_WEIGHT_RANGE = (10.0, 50.0)
BCE_WEIGHT       = 0.3   # было 0.5 — снизили: BCE не должен дожимать всё в ноль
DICE_WEIGHT      = 0.7   # было 0.5 — повысили: Dice устойчивее к imbalance


# ─── АРХИТЕКТУРА ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None,
                 use_checkpoint=False):
        super().__init__()
        if features is None:
            features = FEATURES
        self.use_checkpoint = use_checkpoint
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.decoder.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = []
        for enc in self.encoder:
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(enc, x, use_reentrant=False)
            else:
                x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = TF.resize(x, s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.decoder[i + 1](x)

        return self.head(x)


# ─── DATASET ──────────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(1024, 1024), augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.image_size = image_size
        self.augment    = augment

        self.files = sorted([
            p for ext in ('*.png', '*.jpg', '*.jpeg')
            for p in self.images_dir.glob(ext)
        ])
        print(f"  {images_dir}: {len(self.files)} файлов")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path  = self.files[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')

        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            mask = np.zeros(self.image_size[::-1], dtype=np.uint8)

        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img,  self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img  = torch.from_numpy(img.astype(np.float32)  / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img, mask

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            img  = cv2.flip(img,  0)
            mask = cv2.flip(mask, 0)
        k = np.random.randint(4)
        if k:
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()
        # Яркость/контраст
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask


# ─── LOSS & METRICS ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_w=BCE_WEIGHT, dice_w=DICE_WEIGHT):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_w  = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets):
        return (self.bce_w  * self.bce(logits.clamp(-10, 10), targets)
              + self.dice_w * self.dice(logits, targets))


def compute_pos_weight(masks_dir, image_size, max_files=400):
    """neg/pos на train-масках (уменьшенных до image_size), клампим в POS_WEIGHT_RANGE."""
    files = sorted(Path(masks_dir).glob('*.png'))[:max_files]
    pos = neg = 0
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, image_size, interpolation=cv2.INTER_NEAREST)
        b = (m > 127)
        pos += int(b.sum())
        neg += int((~b).sum())
    if pos == 0:
        print(f"  ⚠️  pos pixels = 0 в {masks_dir} — pos_weight=1.0")
        return 1.0
    raw = neg / pos
    pw  = float(np.clip(raw, POS_WEIGHT_RANGE[0], POS_WEIGHT_RANGE[1]))
    print(f"  pos pixels: {pos:,}  neg pixels: {neg:,}")
    print(f"  raw neg/pos: {raw:.1f}  →  pos_weight (clamped): {pw:.2f}")
    return pw


def iou_score(logits, targets, threshold=0.5):
    p = (torch.sigmoid(logits) > threshold).float().view(-1)
    t = (targets > threshold).float().view(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + 1e-6) / (union + 1e-6) if union > 0 else torch.tensor(1.0)


def dice_score(logits, targets, threshold=0.5):
    p = (torch.sigmoid(logits) > threshold).float().view(-1)
    t = (targets > threshold).float().view(-1)
    inter = (p * t).sum()
    total = p.sum() + t.sum()
    return (2 * inter + 1e-6) / (total + 1e-6) if total > 0 else torch.tensor(1.0)


# ─── TRAIN / VALIDATE ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss = total_iou = total_dice = 0.0
    max_sig_epoch = 0.0      # max sigmoid за всю эпоху → видно, что модель не в "all-zeros"
    pred_cov_sum  = 0.0      # доля пикселей, где pred>0.5, усреднённая по батчам
    n_done = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    pbar = tqdm(loader, desc='train' if train else 'val  ', leave=False)
    with ctx:
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                out  = model(imgs)
                loss = criterion(out, masks)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                probs  = torch.sigmoid(out.detach())
                iou_b  = iou_score(out.detach(),  masks).item()
                dice_b = dice_score(out.detach(), masks).item()
                max_sig_epoch = max(max_sig_epoch, float(probs.max().item()))
                pred_cov_sum += float((probs > 0.5).float().mean().item())

            total_loss += loss.item()
            total_iou  += iou_b
            total_dice += dice_b
            n_done     += 1

            pbar.set_postfix(
                loss=f'{total_loss / n_done:.4f}',
                iou=f'{total_iou / n_done:.4f}',
                dice=f'{total_dice / n_done:.4f}',
                maxσ=f'{max_sig_epoch:.2f}',
            )

    n = len(loader)
    return (total_loss / n, total_iou / n, total_dice / n,
            max_sig_epoch, pred_cov_sum / n)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # TF32 безопасный (не делает поиск kernels). cudnn.benchmark НЕ включаем —
    # на 1024² + AMP он надолго подвисает на первой итерации, перебирая алгоритмы.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"U-NET {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}  |  device: {device}  |  AMP: {device.type == 'cuda'}")
    print(f"features: {FEATURES}  |  lr: {args.lr}  |  batch: {args.batch_size}  |  workers: {args.num_workers}")
    print(f"checkpoint: {args.gradient_checkpoint}")
    print(f"{'='*70}\n")

    train_ds = DefectDataset(args.train_images, args.train_masks,
                             IMAGE_SIZE, augment=True)
    val_ds   = DefectDataset(args.val_images, args.val_masks,
                             IMAGE_SIZE, augment=False)

    nw = args.num_workers
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True, drop_last=False)

    model = UNet(in_channels=3, out_channels=1,
                 use_checkpoint=args.gradient_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Параметров модели: {n_params:.1f}M\n")

    print("Считаю pos_weight по train-маскам...")
    pw_value = compute_pos_weight(args.train_masks, IMAGE_SIZE)
    pos_weight = torch.tensor([pw_value], device=device)

    criterion = CombinedLoss(pos_weight=pos_weight)
    print(f"Loss = {BCE_WEIGHT}·BCE(pos_weight={pw_value:.2f}) + {DICE_WEIGHT}·Dice\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler    = GradScaler(enabled=(device.type == 'cuda'))

    best_iou  = 0.0
    history   = []

    for epoch in range(1, args.epochs + 1):
        (tr_loss, tr_iou, tr_dice,
         tr_max_sig, tr_cov) = run_epoch(model, train_ld, criterion, optimizer,
                                         scaler, device, train=True)
        (va_loss, va_iou, va_dice,
         va_max_sig, va_cov) = run_epoch(model, val_ld,   criterion, optimizer,
                                         scaler, device, train=False)
        scheduler.step()

        row = {'epoch': epoch,
               'tr_loss': tr_loss, 'tr_iou': tr_iou, 'tr_dice': tr_dice,
               'va_loss': va_loss, 'va_iou': va_iou, 'va_dice': va_dice,
               'tr_max_sig': tr_max_sig, 'va_max_sig': va_max_sig,
               'tr_pred_cov': tr_cov,   'va_pred_cov': va_cov,
               'lr': optimizer.param_groups[0]['lr']}
        history.append(row)

        improved = '✅' if va_iou > best_iou else '  '
        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"tr loss={tr_loss:.4f} iou={tr_iou:.4f} dice={tr_dice:.4f}  "
              f"va loss={va_loss:.4f} iou={va_iou:.4f} dice={va_dice:.4f}  {improved}")
        print(f"           diag: tr maxσ={tr_max_sig:.3f} cov={tr_cov*100:.2f}%   "
              f"va maxσ={va_max_sig:.3f} cov={va_cov*100:.2f}%   "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_iou':    va_iou,
                'image_size': IMAGE_SIZE,
                'features':   FEATURES,
            }, out / 'best_model_1024.pth')

        torch.save(model.state_dict(), out / 'last_model_1024.pth')

    print(f"\n🎉 Лучший val IoU: {best_iou:.4f}")
    print(f"   Модель: {out / 'best_model_1024.pth'}")

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', default=TRAIN_IMAGES_DIR)
    parser.add_argument('--train-masks',  default=TRAIN_MASKS_DIR)
    parser.add_argument('--val-images',   default=VAL_IMAGES_DIR)
    parser.add_argument('--val-masks',    default=VAL_MASKS_DIR)
    parser.add_argument('--output',       default=OUTPUT_DIR)
    parser.add_argument('--epochs',       type=int,   default=EPOCHS)
    parser.add_argument('--batch-size',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--num-workers',  type=int,   default=NUM_WORKERS)
    parser.add_argument('--image-size',   type=int,   default=IMAGE_SIZE[0],
                        help='Сторона квадратного входа (1024, 1536, 2048)')
    parser.add_argument('--lr',           type=float, default=LR)
    parser.add_argument('--no-gradient-checkpoint', action='store_true',
                        help='Отключить gradient checkpointing (только если VRAM > 12 GB)')
    args = parser.parse_args()
    args.gradient_checkpoint = not args.no_gradient_checkpoint
    # Переопределим IMAGE_SIZE из аргумента (модуль-уровневая константа)
    IMAGE_SIZE = (args.image_size, args.image_size)
    globals()['IMAGE_SIZE'] = IMAGE_SIZE
    main(args)
