"""
Обучение U-Net на red-датасете (D:/train/prepared_red_2048/dataset)
с улучшениями относительно TRAIN_1024.py:

  Tier 1 (must-have):
    - BatchNorm2d (вернули после теста GN: GN+AMP+grad_chk даёт ×8 замедление)
    - EMA весов (decay=0.999)           — гладкие финальные веса
    - weight_decay 1e-4 (было 1e-5)     — регуляризация для малого датасета
    - NaN-guard перед backward          — защита от AMP fp16 переполнений

  Tier 2 (рекомендую):
    - Loss = 0.3·Focal(γ=2) + 0.7·Dice  — Focal сам ловит class imbalance
                                          (pos_weight больше не нужен)
    - CoarseDropout (online)            — регуляризация: выкидываем patches img
    - OneCycleLR                        — warmup → пик → cooldown

  Дополнительно:
    - test-сплит: финальная оценка best и EMA на отложенных 118 кадрах

Запуск:
  python D:/train/TRAIN_1024_red.py
  python D:/train/TRAIN_1024_red.py --epochs 60
  python D:/train/TRAIN_1024_red.py --loss bce  # вернуться к BCE+Dice если нужно
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

TRAIN_IMAGES_DIR = 'D:/train/prepared_red_2048/dataset/train/images'
TRAIN_MASKS_DIR  = 'D:/train/prepared_red_2048/dataset/train/masks'
VAL_IMAGES_DIR   = 'D:/train/prepared_red_2048/dataset/val/images'
VAL_MASKS_DIR    = 'D:/train/prepared_red_2048/dataset/val/masks'
TEST_IMAGES_DIR  = 'D:/train/prepared_red_2048/dataset/test/images'
TEST_MASKS_DIR   = 'D:/train/prepared_red_2048/dataset/test/masks'
OUTPUT_DIR       = 'D:/AI/defect_model_red_1024'

EPOCHS      = 80
BATCH_SIZE  = 2
LR          = 3e-4              # initial для OneCycleLR (max_lr = 3·LR)
IMAGE_SIZE  = (1024, 1024)
FEATURES    = [64, 128, 256, 512]
NUM_WORKERS = 4

WEIGHT_DECAY = 1e-4             # было 1e-5
EMA_DECAY    = 0.999

FOCAL_GAMMA  = 2.0
FOCAL_WEIGHT = 0.3
DICE_WEIGHT  = 0.7


# ─── АРХИТЕКТУРА ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    BatchNorm-вариант. На batch=2 BN статистика шумная, но GroupNorm + AMP +
    gradient_checkpoint на 1024² на нашем GPU даёт ×8 замедление —
    непригодно по времени. Вернули BN.
    """
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
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.decoder[i + 1](x)
        return self.head(x)


# ─── DATASET ──────────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    """
    Loader для пар (image, mask). Аугментации:
      - geometric (flip, rot90)        — всегда p=0.5
      - яркость/контраст               — всегда (alpha 0.7-1.3, beta -30..30)
      - CoarseDropout (только image)   — p=0.3, 2-5 holes размером 30-120 px

    Маска НЕ обнуляется в hole'е — модель учится "не предсказывать дефект там
    где image нулевая" не отучивается от настоящих дефектов в редких случаях
    когда дефект попал в hole.
    """
    def __init__(self, images_dir, masks_dir, image_size=(1024, 1024), augment=False,
                 coarse_dropout=False):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.image_size = image_size
        self.augment    = augment
        self.coarse_dropout = coarse_dropout

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
            img, mask = self._geo_aug(img, mask)
            img       = self._photo_aug(img)
            if self.coarse_dropout:
                img = self._coarse_dropout(img)

        img  = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img, mask

    def _geo_aug(self, img, mask):
        if np.random.rand() < 0.5:
            img  = cv2.flip(img,  1); mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            img  = cv2.flip(img,  0); mask = cv2.flip(mask, 0)
        k = np.random.randint(4)
        if k:
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    def _photo_aug(self, img):
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-30, 30)
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img

    def _coarse_dropout(self, img, p=0.3, n_holes=(2, 5), hole_size=(30, 120)):
        if np.random.rand() > p:
            return img
        h, w = img.shape[:2]
        n = np.random.randint(n_holes[0], n_holes[1] + 1)
        for _ in range(n):
            hh = np.random.randint(*hole_size)
            ww = np.random.randint(*hole_size)
            y  = np.random.randint(0, max(1, h - hh))
            x  = np.random.randint(0, max(1, w - ww))
            img[y:y+hh, x:x+ww] = 0
        return img


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


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss (binary). gamma=2 — стандарт.
    Без alpha (балансировка идёт через сам γ; для extreme imbalance работает).
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # pt = вероятность правильного класса (стабильно через -bce)
        pt  = torch.exp(-bce.clamp(max=20.0))
        focal = (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


class CombinedFocalDice(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, focal_w=FOCAL_WEIGHT, dice_w=DICE_WEIGHT):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice  = DiceLoss()
        self.fw, self.dw = focal_w, dice_w
    def forward(self, logits, targets):
        return self.fw * self.focal(logits.clamp(-10, 10), targets) \
             + self.dw * self.dice(logits, targets)


class CombinedBceDice(nn.Module):
    """Fallback на старый loss — BCE+Dice (для сравнения через --loss bce)."""
    def __init__(self, pos_weight=None, bce_w=0.3, dice_w=0.7):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bw, self.dw = bce_w, dice_w
    def forward(self, logits, targets):
        return self.bw * self.bce(logits.clamp(-10, 10), targets) \
             + self.dw * self.dice(logits, targets)


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


# ─── EMA ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Теневая EMA копия модели. На инференс/валидацию подставляем ema.module.
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


# ─── TRAIN / VALIDATE ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device,
              train=True, ema: ModelEMA = None):
    """
    Одна эпоха. Если train=True и scheduler передан (OneCycleLR),
    делаем step после каждого optimizer.step() (как требует OneCycleLR).
    """
    model.train(train)
    total_loss = total_iou = total_dice = 0.0
    max_sig_epoch = 0.0
    pred_cov_sum  = 0.0
    n_done = 0
    n_skipped_nan = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    pbar = tqdm(loader, desc='train' if train else 'val  ', leave=False)
    with ctx:
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                out  = model(imgs)
                loss = criterion(out, masks)

            if train:
                # NaN-guard: AMP fp16 + extreme imbalance иногда даёт inf/nan на random batch.
                # Без этого один бракованный batch портит ВСЕ веса навсегда.
                if not torch.isfinite(loss):
                    n_skipped_nan += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            with torch.no_grad():
                probs  = torch.sigmoid(out.detach())
                iou_b  = iou_score(out.detach(),  masks).item()
                dice_b = dice_score(out.detach(), masks).item()
                max_sig_epoch = max(max_sig_epoch, float(probs.max().item()))
                pred_cov_sum += float((probs > 0.5).float().mean().item())

            total_loss += float(loss.item()) if torch.isfinite(loss) else 0.0
            total_iou  += iou_b
            total_dice += dice_b
            n_done     += 1

            pbar.set_postfix(
                loss=f'{total_loss / max(1, n_done):.4f}',
                iou=f'{total_iou / max(1, n_done):.4f}',
                dice=f'{total_dice / max(1, n_done):.4f}',
                maxσ=f'{max_sig_epoch:.2f}',
            )

    n = max(1, len(loader))
    if n_skipped_nan > 0:
        print(f"  ⚠️  пропущено batches с NaN/inf loss: {n_skipped_nan}")
    return (total_loss / n, total_iou / n, total_dice / n,
            max_sig_epoch, pred_cov_sum / n)


@torch.no_grad()
def evaluate(model, loader, device):
    """Финальная оценка на test split — IoU + Dice."""
    model.eval()
    total_iou = total_dice = 0.0
    n = 0
    for imgs, masks in tqdm(loader, desc='test ', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast():
            out = model(imgs)
        total_iou  += iou_score(out,  masks).item()
        total_dice += dice_score(out, masks).item()
        n += 1
    n = max(1, n)
    return total_iou / n, total_dice / n


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"U-NET RED {args.image_size}×{args.image_size}  |  device: {device}  |  AMP: {device.type == 'cuda'}")
    print(f"features: {FEATURES}  |  lr: {args.lr}  |  batch: {args.batch_size}  |  workers: {args.num_workers}")
    print(f"weight_decay: {WEIGHT_DECAY}  |  EMA decay: {EMA_DECAY}")
    print(f"loss: {args.loss}  |  scheduler: {args.scheduler}")
    print(f"checkpoint: {args.gradient_checkpoint}")
    print(f"{'='*70}\n")

    image_size = (args.image_size, args.image_size)

    train_ds = DefectDataset(args.train_images, args.train_masks,
                             image_size, augment=True, coarse_dropout=True)
    val_ds   = DefectDataset(args.val_images,   args.val_masks,
                             image_size, augment=False)
    test_ds  = DefectDataset(args.test_images,  args.test_masks,
                             image_size, augment=False) if Path(args.test_images).exists() else None

    nw = args.num_workers
    persistent = nw > 0
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True,
                          persistent_workers=persistent, prefetch_factor=2 if nw else None)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True, drop_last=False,
                          persistent_workers=persistent, prefetch_factor=2 if nw else None)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True, drop_last=False,
                          persistent_workers=persistent, prefetch_factor=2 if nw else None) if test_ds else None

    model = UNet(in_channels=3, out_channels=1,
                 use_checkpoint=args.gradient_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Параметров модели: {n_params:.1f}M  (BatchNorm)\n")

    # Loss
    if args.loss == 'focal':
        criterion = CombinedFocalDice()
        print(f"Loss = {FOCAL_WEIGHT}·Focal(γ={FOCAL_GAMMA}) + {DICE_WEIGHT}·Dice  (без pos_weight — Focal сам ловит imbalance)\n")
    else:
        # BCE+Dice fallback
        from glob import glob
        # для совместимости — pos_weight не считаем, ставим 50 (как в старой модели)
        pw = torch.tensor([50.0], device=device)
        criterion = CombinedBceDice(pos_weight=pw)
        print(f"Loss = 0.3·BCE(pos_weight=50) + 0.7·Dice  (legacy)\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Scheduler
    if args.scheduler == 'onecycle':
        steps_per_epoch = max(1, len(train_ld))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 3.0,           # пик в 3× от base lr
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.10,                 # 10% warmup
            div_factor=25.0,                # initial = max/25
            final_div_factor=1e4,           # final   = initial/1e4
            anneal_strategy='cos',
        )
        scheduler_step_per_batch = True
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        scheduler_step_per_batch = False

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    ema    = ModelEMA(model, decay=EMA_DECAY)

    best_iou      = 0.0
    best_iou_ema  = 0.0
    history       = []
    start_epoch   = 1

    # ── Resume: подхват полного состояния если есть resume_state.pth ──
    resume_path = out / 'resume_state.pth'
    if resume_path.exists() and not args.no_resume:
        print(f"📂 Найден {resume_path.name} — продолжаем обучение")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema.module.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        best_iou      = ckpt['best_iou']
        best_iou_ema  = ckpt['best_iou_ema']
        history       = ckpt.get('history', [])
        start_epoch   = ckpt['epoch'] + 1
        # Восстановим RNG чтобы порядок батчей продолжился
        # torch.set_rng_state ждёт CPU ByteTensor; map_location=device мог перенести на CUDA.
        if 'rng_torch' in ckpt:
            rng_t = ckpt['rng_torch']
            if hasattr(rng_t, 'cpu'):
                rng_t = rng_t.cpu().to(torch.uint8)
            torch.set_rng_state(rng_t)
        if ckpt.get('rng_cuda') is not None and torch.cuda.is_available():
            rng_c = ckpt['rng_cuda']
            rng_c = [t.cpu().to(torch.uint8) if hasattr(t, 'cpu') else t for t in rng_c]
            torch.cuda.set_rng_state_all(rng_c)
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        print(f"   epoch {ckpt['epoch']} → продолжаем с {start_epoch}")
        print(f"   best raw={best_iou:.4f}  best ema={best_iou_ema:.4f}")
        if start_epoch > args.epochs:
            print(f"   ⚠️  Все {args.epochs} эпох уже отработаны. Удали resume_state.pth или увеличь --epochs.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        # train
        sched_for_run = scheduler if scheduler_step_per_batch else None
        (tr_loss, tr_iou, tr_dice, tr_max_sig, tr_cov) = run_epoch(
            model, train_ld, criterion, optimizer, scaler, sched_for_run,
            device, train=True, ema=ema)

        # val (raw model)
        (va_loss, va_iou, va_dice, va_max_sig, va_cov) = run_epoch(
            model, val_ld, criterion, optimizer, scaler, None,
            device, train=False, ema=None)

        # val (EMA model)
        (eva_loss, eva_iou, eva_dice, eva_max_sig, eva_cov) = run_epoch(
            ema.module, val_ld, criterion, optimizer, scaler, None,
            device, train=False, ema=None)

        if not scheduler_step_per_batch:
            scheduler.step()

        row = {'epoch': epoch,
               'tr_loss': tr_loss, 'tr_iou': tr_iou, 'tr_dice': tr_dice,
               'va_loss': va_loss, 'va_iou': va_iou, 'va_dice': va_dice,
               'eva_loss': eva_loss, 'eva_iou': eva_iou, 'eva_dice': eva_dice,
               'tr_max_sig': tr_max_sig, 'va_max_sig': va_max_sig,
               'tr_pred_cov': tr_cov,   'va_pred_cov': va_cov,
               'lr': optimizer.param_groups[0]['lr']}
        history.append(row)

        flag_raw = '✅' if va_iou  > best_iou     else '  '
        flag_ema = '🌟' if eva_iou > best_iou_ema else '  '
        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"tr loss={tr_loss:.4f} iou={tr_iou:.4f}  "
              f"va loss={va_loss:.4f} iou={va_iou:.4f} {flag_raw}  "
              f"ema iou={eva_iou:.4f} {flag_ema}")
        print(f"           diag: tr maxσ={tr_max_sig:.2f} cov={tr_cov*100:.2f}%   "
              f"va maxσ={va_max_sig:.2f} cov={va_cov*100:.2f}%   "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_iou':    va_iou,
                'image_size': image_size,
                'features':   FEATURES,
                'norm':       'BatchNorm',
            }, out / 'best_model_red_1024.pth')

        if eva_iou > best_iou_ema:
            best_iou_ema = eva_iou
            torch.save({
                'epoch':      epoch,
                'state_dict': ema.module.state_dict(),
                'val_iou':    eva_iou,
                'image_size': image_size,
                'features':   FEATURES,
                'norm':       'BatchNorm',
                'ema':        True,
            }, out / 'best_model_red_1024_ema.pth')

        torch.save(model.state_dict(),       out / 'last_model_red_1024.pth')
        torch.save(ema.module.state_dict(),  out / 'last_model_red_1024_ema.pth')

        # Полный resume state — чтобы можно было Ctrl+C и продолжить с этой эпохи
        torch.save({
            'epoch':        epoch,
            'model':        model.state_dict(),
            'ema':          ema.module.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'scaler':       scaler.state_dict(),
            'best_iou':     best_iou,
            'best_iou_ema': best_iou_ema,
            'history':      history,
            'rng_torch':    torch.get_rng_state(),
            'rng_cuda':     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'rng_numpy':    np.random.get_state(),
        }, str(resume_path))

    print(f"\n🎉 Best val IoU (raw):  {best_iou:.4f}")
    print(f"🌟 Best val IoU (EMA):  {best_iou_ema:.4f}")

    # ── Финальная оценка на test ──
    if test_ld is not None:
        print(f"\n{'─'*60}")
        print("Финальная оценка на test (best checkpoints):")

        # raw best
        ckpt = torch.load(out / 'best_model_red_1024.pth', map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        test_iou,  test_dice  = evaluate(model, test_ld, device)
        print(f"  raw  best (ep {ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}):  test_iou={test_iou:.4f}  test_dice={test_dice:.4f}")

        # ema best
        ckpt_ema = torch.load(out / 'best_model_red_1024_ema.pth', map_location=device)
        ema.module.load_state_dict(ckpt_ema['state_dict'])
        test_iou_ema, test_dice_ema = evaluate(ema.module, test_ld, device)
        print(f"  ema  best (ep {ckpt_ema['epoch']}, val_iou={ckpt_ema['val_iou']:.4f}):  test_iou={test_iou_ema:.4f}  test_dice={test_dice_ema:.4f}")

        history.append({'test_iou_raw': test_iou, 'test_dice_raw': test_dice,
                        'test_iou_ema': test_iou_ema, 'test_dice_ema': test_dice_ema})

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory: {out / 'history.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', default=TRAIN_IMAGES_DIR)
    parser.add_argument('--train-masks',  default=TRAIN_MASKS_DIR)
    parser.add_argument('--val-images',   default=VAL_IMAGES_DIR)
    parser.add_argument('--val-masks',    default=VAL_MASKS_DIR)
    parser.add_argument('--test-images',  default=TEST_IMAGES_DIR)
    parser.add_argument('--test-masks',   default=TEST_MASKS_DIR)
    parser.add_argument('--output',       default=OUTPUT_DIR)
    parser.add_argument('--epochs',       type=int,   default=EPOCHS)
    parser.add_argument('--batch-size',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--num-workers',  type=int,   default=NUM_WORKERS)
    parser.add_argument('--image-size',   type=int,   default=IMAGE_SIZE[0])
    parser.add_argument('--lr',           type=float, default=LR)
    parser.add_argument('--loss',         choices=['focal', 'bce'], default='focal',
                        help='focal = Focal+Dice (default), bce = BCE+Dice (legacy)')
    parser.add_argument('--scheduler',    choices=['onecycle', 'cosine'], default='onecycle')
    parser.add_argument('--no-gradient-checkpoint', action='store_true',
                        help='Отключить gradient checkpointing (только если VRAM > 12 GB)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Игнорировать resume_state.pth и начать с нуля')
    args = parser.parse_args()
    args.gradient_checkpoint = not args.no_gradient_checkpoint
    main(args)
