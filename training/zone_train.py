import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import cv2
from torchvision.transforms import InterpolationMode

# ======== Проверка устройства ========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True  # ускорение при одинаковом размере входа
else:
    device = torch.device("cpu")
    print("⚠️ CUDA не найдена, используется CPU")

# ======== Модель UNet ========
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.center = nn.Sequential(CBR(512, 1024), CBR(1024, 512))
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 256))
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 128))
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.dec1 = nn.Sequential(CBR(128, 64), nn.Conv2d(64, out_channels, 1))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([F.interpolate(c, scale_factor=2, mode='bilinear', align_corners=False), e4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], 1))
        return d1  # logits

# ======== Датасет ========
class DefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = files

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Image transform: bilinear resize, tensor in [0,1]
        img = transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)(img)
        img = transforms.ToTensor()(img)

        # Mask transform: nearest resize to keep labels, binarize to {0,1}
        mask = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)(mask)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return img, mask

    def __len__(self):
        return len(self.files)

# ======== Настройки и разбиение train/val ========
img_dir = "ct_images/"
mask_dir = "masks_f/"

# Сопоставляем только те файлы, для которых есть и изображение, и маска
img_files = {f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
common_files = sorted(list(img_files & mask_files))

# Простейшее разбиение 90/10
num_total = len(common_files)
num_val = max(1, int(0.1 * num_total))
train_files = common_files[num_val:]
val_files = common_files[:num_val]

train_dataset = DefectDataset(img_dir, mask_dir, train_files)
val_dataset = DefectDataset(img_dir, mask_dir, val_files)

loader_kwargs = dict(batch_size=16, shuffle=True, num_workers=0,
                     pin_memory=True if device.type == "cuda" else False,
                     persistent_workers=False)

train_loader = DataLoader(train_dataset, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=loader_kwargs["pin_memory"], persistent_workers=False)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Используем BCE с логитами + Dice для сегментации
criterion_bce = nn.BCEWithLogitsLoss()

def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = torch.sum(probs * targets, dim=(1,2,3))
    union = torch.sum(probs, dim=(1,2,3)) + torch.sum(targets, dim=(1,2,3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ======== Обучение ========
epochs = 100
best_val_dice = 0.0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(img)
            loss = criterion_bce(logits, mask) + dice_loss_with_logits(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    train_avg_loss = total_loss / max(1, len(train_loader))

    # Валидация
    model.eval()
    val_dice_accum = 0.0
    val_bce_accum = 0.0
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(img)
            val_bce = criterion_bce(logits, mask).item()
            val_dice = 1.0 - dice_loss_with_logits(logits, mask).item()
            val_bce_accum += val_bce
            val_dice_accum += val_dice

    val_avg_bce = val_bce_accum / max(1, len(val_loader))
    val_avg_dice = val_dice_accum / max(1, len(val_loader))

    print(f"Epoch {epoch+1}: train_loss={train_avg_loss:.4f} | val_bce={val_avg_bce:.4f} | val_dice={val_avg_dice:.4f}")

    # Сохранение лучшей модели по Dice
    if val_avg_dice >= best_val_dice:
        best_val_dice = val_avg_dice
        torch.save(model.state_dict(), "defect_unet_best.pth")

    # Сохраняем последнюю модель каждый эпоху
    torch.save(model.state_dict(), "defect_unet_last.pth")

# ======== Итог ========
torch.save(model.state_dict(), "defect_unet_final.pth")
print("\n✅ Обучение завершено. Лучший чекпоинт: defect_unet_best.pth; последний: defect_unet_last.pth; финальный дамп: defect_unet_final.pth")
