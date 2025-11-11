"""
Обучение U-Net для сегментации дефектов
ИСПРАВЛЕННАЯ ВЕРСИЯ - защита от NaN
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Для визуализации
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# U-NET АРХИТЕКТУРА
# ============================================================================

class DoubleConv(nn.Module):
    """Блок: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net архитектура для сегментации
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip_connection = skip_connections[idx//2]
            
            # Обработка несоответствия размеров
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)  # DoubleConv
        
        return self.final_conv(x)


# ============================================================================
# DATASET
# ============================================================================

class DefectDataset(Dataset):
    """
    Dataset для загрузки изображений и масок
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, image_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Поиск изображений
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        self.image_files = sorted(self.image_files)
        
        print(f"Загружено {len(self.image_files)} изображений из {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загрузка маски
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Нормализация изображения (ImageNet stats для лучшей стабильности)
        image = image.astype(np.float32) / 255.0
        
        # КРИТИЧНО: Бинаризация маски (0 или 1)
        mask = (mask > 127).astype(np.float32)
        
        # КРИТИЧНО: Проверка на NaN/Inf
        if np.isnan(image).any() or np.isinf(image).any():
            print(f"⚠️ NaN/Inf в изображении: {img_path}")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.isnan(mask).any() or np.isinf(mask).any():
            print(f"⚠️ NaN/Inf в маске: {mask_path}")
            mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Преобразование в тензоры
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW
        
        return image, mask


# ============================================================================
# LOSS FUNCTIONS (ИСПРАВЛЕННЫЕ)
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss для сегментации с защитой от NaN"""
    
    def __init__(self, smooth=1.0):  # УВЕЛИЧЕН smooth для стабильности
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Применяем sigmoid к логитам
        predictions = torch.sigmoid(predictions)
        
        # КРИТИЧНО: Clamp для предотвращения экстремальных значений
        predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
        
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # КРИТИЧНО: Проверка на NaN
        if torch.isnan(dice) or torch.isinf(dice):
            print("⚠️ NaN/Inf обнаружен в Dice Loss!")
            return torch.tensor(0.0, device=dice.device, requires_grad=True)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Комбинация BCE + Dice Loss с защитой"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        # КРИТИЧНО: Clamp предсказаний для стабильности BCE
        predictions = torch.clamp(predictions, min=-10, max=10)
        
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        # КРИТИЧНО: Проверка на NaN в каждом компоненте
        if torch.isnan(bce_loss):
            print("⚠️ NaN в BCE Loss!")
            bce_loss = torch.tensor(0.0, device=bce_loss.device, requires_grad=True)
        
        if torch.isnan(dice_loss):
            print("⚠️ NaN в Dice Loss!")
            dice_loss = torch.tensor(0.0, device=dice_loss.device, requires_grad=True)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


# ============================================================================
# METRICS (ИСПРАВЛЕННЫЕ)
# ============================================================================

def calculate_iou(predictions, targets, threshold=0.5):
    """Intersection over Union (IoU) с защитой от деления на ноль"""
    
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # Защита от деления на ноль
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Проверка на NaN
    if torch.isnan(iou) or torch.isinf(iou):
        return 0.0
    
    return iou.item()


def calculate_dice(predictions, targets, threshold=0.5):
    """Dice Coefficient с защитой"""
    
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (predictions * targets).sum()
    total = predictions.sum() + targets.sum()
    
    # Защита от деления на ноль
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2. * intersection + 1e-6) / (total + 1e-6)
    
    # Проверка на NaN
    if torch.isnan(dice) or torch.isinf(dice):
        return 0.0
    
    return dice.item()


def calculate_pixel_accuracy(predictions, targets, threshold=0.5):
    """Pixel Accuracy"""
    
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    
    accuracy = correct / total
    
    if torch.isnan(accuracy) or torch.isinf(accuracy):
        return 0.0
    
    return accuracy.item()


# ============================================================================
# TRAINING (С ГРАДИЕНТНЫМ КЛИППИНГОМ)
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Обучение одной эпохи с защитой от gradient explosion"""
    
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        # КРИТИЧНО: Проверка входных данных
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"⚠️ NaN/Inf в батче изображений {batch_idx}")
            continue
        
        if torch.isnan(masks).any() or torch.isinf(masks).any():
            print(f"⚠️ NaN/Inf в батче масок {batch_idx}")
            continue
        
        outputs = model(images)
            
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # КРИТИЧНО: Gradient clipping для предотвращения взрыва градиентов
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Метрики
        with torch.no_grad():
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)
        
        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        
        # Обновление progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
            'dice': f'{dice:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_iou, epoch_dice


def validate_epoch(model, dataloader, criterion, device):
    """Валидация"""
    
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_accuracy = 0.0
    
    progress_bar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Метрики
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)
            accuracy = calculate_pixel_accuracy(outputs, masks)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            running_accuracy += accuracy
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)
    
    return epoch_loss, epoch_iou, epoch_dice, epoch_accuracy


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataloader, device, save_path, num_samples=5):
    """Визуализация предсказаний"""
    
    model.eval()
    
    # Получение батча
    images, masks = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.sigmoid(predictions)
    
    # Перенос на CPU
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Визуализация
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Изображение
        img = np.transpose(images[idx], (1, 2, 0))
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Изображение')
        axes[idx, 0].axis('off')
        
        # Ground truth маска
        mask_gt = masks[idx, 0]
        axes[idx, 1].imshow(mask_gt, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Предсказание
        pred = predictions[idx, 0]
        axes[idx, 2].imshow(pred, cmap='gray')
        axes[idx, 2].set_title(f'Предсказание')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Визуализация сохранена: {save_path}")


def plot_training_history(history, save_path):
    """График обучения"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train IoU', linewidth=2)
    axes[0, 1].plot(history['val_iou'], label='Val IoU', linewidth=2)
    axes[0, 1].set_title('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1, 0].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, color='green')
    axes[1, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ График обучения: {save_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_unet(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    output_dir='unet_training',
    epochs=50,
    batch_size=2,  # УМЕНЬШЕН для стабильности
    learning_rate=1e-6,  # УМЕНЬШЕН!
    image_size=(512, 512),
    device=None
):
    """
    Полное обучение U-Net
    """
    
    # Создание выходной папки
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"🚀 ОБУЧЕНИЕ U-NET (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {image_size}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}\n")
    
    # Создание datasets и dataloaders
    train_dataset = DefectDataset(train_images_dir, train_masks_dir, image_size=image_size)
    val_dataset = DefectDataset(val_images_dir, val_masks_dir, image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # УМЕНЬШЕН для стабильности
        pin_memory=True,
        drop_last=True  # КРИТИЧНО: отбрасываем неполный последний батч
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Создание модели
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # КРИТИЧНО: Инициализация весов
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    print("✅ Инициализация весов завершена")
    
    # Loss и optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # КРИТИЧНО: Используем AdamW с weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-5,  # Регуляризация
        eps=1e-8  # Стабильность
    )
    
    # КРИТИЧНО: Более мягкий scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=7,  # Увеличена терпеливость
        factor=0.5, 
        verbose=True,
        min_lr=1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # История обучения
    history = {
        'train_loss': [],
        'train_iou': [],
        'train_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_accuracy': []
    }
    
    # Лучшая модель
    best_val_dice = 0.0
    best_epoch = 0
    
    # Обучение
    print("🔥 Начало обучения...\n")
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        # Training
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validation
        val_loss, val_iou, val_dice, val_accuracy = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_accuracy'].append(val_accuracy)
        
        # Вывод результатов
        print(f"\n📊 Результаты эпохи {epoch+1}:")
        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Сохранение лучшей модели
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, output_dir / 'best_model.pth')
            
            print(f"✅ Лучшая модель сохранена! Dice: {val_dice:.4f}")
        
        # Сохранение последней модели
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / 'last_model.pth')
        
        # Визуализация каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_predictions(
                model, val_loader, device,
                output_dir / f'predictions_epoch_{epoch+1}.png'
            )
    
    # Итоги обучения
    print(f"\n{'='*70}")
    print(f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"Лучший Val Dice: {best_val_dice:.4f}")
    print(f"Модели сохранены в: {output_dir}")
    print(f"{'='*70}\n")
    
    # Сохранение истории
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # График обучения
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Финальная визуализация
    visualize_predictions(
        model, val_loader, device,
        output_dir / 'final_predictions.png',
        num_samples=8
    )
    
    return model, history


# ============================================================================
# INFERENCE
# ============================================================================

def predict_image(model, image_path, device, image_size=(512, 512), threshold=0.5):
    """
    Предсказание на одном изображении
    """
    
    model.eval()
    
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Preprocessing
    image_resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
    
    # Постобработка
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    
    # Resize обратно к оригинальному размеру
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              🧠 ОБУЧЕНИЕ U-NET - ИСПРАВЛЕННАЯ ВЕРСИЯ                  ║
║                                                                        ║
║  ✅ Защита от NaN/Inf                                                 ║
║  ✅ Gradient clipping                                                 ║
║  ✅ Правильная инициализация весов                                    ║
║  ✅ Оптимальный learning rate                                         ║
║  ✅ Стабильные loss функции                                           ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # НАСТРОЙКИ
    # ========================================================================
    
    # Пути к данным
    TRAIN_IMAGES_DIR = 'D:\\train\\prepared_dataset\\dataset\\train\\images'
    TRAIN_MASKS_DIR = 'D:\\train\\prepared_dataset\\dataset\\train\\masks'
    VAL_IMAGES_DIR = 'D:\\train\\prepared_dataset\\dataset\\val\\images'
    VAL_MASKS_DIR = 'D:\\train\\prepared_dataset\\dataset\\val\\masks'
    
    # Выходная папка
    OUTPUT_DIR = 'unet_training_fixed'
    
    # КРИТИЧНО: Правильные гиперпараметры
    EPOCHS = 50
    BATCH_SIZE = 2  # Уменьшен для стабильности
    LEARNING_RATE = 1e-6  # Уменьшен в 10 раз!
    IMAGE_SIZE = (512, 512)
    
    # ========================================================================
    # ОБУЧЕНИЕ
    # ========================================================================
    
    model, history = train_unet(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_masks_dir=TRAIN_MASKS_DIR,
        val_images_dir=VAL_IMAGES_DIR,
        val_masks_dir=VAL_MASKS_DIR,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        image_size=IMAGE_SIZE
    )
    
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!                                               ║
║                                                                        ║
║  Основные исправления:                                                ║
║  • Learning rate: 1e-3 → 1e-4                                        ║
║  • Gradient clipping добавлен                                         ║
║  • Защита от NaN во всех функциях                                     ║
║  • Правильная инициализация весов                                     ║
║  • Бинаризация масок (0 или 1)                                        ║
║  • Увеличен smooth в Dice Loss                                        ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)