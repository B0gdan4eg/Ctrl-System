import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ======== UNet ========
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

# ======== Настройки ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

model_path = "defect_unet_final.pth"
img_folder = r"ct_images_test"

# ======== Загружаем модель ========
model = UNet().to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ======== Визуализация с контурами ========
for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)
    img = Image.open(img_path).convert("L")
    orig_width, orig_height = img.size

    # Масштабирование с сохранением пропорций
    scale = 256 / max(orig_width, orig_height)
    new_w, new_h = int(orig_width * scale), int(orig_height * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Паддинг до 256x256
    pad_w = 256 - new_w
    pad_h = 256 - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    img_padded = np.array(img_resized)
    img_padded = cv2.copyMakeBorder(img_padded, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Прогон через модель
    with torch.no_grad():
        logits = model(img_tensor).squeeze().cpu().numpy()
        pred = 1.0 / (1.0 + np.exp(-logits))

    # Убираем паддинг
    pred_cropped = pred[top:256 - bottom, left:256 - right]

    # Масштабируем обратно к исходному размеру
    mask_resized = cv2.resize(pred_cropped, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

    # Преобразуем изображение в цветное для рисования контуров
    img_cv = np.array(img)
    img_color = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    # Находим контуры и рисуем красным
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)

    # Отображаем
    plt.figure(figsize=(8, 8))
    plt.imshow(img_color)
    plt.title(f"Defects detected: {img_name}")
    plt.axis("off")
    plt.show()
