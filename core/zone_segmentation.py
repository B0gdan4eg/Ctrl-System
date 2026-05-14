"""
Нейросетевая сегментация рабочих зон фильтра.

ZoneSegmenter оборачивает U-Net (обученную на синтетической разметке от
find_top_zones_improved + Otsu refinement). На вход — grayscale uint8,
на выход — список контуров зон, совместимый с find_top_zones() для
drop-in замены в ProcessingWorker.

Использование:
    seg = ZoneSegmenter(model_path="models/zone_unet.pth")
    contours = seg.predict_contours(img_gray, top_n=5)

Модель грузится лениво при первом вызове — ничего не делает на старте.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from models.unet import UNet


class ZoneSegmenter:
    """U-Net сегментация зон с автоматическим выбором CPU/GPU."""

    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[UNet] = None
        self._image_size: int = 512
        self._loaded = False
        self._load_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """True если модель доступна и загрузилась без ошибок."""
        if self._loaded:
            return self._model is not None
        return self.model_path.exists() and self._load_error is None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not self.model_path.exists():
            self._load_error = f"Файл модели не найден: {self.model_path}"
            return False
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
            features = ckpt.get('features', [32, 64, 128, 256])
            self._image_size = ckpt.get('image_size', 512)
            self._model = UNet(in_channels=1, out_channels=1, features=features).to(self.device)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.eval()
            return True
        except Exception as e:
            self._load_error = f"Ошибка загрузки модели: {e}"
            self._model = None
            return False

    @torch.no_grad()
    def predict_mask(self, img_gray: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        """Возвращает бинарную маску зон в исходном разрешении или None при ошибке."""
        if not self._ensure_loaded():
            return None
        if img_gray is None or img_gray.size == 0:
            return None
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        if img_gray.dtype != np.uint8:
            img_gray = self._to_uint8(img_gray)

        h, w = img_gray.shape
        resized = cv2.resize(img_gray, (self._image_size, self._image_size),
                             interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        logits = self._model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

        mask_small = (probs > threshold).astype(np.uint8) * 255
        return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    def predict_contours(
        self,
        img_gray: np.ndarray,
        top_n: int = 5,
        min_area_ratio: float = 0.3,
        max_area_ratio: float = 0.95,
        min_absolute_area: int = 10000,
        threshold: float = 0.5,
    ) -> List[np.ndarray]:
        """Drop-in замена find_top_zones: возвращает контуры зон в порядке слева→направо."""
        mask = self.predict_mask(img_gray, threshold=threshold)
        if mask is None:
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        img_area = img_gray.shape[0] * img_gray.shape[1]
        good = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_absolute_area:
                continue
            if area / img_area > max_area_ratio:
                continue
            good.append(cnt)

        if not good:
            return []

        good.sort(key=cv2.contourArea, reverse=True)

        # Отсев слишком мелких относительно крупнейшего
        max_area = cv2.contourArea(good[0])
        good = [c for c in good if cv2.contourArea(c) >= max_area * min_area_ratio]

        good = good[:top_n]
        good.sort(key=lambda c: cv2.boundingRect(c)[0])
        return good

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        if p99 <= p1:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = (img.astype(np.float32) - p1) / (p99 - p1) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
