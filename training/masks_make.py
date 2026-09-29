import cv2
import numpy as np
import os

os.makedirs("masks", exist_ok=True)

for file in os.listdir("ct_images"):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join("ct_images", file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Выделяем яркие области (порог можно подбирать)
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Убираем шум
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    cv2.imwrite(os.path.join("masks", file), mask)

print("✅ Маски сгенерированы и сохранены в папку masks/")
