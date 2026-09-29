import os
import shutil
from labelme import utils
import json
import cv2
import numpy as np

# Папка с JSON файлами LabelMe
json_folder = r"D:\AI\masks"  # <-- твоя папка с .json
# Папка для готовых масок
mask_folder = r"D:\AI\masks_f"  # <-- папка для масок

os.makedirs(mask_folder, exist_ok=True)

# Проходим по всем JSON файлам
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(json_folder, filename)

        # Загружаем JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Извлекаем изображение
        imageData = data.get("imageData")
        if imageData is None:
            # Если imageData нет, пробуем прочитать из файла
            img_file = os.path.join(json_folder, data["imagePath"])
            image = cv2.imread(img_file)
        else:
            image = utils.img_b64_to_arr(imageData)

        # Генерируем словарь label_name_to_value
        label_name_to_value = {shape["label"]: 1 for shape in data["shapes"]}
        if not label_name_to_value:
            label_name_to_value = {"defect": 1}  # На всякий случай

        # Генерируем маску
        lbl, _ = utils.shapes_to_label(image.shape, data["shapes"], label_name_to_value)
        mask = (lbl > 0).astype(np.uint8) * 255  # Белый = дефект, черный = фон

        # Сохраняем маску
        mask_name = filename.replace(".json", ".png")
        mask_path = os.path.join(mask_folder, mask_name)
        cv2.imwrite(mask_path, mask)
        print(f"Маска создана: {mask_path}")
