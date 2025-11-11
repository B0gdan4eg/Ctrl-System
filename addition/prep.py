"""
Простой скрипт предобработки для ручной разметки в LabelImg
Делает: шумоподавление + resize
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def preprocess_image(image_path, target_size=(512, 512)):
    """
    Предобработка изображения
    1. Шумоподавление (bilateral filter)
    2. Resize до target_size
    """
    # Загрузка
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Ошибка загрузки: {image_path}")
        return None
    
    # 1. Шумоподавление
    img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 2. Resize
    img_resized = cv2.resize(img_denoised, target_size, interpolation=cv2.INTER_AREA)
    
    return img_resized


def process_directory(input_dir, output_dir, target_size=(512, 512)):
    """
    Обработка всей папки с изображениями
    
    Args:
        input_dir: папка с исходными изображениями
        output_dir: папка для сохранения обработанных
        target_size: размер выходных изображений
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Поиск всех изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', 
                       '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(input_dir.glob(ext)))
    
    if len(all_images) == 0:
        print(f"❌ Не найдено изображений в {input_dir}")
        return
    
    print(f"\n📁 Найдено изображений: {len(all_images)}")
    print(f"📐 Размер выходных изображений: {target_size}")
    print(f"💾 Сохранение в: {output_dir}")
    print()
    
    # Обработка
    processed_count = 0
    failed_count = 0
    
    for img_path in tqdm(all_images, desc="Обработка"):
        try:
            # Предобработка
            processed_img = preprocess_image(img_path, target_size)
            
            if processed_img is not None:
                # Сохранение с тем же именем
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), processed_img)
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {img_path.name}: {e}")
            failed_count += 1
    
    # Статистика
    print(f"\n" + "="*60)
    print(f"✅ ГОТОВО!")
    print(f"="*60)
    print(f"Обработано успешно: {processed_count}")
    print(f"Ошибок:             {failed_count}")
    print(f"Сохранено в:        {output_dir}")
    print(f"="*60)
    
    # Сохранение информации
    info = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'target_size': target_size,
        'processed': processed_count,
        'failed': failed_count
    }
    
    with open(output_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def create_labelimg_classes(output_dir):
    """
    Создание файла classes.txt для LabelImg
    """
    output_dir = Path(output_dir)
    
    # Файл с классами (для YOLO формата в LabelImg)
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        f.write('defect\n')  # один класс - "defect"
    
    print(f"✅ Создан файл классов: {classes_file}")
    print(f"   Класс: defect (id=0)")


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              📐 ПРЕДОБРАБОТКА ДЛЯ РУЧНОЙ РАЗМЕТКИ                     ║
║                                                                        ║
║  Что делается:                                                        ║
║  1. Шумоподавление (bilateral filter)                                 ║
║  2. Resize до 512x512                                                 ║
║  3. Сохранение в указанную папку                                      ║
║                                                                        ║
║  После этого:                                                         ║
║  → Откройте LabelImg                                                  ║
║  → Загрузите обработанные изображения                                 ║
║  → Размечайте дефекты                                                 ║
║  → LabelImg создаст .txt файлы с аннотациями                          ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # НАСТРОЙКИ - ИЗМЕНИТЕ ПОД ВАШИ ДАННЫЕ
    # ========================================================================
    
    INPUT_DIR = 'D:\\train\\data'        # 📁 Папка с исходными изображениями
    OUTPUT_DIR = 'D:\\train\\prep_data'       # 📁 Папка для обработанных изображений
    TARGET_SIZE = (512, 512)          # 📐 Размер (ширина, высота)
    
    # ========================================================================
    # ОБРАБОТКА
    # ========================================================================
    
    process_directory(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
    
    # Создание файла классов для LabelImg
    create_labelimg_classes(OUTPUT_DIR)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  🎯 СЛЕДУЮЩИЙ ШАГ - Разметка в LabelImg:                             ║
║                                                                        ║
║  1. Установите LabelImg:                                              ║
║     pip install labelImg                                              ║
║                                                                        ║
║  2. Запустите:                                                        ║
║     labelImg                                                          ║
║                                                                        ║
║  3. В LabelImg:                                                       ║
║     • Open Dir → выберите папку: {OUTPUT_DIR}
║     • Change Save Dir → выберите ту же папку                          ║
║     • View → Auto Save mode (включите)                                ║
║     • Формат: YOLO (слева внизу)                                      ║
║                                                                        ║
║  4. Размечайте:                                                       ║
║     • W - создать bounding box                                        ║
║     • D - следующее изображение                                       ║
║     • A - предыдущее изображение                                      ║
║     • Ctrl+S - сохранить                                              ║
║                                                                        ║
║  5. LabelImg автоматически создаст .txt файлы рядом с изображениями   ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)