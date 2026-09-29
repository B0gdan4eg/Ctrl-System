"""
Скрипт для подготовки данных для обучения модели детекции дефектов
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split
import json


class DefectDataPreprocessor:
    """Класс для предобработки изображений с дефектами"""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        
    def normalize_image(self, image, method='standard'):
        """
        Нормализация изображения
        
        Args:
            image: входное изображение
            method: метод нормализации ('standard', 'minmax', 'clahe', 'adaptive')
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if method == 'standard':
            # Стандартизация (mean=0, std=1)
            normalized = (gray - np.mean(gray)) / (np.std(gray) + 1e-7)
            normalized = ((normalized - normalized.min()) * 255 / 
                         (normalized.max() - normalized.min())).astype(np.uint8)
            
        elif method == 'minmax':
            # Min-Max нормализация [0, 255]
            normalized = ((gray - gray.min()) * 255 / 
                         (gray.max() - gray.min() + 1e-7)).astype(np.uint8)
            
        elif method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(gray)
            
        elif method == 'adaptive':
            # Адаптивная нормализация
            normalized = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        return normalized
    
    def remove_noise(self, image):
        """Удаление шума"""
        # Bilateral filter сохраняет края
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def enhance_defects(self, image):
        """Усиление контраста дефектов"""
        # Морфологические операции для выделения дефектов
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Top-hat transform для выделения темных дефектов
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Комбинирование
        enhanced = cv2.subtract(image, blackhat)
        
        return enhanced
    
    def preprocess_pipeline(self, image_path, save_steps=False):
        """
        Полный пайплайн предобработки
        
        Args:
            image_path: путь к изображению
            save_steps: сохранять ли промежуточные шаги
        """
        # Загрузка
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        steps = {'original': img.copy()}
        
        # 1. Удаление шума
        img_denoised = self.remove_noise(img)
        steps['denoised'] = img_denoised
        
        # 2. Нормализация CLAHE
        img_normalized = self.normalize_image(img_denoised, method='clahe')
        steps['normalized'] = img_normalized
        
        # 3. Усиление дефектов
        img_enhanced = self.enhance_defects(img_normalized)
        steps['enhanced'] = img_enhanced
        
        # 4. Изменение размера
        img_resized = cv2.resize(img_enhanced, self.target_size, 
                                interpolation=cv2.INTER_AREA)
        steps['final'] = img_resized
        
        if save_steps:
            return img_resized, steps
        return img_resized


class DefectAugmentor:
    """Класс для аугментации данных"""
    
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            # Геометрические трансформации
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=45, 
                p=0.5
            ),
            
            # Изменение яркости и контраста
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            
            # Добавление шума
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Размытие
            A.OneOf([
                A.MotionBlur(p=1),
                A.GaussianBlur(p=1),
            ], p=0.3),
            
            # Изменение гаммы
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Elastic transform для имитации деформаций
            A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                alpha_affine=50, 
                p=0.2
            ),
            
            # Grid distortion
            A.GridDistortion(p=0.2),
        ])
        
    def augment(self, image, mask=None, num_augmentations=5):
        """
        Применить аугментацию
        
        Args:
            image: входное изображение
            mask: маска дефектов (опционально)
            num_augmentations: количество аугментированных версий
        """
        augmented_images = []
        augmented_masks = []
        
        for _ in range(num_augmentations):
            if mask is not None:
                augmented = self.augmentation_pipeline(image=image, mask=mask)
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])
            else:
                augmented = self.augmentation_pipeline(image=image)
                augmented_images.append(augmented['image'])
        
        if mask is not None:
            return augmented_images, augmented_masks
        return augmented_images


class DatasetOrganizer:
    """Класс для организации датасета"""
    
    def __init__(self, output_dir='processed_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Создание структуры папок
        (self.output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train' / 'masks').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'val' / 'masks').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'test' / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'test' / 'masks').mkdir(parents=True, exist_ok=True)
        
    def split_dataset(self, image_paths, train_ratio=0.7, val_ratio=0.15):
        """Разделение на train/val/test"""
        test_ratio = 1 - train_ratio - val_ratio
        
        # Train и temp (val+test)
        train_imgs, temp_imgs = train_test_split(
            image_paths, 
            test_size=(1-train_ratio), 
            random_state=42
        )
        
        # Val и test
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            test_size=(test_ratio/(test_ratio+val_ratio)), 
            random_state=42
        )
        
        return train_imgs, val_imgs, test_imgs
    
    def save_dataset_info(self, train_imgs, val_imgs, test_imgs):
        """Сохранение информации о датасете"""
        info = {
            'train_size': len(train_imgs),
            'val_size': len(val_imgs),
            'test_size': len(test_imgs),
            'total_size': len(train_imgs) + len(val_imgs) + len(test_imgs),
            'train_images': [str(p) for p in train_imgs],
            'val_images': [str(p) for p in val_imgs],
            'test_images': [str(p) for p in test_imgs],
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return info


def visualize_preprocessing_steps(preprocessor, image_path, output_path='preprocessing_steps.png'):
    """Визуализация этапов предобработки"""
    _, steps = preprocessor.preprocess_pipeline(image_path, save_steps=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    step_names = ['original', 'denoised', 'normalized', 'enhanced', 'final']
    titles = ['Оригинал', 'Шумоподавление', 'Нормализация (CLAHE)', 
              'Усиление дефектов', 'Финальный результат']
    
    for idx, (step_name, title) in enumerate(zip(step_names, titles)):
        if idx < len(axes) - 1:
            axes[idx].imshow(steps[step_name], cmap='gray')
            axes[idx].set_title(title, fontsize=12)
            axes[idx].axis('off')
    
    # Гистограмма финального изображения
    axes[-1].hist(steps['final'].ravel(), bins=50, color='blue', alpha=0.7)
    axes[-1].set_title('Гистограмма финального изображения')
    axes[-1].set_xlabel('Интенсивность пикселей')
    axes[-1].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Визуализация сохранена: {output_path}")
    plt.close()


def visualize_augmentations(augmentor, image, output_path='augmentations.png'):
    """Визуализация аугментаций"""
    augmented_images = augmentor.augment(image, num_augmentations=8)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    # Оригинал
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Оригинал', fontsize=12)
    axes[0].axis('off')
    
    # Аугментированные версии
    for idx, aug_img in enumerate(augmented_images):
        axes[idx + 1].imshow(aug_img, cmap='gray')
        axes[idx + 1].set_title(f'Аугментация {idx + 1}', fontsize=12)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Визуализация аугментаций сохранена: {output_path}")
    plt.close()


# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("Подготовка данных для детекции дефектов")
    print("=" * 60)
    
    # Инициализация
    preprocessor = DefectDataPreprocessor(target_size=(512, 512))
    augmentor = DefectAugmentor()
    organizer = DatasetOrganizer(output_dir='processed_dataset')
    
    # Пример работы с одним изображением
    test_image = '/mnt/user-data/uploads/1761639702426_image.png'
    
    print("\n1. Предобработка изображения...")
    processed_img = preprocessor.preprocess_pipeline(test_image)
    print(f"   Размер обработанного изображения: {processed_img.shape}")
    
    print("\n2. Визуализация этапов предобработки...")
    visualize_preprocessing_steps(preprocessor, test_image)
    
    print("\n3. Аугментация данных...")
    augmented = augmentor.augment(processed_img, num_augmentations=8)
    print(f"   Создано {len(augmented)} аугментированных версий")
    
    print("\n4. Визуализация аугментаций...")
    visualize_augmentations(augmentor, processed_img)
    
    print("\n" + "=" * 60)
    print("Готово! Проверьте созданные визуализации.")
    print("=" * 60)