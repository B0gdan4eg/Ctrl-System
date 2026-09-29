"""
Инференс U-Net на test выборке
Автоматическая обработка всех изображений из test папки
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import sys
sys.path.append('.')
from TRAIN import UNet


def load_model(model_path, device=None):
    """Загрузка обученной модели"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def predict_image(model, image_path, device, image_size=(512, 512), threshold=0.5):
    """Предсказание на одном изображении"""
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    
    image_resized = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
    
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return image_rgb, mask


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """Наложение маски на изображение"""
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return result


def calculate_metrics(pred_mask, gt_mask):
    """Расчет метрик для одного изображения"""
    pred_bin = (pred_mask > 127).astype(np.float32)
    gt_bin = (gt_mask > 127).astype(np.float32)
    
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-6)
    
    total_pixels = pred_bin.size
    correct_pixels = np.sum(pred_bin == gt_bin)
    accuracy = correct_pixels / total_pixels
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy)
    }


def process_test_dataset(
    model_path,
    test_images_dir,
    test_masks_dir,
    output_dir,
    threshold=0.5,
    save_visualizations=True,
    max_visualizations=20
):
    """Обработка всей test выборки"""
    
    test_images_dir = Path(test_images_dir)
    test_masks_dir = Path(test_masks_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    predictions_dir = output_dir / 'predictions'
    overlays_dir = output_dir / 'overlays'
    predictions_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 ИНФЕРЕНС НА TEST ВЫБОРКЕ")
    print(f"{'='*70}")
    print(f"Модель: {model_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Test masks: {test_masks_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {threshold}")
    print(f"{'='*70}\n")
    
    model, device = load_model(model_path)
    print(f"✅ Модель загружена на {device}\n")
    
    image_files = list(test_images_dir.glob('*.png')) + list(test_images_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"❌ Не найдено изображений в {test_images_dir}")
        return
    
    print(f"Найдено изображений: {len(image_files)}\n")
    
    results = []
    vis_count = 0
    
    for img_path in tqdm(image_files, desc="Обработка"):
        try:
            image_rgb, pred_mask = predict_image(model, img_path, device, threshold=threshold)
            
            gt_mask_path = test_masks_dir / f"{img_path.stem}.png"
            if gt_mask_path.exists():
                gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (image_rgb.shape[1], image_rgb.shape[0]))
                
                metrics = calculate_metrics(pred_mask, gt_mask)
                has_gt = True
            else:
                metrics = None
                gt_mask = None
                has_gt = False
            
            cv2.imwrite(str(predictions_dir / f"{img_path.stem}_pred.png"), pred_mask)
            
            overlay = overlay_mask(image_rgb, pred_mask)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlays_dir / f"{img_path.stem}_overlay.png"), overlay_bgr)
            
            defect_pixels = np.sum(pred_mask > 0)
            total_pixels = pred_mask.size
            defect_percentage = (defect_pixels / total_pixels) * 100
            
            result = {
                'filename': img_path.name,
                'defect_pixels': int(defect_pixels),
                'total_pixels': int(total_pixels),
                'defect_percentage': float(defect_percentage),
                'has_ground_truth': has_gt
            }
            
            if metrics:
                result.update(metrics)
            
            results.append(result)
            
            if save_visualizations and vis_count < max_visualizations:
                fig, axes = plt.subplots(1, 4 if has_gt else 3, figsize=(16 if has_gt else 12, 4))
                
                axes[0].imshow(image_rgb)
                axes[0].set_title(f'{img_path.name}')
                axes[0].axis('off')
                
                if has_gt:
                    axes[1].imshow(gt_mask, cmap='gray')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_mask, cmap='gray')
                    axes[2].set_title(f'Prediction\nDice: {metrics["dice"]:.3f}')
                    axes[2].axis('off')
                    
                    axes[3].imshow(overlay)
                    axes[3].set_title('Overlay')
                    axes[3].axis('off')
                else:
                    axes[1].imshow(pred_mask, cmap='gray')
                    axes[1].set_title('Prediction')
                    axes[1].axis('off')
                    
                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'vis_{img_path.stem}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                vis_count += 1
        
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {img_path.name}: {e}")
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"{'='*70}")
    print(f"Обработано изображений: {len(results)}")
    
    if len(results) > 0:
        avg_defect = np.mean([r['defect_percentage'] for r in results])
        print(f"Средний процент дефектов: {avg_defect:.2f}%")
        
        with_defects = sum(1 for r in results if r['defect_percentage'] > 0.1)
        print(f"Изображений с дефектами: {with_defects}")
        print(f"Изображений без дефектов: {len(results) - with_defects}")
        
        results_with_gt = [r for r in results if r.get('iou') is not None]
        if len(results_with_gt) > 0:
            print(f"\n📊 Метрики (на {len(results_with_gt)} изображениях с ground truth):")
            avg_iou = np.mean([r['iou'] for r in results_with_gt])
            avg_dice = np.mean([r['dice'] for r in results_with_gt])
            avg_acc = np.mean([r['accuracy'] for r in results_with_gt])
            
            print(f"   Средний IoU: {avg_iou:.4f}")
            print(f"   Средний Dice: {avg_dice:.4f}")
            print(f"   Средняя Accuracy: {avg_acc:.4f}")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            iou_values = [r['iou'] for r in results_with_gt]
            dice_values = [r['dice'] for r in results_with_gt]
            acc_values = [r['accuracy'] for r in results_with_gt]
            
            axes[0].hist(iou_values, bins=20, edgecolor='black')
            axes[0].axvline(avg_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_iou:.3f}')
            axes[0].set_title('IoU Distribution')
            axes[0].set_xlabel('IoU')
            axes[0].set_ylabel('Count')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].hist(dice_values, bins=20, edgecolor='black')
            axes[1].axvline(avg_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dice:.3f}')
            axes[1].set_title('Dice Distribution')
            axes[1].set_xlabel('Dice')
            axes[1].set_ylabel('Count')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].hist(acc_values, bins=20, edgecolor='black')
            axes[2].axvline(avg_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_acc:.3f}')
            axes[2].set_title('Accuracy Distribution')
            axes[2].set_xlabel('Accuracy')
            axes[2].set_ylabel('Count')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n📈 График распределения метрик: {output_dir / 'metrics_distribution.png'}")
    
    print(f"\n📁 Результаты сохранены в: {output_dir}")
    print(f"   • predictions/ - предсказанные маски")
    print(f"   • overlays/ - изображения с наложением")
    print(f"   • results.json - детальная статистика")
    if save_visualizations:
        print(f"   • vis_*.png - визуализации ({vis_count} шт.)")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    
    MODEL_PATH = 'D:\\train\\unet_training_fixed\\best_model.pth'
    TEST_IMAGES_DIR = 'D:\\train\\prepared_dataset\\dataset\\test\\images'
    TEST_MASKS_DIR = 'D:\\train\\prepared_dataset\\dataset\\test\\masks'
    OUTPUT_DIR = 'test_results'
    THRESHOLD = 0.5
    
    results = process_test_dataset(
        model_path=MODEL_PATH,
        test_images_dir=TEST_IMAGES_DIR,
        test_masks_dir=TEST_MASKS_DIR,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        save_visualizations=True,
        max_visualizations=20
    )