"""
Простой тест камеры
"""

import cv2
import numpy as np
import time

def test_camera_simple():
    """Простой тест камеры с выводом окна"""
    print("🎥 Запуск теста камеры...")
    
    # Пробуем разные backends для Windows
    backends = [
        ('DSHOW', cv2.CAP_DSHOW),
        ('MSMF', cv2.CAP_MSMF),
        ('ANY', cv2.CAP_ANY),
    ]
    
    cap = None
    backend_name = None
    
    for name, backend in backends:
        print(f"🔄 Пробую {name}...")
        cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            time.sleep(1.0)  # Инициализация
            
            # Проверяем несколько кадров
            valid = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    std_val = np.std(frame)
                    mean_val = np.mean(frame)
                    print(f"  Кадр: std={std_val:.1f}, mean={mean_val:.1f}")
                    
                    if std_val > 10 and 10 < mean_val < 245:
                        valid += 1
                time.sleep(0.1)
            
            if valid >= 3:
                backend_name = name
                print(f"✅ Камера открыта через {name}")
                break
            else:
                print(f"❌ {name} - невалидные кадры")
                cap.release()
                cap = None
    
    if cap is None or not cap.isOpened():
        print("❌ Не удалось открыть камеру")
        return
    
    print(f"\n📹 Камера работает через {backend_name}")
    print("Нажмите 'q' для выхода, 's' для скриншота")
    
    # Основной цикл
    frame_count = 0
    valid_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            
            # Статистика кадра
            std_val = np.std(frame)
            mean_val = np.mean(frame)
            
            if std_val > 10:
                valid_count += 1
                status = "✓ OK"
                color = (0, 255, 0)
            else:
                status = "✗ NOISE"
                color = (0, 0, 255)
            
            # Информация на кадре
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Valid: {valid_count}/{frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"STD: {std_val:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Mean: {mean_val:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Camera Test', frame)
            
            # Каждые 30 кадров - статистика
            if frame_count % 30 == 0:
                ratio = (valid_count / frame_count) * 100
                print(f"📊 Кадров: {frame_count}, валидных: {valid_count} ({ratio:.1f}%)")
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Сохранено: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Финальная статистика:")
    print(f"  Всего кадров: {frame_count}")
    print(f"  Валидных: {valid_count}")
    print(f"  Процент успеха: {(valid_count/frame_count)*100:.1f}%")


if __name__ == "__main__":
    test_camera_simple()