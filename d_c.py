#!/usr/bin/env python3
"""
Диагностика камеры

Copyright (c) 2024 Defect Detection System
Licensed under the MIT License. See LICENSE file for details.
"""

import cv2
import sys
import platform


def diagnose_camera():
    """Полная диагностика камеры"""
    
    print("="*70)
    print("ДИАГНОСТИКА КАМЕРЫ")
    print("="*70)
    
    # Информация о системе
    print(f"\n📋 Система:")
    print(f"   ОС: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   OpenCV: {cv2.__version__}")
    
    # Проверка доступных backends
    print(f"\n🔌 Доступные backends:")
    backends = [
        ('CAP_DSHOW', cv2.CAP_DSHOW),      # DirectShow (Windows)
        ('CAP_MSMF', cv2.CAP_MSMF),        # Media Foundation (Windows)
        ('CAP_V4L2', cv2.CAP_V4L2),        # Video4Linux (Linux)
        ('CAP_AVFOUNDATION', cv2.CAP_AVFOUNDATION),  # AVFoundation (macOS)
        ('CAP_ANY', cv2.CAP_ANY),          # Auto
    ]
    
    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"   ✅ {name}")
                cap.release()
            else:
                print(f"   ❌ {name}")
        except:
            print(f"   ❌ {name} (not available)")
    
    # Поиск камер
    print(f"\n🔍 Поиск камер (ID 0-10):")
    found_cameras = []
    
    for i in range(11):
        # Пробуем разные backends
        for backend_name, backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Пытаемся прочитать кадр
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        print(f"   ✅ Камера {i} ({backend_name}): {width}x{height} @ {fps}fps")
                        found_cameras.append((i, backend, backend_name, width, height, fps))
                        cap.release()
                        break
                    cap.release()
            except Exception as e:
                pass
    
    if not found_cameras:
        print("   ❌ Камеры не найдены!")
        print("\n💡 Возможные причины:")
        print("   1. Камера не подключена")
        print("   2. Камера используется другим приложением")
        print("   3. Нет разрешений на доступ к камере")
        print("   4. Драйверы камеры не установлены")
        
        if platform.system() == "Linux":
            print("\n🔧 Для Linux попробуйте:")
            print("   sudo apt-get install v4l-utils")
            print("   v4l2-ctl --list-devices")
            print("   ls -l /dev/video*")
        
        elif platform.system() == "Windows":
            print("\n🔧 Для Windows попробуйте:")
            print("   1. Проверьте Диспетчер устройств")
            print("   2. Закройте другие программы (Skype, Teams, Zoom)")
            print("   3. Установите драйверы камеры")
        
        elif platform.system() == "Darwin":
            print("\n🔧 Для macOS попробуйте:")
            print("   1. Системные настройки → Защита и безопасность → Камера")
            print("   2. Дайте разрешение Python на использование камеры")
        
        return
    
    print(f"\n✅ Найдено камер: {len(found_cameras)}")
    
    # Тестирование первой найденной камеры
    print(f"\n🎥 Тестирование камеры 0...")
    camera_id, backend, backend_name, width, height, fps = found_cameras[0]
    
    cap = cv2.VideoCapture(camera_id, backend)
    
    if not cap.isOpened():
        print("   ❌ Не удалось открыть камеру")
        return
    
    print("   ✅ Камера открыта")
    print(f"   Backend: {backend_name}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Захват тестовых кадров
    print("\n📸 Захват 10 тестовых кадров...")
    success_count = 0
    
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
            print(f"   Кадр {i+1}: ✅ {frame.shape}")
        else:
            print(f"   Кадр {i+1}: ❌ Ошибка")
    
    cap.release()
    
    print(f"\n📊 Результат: {success_count}/10 успешных кадров")
    
    if success_count == 10:
        print("\n✅ Камера работает отлично!")
    elif success_count > 5:
        print("\n⚠️ Камера работает, но есть проблемы с захватом кадров")
    else:
        print("\n❌ Камера не работает стабильно")
    
    # Рекомендации
    print("\n💡 Рекомендации для приложения:")
    if found_cameras:
        cam_id, backend, backend_name, _, _, _ = found_cameras[0]
        print(f"   Используйте: cv2.VideoCapture({cam_id}, {backend_name})")
        print(f"\n   В коде:")
        print(f"   cap = cv2.VideoCapture({cam_id}, cv2.{backend_name})")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        diagnose_camera()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()