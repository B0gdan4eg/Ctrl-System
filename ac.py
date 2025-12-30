#!/usr/bin/env python3
"""
Тест различных способов открытия камеры
"""

import cv2
import numpy as np

def test_method_1():
    """Метод 1: DSHOW без параметров"""
    print("\n" + "="*70)
    print("МЕТОД 1: CAP_DSHOW (базовый)")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    # Пропуск кадров
    for i in range(50):
        cap.read()
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Яркость: {brightness:.1f}")
        result = brightness > 10
        cap.release()
        return result
    
    cap.release()
    return False


def test_method_2():
    """Метод 2: DSHOW с принудительными настройками"""
    print("\n" + "="*70)
    print("МЕТОД 2: CAP_DSHOW + настройки")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    # Принудительные настройки
    cap.set(cv2.CAP_PROP_SETTINGS, 1)  # Открыть диалог настроек
    
    cap.release()
    return False


def test_method_3():
    """Метод 3: MSMF"""
    print("\n" + "="*70)
    print("МЕТОД 3: CAP_MSMF")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    # Пропуск кадров
    for i in range(50):
        cap.read()
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Яркость: {brightness:.1f}")
        result = brightness > 10
        cap.release()
        return result
    
    cap.release()
    return False


def test_method_4():
    """Метод 4: DSHOW с auto-exposure off"""
    print("\n" + "="*70)
    print("МЕТОД 4: CAP_DSHOW + отключение авто-экспозиции")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    # Отключаем автоэкспозицию
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Экспозиция
    
    # Пропуск кадров
    for i in range(50):
        cap.read()
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Яркость: {brightness:.1f}")
        result = brightness > 10
        cap.release()
        return result
    
    cap.release()
    return False


def test_method_5():
    """Метод 5: Без backend (AUTO)"""
    print("\n" + "="*70)
    print("МЕТОД 5: CAP_ANY (авто)")
    print("="*70)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    backend = cap.getBackendName()
    print(f"📌 Используется backend: {backend}")
    
    # Пропуск кадров
    for i in range(50):
        cap.read()
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Яркость: {brightness:.1f}")
        result = brightness > 10
        cap.release()
        return result
    
    cap.release()
    return False


def test_method_6():
    """Метод 6: DSHOW + FourCC принудительно"""
    print("\n" + "="*70)
    print("МЕТОД 6: CAP_DSHOW + MJPG codec")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    # Устанавливаем MJPG codec
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Пропуск кадров
    for i in range(50):
        cap.read()
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Яркость: {brightness:.1f}")
        result = brightness > 10
        cap.release()
        return result
    
    cap.release()
    return False


def test_method_7():
    """Метод 7: Большой прогрев"""
    print("\n" + "="*70)
    print("МЕТОД 7: CAP_DSHOW + большой прогрев (100 кадров)")
    print("="*70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Не открылась")
        return False
    
    print("⏳ Прогрев камеры (100 кадров)...")
    for i in range(100):
        ret, frame = cap.read()
        if i % 10 == 0:
            if ret and frame is not None:
                brightness = frame.mean()
                print(f"   Кадр {i}: яркость {brightness:.1f}")
    
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"✅ Итоговая яркость: {brightness:.1f}")
        
        if brightness > 10:
            cv2.imwrite('camera_test_working.png', frame)
            print("📸 Сохранен кадр: camera_test_working.png")
            result = True
        else:
            result = False
        
        cap.release()
        return result
    
    cap.release()
    return False


def main():
    """Запуск всех тестов"""
    
    print("="*70)
    print("ТЕСТИРОВАНИЕ РАЗЛИЧНЫХ МЕТОДОВ ОТКРЫТИЯ КАМЕРЫ")
    print("="*70)
    
    methods = [
        # ("Метод 1: DSHOW базовый", test_method_1),
        # ("Метод 3: MSMF", test_method_3),
        ("Метод 4: DSHOW + авто-экспозиция", test_method_4),
        ("Метод 5: AUTO backend", test_method_5),
        ("Метод 6: DSHOW + MJPG", test_method_6),
        ("Метод 7: DSHOW + большой прогрев", test_method_7),
    ]
    
    working_methods = []
    
    for name, method in methods:
        try:
            result = method()
            if result:
                working_methods.append(name)
                print(f"\n✅✅✅ {name} - РАБОТАЕТ!")
        except Exception as e:
            print(f"\n❌ {name} - Ошибка: {e}")
    
    print("\n" + "="*70)
    print("ИТОГИ:")
    print("="*70)
    
    if working_methods:
        print(f"\n✅ Рабочие методы ({len(working_methods)}):")
        for method in working_methods:
            print(f"   • {method}")
    else:
        print("\n❌ НИ ОДИН метод не работает!")
        print("\n💡 Дополнительные действия:")
        print("   1. Переустановите opencv-python:")
        print("      pip uninstall opencv-python")
        print("      pip install opencv-python")
        print("\n   2. Попробуйте opencv-contrib:")
        print("      pip install opencv-contrib-python")
        print("\n   3. Проверьте антивирус - он может блокировать доступ")

if __name__ == '__main__':
    main()