"""
Скрипт автоматической сборки Ctrl-System в .exe файл
Использует PyInstaller для создания standalone приложения

Использование:
    python build_exe.py

После успешной сборки .exe файл будет в папке dist/Ctrl-System/
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Исправление кодировки для Windows консоли
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def print_header(text):
    """Печать заголовка"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num, text):
    """Печать шага"""
    print(f"\n[{step_num}] {text}...")


def check_files():
    """Проверка наличия необходимых файлов"""
    print_step(1, "Проверка необходимых файлов")

    required_files = [
        'main.py',
        'best_model.pth',
        'Ctrl-System.spec',
        'config/settings.py',
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"  ❌ Отсутствует: {file}")
        else:
            print(f"  ✅ Найден: {file}")

    if missing_files:
        print(f"\n⚠️  ОШИБКА: Отсутствуют необходимые файлы!")
        return False

    print("\n✅ Все необходимые файлы найдены")
    return True


def check_dependencies():
    """Проверка установки PyInstaller"""
    print_step(2, "Проверка зависимостей")

    try:
        import PyInstaller
        print(f"  ✅ PyInstaller установлен (версия {PyInstaller.__version__})")
        return True
    except ImportError:
        print("  ❌ PyInstaller не установлен!")
        print("\n  Для установки выполните:")
        print("  pip install pyinstaller")
        return False


def clean_build_dirs():
    """Очистка старых build директорий"""
    print_step(3, "Очистка старых build директорий")

    dirs_to_clean = ['build', 'dist']

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"  🗑️  Удаление {dir_name}/")
            shutil.rmtree(dir_name)

    print("  ✅ Очистка завершена")


def build_executable():
    """Запуск PyInstaller для сборки"""
    print_step(4, "Сборка .exe файла с помощью PyInstaller")
    print("\n  ⏳ Это может занять 5-15 минут...")
    print("  📦 PyInstaller упаковывает PyTorch, OpenCV, PySide6...\n")

    # Запуск PyInstaller
    cmd = [sys.executable, '-m', 'PyInstaller', 'Ctrl-System.spec', '--clean']

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Показываем вывод в реальном времени
            text=True
        )

        print("\n  ✅ Сборка завершена успешно!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Ошибка при сборке!")
        print(f"  Код ошибки: {e.returncode}")
        return False


def copy_required_files():
    """Копирование необходимых файлов из _internal в корень"""
    print_step(5, "Копирование необходимых файлов")

    dist_root = Path('dist/Ctrl-System')
    internal_dir = dist_root / '_internal'

    files_to_copy = [
        ('best_model.pth', 'Модель U-Net'),
        ('icon.ico', 'Иконка приложения'),
    ]

    copied_count = 0
    for filename, description in files_to_copy:
        source = internal_dir / filename
        dest = dist_root / filename

        if source.exists():
            try:
                shutil.copy2(source, dest)
                print(f"  ✅ Скопировано: {description} ({filename})")
                # Удаляем дубликат из _internal чтобы не занимал место
                source.unlink()
                print(f"  🗑️  Удален дубликат из _internal: {filename}")
                copied_count += 1
            except Exception as e:
                print(f"  ⚠️  Ошибка копирования {filename}: {e}")
        else:
            # Проверим может файл уже в корне
            if dest.exists():
                print(f"  ✅ Уже на месте: {description} ({filename})")
                copied_count += 1
            else:
                print(f"  ⚠️  Файл не найден: {filename}")

    print(f"\n  📦 Скопировано файлов: {copied_count}/{len(files_to_copy)}")
    return copied_count == len(files_to_copy)


def check_result():
    """Проверка результата сборки"""
    print_step(6, "Проверка результата")

    exe_path = Path('dist/Ctrl-System/Ctrl-System.exe')

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Исполняемый файл создан!")
        print(f"  📍 Путь: {exe_path.absolute()}")
        print(f"  📏 Размер: {size_mb:.1f} MB")

        # Проверка модели в корне
        model_path = Path('dist/Ctrl-System/best_model.pth')
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Модель на месте ({model_size_mb:.0f} MB)")
        else:
            print(f"  ⚠️  Предупреждение: Файл модели не найден в сборке!")

        # Проверка иконки
        icon_path = Path('dist/Ctrl-System/icon.ico')
        if icon_path.exists():
            print(f"  ✅ Иконка на месте")

        return True
    else:
        print(f"  ❌ Исполняемый файл не найден!")
        print(f"  Ожидался по пути: {exe_path.absolute()}")
        return False


def print_instructions():
    """Вывод инструкций по использованию"""
    print_header("📋 ИНСТРУКЦИИ")

    print("✅ Сборка завершена успешно!\n")
    print("📂 Готовое приложение находится в:")
    print("   dist/Ctrl-System/\n")
    print("🚀 Для запуска:")
    print("   1. Перейдите в папку dist/Ctrl-System/")
    print("   2. Запустите Ctrl-System.exe\n")
    print("📦 Для распространения:")
    print("   - Скопируйте ВСЮ папку dist/Ctrl-System/")
    print("   - Можно заархивировать в .zip")
    print("   - Приложение работает БЕЗ установленного Python\n")
    print("⚠️  ВАЖНО:")
    print("   - Не удаляйте файлы из папки Ctrl-System/")
    print("   - Файл best_model.pth должен быть в той же папке что и .exe")
    print("   - Все библиотеки (PyTorch, OpenCV) уже включены\n")


def main():
    """Главная функция сборки"""
    print_header("🔨 СБОРКА CTRL-SYSTEM В .EXE")

    # Проверка текущей директории
    if not os.path.exists('main.py'):
        print("❌ ОШИБКА: Запустите скрипт из корня проекта Ctrl-System!")
        print("   Текущая директория:", os.getcwd())
        sys.exit(1)

    # Шаг 1: Проверка файлов
    if not check_files():
        sys.exit(1)

    # Шаг 2: Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)

    # Шаг 3: Очистка
    clean_build_dirs()

    # Шаг 4: Сборка
    if not build_executable():
        print("\n❌ Сборка завершилась с ошибками!")
        sys.exit(1)

    # Шаг 5: Копирование файлов
    copy_required_files()

    # Шаг 6: Проверка результата
    if not check_result():
        print("\n❌ Не удалось найти собранный .exe файл!")
        sys.exit(1)

    # Инструкции
    print_instructions()

    print("=" * 70)
    print("🎉 ГОТОВО!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Сборка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
