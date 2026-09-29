"""
Настройки приложения

Нейросетевые параметры (модель, зоны, пороги детекции U-Net) архивированы
вместе с legacy-приложением — см. archive/legacy_unet_app/. Текущая версия
работает по расчёту энергетики/яркости (scripts/run_pipeline.py).
"""

# UI настройки
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
MIN_IMAGE_VIEWER_SIZE = 400

# Параметры калибровки (размер пикселя)
PIXEL_SIZE_MM = 0.0074          # Размер одного пикселя в миллиметрах (JAI RM-4200GE: 7.4 μm)  # = JAI_RM4200GE_PIXEL_SIZE_UM / 1000

# Базовая папка с образцами (стартовая для диалога «Выбрать папку детали»)
SPECIMEN_BASE_DIR = r"D:\Ctrl-System v2\WTS 05.08.2026"

# eBUS SDK (Pleora) - для JAI GigE камер
EBUS_SDK_PATH = r"C:\Program Files\Common Files\Pleora\eBUS SDK"