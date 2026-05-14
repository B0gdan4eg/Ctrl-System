"""
Настройки приложения
"""

# Параметры обработки по умолчанию
DEFAULT_NUM_ZONES = 5
DEFAULT_MIN_AREA_RATIO = 0.3
DEFAULT_MAX_AREA_RATIO = 0.95
DEFAULT_STD_RANGE = 3.0
DEFAULT_DETECTION_THRESHOLD = 0.01
DEFAULT_MODEL_SIZE = 256

# Пути к файлам
MODEL_PATH = "best_model.pth"
ZONE_MODEL_PATH = "models/zone_unet.pth"  # U-Net сегментация зон фильтра
USE_AI_ZONE_DETECTION = True              # True = нейросеть, фолбэк — классика

# UI настройки
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
MIN_IMAGE_VIEWER_SIZE = 400

# Параметры модели
MODEL_IN_CHANNELS = 3
MODEL_OUT_CHANNELS = 1
MODEL_FEATURES = [64, 128, 256, 512]

# Параметры зон
CLIPL_LIMIT_VALUES = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
THRESHOLD_VALUES = [30, 25, 20, 15, 10]

# Moved to archive/unused_settings.py

# Параметры калибровки (размер пикселя)
PIXEL_SIZE_MM = 0.0074          # Размер одного пикселя в миллиметрах (JAI RM-4200GE: 7.4 μm)  # = JAI_RM4200GE_PIXEL_SIZE_UM / 1000

# Параметры предобработки
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75

# Цвета визуализации
COLOR_DEFECT = (0, 0, 255)  # Красный для дефектов
COLOR_ZONE_BORDER = (0, 255, 0)  # Зеленый для границ зон
DEFECT_OVERLAY_ALPHA = 0.5

# eBUS SDK (Pleora) - для JAI GigE камер
EBUS_SDK_PATH = r"C:\Program Files\Common Files\Pleora\eBUS SDK"
# Moved to archive/unused_settings.py