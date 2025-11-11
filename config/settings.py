"""
Настройки приложения
"""

# Параметры обработки по умолчанию
DEFAULT_NUM_ZONES = 5
DEFAULT_MIN_AREA_RATIO = 0.3
DEFAULT_MAX_AREA_RATIO = 0.95
DEFAULT_STD_RANGE = 1.0
DEFAULT_DETECTION_THRESHOLD = 0.1
DEFAULT_MODEL_SIZE = 512

# Пути к файлам
MODEL_PATH = "best_model.pth"

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

# Параметры предобработки
BILATERAL_FILTER_D = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75

# Цвета визуализации
COLOR_DEFECT = (0, 0, 255)  # Красный для дефектов
COLOR_ZONE_BORDER = (0, 255, 0)  # Зеленый для границ зон
DEFECT_OVERLAY_ALPHA = 0.5